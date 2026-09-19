"""
Calibrate `kappa`, the ratio of kick spacing to snapshot spacing.

Why kappa needs a DIFFERENT experiment from tau_frac
----------------------------------------------------
Every other `auto` constant controls the SNAPSHOT grid, so it changes the density history and
therefore the wake computed at a given s -- which is why §6x and §11l could calibrate by comparing
a wake cut at one position. `kappa` changes none of that. The wake at s_i is identical however the
kicks are spaced. What kappa controls is the QUADRATURE of the kick along s: each kick carries an
arc `L_kick` and applies its wake across that whole interval as if W were constant there.

So a wake cut is exactly the wrong observable -- it is the thing kappa provably does not move. The
error kappa controls is in `integral of W ds`, which only appears in the BEAM: energy loss, induced
energy spread, emittance growth. This measures those, over the whole lattice, with CSR applied.

The experiment
--------------
* A gentle 0.1 rad bend and an UNSHEARED beam, chosen so `auto` returns 129 nodes at every kappa
  (verified in-run). The snapshot grid is therefore held FIXED and kappa is the only variable --
  the §6x waist case would have been useless here, because its collapsing frame forces sub-mm
  snapshot spacing for reasons that have nothing to do with kick quadrature.
* `kick_interval: midpoint`, since that is the O(h^2) rule §11d added and the one worth calibrating.
* The reference is `kappa` small enough that every snapshot node is a kick, i.e. the finest
  quadrature the fixed snapshot grid admits.
* `CSR_integration` at 100x100, which §11k measured converged (0.09 % against 300x300) and which
  costs 1.48x less than 200x200.

Cost. One wake mesh is ~19 s serial, ~3.4 s at 10 MPI ranks (measured 5.6x). The sweep is
~160 meshes, so it is run under MPI; serially it would be ~50 min of pure mesh time.

Run:
    mpirun -np 10 python test/test_kappa_calibrate.py
"""
import gc
import os
import sys
import time

import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from mpi4py import MPI
from pyDFCSR_2D.CSR import CSR2D

EXAMPLE_DIR = os.path.join(os.path.dirname(__file__), '..', 'example')
RESULT_DIR = os.path.join(os.path.dirname(__file__), 'benchmark_results', 'kappa')

DRIFT = 0.35
BEND_L, BEND_ANGLE = 1.0, 0.1
POST = 0.5
LAT_LEN = DRIFT + BEND_L + POST

KAPPAS = (32.0, 16.0, 8.0, 4.0, 2.0, 1.0)
REF_KAPPA = 0.0          # 0 => every snapshot node is a kick: the finest quadrature available

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()


def write_inputs(tag, kappa):
    """
    Rank 0 writes; everyone waits. Letting every rank write the same YAML is a race -- a rank can
    read a file another rank has truncated but not yet filled, which showed up as
    `TypeError: 'NoneType' object is not iterable` from parse_yaml returning None.
    """
    p = f'input/kap_config_{tag}.yaml'
    if RANK != 0:
        COMM.Barrier()
        return p

    beam = {
        'n_particle': 200000, 'species': 'electron',
        'px_dist': {'sigma_px': {'units': 'keV/c', 'value': 25.0}, 'type': 'gaussian'},
        'py_dist': {'sigma_py': {'units': 'keV/c', 'value': 25.0}, 'type': 'gaussian'},
        'pz_dist': {'avg_pz': {'units': 'GeV/c', 'value': 5},
                    'sigma_pz': {'units': 'MeV/c', 'value': 0.5}, 'type': 'gaussian'},
        'x_dist': {'sigma_x': {'units': 'um', 'value': 50}, 'type': 'gaussian'},
        'y_dist': {'sigma_y': {'units': 'um', 'value': 5}, 'type': 'gaussian'},
        'random': {'type': 'hammersley'},
        'start': {'tstart': {'units': 'sec', 'value': 0}, 'type': 'time'},
        'total_charge': {'units': 'nC', 'value': 1},
        'z_dist': {'avg_z': {'units': 'mm', 'value': 0},
                   'sigma_z': {'units': 'um', 'value': 50}, 'type': 'gaussian'},
    }
    bp = f'input/kap_beam_{tag}.yaml'
    with open(os.path.join(EXAMPLE_DIR, bp), 'w') as f:
        yaml.dump(beam, f, default_flow_style=False, sort_keys=False)

    lat = {'step_size': 0.05,
           'element_1': {'type': 'drift', 'L': DRIFT, 'nsep': 1},
           'element_2': {'type': 'dipole', 'L': BEND_L, 'angle': BEND_ANGLE, 'nsep': 1},
           'element_3': {'type': 'drift', 'L': POST, 'nsep': 1}}
    lp = f'input/kap_lat_{tag}.yaml'
    with open(os.path.join(EXAMPLE_DIR, lp), 'w') as f:
        yaml.dump(lat, f, default_flow_style=False, sort_keys=False)

    cfg = {
        'input_beam': {'style': 'distgen', 'distgen_input_file': bp},
        'input_lattice': {'lattice_input_file': lp},
        'particle_deposition': {'method': 'bspline_comoving',
                                'xbins': 128, 'zbins': 128,
                                'xlim': 5, 'zlim': 5, 'smoothing_sigma': 3.0,
                                'poly_degree': 1, 'velocity_threhold': 1000},
        # 100x100: §11k measured converged (0.09 % vs 300x300) at 1.48x less cost than 200x200
        'CSR_integration': {'n_formation_length': 1.5, 'zbins': 100, 'xbins': 100},
        'CSR_computation': {'compute_CSR': 1, 'apply_CSR': 1, 'transverse_on': 1,
                            'xbins': 21, 'zbins': 51, 'xlim': 3, 'zlim': 3,
                            'write_beam': [], 'write_wakes': False,
                            'write_name': f'kap_{tag}', 'workdir': './output'},
        'step_control': {'mode': 'auto', 'kick_interval': 'midpoint', 'n_sub': 400,
                         'kappa': float(kappa)},
    }
    with open(os.path.join(EXAMPLE_DIR, p), 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    COMM.Barrier()
    return p


def run_case(tag, kappa):
    """
    One full-lattice run with CSR applied. Cached, since these runs are the expensive part.

    The cache hit must be decided COLLECTIVELY. If rank 0 saw the file and others did not, the
    ranks would take different branches and deadlock on the barrier inside write_inputs.
    """
    cache = os.path.join(RESULT_DIR, f'stat_{tag}.npz')
    hit = COMM.bcast(os.path.exists(cache) if RANK == 0 else None, root=0)
    if hit:
        d = np.load(cache, allow_pickle=True)
        return d['stat'].item(), d['meta'].item()

    cfg = write_inputs(tag, kappa)
    t0 = time.time()
    csr = CSR2D(input_file=cfg, parallel=True)
    sch = csr.lattice.schedule
    csr.run(debug=True)
    st = csr.statistics
    s = np.asarray(sch.s_nodes[:len(st['sigma_z'])], float)
    stat = dict(s=s,
                sigma_z=np.array(st['sigma_z']),
                sigma_x=np.array(st['sigma_x']),
                sigma_energy=np.array(st['sigma_energy']),
                mean_energy=np.array(st['mean_energy']),
                norm_emit_x=np.array(st['twiss']['norm_emit_x']))
    m = dict(tag=tag, kappa=kappa, nodes=int(sch.n_nodes), snaps=int(sch.n_snap),
             kicks=int(sch.n_kick), seconds=time.time() - t0,
             arc_kicked=float(sch.total_arc_kicked()))
    if RANK == 0:
        os.makedirs(RESULT_DIR, exist_ok=True)
        np.savez(cache, stat=np.array(stat, dtype=object), meta=np.array(m, dtype=object))
    COMM.Barrier()
    del csr
    gc.collect()
    return stat, m


def main():
    os.chdir(EXAMPLE_DIR)
    lines = []

    def emit(t=''):
        if RANK == 0:
            print(t, flush=True)
            lines.append(t)

    emit('=' * 104)
    emit('Calibrating kappa: kick spacing against the INTEGRATED effect on the beam')
    emit('=' * 104)
    emit(f'unsheared beam, {BEND_ANGLE} rad bend, drift {DRIFT} m + bend {BEND_L} m + drift {POST} m')
    emit('CSR computed AND applied. kappa does not change the wake at a given s -- it changes the')
    emit('quadrature of integral W ds -- so the observables are beam quantities, not a wake cut.')
    emit(f'running on {COMM.Get_size()} MPI ranks')
    emit('')

    ref_stat, ref_m = run_case('ref', REF_KAPPA)
    emit(f"  reference kappa={REF_KAPPA:g} (every snapshot node is a kick): "
         f"{ref_m['snaps']} snapshots, {ref_m['kicks']} kicks, {ref_m['seconds']:.0f} s")
    emit(f"    kick intervals tile {ref_m['arc_kicked']:.9f} m of a {LAT_LEN:.9f} m lattice")
    emit('')

    rows = []
    for kap in KAPPAS:
        tag = 'k' + str(kap).replace('.', 'p')
        stat, m = run_case(tag, kap)
        assert m['snaps'] == ref_m['snaps'], (
            f"kappa={kap} changed the SNAPSHOT grid ({m['snaps']} vs {ref_m['snaps']}), so this "
            f"case does not isolate kick quadrature")
        rows.append((m, stat))
        emit(f"  ran kappa={kap:<6g} {m['kicks']:>4} kicks, {m['seconds']:>6.0f} s")
    emit('')

    def final(st, key):
        return float(st[key][-1])

    emit(f"  final beam at the lattice end, against the kappa={REF_KAPPA:g} reference")
    emit(f"    {'kappa':>7} {'kicks':>6} {'sec':>7} {'dE_mean MeV':>13} {'rel err':>9} "
         f"{'sig_E keV':>11} {'rel err':>9} {'emit_x nm':>11} {'rel err':>9}")
    e0 = final(ref_stat, 'mean_energy')
    dE0 = e0 - float(ref_stat['mean_energy'][0])
    sE0 = final(ref_stat, 'sigma_energy')
    ex0 = final(ref_stat, 'norm_emit_x')
    emit(f"    {REF_KAPPA:>7g} {ref_m['kicks']:>6} {ref_m['seconds']:>7.0f} "
         f"{dE0 / 1e6:>13.6f} {'reference':>9} {sE0 / 1e3:>11.4f} {'reference':>9} "
         f"{ex0 * 1e9:>11.5f} {'reference':>9}")
    for m, st in rows:
        dE = final(st, 'mean_energy') - float(st['mean_energy'][0])
        sE = final(st, 'sigma_energy')
        ex = final(st, 'norm_emit_x')
        emit(f"    {m['kappa']:>7g} {m['kicks']:>6} {m['seconds']:>7.0f} "
             f"{dE / 1e6:>13.6f} {abs(dE - dE0) / abs(dE0):>9.5f} "
             f"{sE / 1e3:>11.4f} {abs(sE - sE0) / abs(sE0):>9.5f} "
             f"{ex * 1e9:>11.5f} {abs(ex - ex0) / abs(ex0):>9.5f}")
    emit('')

    # ---- the decomposition that resolves the convergence-order question -------------------
    # Taking max|error| over s conflates TWO different errors with different orders:
    #
    #   1. a SAWTOOTH. Between kicks the beam gets no CSR at all, so sigma_E falls behind the
    #      continuously-kicked reference; then a kick lands carrying the whole arc L_kick at once
    #      and it overshoots. A staircase chasing a ramp. The ripple amplitude is how much sigma_E
    #      grows over one kick interval, so it is O(h) and NO centring can remove it -- it is
    #      intrinsic to applying a continuous process as discrete impulses.
    #
    #   2. an accumulated BIAS, the net quadrature error of int(W ds). This is what the midpoint
    #      rule fixes, and it is the one that is O(h^2).
    #
    # max|.| over s samples the ripple, so it reports (1) and hides (2). The signed mean cancels
    # the ripple and exposes (2). Measured before this was understood: max|.| gave order ~1.0 and
    # looked like it contradicted §11d's O(h^2) claim for midpoint intervals. It does not.
    #
    # Practical consequence: the observable for choosing kappa is the beam AFTER the bend, not the
    # worst point inside it -- a mid-bend sample catches the beam mid-sawtooth, which reflects
    # WHEN you looked rather than an error in the delivered beam.
    emit('  error decomposition through the bend: sawtooth (O(h)) vs accumulated bias (O(h^2))')
    emit(f"    {'kappa':>7} {'sawtooth pp':>12} {'order':>7} {'|mean bias|':>12} {'order':>7} "
         f"{'sign flips':>11}")
    s_ref = np.asarray(ref_stat['s'], float)
    prev = None
    for m, st in rows:
        a = np.asarray(st['sigma_energy'], float)
        b = np.asarray(ref_stat['sigma_energy'], float)
        n = min(a.size, b.size)
        e = (a[:n] - b[:n]) / np.maximum(np.abs(b[:n]), 1e-300)   # SIGNED, deliberately
        inside = ((s_ref[:n] > DRIFT + 0.01) & (s_ref[:n] < DRIFT + BEND_L - 0.01)
                  & np.isfinite(e))
        ev = e[inside]
        pp = float(np.nanmax(ev) - np.nanmin(ev))
        bias = abs(float(np.nanmean(ev)))
        flips = int(np.sum(np.diff(np.sign(ev)) != 0))
        if prev is None:
            o_pp, o_b = '-', '-'
        else:
            o_pp = f'{np.log2(prev[0] / pp):.2f}' if pp > 0 else '-'
            o_b = f'{np.log2(prev[1] / bias):.2f}' if bias > 0 else '-'
        emit(f"    {m['kappa']:>7g} {pp:>12.6f} {o_pp:>7} {bias:>12.6f} {o_b:>7} {flips:>11}")
        prev = (pp, bias)
    emit('    a sawtooth shows many sign flips and pp >> |mean|; a systematic bias would show')
    emit('    neither. Order 1 in the pp column and 2 in the bias column is the expected result.')
    emit('')

    # At the bend EXIT every kick inside the bend has landed, so the sawtooth has completed its
    # last cycle and what remains is the error actually carried forward.
    j_exit = int(np.flatnonzero(s_ref <= DRIFT + BEND_L)[-1])
    emit(f'  at the bend exit (s = {s_ref[j_exit]:.4f} m), after every in-bend kick has landed')
    emit(f"    {'kappa':>7} {'sigma_E err':>12} {'emit_x err':>11}  (compare to the worst-in-bend "
         f"values below)")
    for m, st in rows:
        line = []
        for key in ('sigma_energy', 'norm_emit_x'):
            a, b = np.asarray(st[key], float), np.asarray(ref_stat[key], float)
            line.append(abs(a[j_exit] - b[j_exit]) / max(abs(b[j_exit]), 1e-300))
        emit(f"    {m['kappa']:>7g} {line[0]:>12.5f} {line[1]:>11.5f}")
    emit('')

    # the whole trajectory, not just the endpoint: a coarse kappa can land the right final
    # number by cancelling errors along the way
    emit('  worst relative deviation ALONG the lattice (not just at the end)')
    emit(f"    {'kappa':>7} {'sigma_energy':>13} {'norm_emit_x':>12} {'sigma_z':>10} {'sigma_x':>10}")
    for m, st in rows:
        w = {}
        for key in ('sigma_energy', 'norm_emit_x', 'sigma_z', 'sigma_x'):
            a, b = np.asarray(st[key], float), np.asarray(ref_stat[key], float)
            n = min(a.size, b.size)
            d = np.abs(a[:n] - b[:n]) / np.maximum(np.abs(b[:n]), 1e-300)
            w[key] = float(np.nanmax(d))
        emit(f"    {m['kappa']:>7g} {w['sigma_energy']:>13.5f} {w['norm_emit_x']:>12.5f} "
             f"{w['sigma_z']:>10.5f} {w['sigma_x']:>10.5f}")
    emit('')

    emit('  cost')
    for m, st in rows:
        emit(f"    kappa={m['kappa']:<6g} {m['kicks'] / ref_m['kicks']:.3f}x the kicks, "
             f"{m['seconds'] / ref_m['seconds']:.3f}x the time")

    if RANK != 0:
        return

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    for ax, key, lab, sc in ((axes[0, 0], 'sigma_energy', 'sigma_E  [keV]', 1e-3),
                             (axes[0, 1], 'norm_emit_x', 'norm emit_x  [nm]', 1e9),
                             (axes[0, 2], 'sigma_z', 'sigma_z  [um]', 1e6)):
        ax.plot(ref_stat['s'], np.asarray(ref_stat[key]) * sc, 'k-', lw=2.4,
                label=f'kappa={REF_KAPPA:g} (ref)')
        for m, st in rows:
            ax.plot(st['s'], np.asarray(st[key]) * sc, lw=1.0, label=f"kappa={m['kappa']:g}")
        ax.axvspan(DRIFT, DRIFT + BEND_L, color='C1', alpha=0.10)
        ax.set_xlabel('s  [m]')
        ax.set_ylabel(lab)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    axes[0, 0].set_title('induced energy spread (shaded = dipole)')
    axes[0, 1].set_title('emittance growth')
    axes[0, 2].set_title('bunch length')

    kap = np.array([m['kappa'] for m, _ in rows])
    for ax, key, lab in ((axes[1, 0], 'sigma_energy', 'sigma_E'),
                         (axes[1, 1], 'norm_emit_x', 'norm emit_x')):
        fin = np.array([abs(float(st[key][-1]) - float(ref_stat[key][-1]))
                        / abs(float(ref_stat[key][-1])) for _, st in rows])
        wor = []
        for _, st in rows:
            a, b = np.asarray(st[key], float), np.asarray(ref_stat[key], float)
            n = min(a.size, b.size)
            wor.append(float(np.nanmax(np.abs(a[:n] - b[:n])
                                       / np.maximum(np.abs(b[:n]), 1e-300))))
        ax.loglog(kap, np.maximum(fin, 1e-12), 'o-', label='at the lattice end')
        ax.loglog(kap, np.maximum(wor, 1e-12), 's--', label='worst along s')
        ax.loglog(kap, fin[-1] * (kap / kap[-1]) ** 2, ':', color='0.6',
                  label='O(kappa^2) for reference')
        ax.invert_xaxis()
        ax.set_xlabel('kappa  (finer ->)')
        ax.set_ylabel(f'rel error in {lab}')
        ax.set_title(f'{lab} against kappa')
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3, which='both')

    ax = axes[1, 2]
    kicks = np.array([m['kicks'] for m, _ in rows], float)
    fin = np.array([abs(float(st['sigma_energy'][-1]) - sE0) / abs(sE0) for _, st in rows])
    ax.loglog(kicks, np.maximum(fin, 1e-12), 'o-')
    for k, (m, _) in enumerate(rows):
        ax.annotate(f"{m['kappa']:g}", (kicks[k], max(fin[k], 1e-12)), fontsize=7,
                    textcoords='offset points', xytext=(4, 4))
    ax.axvline(ref_m['kicks'], color='k', ls='--', lw=1,
               label=f"reference: {ref_m['kicks']} kicks")
    ax.set_xlabel('kicks over the lattice')
    ax.set_ylabel('rel error in sigma_E at the end')
    ax.set_title('cost against accuracy')
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, which='both')

    fig.tight_layout()
    os.makedirs(RESULT_DIR, exist_ok=True)
    fig.savefig(os.path.join(RESULT_DIR, 'kappa_calibrate.png'), dpi=130)
    with open(os.path.join(RESULT_DIR, 'kappa_log.txt'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'\nwrote {RESULT_DIR}/kappa_calibrate.png')


if __name__ == '__main__':
    main()
