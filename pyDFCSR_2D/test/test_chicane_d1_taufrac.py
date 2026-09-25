"""
Is the D1 tail roughness a `tau` sampling error? Refine `tau_frac` and see.

§11q found the wake in the chicane's first long drift (D1) rough, and §11r-diagnostics traced it to
the bunch TAILS (100 % of the second-difference norm at s = 5.464 comes from |z| > 1.5 sigma_z, none
from the core) with roughness correlating with the tilt amplification `sigma_x/sigma_xi` at **0.904**
within D1 alone. Amplification sweeps 2.1 -> 64 through that drift.

That correlation suggests a specific mechanism, and it is falsifiable. §6r established that a `tau`
error displaces the integration band CENTRE by `dtau * (z_ret - z_bar)` while the band half-width is
only `margin * xlim * sigma_xi`. Two consequences:

  1. the displacement grows with |z - z_bar|, so it is WORST IN THE TAILS -- which is exactly where
     the roughness is;
  2. it is controlled by `tau_frac`, which bounds the band-centre drift per step.

So if the mechanism is right, **refining `tau_frac` must reduce the tail roughness specifically**,
and reduce it more than the core roughness. If tail roughness is flat against `tau_frac`, the
mechanism is wrong and the correlation with amplification was coincidental -- D1 and D3 would then
both need another explanation.

This is a genuine test rather than a demonstration: §11l measured the wake to be FLAT against
`tau_frac` over a 32x range at amplifications of 20x and 109x, which argues the answer here will be
"no change". But §11l measured a mid-x wake CUT on a single-dipole lattice, not tail structure in a
dispersive chicane drift at 64x, so it does not settle this.

Cost: tracking stops at s = 5.5 m (the end of D1), so only 28-86 kicks per rung rather than 105-302.
"""
import gc
import os
import sys
import time

import matplotlib
import numpy as np
import yaml

matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from mpi4py import MPI
from pyDFCSR_2D.CSR import CSR2D

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
PARALLEL = COMM.Get_size() > 1

EXAMPLE_DIR = os.path.join(os.path.dirname(__file__), '..', 'example')
RESULT_DIR = os.path.join(os.path.dirname(__file__), 'benchmark_results', 'chicane_d1_tau')

TAU_FRACS = (2.0, 1.0, 0.5, 0.25)
S_END = 5.5                 # end of D1; B2 starts at 5.6060
# Observation points through D1, chosen to span the amplification sweep. Pinned with force_nodes so
# every rung is compared at the same s -- §11o measured that an unpinned point is itself an error.
S_OBS = (3.0, 4.5, 5.5)
MESH_XBINS, MESH_ZBINS = 10, 30
DEP_BINS = 300
TAIL_SIGMA = 1.5            # |z| beyond this counts as tail, as in the §11q decomposition


def rough_parts(w, z, sigma_z):
    """
    Second-difference norm split into core and tail contributions.

    Returned relative to the curve's own amplitude, so it is a shape measure (§11q), and split at
    |z| = TAIL_SIGMA because that is where the §11q decomposition located the roughness.
    """
    w = np.asarray(w, float)
    amp = np.abs(w).max()
    if amp <= 0:
        return 0.0, 0.0, 0.0
    d2 = w[2:] - 2.0 * w[1:-1] + w[:-2]
    zc = np.abs(z[1:-1]) / max(sigma_z, 1e-30)
    tail = zc > TAIL_SIGMA
    tot = float(np.linalg.norm(d2) / amp)
    core_n = float(np.linalg.norm(d2[~tail]) / amp)
    tail_n = float(np.linalg.norm(d2[tail]) / amp)
    return tot, core_n, tail_n


def write_config(tau_frac):
    tag = f'tf{tau_frac:g}'.replace('.', 'p')
    p = f'input/d1tau_{tag}.yaml'
    if RANK != 0:
        COMM.Barrier()
        return p
    cfg = {
        'input_beam': {'style': 'distgen',
                       'distgen_input_file': 'input/chicane_init_beam.yaml'},
        'input_lattice': {'lattice_input_file': 'input/chicane_lattice.yaml'},
        'particle_deposition': {'method': 'bspline_comoving',
                                'xbins': DEP_BINS, 'zbins': DEP_BINS,
                                'xlim': 5, 'zlim': 5, 'smoothing_sigma': 3.0,
                                'poly_degree': 1, 'velocity_threhold': 1000},
        'CSR_integration': {'n_formation_length': 1, 'zbins': 100, 'xbins': 100},
        'CSR_computation': {'compute_CSR': 1, 'apply_CSR': 1, 'transverse_on': 1,
                            'xbins': MESH_XBINS, 'zbins': MESH_ZBINS, 'xlim': 3, 'zlim': 3,
                            'write_beam': [], 'write_wakes': False,
                            'write_name': f'd1tau_{tag}', 'workdir': './output'},
        'step_control': {'mode': 'auto', 'kick_interval': 'midpoint', 'n_sub': 400,
                         'tau_frac': float(tau_frac),
                         'force_nodes': [float(x) for x in S_OBS]},
    }
    with open(os.path.join(EXAMPLE_DIR, p), 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    if PARALLEL:
        COMM.Barrier()
    return p


def run_rung(tau_frac):
    tag = f'tf{tau_frac:g}'.replace('.', 'p')
    cache = os.path.join(RESULT_DIR, f'cut_{tag}.npz')
    hit = COMM.bcast(os.path.exists(cache) if RANK == 0 else None, root=0) if PARALLEL \
        else os.path.exists(cache)
    if hit:
        d = np.load(cache, allow_pickle=True)
        return d['cuts'].item(), d['meta'].item()

    cfg = write_config(tau_frac)
    cuts, meta = {}, dict(tau_frac=tau_frac)
    t0 = time.time()
    for s_obs in S_OBS:
        csr = CSR2D(input_file=cfg, parallel=PARALLEL)
        sch = csr.lattice.schedule
        meta['nodes'] = int(sch.n_nodes)
        csr.run(stop_time=s_obs, debug=False)
        assert abs(csr.beam.position - s_obs) < 1e-9, \
            f'tau_frac {tau_frac} landed {csr.beam.position} not {s_obs}'
        csr.get_CSR_mesh()
        if PARALLEL:
            csr.calculate_2D_CSR_parallel()
            dE = csr.dE_dct.copy()
        else:
            csr.calculate_2D_CSR()
            dE = csr.dE_dct.copy()
        nx, nz = MESH_XBINS, MESH_ZBINS
        b = csr.beam
        cuts[s_obs] = dict(dE=dE.reshape(nx, nz),
                           zz=csr.CSR_zmesh.reshape(nx, nz),
                           sigma_z=float(b._sigma_z),
                           sigma_xi=float(b._sigma_x_transform),
                           sigma_x=float(b._sigma_x),
                           tau=float(b._slope[0]),
                           snaps=len(csr.DF_tracker.time_log))
        meta[f'kicks_{s_obs}'] = int(sch.is_kick[sch.s_nodes <= s_obs + 1e-12].sum())
        del csr
        gc.collect()
    meta['seconds'] = time.time() - t0
    if RANK == 0:
        os.makedirs(RESULT_DIR, exist_ok=True)
        np.savez(cache, cuts=np.array(cuts, dtype=object), meta=np.array(meta, dtype=object))
    if PARALLEL:
        COMM.Barrier()
    return cuts, meta


def main():
    os.chdir(EXAMPLE_DIR)
    lines = []

    def emit(t=''):
        if RANK == 0:
            print(t, flush=True)
            lines.append(t)

    emit('=' * 100)
    emit('Is the D1 tail roughness a tau sampling error? Refining tau_frac to find out')
    emit('=' * 100)
    emit('chicane D1 (the 5 m dispersive drift after B1), tracking to s = 5.5 m')
    emit(f'roughness split at |z| = {TAIL_SIGMA} sigma_z, as in the §11q decomposition')
    emit('PREDICTION if the mechanism is a tau error: tail roughness falls with tau_frac,')
    emit('and falls more than core roughness. If it is flat, the mechanism is wrong.')
    emit('')

    rows = []
    for tf in TAU_FRACS:
        cuts, meta = run_rung(tf)
        rows.append((tf, cuts, meta))
        emit(f"  ran tau_frac={tf:<5g} {meta['nodes']:>5} nodes total, "
             f"{meta['seconds']:>5.0f} s")
    emit('')

    for s_obs in S_OBS:
        c0 = rows[0][1][s_obs]
        amp = c0['sigma_x'] / max(c0['sigma_xi'], 1e-30)
        emit(f'  s = {s_obs} m   sigma_z {c0["sigma_z"] * 1e6:.1f} um, '
             f'sigma_xi {c0["sigma_xi"] * 1e6:.1f} um, tau {c0["tau"]:+.3f}, '
             f'amplification {amp:.1f}x')
        emit(f"    {'tau_frac':>9} {'kicks':>6} {'snaps':>6} {'|dE| peak':>11} "
             f"{'rough tot':>10} {'core':>8} {'TAIL':>8}")
        for tf, cuts, meta in rows:
            c = cuts[s_obs]
            mid = c['dE'][c['dE'].shape[0] // 2, :]
            zmid = c['zz'][c['zz'].shape[0] // 2, :]
            tot, core, tail = rough_parts(mid, zmid, c['sigma_z'])
            emit(f"    {tf:>9g} {meta[f'kicks_{s_obs}']:>6} {c['snaps']:>6} "
                 f"{np.abs(mid).max():>11.5f} {tot:>10.4f} {core:>8.4f} {tail:>8.4f}")
        emit('')

    # verdict
    emit('  VERDICT')
    for s_obs in S_OBS:
        t_first = rough_parts(rows[0][1][s_obs]['dE'][MESH_XBINS // 2, :],
                              rows[0][1][s_obs]['zz'][MESH_XBINS // 2, :],
                              rows[0][1][s_obs]['sigma_z'])[2]
        t_last = rough_parts(rows[-1][1][s_obs]['dE'][MESH_XBINS // 2, :],
                             rows[-1][1][s_obs]['zz'][MESH_XBINS // 2, :],
                             rows[-1][1][s_obs]['sigma_z'])[2]
        chg = (t_last - t_first) / t_first if t_first > 0 else float('nan')
        emit(f'    s = {s_obs}: tail roughness {t_first:.4f} -> {t_last:.4f} over '
             f'tau_frac {TAU_FRACS[0]:g} -> {TAU_FRACS[-1]:g}  ({chg * 100:+.0f} %)')
    emit('')

    if RANK != 0:
        return
    os.makedirs(RESULT_DIR, exist_ok=True)

    fig, axes = plt.subplots(2, len(S_OBS), figsize=(6 * len(S_OBS), 8.5))
    for c, s_obs in enumerate(S_OBS):
        ax = axes[0, c]
        for tf, cuts, _ in rows:
            cc = cuts[s_obs]
            mid = cc['dE'][cc['dE'].shape[0] // 2, :]
            zmid = cc['zz'][cc['zz'].shape[0] // 2, :]
            ax.plot(zmid / cc['sigma_z'], mid, lw=1.2, label=f'tau_frac={tf:g}')
        for sgn in (-1, 1):
            ax.axvline(sgn * TAIL_SIGMA, color='0.6', ls=':', lw=1)
        amp = cuts[s_obs]['sigma_x'] / max(cuts[s_obs]['sigma_xi'], 1e-30)
        ax.set_title(f's = {s_obs} m, amplification {amp:.0f}x', fontsize=9)
        ax.set_xlabel('z / sigma_z', fontsize=8)
        ax.set_ylabel('dE/ds  [MeV/m]  (mid-x row)', fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

        ax = axes[1, c]
        tfs = np.array([r[0] for r in rows])
        tots, cores, tails = [], [], []
        for tf, cuts_, _ in rows:
            cc = cuts_[s_obs]
            t, co, ta = rough_parts(cc['dE'][cc['dE'].shape[0] // 2, :],
                                    cc['zz'][cc['zz'].shape[0] // 2, :], cc['sigma_z'])
            tots.append(t)
            cores.append(co)
            tails.append(ta)
        ax.loglog(tfs, np.maximum(tots, 1e-6), 'o-', label='total')
        ax.loglog(tfs, np.maximum(cores, 1e-6), 's--', label=f'core |z|<{TAIL_SIGMA}')
        ax.loglog(tfs, np.maximum(tails, 1e-6), '^-', lw=2, label=f'TAIL |z|>{TAIL_SIGMA}')
        ax.invert_xaxis()
        ax.set_xlabel('tau_frac  (finer ->)', fontsize=8)
        ax.set_ylabel('relative second-difference norm', fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3, which='both')
    fig.suptitle('D1 tail roughness against tau_frac: does refining the tilt sampling help?',
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    fig.savefig(os.path.join(RESULT_DIR, 'chicane_d1_taufrac.png'), dpi=130)
    plt.close(fig)
    emit('  wrote chicane_d1_taufrac.png')

    with open(os.path.join(RESULT_DIR, 'chicane_d1_taufrac_log.txt'), 'w') as f:
        f.write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
