"""
x-z wake maps through a whole dipole, for x-z shear = 0, 10, 20, 50, with the CURRENT code.

Why this is worth redoing
------------------------
Every x-z map in this work (§6f, §6s, §6u, §6y) predates the calibrated scheduler, and most were
computed at a uniform `step_size = 0.05 m`. §6x measured that this is **79 % wrong** at a waist.
The shear cases each put a waist at a different place, needing wildly different step sizes:

    shear  0   no waist in the dipole
    shear 10   waist 0.099 m in, width 9.79 mm  -> needs step <= 0.0049 m
    shear 20   waist 0.050 m in, width 2.39 mm  -> needs step <= 0.0012 m
    shear 50   waist 0.020 m in, width 0.34 mm  -> needs step <= 0.00017 m

A single uniform step that serves shear 50 would need ~10^4 steps over the lattice. This is exactly
what `mode: auto` exists for, so these maps are computed with it -- and the schedule it picks is
reported per shear, because "the map is trustworthy" is a claim that rests on it:

    shear  0    92 nodes     shear 20   319 nodes
    shear 10   203 nodes     shear 50  1037 nodes, min h 1.95e-04 m, history <= 0.67 GB

Observation points, chosen in units of the TRANSIENT scales rather than round numbers. For this
dipole (R = 1 m, phi = 1 rad, sigma_z = 50 um) §11h's two scales are
`L_entrance = (24 R^2 5 sigma_z)^(1/3) = 0.1817 m` and `L_exit = R phi/2 = 0.5 m`:

    s_dip 0.02   0.11 L_entrance   deep in the entrance transient
    s_dip 0.06   0.33 L_entrance   transient building
    s_dip 0.18   0.99 L_entrance   one overtaking length: transient essentially complete
    s_dip 0.60   3.30 L_entrance   steady state
    exit +0.05   0.10 L_exit       just outside, decay beginning
    exit +0.25   0.50 L_exit       amplitude halved (Stupakov & Emma Eq. 10)

`force_nodes` pins every one of them exactly, so the same s is compared across shears -- §11o
measured the entrance wake varying 62 % over 5.6 mm, so an unpinned observation point is itself a
several-percent error.

Colour scales are PER PANEL, so every map reaches saturation and its shape is readable. The wake
spans roughly four orders of magnitude between the entrance transient and the exit decay
(dE peak 20.7 MeV/m at shear 50 just inside the entrance, 0.005 MeV/m at exit + 0.25 m), so any
shared scale leaves most panels blank. The trade-off is that amplitudes cannot be compared by eye,
so each panel is annotated with the peak it is normalised to, and `shear_xz_amplitude.png` plots
peak amplitude against s for the amplitude comparison.
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
from pyDFCSR_2D.waist import scan_waists, sigma_from_coords

EXAMPLE_DIR = os.path.join(os.path.dirname(__file__), '..', 'example')
RESULT_DIR = os.path.join(os.path.dirname(__file__), 'benchmark_results', 'shear_xz')

DRIFT, BEND_L, BEND_ANGLE, POST = 0.35, 1.0, 1.0, 0.5
S_EXIT = DRIFT + BEND_L
SHEARS = (0.0, 10.0, 20.0, 50.0)
DEP_BINS = 128
MESH_XBINS, MESH_ZBINS = 21, 51

R_BEND = BEND_L / BEND_ANGLE
L_ENTRANCE = (24.0 * R_BEND ** 2 * 5.0 * 50e-6) ** (1.0 / 3.0)
L_EXIT = 0.5 * R_BEND * BEND_ANGLE

# (absolute s, short label, longer description)
POINTS = [
    (DRIFT + 0.02, 'entrance +0.02', f'{0.02 / L_ENTRANCE:.2f} L_ent, deep in the transient'),
    (DRIFT + 0.06, 'entrance +0.06', f'{0.06 / L_ENTRANCE:.2f} L_ent, building'),
    (DRIFT + 0.18, 'entrance +0.18', f'{0.18 / L_ENTRANCE:.2f} L_ent, transient complete'),
    (DRIFT + 0.60, 'mid-bend +0.60', f'{0.60 / L_ENTRANCE:.2f} L_ent, steady state'),
    (S_EXIT + 0.05, 'exit +0.05', f'{0.05 / L_EXIT:.2f} L_exit, decay begins'),
    (S_EXIT + 0.25, 'exit +0.25', f'{0.25 / L_EXIT:.2f} L_exit, amplitude halved'),
]


def write_inputs(shear):
    """Rank 0 writes, everyone waits -- concurrent writers race, see §11m."""
    tag = f's{shear:g}'
    p = f'input/sxz_config_{tag}.yaml'
    if RANK != 0:
        COMM.Barrier()          # only reached when PARALLEL, since RANK is 0 otherwise
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
    if shear:
        beam['transforms'] = {'s1': {'shear_coefficient':
                                     {'units': 'dimensionless', 'value': float(shear)},
                                     'type': 'shear z:x'}}
    bp = f'input/sxz_beam_{tag}.yaml'
    with open(os.path.join(EXAMPLE_DIR, bp), 'w') as f:
        yaml.dump(beam, f, default_flow_style=False, sort_keys=False)

    lat = {'step_size': 0.05,
           'element_1': {'type': 'drift', 'L': DRIFT, 'nsep': 1},
           'element_2': {'type': 'dipole', 'L': BEND_L, 'angle': BEND_ANGLE, 'nsep': 1},
           'element_3': {'type': 'drift', 'L': POST, 'nsep': 1}}
    lp = f'input/sxz_lat_{tag}.yaml'
    with open(os.path.join(EXAMPLE_DIR, lp), 'w') as f:
        yaml.dump(lat, f, default_flow_style=False, sort_keys=False)

    cfg = {
        'input_beam': {'style': 'distgen', 'distgen_input_file': bp},
        'input_lattice': {'lattice_input_file': lp},
        'particle_deposition': {'method': 'bspline_comoving',
                                'xbins': DEP_BINS, 'zbins': DEP_BINS,
                                'xlim': 5, 'zlim': 5, 'smoothing_sigma': 3.0,
                                'poly_degree': 1, 'velocity_threhold': 1000},
        'CSR_integration': {'n_formation_length': 1.5, 'zbins': 100, 'xbins': 100},
        'CSR_computation': {'compute_CSR': 0, 'apply_CSR': 0, 'transverse_on': 1,
                            'xbins': MESH_XBINS, 'zbins': MESH_ZBINS, 'xlim': 3, 'zlim': 3,
                            'write_beam': [], 'write_wakes': False,
                            'write_name': f'sxz_{tag}', 'workdir': './output'},
        # auto, because no single uniform step serves shear 0 and shear 50 (see the docstring)
        'step_control': {'mode': 'auto', 'kick_interval': 'midpoint', 'n_sub': 400,
                         'force_nodes': [float(s) for s, _, _ in POINTS]},
    }
    with open(os.path.join(EXAMPLE_DIR, p), 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    if PARALLEL:
        COMM.Barrier()
    return p


def schedule_profile(shear):
    """
    The linear-optics frame along the lattice plus the realised node positions.

    Taken from the schedule the run will actually use, so the plotted nodes ARE the nodes, not a
    reconstruction. Costs nothing -- no tracking, just the pre-computed scan.
    """
    # Do NOT call write_inputs here: it contains a collective barrier, and this runs after the
    # `RANK != 0: return` guard, so rank 0 would wait alone for ever. The config file already
    # exists by this point.
    cfg = f'input/sxz_config_s{shear:g}.yaml'
    csr = CSR2D(input_file=cfg)
    sch = csr.lattice.schedule
    out = dict(s_scan=np.asarray(sch.auto_s_scan, float),
               h_eff=np.asarray(sch.auto_h_eff, float),
               s_nodes=np.asarray(sch.s_nodes, float),
               kick_s=np.asarray(sch.s_nodes[sch.is_kick], float))
    from pyDFCSR_2D.waist import propagate_frame
    b = csr.beam
    sigma0 = sigma_from_coords(b.x, b.px, b.particle.y, b.particle.py, b.z, b.pz)
    s, sz, sx, tau, sxi, rho = propagate_frame(csr.lattice.lattice_config, sigma0, n_sub=400)
    out.update(fs=s, sz=sz, sx=sx, sxi=sxi, tau=tau)
    del csr
    gc.collect()
    return out


def run_shear(shear):
    """All observation points for one shear. One fresh CSR2D per point: run() is not resumable."""
    tag = f's{shear:g}'
    cache = os.path.join(RESULT_DIR, f'maps_{tag}.npz')
    hit = COMM.bcast(os.path.exists(cache) if RANK == 0 else None, root=0) if PARALLEL \
        else os.path.exists(cache)
    if hit:
        d = np.load(cache, allow_pickle=True)
        return d['maps'].item(), d['meta'].item()

    cfg = write_inputs(shear)
    maps, meta = {}, dict(shear=shear)
    t0 = time.time()
    for s_obs, label, _ in POINTS:
        csr = CSR2D(input_file=cfg, parallel=PARALLEL)
        sch = csr.lattice.schedule
        meta['nodes'] = int(sch.n_nodes)
        meta['kicks'] = int(sch.n_kick)
        meta['min_h'] = float(sch.auto_h_eff.min())
        csr.run(stop_time=s_obs, debug=True)
        assert abs(csr.beam.position - s_obs) < 1e-9, \
            f'shear {shear} landed {csr.beam.position} not {s_obs}'
        csr.get_CSR_mesh()
        b = csr.beam
        nx, nz = MESH_XBINS, MESH_ZBINS
        if PARALLEL:
            csr.calculate_2D_CSR_parallel()      # splits the mesh across ranks
            dE, xk = csr.dE_dct.copy(), csr.x_kick.copy()
        else:
            dE = np.zeros(nx * nz)
            xk = np.zeros(nx * nz)
            for i in range(nx * nz):
                dE[i], xk[i] = csr.get_CSR_wake(b.position + csr.CSR_zmesh[i], csr.CSR_xmesh[i])
        assert np.all(np.isfinite(dE)) and np.all(np.isfinite(xk)), \
            f'shear {shear} at s={s_obs}: non-finite wake'
        zz = csr.CSR_zmesh.reshape(nx, nz)
        xx = csr.CSR_xmesh.reshape(nx, nz)
        maps[s_obs] = dict(dE=dE.reshape(nx, nz), xk=xk.reshape(nx, nz),
                           zz=zz, xx=xx, xt=xx - np.polyval(b.slope, zz),
                           sigma_z=float(b._sigma_z), sigma_xi=float(b._sigma_x_transform),
                           sigma_x=float(b._sigma_x), tau=float(b._slope[0]),
                           snapshots=len(csr.DF_tracker.time_log))
        del csr
        gc.collect()
    meta['seconds'] = time.time() - t0
    if RANK == 0:
        os.makedirs(RESULT_DIR, exist_ok=True)
        np.savez(cache, maps=np.array(maps, dtype=object), meta=np.array(meta, dtype=object))
    if PARALLEL:
        COMM.Barrier()
    return maps, meta


def main():
    os.makedirs(RESULT_DIR, exist_ok=True)
    os.chdir(EXAMPLE_DIR)
    lines = []

    def emit(t=''):
        if RANK == 0:
            print(t, flush=True)
            lines.append(t)

    emit('=' * 108)
    emit('x-z wake maps through a dipole vs x-z shear, with the calibrated auto scheduler')
    emit('=' * 108)
    emit(f'dipole R = {R_BEND:g} m, phi = {BEND_ANGLE:g} rad; drift {DRIFT} + bend {BEND_L} '
         f'+ drift {POST} m')
    emit(f'transient scales: L_entrance = {L_ENTRANCE:.4f} m, L_exit = {L_EXIT:.4f} m')
    emit(f'deposition {DEP_BINS}^2, mesh {MESH_XBINS}x{MESH_ZBINS}, all points pinned with '
         f'force_nodes')
    emit('')

    # where the waist is, per shear -- this is why auto is needed
    emit('  the waist each shear puts inside the dipole (linear optics)')
    emit(f"    {'shear':>6} {'waist s':>9} {'into bend':>10} {'width mm':>9} {'needs step <=':>14}")
    for sh in SHEARS:
        cfg = write_inputs(sh)
        csr = CSR2D(input_file=cfg)
        b = csr.beam
        rep = scan_waists(csr.lattice.lattice_config,
                          sigma_from_coords(b.x, b.px, b.particle.y, b.particle.py, b.z, b.pz),
                          0.05)
        ws = [w for w in rep['waists'] if DRIFT <= w['s'] <= S_EXIT]
        if ws:
            emit(f"    {sh:>6g} {ws[0]['s']:>9.4f} {ws[0]['s'] - DRIFT:>10.3f} "
                 f"{ws[0]['width'] * 1e3:>9.4f} {ws[0]['width'] / 2:>14.5f}")
        else:
            emit(f"    {sh:>6g} {'-':>9} {'-':>10} {'-':>9} {'no waist':>14}")
        del csr
        gc.collect()
    emit('')

    results = {}
    for sh in SHEARS:
        maps, meta = run_shear(sh)
        results[sh] = (maps, meta)
        emit(f"  ran shear={sh:<5g} {meta['nodes']:>5} nodes, {meta['kicks']:>4} kicks, "
             f"min h {meta['min_h']:.2e} m, {meta['seconds']:>5.0f} s")
    emit('')

    # the frame, and the wake amplitude, at every point
    for s_obs, label, desc in POINTS:
        emit(f'  s = {s_obs:.2f} m -- {label} ({desc})')
        emit(f"    {'shear':>6} {'snaps':>6} {'sigma_z um':>11} {'sigma_xi um':>12} "
             f"{'tau':>9} {'amp':>7} {'dE range MeV/m':>26} {'x_kick range':>24}")
        for sh in SHEARS:
            m = results[sh][0][s_obs]
            amp = m['sigma_x'] / max(m['sigma_xi'], 1e-30)
            emit(f"    {sh:>6g} {m['snapshots']:>6} {m['sigma_z'] * 1e6:>11.3f} "
                 f"{m['sigma_xi'] * 1e6:>12.4f} {m['tau']:>9.3f} {amp:>6.1f}x "
                 f"[{m['dE'].min():+11.4f},{m['dE'].max():+11.4f}] "
                 f"[{m['xk'].min():+10.4f},{m['xk'].max():+10.4f}]")
        emit('')

    # Everything below writes files. Only rank 0 may do it: with 10 ranks all of them ran the
    # plotting and the log write, and whichever finished last overwrote shear_xz_log.txt with its
    # own EMPTY `lines` (emit() appends only on rank 0). The figures survived because every rank
    # computed identical content, but the log came out blank.
    if RANK != 0:
        return

    # ---- figures: one per quantity, rows = observation point, cols = shear ------------------
    for quant, qlabel, fname in (('dE', 'dE/ds  [MeV/m]', 'shear_xz_longitudinal.png'),
                                 ('xk', 'transverse kick  [MeV/m]', 'shear_xz_transverse.png')):
        nr, nc = len(POINTS), len(SHEARS)
        fig, axes = plt.subplots(nr, nc, figsize=(3.6 * nc, 2.9 * nr))
        for r, (s_obs, label, desc) in enumerate(POINTS):
            for c, sh in enumerate(SHEARS):
                m = results[sh][0][s_obs]
                # PER-PANEL colour scale, so each map reaches saturation and its structure is
                # visible. The cost is that amplitudes are no longer comparable by eye -- the
                # wake spans four orders of magnitude between the entrance and the exit -- so
                # each panel is annotated with its own range, and shear_xz_amplitude.png carries
                # the amplitude comparison instead.
                #
                # Diverging-symmetric ONLY where the panel actually changes sign. Many panels do
                # not: the transverse kick is single-signed almost everywhere (0 % negative at
                # shear 0), and dE at shear 20 just inside the entrance is 100 % negative. Forcing
                # a symmetric scale there throws away half the colormap and leaves the panel pale.
                a = m[quant]
                lo, hi = float(a.min()), float(a.max())
                signed = lo < 0.0 and hi > 0.0
                if signed:
                    v = max(abs(lo), abs(hi)) or 1.0
                    kw = dict(cmap='RdBu_r', vmin=-v, vmax=v)
                else:
                    # sequential, spanning the data: full colour range on a one-signed field
                    kw = (dict(cmap='Reds', vmin=0.0, vmax=hi if hi > 0 else 1.0) if hi > 0
                          else dict(cmap='Blues_r', vmin=lo, vmax=0.0))
                ax = axes[r, c]
                # plot against the TILT-REMOVED x, which is the frame the deposition uses
                im = ax.pcolormesh(m['zz'] * 1e3, m['xt'] * 1e6, a, shading='auto', **kw)
                if r == 0:
                    ax.set_title(f'shear {sh:g}', fontsize=10)
                if c == 0:
                    ax.set_ylabel(f'{label}\nx - tau z  [um]', fontsize=8)
                if r == nr - 1:
                    ax.set_xlabel('z  [mm]', fontsize=8)
                ax.tick_params(labelsize=7)
                # the range this panel is normalised to, so the scale is never guessed
                ax.text(0.03, 0.96, f'[{lo:.3g}, {hi:.3g}]', transform=ax.transAxes,
                        fontsize=6.0, va='top', ha='left',
                        bbox=dict(fc='white', ec='0.6', alpha=0.85, pad=1.4))
                fig.colorbar(im, ax=ax, fraction=0.046)
        fig.suptitle(f'{qlabel}   |   rows = position, columns = x-z shear\n'
                     f'each panel self-normalised to its own range (annotated); '
                     f'amplitudes are compared in shear_xz_amplitude.png',
                     fontsize=10)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        fig.savefig(os.path.join(RESULT_DIR, fname), dpi=120)
        plt.close(fig)
        emit(f'  wrote {fname}')

    # ---- amplitude vs position, to show the transients as curves --------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    sp = np.array([s for s, _, _ in POINTS])
    for quant, ax, lab in (('dE', axes[0], 'peak |dE/ds|  [MeV/m]'),
                           ('xk', axes[1], 'peak |x_kick|  [MeV/m]')):
        for sh in SHEARS:
            pk = [np.abs(results[sh][0][s][quant]).max() for s in sp]
            ax.semilogy(sp, pk, 'o-', label=f'shear {sh:g}')
        ax.axvspan(DRIFT, S_EXIT, color='C1', alpha=0.10)
        ax.axvline(DRIFT + L_ENTRANCE, ls=':', color='0.4', lw=1,
                   label='entrance + L_entrance')
        ax.axvline(S_EXIT + L_EXIT, ls='--', color='0.4', lw=1, label='exit + L_exit')
        ax.set_xlabel('s  [m]')
        ax.set_ylabel(lab)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3, which='both')
    axes[0].set_title('longitudinal wake amplitude (shaded = dipole)')
    axes[1].set_title('transverse wake amplitude')
    fig.tight_layout()
    fig.savefig(os.path.join(RESULT_DIR, 'shear_xz_amplitude.png'), dpi=130)
    plt.close(fig)
    emit('  wrote shear_xz_amplitude.png')

    # ---- beam size along the lattice, with the adaptive nodes marked -----------------------
    # One row per shear. The point is to show WHERE auto spends its steps against WHY: the node
    # rug should crowd exactly where sigma_z collapses, and at the bend faces.
    profs = {sh: schedule_profile(sh) for sh in SHEARS}
    nr = len(SHEARS)
    fig, axes = plt.subplots(nr, 1, figsize=(13, 3.1 * nr), sharex=True)
    for r, sh in enumerate(SHEARS):
        p = profs[sh]
        ax = axes[r]
        ax.semilogy(p['fs'], p['sz'] * 1e6, '-', color='C0', lw=1.8, label='sigma_z')
        ax.semilogy(p['fs'], p['sxi'] * 1e6, '-', color='C2', lw=1.4, label='sigma_xi')
        ax.semilogy(p['fs'], p['sx'] * 1e6, '-', color='C7', lw=1.0, alpha=0.8, label='sigma_x')
        ax.axvspan(DRIFT, S_EXIT, color='C1', alpha=0.10)
        lo, hi = ax.get_ylim()
        # node rug along the bottom; kicks a little above, so both are legible
        ax.plot(p['s_nodes'], np.full_like(p['s_nodes'], lo * 1.10), '|',
                color='k', ms=5, alpha=0.5,
                label=f"snapshot nodes ({p['s_nodes'].size})")
        ax.plot(p['kick_s'], np.full_like(p['kick_s'], lo * 1.6), '|',
                color='C3', ms=8, label=f"kicks ({p['kick_s'].size})")
        ax.set_ylim(lo, hi)
        ax2 = ax.twinx()
        ax2.semilogy(p['s_scan'], p['h_eff'], '--', color='C4', lw=1.2, label='h_eff')
        ax2.set_ylabel('h_eff  [m]', color='C4', fontsize=8)
        ax2.tick_params(axis='y', colors='C4', labelsize=7)
        ax.set_ylabel(f'shear {sh:g}\nbeam size  [um]', fontsize=9)
        ax.grid(alpha=0.3, which='both')
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=7, ncol=3, loc='upper left')
        for s_obs, _, _ in POINTS:
            ax.axvline(s_obs, color='0.5', ls=':', lw=0.8)
    axes[-1].set_xlabel('s  [m]')
    fig.suptitle('beam size and the adaptive step schedule (shaded = dipole, '
                 'dotted verticals = wake-map positions)', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(os.path.join(RESULT_DIR, 'shear_xz_schedule.png'), dpi=130)
    plt.close(fig)
    emit('  wrote shear_xz_schedule.png')

    emit('')
    emit('  where auto spends its steps: node COUNT inside each region')
    emit(f"    {'shear':>6} {'pre-drift':>10} {'bend':>7} {'post-drift':>11} "
         f"{'min h in bend':>14} {'h_max':>8}")
    for sh in SHEARS:
        p = profs[sh]
        n = p['s_nodes']
        pre = int((n < DRIFT).sum())
        inb = int(((n >= DRIFT) & (n <= S_EXIT)).sum())
        post = int((n > S_EXIT).sum())
        inband = (p['s_scan'] >= DRIFT) & (p['s_scan'] <= S_EXIT)
        emit(f"    {sh:>6g} {pre:>10} {inb:>7} {post:>11} "
             f"{p['h_eff'][inband].min():>14.2e} {p['h_eff'].max():>8.4f}")

    with open(os.path.join(RESULT_DIR, 'shear_xz_log.txt'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'\nwrote {RESULT_DIR}/')


if __name__ == '__main__':
    main()
