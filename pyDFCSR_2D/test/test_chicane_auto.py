"""
The example_chicane.ipynb case, run with the CURRENT code.

Same beam and lattice as the notebook -- this script READS
`example/input/chicane_init_beam.yaml` and `example/input/chicane_lattice.yaml` rather than
restating them, so it cannot drift out of step with the example. What differs is the
configuration, which uses what has been built and calibrated since:

    particle_deposition.method  bspline_comoving   (the co-moving interpolant, Step 5)
    step_control.mode           auto               (§11f, with every constant calibrated)
    step_control.kick_interval  midpoint           (§11d, O(h^2) kick quadrature)
    CSR_integration             100 x 100          (§11k measured converged: 0.09 % against
                                                    300x300, at 1.48x less cost than 200x200)

The notebook's own settings were `step_size = 0.1` uniform with `nsep = 5` in the long drifts,
legacy CIC deposition at 300^2, and 200x200 integration.

Why `auto` matters here rather than being cosmetic: this is a real compressing chicane, and the
schedule it picks is reported so the cost is visible up front rather than discovered at run time.

Note on the notebook itself: `example_chicane.ipynb` imports `pyDFCSR_2D` with the working
directory set to `example/`, and until 2026-09-21 a STALE non-editable copy of the package sat in
site-packages (dated Feb 4, v0.1.dev90) that shadowed the repo in exactly that situation. So the
notebook was silently running February code -- no scheduler, no nan fix, old frame_blend default.
Fixed by `pip install -e .`; this script uses an explicit sys.path anyway, as all the tests here do.
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
RESULT_DIR = os.path.join(os.path.dirname(__file__), 'benchmark_results', 'chicane_auto')

# the notebook's own inputs, reused verbatim
BEAM_YAML = 'input/chicane_init_beam.yaml'
LATTICE_YAML = 'input/chicane_lattice.yaml'
CONFIG_OUT = 'input/chicane_auto_current.yaml'

DEP_BINS = 300          # as the notebook: 300 x 300 deposition

# The notebook's own wake mesh, kept deliberately. The wake is evaluated at every mesh point at
# every kick, and that IS the cost of the run: 10x30 = 300 points gives 6.5 min on 10 ranks, while
# the 21x51 = 1071 used by the other x-z studies (§6y, §11p) would be 3.57x the work and ~23 min.
# Since the maps here are read qualitatively rather than compared against a converged reference,
# paying 3.6x on all 105 kicks to improve 8 plotted panels is not worth it. The consequence to
# remember: 10 points across x is coarse for a MAP even though it is adequate for the KICK, which
# is interpolated onto particles.
MESH_XBINS, MESH_ZBINS = 10, 30

# Representative positions to plot, one per regime of a chicane. Element spans:
#   D0 0.000-0.100  B1 0.100-0.600  D1 0.600-5.606  B2 5.606-6.106  D2 6.106-7.106
#   B3 7.106-7.606  D3 7.606-12.612 B4 12.612-13.112 Df 13.112-13.312
PLOT_TARGETS = [
    (0.35, 'B1 mid', 'first bend, sigma_z still 200 um'),
    (2.00, 'D1 early', 'dispersive drift, sigma_x growing'),
    (5.00, 'D1 late', 'sigma_x near maximum, 1.8 mm'),
    (5.85, 'B2 mid', 'second bend, compression 200->111 um'),
    (6.60, 'D2 mid', 'central drift between the dipole pairs'),
    (7.35, 'B3 mid', 'third bend, compression 111->20 um'),
    (10.0, 'D3 mid', 'long drift after full compression'),
    (12.86, 'B4 mid', 'fourth bend, dispersion closing'),
]


# A relative metric needs an ABSOLUTE floor on its denominator, which is the lesson §11n drew the
# hard way: the transverse kick vanishes identically in a drift (no centripetal force), and 34 of
# the 105 maps here have |x_kick| peaks of 1e-14 to 1e-19, i.e. pure floating-point residue.
# Normalising roundoff by itself yields a "roughness" of 4, which is meaningless. Anything below
# this fraction of the run's own maximum is reported as negligible rather than rough.
FLOOR_FRAC = 1e-6


def roughness(w, amp_ref=None):
    """
    Second-difference norm along z, relative to amplitude -- a SHAPE measure.

    §6x used the raw second-difference norm to detect an unresolved waist. Raw will not do here:
    the chicane wake amplitude spans 2.2e-3 to 6.1 MeV/m along the lattice, so a raw number would
    just track amplitude. Dividing by the amplitude makes it comparable between positions.

    `amp_ref` is the amplitude to normalise by, defaulting to this curve's own peak. Pass the
    run-wide maximum instead to keep near-zero maps from being amplified into apparent noise.

    Scale: for a smooth curve sampled at n points the second difference is O(1/n^2) per point, so
    at 30 z-bins a value of order 0.1 or below is smooth, and >~ 1 means the shape is not resolved.
    """
    w = np.asarray(w, float)
    amp = np.abs(w).max() if amp_ref is None else amp_ref
    if amp <= 0:
        return 0.0
    d2 = w[2:] - 2.0 * w[1:-1] + w[:-2]
    return float(np.linalg.norm(d2) / amp)


def map_roughness(m, key, floor=0.0):
    """
    Mean and worst-row roughness of one x-z map.

    Returns (nan, nan) when the map's amplitude is below `floor` -- such a map carries no wake to
    be rough, and scoring it would only measure roundoff.
    """
    a = m[key]
    if np.abs(a).max() <= floor:
        return float('nan'), float('nan')
    rows = [roughness(a[i, :]) for i in range(a.shape[0])]
    return float(np.mean(rows)), float(np.max(rows))


def write_config():
    """Build the current-code config around the notebook's beam and lattice. Rank 0 only."""
    if RANK != 0:
        if PARALLEL:
            COMM.Barrier()
        return CONFIG_OUT
    cfg = {
        'input_beam': {'style': 'distgen', 'distgen_input_file': BEAM_YAML},
        'input_lattice': {'lattice_input_file': LATTICE_YAML},
        'particle_deposition': {'method': 'bspline_comoving',
                                'xbins': DEP_BINS, 'zbins': DEP_BINS,
                                'xlim': 5, 'zlim': 5, 'smoothing_sigma': 3.0,
                                'poly_degree': 1, 'velocity_threhold': 1000},
        # 100x100: §11k measured converged against a 300x300 reference
        'CSR_integration': {'n_formation_length': 1, 'zbins': 100, 'xbins': 100},
        'CSR_computation': {'compute_CSR': 1, 'apply_CSR': 1, 'transverse_on': 1,
                            'xbins': MESH_XBINS, 'zbins': MESH_ZBINS,
                            'xlim': 3, 'zlim': 3,
                            'write_beam': [], 'write_wakes': False,
                            'write_name': 'chicane_auto', 'workdir': './output'},
        'step_control': {'mode': 'auto', 'kick_interval': 'midpoint', 'n_sub': 400},
    }
    with open(os.path.join(EXAMPLE_DIR, CONFIG_OUT), 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    if PARALLEL:
        COMM.Barrier()
    return CONFIG_OUT


def main():
    os.chdir(EXAMPLE_DIR)
    lines = []

    def emit(t=''):
        if RANK == 0:
            print(t, flush=True)
            lines.append(t)

    cfg = write_config()

    emit('=' * 100)
    emit('example_chicane.ipynb, run with the current code (auto schedule, comoving deposition)')
    emit('=' * 100)

    csr = CSR2D(input_file=cfg, parallel=PARALLEL)
    sch = csr.lattice.schedule
    lat = csr.lattice
    snap_mb = DEP_BINS * DEP_BINS * 5 * 8 / 1e6

    emit(f'  lattice: {lat.Nelement} elements, {lat.lattice_length:.4f} m')
    emit(f'  beam:    {len(csr.beam.x)} particles, sigma_z {csr.beam._sigma_z * 1e6:.2f} um, '
         f'sigma_x {csr.beam._sigma_x * 1e6:.2f} um')
    emit(f'  schedule: mode={sch.mode}, {sch.n_nodes} nodes, {sch.n_snap} snapshots, '
         f'{sch.n_kick} kicks')
    emit(f'            h_eff {sch.auto_h_eff.min():.3e} .. {sch.auto_h_eff.max():.3e} m '
         f'(ratio {sch.auto_h_eff.max() / sch.auto_h_eff.min():.0f})')
    emit(f'            steps per element {sch.steps_per_element(lat.Nelement)}')
    emit(f'            history at most {sch.n_snap} x {snap_mb:.2f} MB = '
         f'{sch.n_snap * snap_mb / 1024:.2f} GB')
    emit(f'  legacy for comparison: {int(np.ceil(lat.lattice_length / 0.1))} uniform steps '
         f'at step_size 0.1, kicks every nsep')
    emit(f'  running on {COMM.Get_size()} MPI rank(s)')
    emit('')

    # Capture the x-z map at every kick. Wrapping the calculation rather than re-deriving the
    # maps afterwards means these are EXACTLY the wakes that were applied to the beam.
    maps = []
    _orig = csr.calculate_2D_CSR_parallel if PARALLEL else csr.calculate_2D_CSR

    def _capture():
        _orig()
        nx, nz = MESH_XBINS, MESH_ZBINS
        b = csr.beam
        maps.append(dict(s=float(b.position),
                         dE=csr.dE_dct.reshape(nx, nz).copy(),
                         xk=csr.x_kick.reshape(nx, nz).copy(),
                         zz=csr.CSR_zmesh.reshape(nx, nz).copy(),
                         xx=csr.CSR_xmesh.reshape(nx, nz).copy(),
                         slope=np.asarray(b.slope, float).copy(),
                         sigma_z=float(b._sigma_z),
                         sigma_x=float(b._sigma_x),
                         sigma_xi=float(b._sigma_x_transform)))

    if PARALLEL:
        csr.calculate_2D_CSR_parallel = _capture
    else:
        csr.calculate_2D_CSR = _capture

    t0 = time.time()
    csr.run(debug=False)
    wall = time.time() - t0
    emit(f'  captured {len(maps)} wake maps at {MESH_XBINS}x{MESH_ZBINS}')
    emit(f'  RUN COMPLETE in {wall / 60:.1f} min ({wall:.0f} s)')
    emit('')

    st = csr.statistics
    s_nodes = sch.s_nodes[:len(st['sigma_z'])]

    # the nan bug of §11m hit exactly here, at the final node of a full run
    fin = {k: float(np.asarray(st[k], float)[-1]) for k in
           ('sigma_z', 'sigma_x', 'sigma_energy', 'mean_energy')}
    nbad = {k: int(np.isnan(np.asarray(st[k], float)).sum()) for k in st
            if isinstance(st[k], np.ndarray) and st[k].ndim == 1}
    emit('  final beam (the §11m nan bug would have made the energy columns nan here)')
    emit(f"    sigma_z      {fin['sigma_z'] * 1e6:12.4f} um")
    emit(f"    sigma_x      {fin['sigma_x'] * 1e6:12.4f} um")
    emit(f"    sigma_energy {fin['sigma_energy'] / 1e3:12.4f} keV")
    emit(f"    mean_energy  {fin['mean_energy'] / 1e9:12.6f} GeV  "
         f"(loss {(fin['mean_energy'] - float(st['mean_energy'][0])) / 1e6:+.4f} MeV)")
    emit(f"    nan count in the statistics arrays: {sum(nbad.values())}")
    emit('')

    emit('  compression along the lattice')
    emit(f"    {'s [m]':>8} {'sigma_z um':>12} {'sigma_x um':>12} {'sigma_E keV':>12}")
    for target in (0.0, 0.6, 5.6, 6.1, 7.1, 7.6, 12.6, 13.1, lat.lattice_length):
        j = int(np.argmin(np.abs(s_nodes - target)))
        emit(f"    {s_nodes[j]:>8.3f} {st['sigma_z'][j] * 1e6:>12.3f} "
             f"{st['sigma_x'][j] * 1e6:>12.3f} {st['sigma_energy'][j] / 1e3:>12.4f}")
    emit('')

    if RANK != 0:
        return

    os.makedirs(RESULT_DIR, exist_ok=True)

    # bend spans, for shading every figure
    _starts = np.concatenate([[0.0], lat.distance[:-1]])
    _keys = [k for k in lat.lattice_config if k != 'step_size']
    bends = [(_starts[i], lat.distance[i]) for i, k in enumerate(_keys)
             if lat.lattice_config[k].get('type') == 'dipole']

    # ---- wake smoothness, at EVERY kick ------------------------------------------------
    # Checked at all 105 kicks rather than only the plotted ones: a rough wake at an unplotted
    # position is exactly the case a figure would miss.
    emit('  wake smoothness: second-difference norm along z, relative to each map amplitude')
    emit('  (a smooth curve at 30 z-bins is O(0.1); >~ 1 means the shape is not resolved)')
    ms = np.array([m['s'] for m in maps])
    fl_dE = FLOOR_FRAC * max(np.abs(m['dE']).max() for m in maps)
    fl_xk = FLOOR_FRAC * max(np.abs(m['xk']).max() for m in maps)
    rg_dE = np.array([map_roughness(m, 'dE', fl_dE)[0] for m in maps])
    rg_xk = np.array([map_roughness(m, 'xk', fl_xk)[0] for m in maps])
    wr_dE = np.array([map_roughness(m, 'dE', fl_dE)[1] for m in maps])
    n_null = int(np.isnan(rg_xk).sum())
    emit(f'    amplitude floor {FLOOR_FRAC:g} x the run maximum excludes {n_null} of {len(maps)} '
         f'x_kick maps as numerically zero (drifts have no transverse force)')
    emit(f"    dE     mean over kicks {np.nanmean(rg_dE):.4f}, worst map {np.nanmax(rg_dE):.4f} "
         f"at s = {ms[int(np.nanargmax(rg_dE))]:.3f} m")
    emit(f"    x_kick mean over kicks {np.nanmean(rg_xk):.4f}, worst map {np.nanmax(rg_xk):.4f} "
         f"at s = {ms[int(np.nanargmax(rg_xk))]:.3f} m")
    emit(f"    worst SINGLE x row (dE) {np.nanmax(wr_dE):.4f} at s = {ms[int(np.nanargmax(wr_dE))]:.3f} m")
    bad = int(np.nansum(rg_dE > 1.0))
    emit(f"    kicks with dE roughness > 1.0: {bad} of {len(maps)}")
    emit('')
    emit(f"    {'s [m]':>8} {'element':>9} {'dE rough':>9} {'xk rough':>9} {'|dE| peak':>11}")
    for tgt, lab, _ in PLOT_TARGETS:
        j = int(np.argmin(np.abs(ms - tgt)))
        emit(f"    {ms[j]:>8.3f} {lab:>9} {rg_dE[j]:>9.4f} {rg_xk[j]:>9.4f} "
             f"{np.abs(maps[j]['dE']).max():>11.5f}")
    emit('')

    # ---- x-z wake maps at the representative positions ----------------------------------
    sel = [(lab, desc, int(np.argmin(np.abs(ms - tgt)))) for tgt, lab, desc in PLOT_TARGETS]
    for quant, qlabel, fname in (('dE', 'dE/ds  [MeV/m]', 'chicane_wakes_longitudinal.png'),
                                 ('xk', 'transverse kick  [MeV/m]', 'chicane_wakes_transverse.png')):
        fig, axes = plt.subplots(2, 4, figsize=(19, 8.5))
        for ax, (lab, desc, j) in zip(axes.ravel(), sel):
            m = maps[j]
            a = m[quant]
            lo, hi = float(a.min()), float(a.max())
            if lo < 0.0 < hi:                      # diverging only where the sign changes (§11p)
                v = max(abs(lo), abs(hi)) or 1.0
                kw = dict(cmap='RdBu_r', vmin=-v, vmax=v)
            else:
                kw = (dict(cmap='Reds', vmin=0.0, vmax=hi if hi > 0 else 1.0) if hi > 0
                      else dict(cmap='Blues_r', vmin=lo, vmax=0.0))
            xt = m['xx'] - np.polyval(m['slope'], m['zz'])
            im = ax.pcolormesh(m['zz'] * 1e3, xt * 1e6, a, shading='auto', **kw)
            ax.set_title(f"{lab}  s = {m['s']:.2f} m\n{desc}", fontsize=8.5)
            ax.set_xlabel('z  [mm]', fontsize=8)
            ax.set_ylabel('x - tau z  [um]', fontsize=8)
            ax.tick_params(labelsize=7)
            ax.text(0.03, 0.96, f'[{lo:.3g}, {hi:.3g}]\nrough {roughness(a[a.shape[0] // 2, :]):.3f}',
                    transform=ax.transAxes, fontsize=6.5, va='top',
                    bbox=dict(fc='white', ec='0.6', alpha=0.85, pad=1.4))
            fig.colorbar(im, ax=ax, fraction=0.046)
        fig.suptitle(f'chicane {qlabel}: each panel self-normalised, range and mid-row roughness '
                     f'annotated   (mesh {MESH_XBINS}x{MESH_ZBINS})', fontsize=10)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(os.path.join(RESULT_DIR, fname), dpi=130)
        plt.close(fig)
        emit(f'  wrote {fname}')

    # ---- roughness along the lattice ----------------------------------------------------
    fig, ax = plt.subplots(figsize=(13, 4.2))
    ax.semilogy(ms, rg_dE, 'o-', ms=3, lw=1.2, label='dE (mean over x rows)')
    ax.semilogy(ms, rg_xk, 's-', ms=3, lw=1.0, label='x_kick (mean over x rows)')
    ax.axhline(1.0, color='C3', ls='--', lw=1, label='roughness 1.0 (unresolved)')
    for b0, b1 in bends:
        ax.axvspan(b0, b1, color='C1', alpha=0.15)
    ax.set_xlabel('s  [m]')
    ax.set_ylabel('relative second-difference norm')
    ax.set_title('wake smoothness at every kick (shading = dipoles)')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which='both')
    fig.tight_layout()
    fig.savefig(os.path.join(RESULT_DIR, 'chicane_wake_roughness.png'), dpi=130)
    plt.close(fig)
    emit('  wrote chicane_wake_roughness.png')

    np.savez(os.path.join(RESULT_DIR, 'chicane_auto_maps.npz'),
             maps=np.array(maps, dtype=object))
    np.savez(os.path.join(RESULT_DIR, 'chicane_auto_stats.npz'),
             s=s_nodes, sigma_z=np.asarray(st['sigma_z'], float),
             sigma_x=np.asarray(st['sigma_x'], float),
             sigma_energy=np.asarray(st['sigma_energy'], float),
             mean_energy=np.asarray(st['mean_energy'], float),
             norm_emit_x=np.asarray(st['twiss']['norm_emit_x'], float),
             s_nodes=sch.s_nodes, kick_s=sch.s_nodes[sch.is_kick],
             s_scan=sch.auto_s_scan, h_eff=sch.auto_h_eff,
             distance=lat.distance, wall=wall)

    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
    ax = axes[0]
    ax.semilogy(s_nodes, np.asarray(st['sigma_z']) * 1e6, lw=1.8, color='C0', label='sigma_z')
    ax.semilogy(s_nodes, np.asarray(st['sigma_x']) * 1e6, lw=1.4, color='C2', label='sigma_x')
    ax.set_ylabel('beam size  [um]')
    ax.set_title('chicane with the current code: compression, energy spread, and the schedule')
    ax = axes[1]
    ax.plot(s_nodes, np.asarray(st['sigma_energy']) / 1e3, lw=1.8, color='C3')
    ax.set_ylabel('sigma_E  [keV]')
    ax = axes[2]
    ax.semilogy(sch.auto_s_scan, sch.auto_h_eff, '--', color='C4', lw=1.3, label='h_eff')
    lo = sch.auto_h_eff.min()
    ax.plot(sch.s_nodes, np.full(sch.n_nodes, lo * 0.75), '|', color='k', ms=5, alpha=0.5,
            label=f'snapshot nodes ({sch.n_nodes})')
    ax.plot(sch.s_nodes[sch.is_kick], np.full(sch.n_kick, lo * 0.55), '|', color='C3', ms=8,
            label=f'kicks ({sch.n_kick})')
    ax.set_ylabel('step size  [m]')
    ax.set_xlabel('s  [m]')
    for a in axes:
        for b0, b1 in bends:
            a.axvspan(b0, b1, color='C1', alpha=0.15)
        a.grid(alpha=0.3, which='both')
        a.legend(fontsize=8, loc='best')
    fig.tight_layout()
    fig.savefig(os.path.join(RESULT_DIR, 'chicane_auto.png'), dpi=130)
    plt.close(fig)
    emit('  wrote chicane_auto.png  (shading = the four dipoles)')

    with open(os.path.join(RESULT_DIR, 'chicane_auto_log.txt'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'\nwrote {RESULT_DIR}/')


if __name__ == '__main__':
    main()
