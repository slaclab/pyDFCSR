"""
Longitudinal and transverse CSR wake with the final configuration:
co-moving interpolant, two-branch Eq 4.24 bands, polar patch, per-step formation
length, and per-region longitudinal allocation (near_cell = 0.5).

Two studies:

  PART A  at a fixed location near the dipole exit, wakes across shear 0 / 2 / 20 / 50
          (tilt amplification ~1x to ~167x).
  PART B  for shear 20 and 50, wakes at several locations THROUGH the dipole, which
          is where sigma_z, sigma_x and the tilt all evolve strongly.

Each is shown three ways: the 2D map in the physical (x, z) plane, the same map on
the tilt-removed mesh the code actually integrates on, and a 1D cut along z at mid-x.
The physical view shows what the beam sees; at high shear the mesh is a strongly
sheared parallelogram there and detail becomes unreadable, so the tilt-removed view is
the one to judge structure and smoothness on.

Cost note: compute_CSR = 0 with run(debug=True) builds the density history without
computing a wake at every lattice step -- the history build is gated on
`debug or compute_CSR` at CSR.py:311, while the wake is gated on compute_CSR alone.
Getting that wrong costs a factor of ~200 (see section 6f).

run(stop_time=T) stops after the first step that REACHES T, so it overshoots by up to
one step. Actual beam.position is reported and used in every label rather than the
requested value (see section 6i, where assuming otherwise invalidated a study).
"""
import sys
import os
import numpy as np
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from pyDFCSR_2D.CSR import CSR2D

RESULT_DIR = os.path.join(os.path.dirname(__file__), 'benchmark_results', 'wake_evol')
EXAMPLE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'example'))

SHEARS_A = [0.0, 2.0, 20.0, 50.0]
STOP_A = 0.60                      # -> lands near the dipole exit
SHEARS_B = [20.0, 50.0]
STOPS_B = [0.15, 0.35, 0.55, 0.75, 0.95]   # inside the dipole (0.1 .. 1.1 m)

DEP_BINS = 200
N_PARTICLE = 200000
MESH_XBINS, MESH_ZBINS = 21, 51
INT_XBINS = 200


def write_inputs(shear, tag):
    beam = {
        'n_particle': N_PARTICLE, 'species': 'electron',
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
    if shear != 0.0:
        beam['transforms'] = {'s1': {'shear_coefficient':
                                     {'units': 'dimensionless', 'value': float(shear)},
                                     'type': 'shear z:x'}}
    with open(os.path.join(EXAMPLE_DIR, f'input/we_beam_{tag}.yaml'), 'w') as f:
        yaml.dump(beam, f, default_flow_style=False, sort_keys=False)
    cfg = {
        'input_beam': {'style': 'distgen',
                       'distgen_input_file': f'input/we_beam_{tag}.yaml'},
        'input_lattice': {'lattice_input_file': 'input/dipole_lattice.yaml'},
        'particle_deposition': {'method': 'bspline_comoving',
                                'xbins': DEP_BINS, 'zbins': DEP_BINS,
                                'xlim': 5, 'zlim': 5, 'smoothing_sigma': 3.0,
                                'poly_degree': 1, 'velocity_threhold': 1000},
        # near_cell / far_zbins left at their new defaults (0.5 / 200)
        'CSR_integration': {'n_formation_length': 1.5, 'zbins': 400,
                            'xbins': INT_XBINS},
        'CSR_computation': {'compute_CSR': 0, 'apply_CSR': 0, 'transverse_on': 1,
                            'xbins': MESH_XBINS, 'zbins': MESH_ZBINS,
                            'xlim': 3, 'zlim': 3,
                            'write_beam': [], 'write_wakes': False,
                            'write_name': f'we_{tag}', 'workdir': './output'},
    }
    p = f'input/we_config_{tag}.yaml'
    with open(os.path.join(EXAMPLE_DIR, p), 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    return p


CACHE = os.path.join(RESULT_DIR, 'cache')

def wake_map(shear, stop, tag):
    """
    2D wake on the CSR mesh, plus the geometry needed to plot it.

    Cached to .npz per (shear, stop). Three long background runs in this session
    were killed silently part-way, losing everything computed so far; caching makes
    the study restartable and means a re-run only pays for what is missing.
    """
    os.makedirs(CACHE, exist_ok=True)
    cf = os.path.join(CACHE, f'{tag}.npz')
    if os.path.exists(cf):
        d = dict(np.load(cf))
        for k in ('pos', 'slope', 'sig_z', 'sig_x', 'sig_xi', 'amp',
                  'near_nodes', 'far_nodes', 'shear'):
            d[k] = d[k].item()
        return d
    csr = CSR2D(input_file=write_inputs(shear, tag))
    csr.run(stop_time=stop, debug=True)
    csr.get_CSR_mesh()
    b = csr.beam
    nx, nz = MESH_XBINS, MESH_ZBINS
    dE = np.zeros(nx*nz)
    xk = np.zeros(nx*nz)
    for i in range(nx*nz):
        dE[i], xk[i] = csr.get_CSR_wake(b.position + csr.CSR_zmesh[i],
                                        csr.CSR_xmesh[i])
    zz = csr.CSR_zmesh.reshape(nx, nz)
    xx = csr.CSR_xmesh.reshape(nx, nz)
    out = dict(dE=dE.reshape(nx, nz), xk=xk.reshape(nx, nz),
               zz=zz, xx=xx, xt=xx - np.polyval(b.slope, zz),
               pos=b.position, slope=b._slope[0],
               sig_z=b._sigma_z, sig_x=b._sigma_x, sig_xi=b._sigma_x_transform,
               amp=b._sigma_x/b._sigma_x_transform,
               # real counts from the last get_CSR_wake call. An earlier version
               # passed hardcoded dummy bounds here and so reported the node count
               # for a fictitious 10 mm region rather than the actual one.
               near_nodes=csr._last_region_nodes[-1],
               far_nodes=csr._last_region_nodes[0],
               shear=shear)
    np.savez(cf, **out)
    return out


def panel(ax, X, Y, W, title, xlabel, ylabel, label):
    lim = np.abs(W).max()
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-lim, vmax=lim) if lim > 0 else None
    im = ax.pcolormesh(X*1e6, Y*1e6, W, cmap='RdBu_r', norm=norm, shading='auto')
    plt.colorbar(im, ax=ax, label=label)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def maps_figure(runs, fname, suptitle, col_title):
    """2 rows (dE, x_kick) x N columns, physical (x, z) and tilt-removed."""
    for view, ykey, ylab in (('xz', 'xx', r'$x$ ($\mu$m)'),
                             ('transformed', 'xt', r'$x - p(z)$ ($\mu$m)')):
        n = len(runs)
        fig, axs = plt.subplots(2, n, figsize=(4.6*n, 8.4), squeeze=False)
        for j, r in enumerate(runs):
            panel(axs[0][j], r['zz'], r[ykey], r['dE'], col_title(r),
                  r'$z$ ($\mu$m)', ylab, r'$dE/d(ct)$ (MeV/m)')
            panel(axs[1][j], r['zz'], r[ykey], r['xk'], 'transverse kick',
                  r'$z$ ($\mu$m)', ylab, 'transverse kick (a.u.)')
        fig.suptitle(suptitle + ('  —  physical (x, z) plane' if view == 'xz'
                                 else '  —  tilt-removed mesh (as integrated)'),
                     fontsize=12)
        plt.tight_layout()
        p = os.path.join(RESULT_DIR, f'{fname}_{view}.png')
        plt.savefig(p, dpi=125)
        plt.close()
        print(f'  saved {p}')


def cuts_figure(runs, fname, suptitle, label_fn):
    """1D cut along z at mid-x."""
    ixm = MESH_XBINS//2
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.6))
    for r in runs:
        z = r['zz'][ixm]*1e6
        axs[0].plot(z, r['dE'][ixm], label=label_fn(r))
        axs[1].plot(z, r['xk'][ixm], label=label_fn(r))
    for ax, lab in ((axs[0], r'$dE/d(ct)$ (MeV/m)'),
                    (axs[1], 'transverse kick (a.u.)')):
        ax.set_xlabel(r'$z$ ($\mu$m)')
        ax.set_ylabel(lab)
        ax.axhline(0, color='k', lw=0.5, alpha=0.4)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle(suptitle + '  —  cut along z at mid-x', fontsize=12)
    plt.tight_layout()
    p = os.path.join(RESULT_DIR, f'{fname}_zcut.png')
    plt.savefig(p, dpi=130)
    plt.close()
    print(f'  saved {p}')


def main():
    # optional CLI: "A", "B", or "B 50" to run only part of the study
    want = sys.argv[1].upper() if len(sys.argv) > 1 else 'AB'
    only_shear = float(sys.argv[2]) if len(sys.argv) > 2 else None
    os.makedirs(RESULT_DIR, exist_ok=True)
    os.chdir(EXAMPLE_DIR)
    lines = []

    def emit(t=''):
        print(t)
        lines.append(t)

    emit('=' * 92)
    emit('CSR wake with the final configuration (near_cell = 0.5 per-region allocation)')
    emit('=' * 92)
    emit(f'method = bspline_comoving, two-branch bands, polar patch, '
         f'wake mesh {MESH_XBINS} x {MESH_ZBINS}')
    emit('')

    # ---------------- PART A: shear sweep at one location -------------------
    emit('PART A: wakes vs shear, near the dipole exit')
    emit(f"  {'shear':>6} {'s (m)':>8} {'amp':>7} {'slope':>8} {'sigma_z':>9} "
         f"{'sigma_xi':>9} {'near nodes':>11} {'dE range (MeV/m)':>26}")
    runs_a = []
    for sh in (SHEARS_A if 'A' in want else []):
        r = wake_map(sh, STOP_A, f'a{sh:g}')
        runs_a.append(r)
        emit(f"  {sh:>6.0f} {r['pos']:>8.4f} {r['amp']:>7.1f} {r['slope']:>+8.3f} "
             f"{r['sig_z']*1e6:>8.1f}u {r['sig_xi']*1e6:>8.2f}u "
             f"{r['near_nodes']:>11} "
             f"[{r['dE'].min():>+9.4f}, {r['dE'].max():>+9.4f}]")
    emit('')
    if runs_a:
        maps_figure(runs_a, 'partA_shear',
                     'CSR wake vs beam shear near the dipole exit',
                     lambda r: (f"shear {r['shear']:g}, amp {r['amp']:.0f}x\n"
                                f"s = {r['pos']:.3f} m, dx/dz = {r['slope']:+.2f}"))
        cuts_figure(runs_a, 'partA_shear', 'CSR wake vs beam shear',
                    lambda r: f"shear {r['shear']:g} (amp {r['amp']:.0f}x)")
    emit('')

    # ---------------- PART B: through the dipole ---------------------------
    emit('PART B: wakes at several locations through the dipole (0.1 - 1.1 m)')
    b_list = [] if 'B' not in want else (
        [only_shear] if only_shear is not None else SHEARS_B)
    for sh in b_list:
        emit(f'--- shear {sh:g} ---')
        emit(f"  {'requested':>10} {'actual s':>9} {'amp':>7} {'slope':>8} "
             f"{'sigma_z':>9} {'sigma_xi':>9} {'near nodes':>11} "
             f"{'dE range (MeV/m)':>26}")
        runs_b = []
        for st in STOPS_B:
            r = wake_map(sh, st, f'b{sh:g}_{st:g}')
            runs_b.append(r)
            emit(f"  {st:>10.2f} {r['pos']:>9.4f} {r['amp']:>7.1f} "
                 f"{r['slope']:>+8.3f} {r['sig_z']*1e6:>8.1f}u "
                 f"{r['sig_xi']*1e6:>8.2f}u {r['near_nodes']:>11} "
                 f"[{r['dE'].min():>+9.4f}, {r['dE'].max():>+9.4f}]")
        maps_figure(runs_b, f'partB_shear{sh:g}',
                    f'CSR wake through the dipole, shear {sh:g}',
                    lambda r: (f"s = {r['pos']:.3f} m\namp {r['amp']:.0f}x, "
                               f"dx/dz = {r['slope']:+.2f}"))
        cuts_figure(runs_b, f'partB_shear{sh:g}',
                    f'CSR wake through the dipole, shear {sh:g}',
                    lambda r: f"s = {r['pos']:.3f} m")
        emit('')

    with open(os.path.join(RESULT_DIR, 'wake_evolution_log.txt'), 'w') as f:
        f.write('\n'.join(lines))


if __name__ == '__main__':
    main()
