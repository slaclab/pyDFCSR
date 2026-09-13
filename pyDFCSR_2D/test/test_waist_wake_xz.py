"""
The x-z wake at a waist, with a step_size that actually resolves it.

Every x-z wake map in this work so far (§6f, §6s, §6u) was computed at step_size = 0.05 m.
§6x then measured that at the waist point this is **79 % wrong**, with the positive peak 87 %
too large, and that the answer converges only below about step_size = 0.003 m. So none of
those entrance-region maps show the real wake.

This recomputes the full 21 x 51 mesh -- longitudinal AND transverse -- at step sizes
spanning unresolved to resolved, so the difference is visible as a picture rather than as a
single rel-L2 number.

Colour scales are SHARED across the step sizes within each row. Self-normalising every panel
would hide an 87 % amplitude error completely, which is the specific thing being shown.

A control point away from the waist is included. If the coarse and fine maps agree there and
disagree at the waist, the difference is the waist rather than step_size doing something
global.
"""
import sys
import os
import gc
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from pyDFCSR_2D.CSR import CSR2D
from pyDFCSR_2D.waist import scan_waists, sigma_from_coords

EXAMPLE_DIR = os.path.join(os.path.dirname(__file__), '..', 'example')
RESULT_DIR = os.path.join(os.path.dirname(__file__), 'benchmark_results', 'waist_xz')

SHEAR = 20.0
DRIFT = 0.35
DEP_BINS = 128
MESH_XBINS, MESH_ZBINS = 21, 51

# (metres into the dipole, step sizes to compare, label)
CASES = [
    (0.10, (0.05, 0.003, 0.0006), 'waist inside the near region'),
    (0.60, (0.05, 0.0025), 'control, far from the waist'),
]


def write_inputs(step, tag):
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
        'transforms': {'s1': {'shear_coefficient':
                              {'units': 'dimensionless', 'value': SHEAR},
                              'type': 'shear z:x'}},
    }
    with open(os.path.join(EXAMPLE_DIR, f'input/wx_beam_{tag}.yaml'), 'w') as f:
        yaml.dump(beam, f, default_flow_style=False, sort_keys=False)

    # step_size MUST stay first: lattice.py reads it positionally.
    lat = {'step_size': step,
           'element_1': {'type': 'drift', 'L': DRIFT, 'nsep': 1},
           'element_2': {'type': 'dipole', 'L': 1.0, 'angle': 1.0, 'nsep': 1},
           'element_3': {'type': 'drift', 'L': 0.5, 'nsep': 1}}
    lp = f'input/wx_lat_{tag}.yaml'
    with open(os.path.join(EXAMPLE_DIR, lp), 'w') as f:
        yaml.dump(lat, f, default_flow_style=False, sort_keys=False)

    cfg = {
        'input_beam': {'style': 'distgen',
                       'distgen_input_file': f'input/wx_beam_{tag}.yaml'},
        'input_lattice': {'lattice_input_file': lp},
        'particle_deposition': {'method': 'bspline_comoving',
                                'xbins': DEP_BINS, 'zbins': DEP_BINS,
                                'xlim': 5, 'zlim': 5, 'smoothing_sigma': 3.0,
                                'poly_degree': 1, 'velocity_threhold': 1000},
        'CSR_integration': {'n_formation_length': 1.5, 'zbins': 200, 'xbins': 200},
        'CSR_computation': {'compute_CSR': 0, 'apply_CSR': 0, 'transverse_on': 1,
                            'xbins': MESH_XBINS, 'zbins': MESH_ZBINS,
                            'xlim': 3, 'zlim': 3,
                            'write_beam': [], 'write_wakes': False,
                            'write_name': f'wx_{tag}', 'workdir': './output'},
    }
    p = f'input/wx_config_{tag}.yaml'
    with open(os.path.join(EXAMPLE_DIR, p), 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    return p


def reldiff(a, b):
    n = np.linalg.norm(b)
    return np.linalg.norm(a - b) / n if n > 0 else np.nan


def run_case(s_dip, step):
    tag = f'{s_dip:g}_{step:g}'.replace('.', 'p')
    cf = os.path.join(RESULT_DIR, f'mesh_{tag}.npz')
    if os.path.exists(cf):
        return dict(np.load(cf, allow_pickle=True))

    # The step grid starts at s = 0, so the observation point is only reachable exactly
    # when step divides it. 0.95/0.003 = 316.67 overshot to 0.9510 and tripped the
    # assertion below; comparing step sizes at DIFFERENT positions would have been worse,
    # so this is checked before spending the tracking time rather than after.
    target = DRIFT + s_dip
    ratio = target / step
    assert abs(ratio - round(ratio)) < 1e-6, (
        f'step_size {step} does not divide the path to {target} m '
        f'({ratio:.4f} steps); pick one that does, or the rungs land at different s')

    cfg = write_inputs(step, tag)
    t0 = time.time()
    csr = CSR2D(input_file=cfg)
    csr.run(stop_time=target - 0.5 * step, debug=True)
    csr.get_CSR_mesh()
    assert abs(csr.beam.position - target) < 1e-6, csr.beam.position

    b = csr.beam
    nx, nz = MESH_XBINS, MESH_ZBINS
    dE = np.zeros(nx * nz)
    xk = np.zeros(nx * nz)
    for i in range(nx * nz):
        dE[i], xk[i] = csr.get_CSR_wake(b.position + csr.CSR_zmesh[i],
                                        csr.CSR_xmesh[i])
    zz = csr.CSR_zmesh.reshape(nx, nz)
    xx = csr.CSR_xmesh.reshape(nx, nz)
    out = dict(dE=dE.reshape(nx, nz), xk=xk.reshape(nx, nz),
               zz=zz, xx=xx, xt=xx - np.polyval(b.slope, zz),
               snapshots=len(csr.DF_tracker.time_log),
               seconds=time.time() - t0,
               sigma_z=float(b._sigma_z), sigma_xi=float(b._sigma_x_transform),
               tau=float(b._slope[0]))
    os.makedirs(RESULT_DIR, exist_ok=True)
    np.savez(cf, **out)
    del csr
    gc.collect()
    return out


def main():
    os.makedirs(RESULT_DIR, exist_ok=True)
    os.chdir(EXAMPLE_DIR)
    lines = []

    def emit(t=''):
        print(t, flush=True)
        lines.append(t)

    emit('=' * 100)
    emit('x-z CSR wakes at a waist: unresolved vs resolved step_size')
    emit('=' * 100)
    emit(f'shear {SHEAR:g}, drift {DRIFT} m, mesh {MESH_XBINS} x {MESH_ZBINS}, '
         f'deposition {DEP_BINS}^2 fixed')

    # the waist, from the launch distribution (not the tracked beam, which is past it)
    cfg = write_inputs(CASES[0][1][0], 'scan')
    csr0 = CSR2D(input_file=cfg)
    b0 = csr0.beam
    rep = scan_waists(csr0.lattice.lattice_config,
                      sigma_from_coords(b0.x, b0.px, b0.particle.y, b0.particle.py,
                                        b0.z, b0.pz), CASES[0][1][0])
    w = [x for x in rep['waists'] if DRIFT <= x['s'] <= DRIFT + 1.0][0]
    emit(f'waist at s = {w["s"]:.4f} m ({w["s"]-DRIFT:.4f} m into the dipole), '
         f'width {w["width"]*1e3:.4f} mm, sigma_z {w["sigma_z"]*1e6:.3f} um')
    del csr0
    gc.collect()
    emit('')

    data = {}
    for s_dip, steps, label in CASES:
        emit(f'--- {s_dip:.2f} m into the dipole: {label} ---')
        emit(f"  {'step':>9} {'across':>7} {'snaps':>6} {'sec':>7} "
             f"{'dE range':>24} {'x_kick range':>26}")
        for st in steps:
            d = run_case(s_dip, st)
            data[(s_dip, st)] = d
            emit(f'  {st:>9g} {w["width"]/st:>7.2f} {int(d["snapshots"]):>6} '
                 f'{float(d["seconds"]):>7.1f} '
                 f'[{d["dE"].min():+9.4f},{d["dE"].max():+9.4f}] '
                 f'[{d["xk"].min():+11.4e},{d["xk"].max():+11.4e}]')
        fine = steps[-1]
        emit('  vs the finest step at this point:')
        for st in steps[:-1]:
            emit(f'    {st:>9g} : dE rel L2 '
                 f'{reldiff(data[(s_dip, st)]["dE"], data[(s_dip, fine)]["dE"]):.5f}'
                 f'   x_kick rel L2 '
                 f'{reldiff(data[(s_dip, st)]["xk"], data[(s_dip, fine)]["xk"]):.5f}')
        emit('')

    for s_dip, steps, label in CASES:
        n = len(steps)
        fig, axs = plt.subplots(2, n, figsize=(5.3 * n, 8.4), squeeze=False)
        for row, (key, nm, unit) in enumerate(
                (('dE', 'longitudinal $dE/dct$', 'MeV/m'),
                 ('xk', 'transverse kick', 'rad/m'))):
            # one scale across the row: an 87% amplitude error is invisible otherwise
            m = max(np.abs(data[(s_dip, st)][key]).max() for st in steps)
            for j, st in enumerate(steps):
                d = data[(s_dip, st)]
                ax = axs[row][j]
                im = ax.pcolormesh(d['zz'] * 1e6, d['xt'] * 1e6, d[key],
                                   cmap='RdBu_r', vmin=-m, vmax=m, shading='auto')
                plt.colorbar(im, ax=ax, pad=0.01, label=f'({unit})')
                ax.set_xlabel(r'$z$  ($\mu$m)')
                ax.set_ylabel(r'$x - p(z)$  ($\mu$m)')
                res = 'RESOLVED' if w['width'] / st >= rep['min_steps'] else 'unresolved'
                ttl = (f'step_size = {st:g} m  ({w["width"]/st:.2f} steps '
                       f'across the waist, {res})\n' if row == 0 else '')
                ax.set_title(ttl + f'{nm}\nrange '
                             f'[{d[key].min():+.4g}, {d[key].max():+.4g}]',
                             fontsize=9)
        fig.suptitle(f'CSR wake in the tilt-removed x-z plane, shear {SHEAR:g}, '
                     f'{s_dip:.2f} m into the dipole ({label}).\n'
                     f'Colour scale SHARED across each row, so amplitude errors are '
                     f'visible. Waist width {w["width"]*1e3:.3f} mm.', fontsize=11)
        fig.tight_layout()
        p = os.path.join(RESULT_DIR, f'wake_xz_{s_dip:g}'.replace('.', 'p') + '.png')
        fig.savefig(p, dpi=125, bbox_inches='tight')
        plt.close(fig)
        emit(f'Plot saved: {p}')

    with open(os.path.join(RESULT_DIR, 'waist_xz_log.txt'), 'w') as f:
        f.write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
