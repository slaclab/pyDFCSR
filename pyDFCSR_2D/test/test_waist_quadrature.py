"""
Is the waist a QUADRATURE-resolution problem rather than a frame-sampling one?

§6n/§6r concluded the entrance noise is undersampling of the frame HISTORY, because
nothing converged as step_size -> 0 (successive differences of 28-140% even at
step_size = 0.00625 m, i.e. 160 steps through a 1 m drift). That conclusion rests on
having refined the right thing.

§6u surfaced numbers that undercut it. The near-region s' cell is

    ds = max(near_cell * sigma_xi(OBSERVATION time),  near_grade * |s - s'|)

Both terms are set by quantities at the observation point. Neither knows anything about
sigma_z at the RETARDED time -- and at a waist that collapses. At shear 20, s = 0.10 m
into the dipole:

    floor = near_cell * sigma_xi = 0.5 * 49.911 um   = 24.96 um
    compression point is ~50 mm upstream, where the GRADED cell is
            near_grade * u       = 0.05 * 50 mm      =  2.5 mm
    retarded density there has sigma_z                =  2.51 um

so the quadrature cell is ~1000x longer than the longitudinal structure it integrates.
Refining step_size refines the frame but leaves the cell pinned to observation-time
quantities, which is a mechanism for exactly the non-convergence §6r saw.

This script refines the QUADRATURE instead, holding step_size fixed:

    (near_cell, near_grade) down a ladder, ending at a uniform grid at a fine floor

and asks whether the wake converges. A control point where §6r already converged is
included, so a null result there confirms the ladder is not just moving everything.

Reads the retarded sigma_z profile directly and reports cell/sigma_z at its minimum,
which is the quantitative form of the claim.
"""
import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from pyDFCSR_2D.CSR import CSR2D

EXAMPLE_DIR = os.path.join(os.path.dirname(__file__), '..', 'example')
RESULT_DIR = os.path.join(os.path.dirname(__file__), 'benchmark_results', 'waist_quad')

DRIFT = 1.0
STEP = 0.05
MESH_XBINS, MESH_ZBINS = 21, 51

# (shear, metres into dipole, label)
POINTS = [
    (20.0, 0.10, 'waist (shear 20)'),
    (10.0, 0.10, 'waist (shear 10)'),
    (20.0, 0.60, 'control, converged in 6r'),
]

# (near_cell, near_grade). near_grade = 0 is a uniform grid at the floor.
LADDER = [
    (0.5,  0.05),      # shipped default
    (0.5,  0.02),
    (0.5,  0.01),
    (0.5,  0.0),
    (0.2,  0.0),
    (0.1,  0.0),
    (0.05, 0.0),
]


def write_inputs(shear, tag):
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
    if shear != 0.0:
        beam['transforms'] = {'s1': {'shear_coefficient':
                                     {'units': 'dimensionless', 'value': float(shear)},
                                     'type': 'shear z:x'}}
    with open(os.path.join(EXAMPLE_DIR, f'input/wq_beam_{tag}.yaml'), 'w') as f:
        yaml.dump(beam, f, default_flow_style=False, sort_keys=False)

    lat = {'step_size': STEP,
           'element_1': {'type': 'drift', 'L': DRIFT, 'nsep': 1},
           'element_2': {'type': 'dipole', 'L': 1.0, 'angle': 1.0,
                         'E1': 0, 'E2': 0, 'FINT': 0.0, 'FINTX': 0.5,
                         'HGAP': 0.1, 'HGAPX': 0.0, 'FRINGE_AT': 'both_ends',
                         'FRINGE_TYPE': 'linear_edge', 'TILT': 0.0, 'nsep': 1},
           'element_3': {'type': 'drift', 'L': 0.5, 'nsep': 1}}
    lp = f'input/wq_lat_{tag}.yaml'
    with open(os.path.join(EXAMPLE_DIR, lp), 'w') as f:
        yaml.dump(lat, f, default_flow_style=False, sort_keys=False)

    cfg = {
        'input_beam': {'style': 'distgen',
                       'distgen_input_file': f'input/wq_beam_{tag}.yaml'},
        'input_lattice': {'lattice_input_file': lp},
        'particle_deposition': {'method': 'bspline_comoving',
                                'xbins': 200, 'zbins': 200,
                                'xlim': 5, 'zlim': 5, 'smoothing_sigma': 3.0,
                                'poly_degree': 1, 'velocity_threhold': 1000},
        # zbins only sets the node CAP (100*zbins) once near_cell != 0. Raised well
        # above the ladder's demand so the cap cannot silently clip the fine rungs --
        # that clipping is exactly how an under-resolved wake once looked plausible
        # (§6l). The cap warning fires only in the ungraded path, which is where the
        # fine rungs live, so it will speak up if this is still not enough.
        'CSR_integration': {'n_formation_length': 1.5, 'zbins': 4000, 'xbins': 200},
        'CSR_computation': {'compute_CSR': 0, 'apply_CSR': 0, 'transverse_on': 1,
                            'xbins': MESH_XBINS, 'zbins': MESH_ZBINS,
                            'xlim': 3, 'zlim': 3,
                            'write_beam': [], 'write_wakes': False,
                            'write_name': f'wq_{tag}', 'workdir': './output'},
    }
    p = f'input/wq_config_{tag}.yaml'
    with open(os.path.join(EXAMPLE_DIR, p), 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    return p


def reldiff(a, b):
    n = np.linalg.norm(b)
    return np.linalg.norm(a - b) / n if n > 0 else np.nan


def abs_rough(w):
    d2 = w[2:] - 2 * w[1:-1] + w[:-2]
    return np.linalg.norm(d2)


def z_cut(csr):
    nz = csr.CSR_params.zbins
    ix = csr.CSR_params.xbins // 2
    s0 = csr.beam.position
    dE = np.zeros(nz)
    for j in range(nz):
        k = ix * nz + j
        dE[j], _ = csr.get_CSR_wake(s0 + csr.CSR_zmesh[k], csr.CSR_xmesh[k])
    return dE


def main():
    os.makedirs(RESULT_DIR, exist_ok=True)
    os.chdir(EXAMPLE_DIR)
    lines = []

    def emit(t=''):
        print(t, flush=True)
        lines.append(t)

    emit('=' * 104)
    emit('Is the waist a QUADRATURE-resolution problem? Refine the s\' grid, hold '
         'step_size fixed.')
    emit('=' * 104)
    emit(f'step_size {STEP} m throughout (so the frame history is IDENTICAL down each '
         f'ladder), drift {DRIFT} m')
    emit('')

    results = {}
    for shear, s_dip, label in POINTS:
        tag = f's{shear:g}_{s_dip:g}'.replace('.', 'p')
        csr = CSR2D(input_file=write_inputs(shear, tag))
        target = DRIFT + s_dip
        csr.run(stop_time=target - 0.5 * STEP, debug=True)
        csr.get_CSR_mesh()
        assert abs(csr.beam.position - target) < 1e-6, csr.beam.position

        b = csr.beam
        t = b.position
        ip = csr.integration_params
        ix, nz = MESH_XBINS // 2, MESH_ZBINS
        s_mid = t + csr.CSR_zmesh[ix * nz + nz // 2]
        x_mid = csr.CSR_xmesh[ix * nz + nz // 2]

        emit('-' * 104)
        emit(f'{label}: shear {shear:g}, {s_dip:.2f} m into the dipole')
        emit('-' * 104)
        emit(f'  observation frame: tau {csr._frame_tilt():+.3f}, '
             f'sigma_z {b._sigma_z*1e6:.2f} um, sigma_xi {b._sigma_x_transform*1e6:.3f} um')

        # --- the claim: cell size vs retarded sigma_z, on the DEFAULT grid ---
        ip.near_cell, ip.near_grade = LADDER[0]
        bnds = csr._layout_bounds(s_mid, x_mid)
        s3, s4 = bnds[2]
        nodes = csr._near_region_nodes(s_mid, s3, s4)
        prof = np.linspace(s3, s4, 3000)
        # s' is a good proxy for t_ret along x' = x; exact enough to locate the waist
        sz_ret = csr._comoving_frame_at(prof)[4]
        i_min = int(np.argmin(sz_ret))
        s_waist = prof[i_min]
        sz_min = sz_ret[i_min]
        # graded cell at the waist location
        k = int(np.clip(np.searchsorted(nodes, s_waist) - 1, 0, len(nodes) - 2))
        cell_at_waist = nodes[k + 1] - nodes[k]
        emit(f'  near region [{(s3-t)*1e3:+.3f}, {(s4-t)*1e3:+.3f}] mm about s, '
             f'{len(nodes)} default nodes')
        emit(f'  retarded sigma_z minimum: {sz_min*1e6:.3f} um at '
             f'{(s_waist-t)*1e3:+.3f} mm from s '
             f'(sigma_z at s is {b._sigma_z*1e6:.2f} um)')
        emit(f'  DEFAULT graded cell there: {cell_at_waist*1e6:.1f} um'
             f'  ->  cell / retarded sigma_z = '
             f'{cell_at_waist/max(sz_min,1e-30):.1f}x')
        emit('')

        emit(f"  {'near_cell':>10} {'near_grade':>11} {'nodes':>7} "
             f"{'cell@waist/um':>14} {'ratio':>8} {'abs rough':>10} "
             f"{'rel L2 vs finest':>17} {'dE range':>24}")
        cuts = []
        for nc, ng in LADDER:
            ip.near_cell, ip.near_grade = nc, ng
            nodes = csr._near_region_nodes(s_mid, s3, s4)
            k = int(np.clip(np.searchsorted(nodes, s_waist) - 1, 0,
                            max(len(nodes) - 2, 0)))
            cw = nodes[k + 1] - nodes[k] if len(nodes) > 1 else np.nan
            dE = z_cut(csr)
            cuts.append((nc, ng, len(nodes), cw, dE))
        ref = cuts[-1][4]
        for nc, ng, nn, cw, dE in cuts:
            emit(f'  {nc:>10.3g} {ng:>11.3g} {nn:>7} {cw*1e6:>14.2f} '
                 f'{cw/max(sz_min,1e-30):>8.1f} {abs_rough(dE):>10.5f} '
                 f'{reldiff(dE, ref):>17.5f} '
                 f'[{dE.min():+9.4f},{dE.max():+9.4f}]')
        emit('')
        emit('  successive differences down the ladder (does it converge?)')
        for i in range(len(cuts) - 1):
            emit(f'    ({cuts[i][0]:g},{cuts[i][1]:g}) -> '
                 f'({cuts[i+1][0]:g},{cuts[i+1][1]:g}) : '
                 f'{reldiff(cuts[i][4], cuts[i+1][4]):.5f}')
        emit('')
        results[(shear, s_dip)] = dict(cuts=cuts, label=label, sz_min=sz_min,
                                       s_waist=s_waist, t=t)
        ip.near_cell, ip.near_grade = LADDER[0]

    # ---- figure ----------------------------------------------------------------
    n = len(POINTS)
    fig, axs = plt.subplots(2, n, figsize=(5.6 * n, 8.6), squeeze=False)
    for j, (shear, s_dip, label) in enumerate(POINTS):
        r = results[(shear, s_dip)]
        ax = axs[0][j]
        for nc, ng, nn, cw, dE in r['cuts']:
            ax.plot(dE, lw=1.3, label=f'nc={nc:g}, ng={ng:g} ({nn} nodes)')
        ax.set_title(f'{label}\nshear {shear:g}, {s_dip:.2f} m into dipole',
                     fontsize=10)
        ax.set_xlabel('z mesh index')
        ax.set_ylabel('dE/dct (MeV/m)')
        ax.legend(fontsize=7)
        ax.grid(alpha=0.25)

        ax = axs[1][j]
        ref = r['cuts'][-1][4]
        nn = [c[2] for c in r['cuts']]
        err = [reldiff(c[4], ref) for c in r['cuts']]
        ax.loglog(nn[:-1], np.maximum(err[:-1], 1e-12), 'o-', lw=1.7)
        ax.set_xlabel("near-region s' nodes")
        ax.set_ylabel('rel L2 vs finest grid')
        ax.set_title('quadrature convergence', fontsize=10)
        ax.grid(alpha=0.25, which='both')
    fig.suptitle("Refining the s' quadrature at fixed step_size. If the waist is a "
                 "quadrature problem, the left column collapses onto one curve and "
                 "the right column falls.", fontsize=12)
    fig.tight_layout()
    p = os.path.join(RESULT_DIR, 'waist_quadrature.png')
    fig.savefig(p, dpi=125, bbox_inches='tight')
    emit(f'Plot saved: {p}')

    with open(os.path.join(RESULT_DIR, 'waist_quadrature_log.txt'), 'w') as f:
        f.write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
