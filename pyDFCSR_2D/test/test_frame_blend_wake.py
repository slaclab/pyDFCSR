"""
Does frame_blend = 'moment' make the WAKE more accurate across a full-compression
point, or only smoother?

The distinction matters and has bitten this work before (progress 6o): dropping the
(x - xmean) term from the near-region extent made the wake look 4x smoother while
converging to the same answer, i.e. it removed node-layout jitter from the roughness
METRIC rather than error from the result. Roughness alone cannot tell the two apart.

So the test here is CONVERGENCE, not smoothness. The frame blend is an interpolation
between snapshots, so its error must vanish as the snapshot spacing does. Refine
step_size and ask which mode is closer to the finely-sampled reference. A mode that is
merely smoother will not approach the reference faster; a mode that is more accurate
will.

The reference is the finest step_size available, computed in BOTH modes: at fine
sampling they must agree with each other, which is what makes it a reference rather
than a preference. If they disagree at the finest step the comparison is void and the
script says so.

Two observation points:
  0.10 m into the dipole -- the near region contains the compression point
  0.45 m into the dipole -- clean, to confirm no regression away from the waist
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
RESULT_DIR = os.path.join(os.path.dirname(__file__), 'benchmark_results', 'flip')

SHEAR = 20.0
DRIFT = 1.0
STEPS = (0.05, 0.025, 0.0125, 0.00625)
MODES = ('coeff', 'moment', 'orient')
# one colour per mode; a short tuple here silently drops modes from the figure
COLORS = {'coeff': '#1f77b4', 'moment': '#d62728', 'orient': '#2ca02c'}
OBS = (0.10, 0.45)          # metres into the dipole
MESH_XBINS, MESH_ZBINS = 21, 51


def write_inputs(tag, step, frame_blend):
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
    with open(os.path.join(EXAMPLE_DIR, f'input/fbw_beam_{tag}.yaml'), 'w') as f:
        yaml.dump(beam, f, default_flow_style=False, sort_keys=False)

    lat = {'step_size': step,
           'element_1': {'type': 'drift', 'L': DRIFT, 'nsep': 1},
           'element_2': {'type': 'dipole', 'L': 1.0, 'angle': 1.0,
                         'E1': 0, 'E2': 0, 'FINT': 0.0, 'FINTX': 0.5,
                         'HGAP': 0.1, 'HGAPX': 0.0, 'FRINGE_AT': 'both_ends',
                         'FRINGE_TYPE': 'linear_edge', 'TILT': 0.0, 'nsep': 1},
           'element_3': {'type': 'drift', 'L': 0.5, 'nsep': 1}}
    lp = f'input/fbw_lat_{tag}.yaml'
    with open(os.path.join(EXAMPLE_DIR, lp), 'w') as f:
        yaml.dump(lat, f, default_flow_style=False, sort_keys=False)

    cfg = {
        'input_beam': {'style': 'distgen',
                       'distgen_input_file': f'input/fbw_beam_{tag}.yaml'},
        'input_lattice': {'lattice_input_file': lp},
        'particle_deposition': {'method': 'bspline_comoving',
                                'xbins': 200, 'zbins': 200,
                                'xlim': 5, 'zlim': 5, 'smoothing_sigma': 3.0,
                                'poly_degree': 1, 'velocity_threhold': 1000,
                                'frame_blend': frame_blend},
        'CSR_integration': {'n_formation_length': 1.5, 'zbins': 200, 'xbins': 200},
        'CSR_computation': {'compute_CSR': 0, 'apply_CSR': 0, 'transverse_on': 1,
                            'xbins': MESH_XBINS, 'zbins': MESH_ZBINS,
                            'xlim': 3, 'zlim': 3,
                            'write_beam': [], 'write_wakes': False,
                            'write_name': f'fbw_{tag}', 'workdir': './output'},
    }
    p = f'input/fbw_config_{tag}.yaml'
    with open(os.path.join(EXAMPLE_DIR, p), 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    return p


def roughness(w):
    d2 = w[2:] - 2 * w[1:-1] + w[:-2]
    n = np.linalg.norm(w)
    return np.linalg.norm(d2) / n if n > 0 else np.nan


def abs_roughness(w):
    d2 = w[2:] - 2 * w[1:-1] + w[:-2]
    return np.linalg.norm(d2)


def reldiff(a, b):
    n = np.linalg.norm(b)
    return np.linalg.norm(a - b) / n if n > 0 else np.nan


def z_scan(csr):
    """Longitudinal wake cut at mid-x."""
    nz = csr.CSR_params.zbins
    ix = csr.CSR_params.xbins // 2
    s0 = csr.beam.position
    dE = np.zeros(nz)
    xk = np.zeros(nz)
    for j in range(nz):
        k = ix * nz + j
        dE[j], xk[j] = csr.get_CSR_wake(s0 + csr.CSR_zmesh[k], csr.CSR_xmesh[k])
    return dE, xk


def run_case(step, mode, s_dip):
    tag = f'{mode}_{str(step).replace(".", "p")}_{str(s_dip).replace(".", "p")}'
    cfg = write_inputs(tag, step, mode)
    csr = CSR2D(input_file=cfg)
    # debug=True builds the density history even though compute_CSR = 0; the wake is
    # then computed once, here, instead of at every tracking step.
    #
    # run() stops after the first step whose END reaches stop_time, and the tracked
    # position accumulates step by step, so asking for exactly the target can land one
    # step past it (1.0/0.0125 accumulates to 1.09999... < 1.1, which buys another
    # step). Asking half a step early lands on the target for every step_size. This
    # matters more than it looks: comparing step sizes at DIFFERENT observation points
    # would confound the convergence study with the wake's variation along s.
    target = DRIFT + s_dip
    csr.run(stop_time=target - 0.5 * step, debug=True)
    csr.get_CSR_mesh()
    assert abs(csr.beam.position - target) < 1e-6, (
        f'landed at {csr.beam.position}, wanted {target} (step {step})')
    dE, xk = z_scan(csr)
    return dE, xk, csr


def main():
    os.makedirs(RESULT_DIR, exist_ok=True)
    os.chdir(EXAMPLE_DIR)
    lines = []

    def emit(s=''):
        print(s, flush=True)
        lines.append(s)

    emit('=' * 92)
    emit('frame_blend: does moment blending make the wake more ACCURATE, or only '
         'smoother?')
    emit('=' * 92)
    emit(f'shear {SHEAR:g}, drift {DRIFT} m, wake mesh {MESH_XBINS}x{MESH_ZBINS}, '
         f'z-cut at mid-x')
    emit(f'step_size sweep {STEPS}, modes {MODES}')
    emit()

    res = {}
    for s_dip in OBS:
        for step in STEPS:
            for mode in MODES:
                dE, xk, csr = run_case(step, mode, s_dip)
                res[(s_dip, step, mode)] = (dE, xk)
                emit(f'  ran s_dip={s_dip:.3f} step={step:<7g} mode={mode:<7} '
                     f'sig_z={csr.beam._sigma_z*1e6:8.2f}um  '
                     f'dE range [{dE.min():+.4f}, {dE.max():+.4f}]')
    emit()

    fine = STEPS[-1]
    for s_dip in OBS:
        emit('-' * 92)
        emit(f'OBSERVATION POINT: {s_dip:.3f} m into the dipole')
        emit('-' * 92)
        emit('  Deliberately NOT measured against a blend of the modes: averaging')
        emit('  answers that disagree puts each of them half the disagreement from the')
        emit('  "reference" by construction, which says nothing. Instead:')
        emit('    (a) SELF-CONVERGENCE -- successive step sizes, per mode. Every mode')
        emit('        is exact at snapshot times, so every mode must converge as')
        emit('        h -> 0. A mode whose successive differences do not shrink is')
        emit('        broken, however smooth its output looks.')
        emit('    (b) CROSS-MODE GAPS -- modes that converge share a limit, so every')
        emit('        pairwise gap must shrink too.')
        emit()

        emit('  (a) self-convergence, rel L2 between consecutive step sizes')
        emit(f"      {'steps':>18} " + ' '.join(f'{m:>12}' for m in MODES))
        for i in range(len(STEPS) - 1):
            row = []
            for mode in MODES:
                row.append(reldiff(res[(s_dip, STEPS[i], mode)][0],
                                   res[(s_dip, STEPS[i + 1], mode)][0]))
            emit(f'      {STEPS[i]:>8g} -> {STEPS[i+1]:<7g} '
                 + ' '.join(f'{v:>12.5f}' for v in row))
        emit()

        emit('  (b) cross-mode gaps at each step size (all pairs)')
        pairs = [(m1, m2) for i, m1 in enumerate(MODES) for m2 in MODES[i + 1:]]
        emit(f"      {'step':>9} "
             + ' '.join(f'{m1[:3] + "-" + m2[:3]:>12}' for m1, m2 in pairs))
        first, last = {}, {}
        for step in STEPS:
            row = []
            for m1, m2 in pairs:
                g = reldiff(res[(s_dip, step, m1)][0], res[(s_dip, step, m2)][0])
                row.append(g)
                if step == STEPS[0]:
                    first[(m1, m2)] = g
                last[(m1, m2)] = g
            emit(f'      {step:>9g} ' + ' '.join(f'{v:>12.5f}' for v in row))
        for pr in pairs:
            if last[pr] >= first[pr]:
                emit(f'      *** {pr[0]} and {pr[1]} do NOT approach each other '
                     f'({first[pr]:.5f} -> {last[pr]:.5f}). Both are exact at nodes,')
                emit('          so they must share an h -> 0 limit; a gap that will')
                emit('          not close is a defect, not a trade-off.')
        emit()

        emit('  roughness (reported, but NOT used to decide accuracy)')
        emit(f"      {'step':>9} {'mode':>8} {'rel rough':>11} {'abs rough':>11}")
        for step in STEPS:
            for mode in MODES:
                dE = res[(s_dip, step, mode)][0]
                emit(f'      {step:>9g} {mode:>8} {roughness(dE):>11.5f} '
                     f'{abs_roughness(dE):>11.5f}')
        emit()

    # ---- figure ----------------------------------------------------------------
    fig, axes = plt.subplots(2, len(OBS), figsize=(7.2 * len(OBS), 9.0),
                             squeeze=False)
    for c, s_dip in enumerate(OBS):
        ax = axes[0][c]
        for mode in MODES:
            col = COLORS[mode]
            ax.plot(res[(s_dip, STEPS[0], mode)][0], color=col, lw=1.6,
                    label=f'{mode}, step {STEPS[0]:g}')
            ax.plot(res[(s_dip, fine, mode)][0], color=col, lw=1.1, ls='--',
                    label=f'{mode}, step {fine:g}')
        ax.set_title(f'dE/dct along z at mid-x, {s_dip:.2f} m into the dipole',
                     fontsize=10)
        ax.set_xlabel('z mesh index')
        ax.set_ylabel('dE/dct  (MeV/m)')
        ax.legend(fontsize=8.5)
        ax.grid(alpha=0.25)

        ax = axes[1][c]
        for mode in MODES:
            col = COLORS[mode]
            errs = [reldiff(res[(s_dip, STEPS[i], mode)][0],
                            res[(s_dip, STEPS[i + 1], mode)][0])
                    for i in range(len(STEPS) - 1)]
            ax.loglog(STEPS[:-1], errs, 'o-', color=col, lw=1.7,
                      label=f'{mode} self-conv')
        gaps = [reldiff(res[(s_dip, st, 'coeff')][0],
                        res[(s_dip, st, 'orient')][0]) for st in STEPS]
        ax.loglog(STEPS, gaps, 's--', color='k', lw=1.4, label='coeff vs orient')
        ax.set_title('self-convergence and cross-mode gap', fontsize=10)
        ax.set_xlabel('step_size (m)')
        ax.set_ylabel('rel L2')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.25, which='both')

    fig.tight_layout()
    out = os.path.join(RESULT_DIR, 'frame_blend_wake.png')
    fig.savefig(out, dpi=135, bbox_inches='tight')
    emit(f'Plot saved: {out}')

    with open(os.path.join(RESULT_DIR, 'frame_blend_wake_log.txt'), 'w') as f:
        f.write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
