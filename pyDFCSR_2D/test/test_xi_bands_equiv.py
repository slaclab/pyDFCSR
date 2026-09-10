"""
Falsification test for the xi_bands quadrature fix.

The fix restricts the transverse integration domain from a +-20*sigma_x rectangle
in lab x' to the deposition grid's own support, per s' column. The claim is that
this is EXACT, because the interpolant returns identically 0.0 outside that grid,
so the discarded nodes contributed nothing.

At high tilt that claim is hard to check directly: the old bands need an enormous
grid before they resolve a support 264x narrower than themselves, so there is no
cheap trusted reference to compare against.

At LOW tilt there is. When sigma_x ~ sigma_xi the old bands and the deposition
grid are the same size, the old quadrature is well sampled, and both modes must
give the same answer at modest resolution. So:

    shear = 0   -> tilt amplification ~ 1   -> the two modes MUST agree
    shear = 2   -> mild                     -> should still agree
    shear = 20  -> 66x                       -> old bands unconverged; expect a gap

If the two modes disagree at shear = 0, the band construction is losing real
density and the fix is wrong. If they agree at shear = 0 and diverge only where
the old bands are demonstrably undersampled, the gap at high tilt is the old
bands' error, not the new bands'.

This is the test that distinguishes "my fix is right" from "my fix is smooth and
wrong", so it is the one that matters most.
"""
import sys
import os
import numpy as np
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from pyDFCSR_2D.CSR import CSR2D

RESULT_DIR = os.path.join(os.path.dirname(__file__), 'benchmark_results', 'xi_bands')
EXAMPLE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'example'))

SHEARS = [0.0, 2.0, 20.0]

# Which deposition method. 'bspline_comoving' (poly_degree 1) is the only one that
# takes the two-branch band locator of Step 6, so it is the default: this test is
# what distinguishes "the two-branch bands are right" from "smooth and wrong".
METHOD = sys.argv[1] if len(sys.argv) > 1 else 'bspline_comoving'
POLY_DEGREE = 1 if METHOD == 'bspline_comoving' else 3
STOP_S = 0.60
DEP_BINS = 200
N_PARTICLE = 200000
GRIDS = [100, 200, 400, 800]


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
    with open(os.path.join(EXAMPLE_DIR, f'input/xibe_beam_{tag}.yaml'), 'w') as f:
        yaml.dump(beam, f, default_flow_style=False, sort_keys=False)

    config = {
        'input_beam': {'style': 'distgen',
                       'distgen_input_file': f'input/xibe_beam_{tag}.yaml'},
        'input_lattice': {'lattice_input_file': 'input/dipole_lattice.yaml'},
        'particle_deposition': {'method': METHOD,
                                'xbins': DEP_BINS, 'zbins': DEP_BINS,
                                'xlim': 5, 'zlim': 5, 'smoothing_sigma': 3.0,
                                'poly_degree': POLY_DEGREE,
                                'velocity_threhold': 1000},
        'CSR_integration': {'n_formation_length': 1.5, 'zbins': 100, 'xbins': 100},
        'CSR_computation': {'compute_CSR': 1, 'apply_CSR': 0, 'transverse_on': 1,
                            'xbins': 5, 'zbins': 40, 'xlim': 3, 'zlim': 3,
                            'write_beam': [], 'write_wakes': False,
                            'write_name': f'xibe_{tag}', 'workdir': './output'},
    }
    p = os.path.join(EXAMPLE_DIR, f'input/xibe_config_{tag}.yaml')
    with open(p, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    return f'input/xibe_config_{tag}.yaml'


def roughness(w):
    d2 = w[2:] - 2 * w[1:-1] + w[:-2]
    n = np.linalg.norm(w)
    return np.linalg.norm(d2) / n if n > 0 else np.nan


def reldiff(a, b):
    n = np.linalg.norm(b)
    return np.linalg.norm(a - b) / n if n > 0 else np.nan


def run_case(shear, tag):
    cfg = write_inputs(shear, tag)
    os.chdir(EXAMPLE_DIR)
    csr = CSR2D(input_file=cfg)
    csr.run(stop_time=STOP_S)
    csr.get_CSR_mesh()

    s_obs = csr.beam.position
    nx, nz = csr.CSR_params.xbins, csr.CSR_params.zbins
    ix = nx // 2

    sig_x = csr.beam._sigma_x
    xi = csr.beam.x - np.polyval(csr.beam.slope, csr.beam.z)
    sig_xi = np.std(xi)

    out = {'shear': shear, 'amp': sig_x / sig_xi, 'slope': csr.beam._slope[0],
           'zrange': csr.CSR_zrange.copy(), 'scans': {}}

    for xb in (False, True):
        for g in GRIDS:
            csr.integration_params.xi_bands = xb
            csr.integration_params.xbins = g
            csr.integration_params.zbins = g
            dE = np.zeros(nz)
            xk = np.zeros(nz)
            for j in range(nz):
                k = ix * nz + j
                dE[j], xk[j] = csr.get_CSR_wake(s_obs + csr.CSR_zmesh[k],
                                                csr.CSR_xmesh[k])
            out['scans'][(xb, g)] = (dE, xk)
    return out


def main():
    os.makedirs(RESULT_DIR, exist_ok=True)
    lines = []

    def emit(s=''):
        print(s)
        lines.append(s)

    results = []
    for shear in SHEARS:
        tag = f's{shear:g}'.replace('.', 'p')
        print(f'\n{"="*70}\n  shear z:x = {shear}\n{"="*70}')
        results.append(run_case(shear, tag))

    emit('=' * 78)
    emit('xi_bands falsification test: do old and new bands agree where the old')
    emit('bands are well sampled?')
    emit('=' * 78)
    emit('')
    emit('Each mode is refined independently. "old vs new" compares the two at')
    emit('their finest common grid (800^2). At low tilt the old bands resolve the')
    emit('density and the two MUST agree; disagreement there falsifies the fix.')
    emit('')
    emit('')

    for r in results:
        emit(f"--- shear = {r['shear']:g}, |slope| = {abs(r['slope']):.3f}, "
             f"tilt amplification = {r['amp']:.1f} x ---")
        emit(f"  {'grid':>6} {'rough old':>11} {'rough new':>11} "
             f"{'old self-conv':>14} {'new self-conv':>14} {'old vs new':>12}")
        best_old = r['scans'][(False, GRIDS[-1])][0]
        best_new = r['scans'][(True, GRIDS[-1])][0]
        for g in GRIDS:
            o = r['scans'][(False, g)][0]
            n = r['scans'][(True, g)][0]
            so = reldiff(o, best_old)
            sn = reldiff(n, best_new)
            emit(f'  {g:>6} {roughness(o):>11.5f} {roughness(n):>11.5f} '
                 f'{so:>14.5f} {sn:>14.5f} {reldiff(o, n):>12.5f}')
        emit(f'  old vs new at {GRIDS[-1]}^2: '
             f'dE {reldiff(best_old, best_new):.5f}, '
             f"x_kick {reldiff(r['scans'][(False, GRIDS[-1])][1], r['scans'][(True, GRIDS[-1])][1]):.5f}")
        emit('')

    # ---- plot ---------------------------------------------------------------
    fig, axes = plt.subplots(2, len(results), figsize=(6 * len(results), 9))
    for c, r in enumerate(results):
        for row, idx, lab in ((0, 0, 'dE/dct (MeV/m)'), (1, 1, 'x_kick')):
            ax = axes[row, c]
            z = r['zrange'] * 1e6
            ax.plot(z, r['scans'][(False, GRIDS[-1])][idx], 'b-o', ms=3,
                    label=f'old bands {GRIDS[-1]}$^2$')
            ax.plot(z, r['scans'][(True, GRIDS[-1])][idx], 'r-s', ms=3,
                    label=f'xi_bands {GRIDS[-1]}$^2$')
            ax.plot(z, r['scans'][(True, 100)][idx], 'r--', alpha=0.6,
                    label='xi_bands 100$^2$')
            ax.plot(z, r['scans'][(False, 100)][idx], 'b:', alpha=0.6,
                    label='old bands 100$^2$')
            ax.set_xlabel(r'z ($\mu$m)')
            ax.set_ylabel(lab)
            ax.set_title(f"shear = {r['shear']:g}, amplification "
                         f"{r['amp']:.0f}x\n{lab}")
            ax.legend(fontsize=7)
            ax.grid(alpha=0.3)

    fig.suptitle('Old vs new transverse integration bands across tilt\n'
                 'they must agree where the old bands are well sampled',
                 fontsize=13)
    plt.tight_layout()
    p = os.path.join(RESULT_DIR, f'xi_bands_equiv_{METHOD}.png')
    plt.savefig(p, dpi=130)
    plt.close()
    emit(f'Plot saved: {p}')

    with open(os.path.join(RESULT_DIR,
                           f'xi_bands_equiv_log_{METHOD}.txt'), 'w') as f:
        f.write('\n'.join(lines))


if __name__ == '__main__':
    main()
