"""
Step 4 acceptance test: does placing the transverse integration nodes on the
retarded density ribbon fix the high-tilt wake noise?

Step 3 found the noise is a sampling failure of the outer (x', s') quadrature:
get_CSR_wake sizes its transverse bands from sigma_x (lab, tilt included) while
the bspline_fft deposition grid is sized by sigma_xi (tilt removed). At 66x tilt
amplification the band is 264x wider than the density support, so fewer than one
quadrature node lands inside the beam and ~100% of nodes return a hard 0.0.

`xi_bands = True` places the nodes per-s'-column on the deposition grid's own
extent instead. Because the integrand is identically zero outside that grid, this
is an exact restriction of the domain, not an approximation.

Three things are measured, in increasing order of importance:

  1. Sampling.  Fraction of query points that return a hard zero, and the number
     of nodes across the density support. Should go from ~100% / 0.76 to small /
     many.
  2. Noise.     Roughness (relative L2 of the second difference of the wake along
     z). Should drop.
  3. CORRECTNESS. The converged answer must not change. The old bands with a very
     fine grid and the new bands with a modest grid are integrating the same
     function over the same non-zero region, so they must agree. If the new bands
     are smooth but converge to a *different* answer, they are clipping real
     density and the fix is wrong.

The simulation is run once and the wake is then recomputed in both modes, so both
see byte-identical density history.
"""
import sys
import os
import numpy as np
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import pyDFCSR_2D.CSR as CSRmod
from pyDFCSR_2D.CSR import CSR2D

RESULT_DIR = os.path.join(os.path.dirname(__file__), 'benchmark_results', 'xi_bands')
EXAMPLE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'example'))

SHEAR = 20.0
STOP_S = 0.60
DEP_BINS = 200
N_PARTICLE = 200000
GRIDS = [50, 100, 200, 400]
REF_LADDER = [400, 800, 1600]   # old-band refinement for the correctness check


def write_inputs():
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
        'transforms': {'s1': {'shear_coefficient': {'units': 'dimensionless',
                                                    'value': float(SHEAR)},
                              'type': 'shear z:x'}},
    }
    with open(os.path.join(EXAMPLE_DIR, 'input/xib_beam.yaml'), 'w') as f:
        yaml.dump(beam, f, default_flow_style=False, sort_keys=False)

    config = {
        'input_beam': {'style': 'distgen',
                       'distgen_input_file': 'input/xib_beam.yaml'},
        'input_lattice': {'lattice_input_file': 'input/dipole_lattice.yaml'},
        'particle_deposition': {'method': 'bspline_fft',
                                'xbins': DEP_BINS, 'zbins': DEP_BINS,
                                'xlim': 5, 'zlim': 5, 'smoothing_sigma': 3.0,
                                'poly_degree': 3, 'velocity_threhold': 1000},
        'CSR_integration': {'n_formation_length': 1.5, 'zbins': 100, 'xbins': 100},
        'CSR_computation': {'compute_CSR': 1, 'apply_CSR': 0, 'transverse_on': 1,
                            'xbins': 5, 'zbins': 40, 'xlim': 3, 'zlim': 3,
                            'write_beam': [], 'write_wakes': False,
                            'write_name': 'xib', 'workdir': './output'},
    }
    with open(os.path.join(EXAMPLE_DIR, 'input/xib_config.yaml'), 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


# ---------------------------------------------------------------------------
# Sampling diagnostic: how many query points actually hit the density?
# ---------------------------------------------------------------------------
_calls = []


def install_spy():
    orig = CSRmod.interpolate3D_transformed

    def spy(**kw):
        out = orig(**kw)
        if len(kw['xval']) > 1:
            _calls.append({'x': kw['xval'].copy(), 'z': kw['zval'].copy(),
                           't': kw['tval'].copy()})
        return out

    CSRmod.interpolate3D_transformed = spy
    return orig


def hard_zero_fraction(tracker, xq, zq, tq):
    """Fraction of query points for which bilinear_single returns a hard 0.0."""
    n_t, n_xi, n_z = tracker.data_density_interp.shape
    k = np.clip(((tq - tracker.min_x) / tracker.delta_x).astype(int),
                0, max(n_t - 2, 0))

    inside = np.ones(len(xq), dtype=bool)
    for off in (0, 1):
        kk = np.clip(k + off, 0, n_t - 1)
        p = tracker.poly_coeffs_interp[kk]
        xi = xq - np.einsum('ij,ij->i', p, np.vander(zq, p.shape[1]))
        x0 = ((xi - tracker.min_xi_arr[kk]) / tracker.delta_xi_arr[kk]).astype(int)
        y0 = ((zq - tracker.min_z_arr[kk]) / tracker.delta_z_arr[kk]).astype(int)
        inside &= (x0 >= 0) & (y0 >= 0) & (x0 < n_xi - 1) & (y0 < n_z - 1)
    return 1.0 - inside.mean()


def roughness(w):
    d2 = w[2:] - 2 * w[1:-1] + w[:-2]
    n = np.linalg.norm(w)
    return np.linalg.norm(d2) / n if n > 0 else np.nan


def wake_scan(csr, ix, nz, grid, xi_bands):
    csr.integration_params.xbins = grid
    csr.integration_params.zbins = grid
    csr.integration_params.xi_bands = xi_bands
    dE = np.zeros(nz)
    xk = np.zeros(nz)
    for j in range(nz):
        k = ix * nz + j
        dE[j], xk[j] = csr.get_CSR_wake(s_obs_global + csr.CSR_zmesh[k],
                                        csr.CSR_xmesh[k])
    return dE, xk


s_obs_global = None


def main():
    global s_obs_global
    os.makedirs(RESULT_DIR, exist_ok=True)
    write_inputs()
    os.chdir(EXAMPLE_DIR)

    orig_interp = install_spy()

    csr = CSR2D(input_file='input/xib_config.yaml')
    csr.run(stop_time=STOP_S)

    tracker = csr.DF_tracker
    s_obs_global = csr.beam.position
    csr.get_CSR_mesh()
    nx, nz = csr.CSR_params.xbins, csr.CSR_params.zbins
    ix = nx // 2
    zrange = csr.CSR_zrange

    lines = []

    def emit(s=''):
        print(s)
        lines.append(s)

    sig_x = csr.beam._sigma_x
    xi = csr.beam.x - np.polyval(csr.beam.slope, csr.beam.z)
    sig_xi = np.std(xi)

    emit('=' * 78)
    emit('Step 4: transverse integration nodes on the density ribbon (xi_bands)')
    emit('=' * 78)
    emit(f's_obs = {s_obs_global:.4f} m, shear z:x = {SHEAR}, '
         f'|slope| = {abs(csr.beam._slope[0]):.3f}')
    emit(f'sigma_x = {sig_x*1e6:.2f} um, sigma_xi = {sig_xi*1e6:.2f} um, '
         f'tilt amplification = {sig_x/sig_xi:.1f} x')
    emit(f'deposition {tracker.data_density_interp.shape}, '
         f'history {len(tracker.DF_log)} snapshots')
    emit('')

    # ---- (1) sampling --------------------------------------------------------
    emit('--- (1) sampling: fraction of query points returning a hard 0.0 ---')
    emit('')
    k_mid = ix * nz + nz // 2
    s_q = s_obs_global + csr.CSR_zmesh[k_mid]
    x_q = csr.CSR_xmesh[k_mid]

    emit(f"{'mode':<12} {'grid':>6} {'n_query':>10} {'hard-zero %':>13}")
    emit('-' * 44)
    frac = {}
    for xb in (False, True):
        for g in (100, 400):
            csr.integration_params.xbins = g
            csr.integration_params.zbins = g
            csr.integration_params.xi_bands = xb
            _calls.clear()
            csr.get_CSR_wake(s_q, x_q)
            xs = np.concatenate([c['x'] for c in _calls])
            zs = np.concatenate([c['z'] for c in _calls])
            ts = np.concatenate([c['t'] for c in _calls])
            f = hard_zero_fraction(tracker, xs, zs, ts)
            frac[(xb, g)] = f
            emit(f"{'xi_bands' if xb else 'old (sigma_x)':<12} {g:>6} "
                 f"{len(xs):>10} {100*f:>13.2f}")
    emit('')

    # achieved transverse cell size, per s' region
    emit('--- located band width and achieved cell size (xi_bands, 100 nodes) ---')
    emit('The band must cover the UNION of the two snapshots the lab-frame time')
    emit('blend mixes, so at large tilt-rate it is set by the ghost separation')
    emit('rather than by sigma_xi. Fixing the blend would tighten it further.')
    emit('')
    sz = csr.beam._sigma_z
    s2_ = s_q - 500 * sz
    s3_ = s_q - 20 * sz
    s4_ = s_q + 5 * sz
    s1_ = max(0.0, s2_ - csr.integration_params.n_formation_length * csr.formation_length)
    emit(f"  {'region':<10} {'width min':>12} {'width med':>12} {'width max':>12} "
         f"{'cell/s_xi med':>14} {'cell/s_xi max':>14}")
    for nm, (a, b) in zip(('sp1', 'sp2', 'sp3'),
                          ((s1_, s2_), (s2_, s3_), (s3_, s4_))):
        spk = np.linspace(a, b, 100)
        lo_, hi_ = csr._retarded_xi_band(s_q, x_q, s_obs_global, spk)
        w = hi_ - lo_
        cell = w / 99
        emit(f'  {nm:<10} {w.min()*1e6:>11.1f}u {np.median(w)*1e6:>11.1f}u '
             f'{w.max()*1e6:>11.1f}u {np.median(cell)/sig_xi:>14.2f} '
             f'{cell.max()/sig_xi:>14.2f}')
    emit('')
    emit('  (old bands, for comparison: cell = 13.2 sigma_xi at 100 nodes)')
    emit('')

    # The spy retains every query array; at the grids used below that is GBs.
    CSRmod.interpolate3D_transformed = orig_interp
    _calls.clear()

    # ---- (2) noise -----------------------------------------------------------
    emit('--- (2) noise: wake roughness vs integration grid, both modes ---')
    emit('')
    scans = {}
    for xb in (False, True):
        for g in GRIDS:
            scans[(xb, g)] = wake_scan(csr, ix, nz, g, xb)

    emit(f"{'mode':<14} {'grid':>6} {'rough(dE)':>11} {'rough(xk)':>11}")
    emit('-' * 45)
    for xb in (False, True):
        for g in GRIDS:
            dE, xk = scans[(xb, g)]
            emit(f"{'xi_bands' if xb else 'old (sigma_x)':<14} {g:>6} "
                 f"{roughness(dE):>11.5f} {roughness(xk):>11.5f}")
    emit('')

    # ---- (3) correctness -----------------------------------------------------
    # The two modes integrate the same function over the same non-zero region, so
    # they must converge to the same answer. Rather than assume either one is the
    # reference, refine BOTH and ask whether they meet. The old bands need a very
    # fine grid to resolve a density support 264x narrower than the band, so their
    # ladder has to be pushed much further.
    emit('--- (3) correctness: do the two modes converge to the same answer? ---')
    emit('')
    xi_best = wake_scan(csr, ix, nz, 800, True)
    for g in REF_LADDER:
        scans[(False, g)] = wake_scan(csr, ix, nz, g, False)

    emit(f"{'old-band grid':>14} {'cell/s_xi':>10} {'rough(dE)':>11} "
         f"{'vs xi_bands 800^2':>18} {'vs prev old':>12}")
    emit('-' * 70)
    prev = None
    for g in REF_LADDER:
        dE, xk = scans[(False, g)]
        d = np.linalg.norm(dE - xi_best[0]) / np.linalg.norm(xi_best[0])
        pv = ('' if prev is None else
              f'{np.linalg.norm(dE - prev) / np.linalg.norm(dE):12.5f}')
        # region 1 uses 2*xbins nodes over +-20 sigma_x
        cell = 2 * 20 * sig_x / (2 * g) / sig_xi
        emit(f'{g:>14} {cell:>10.2f} {roughness(dE):>11.5f} {d:>18.5f} {pv:>12}')
        prev = dE
    emit('')
    emit('If the old bands walk monotonically toward the xi_bands answer as their')
    emit('grid is refined, the xi_bands answer is the converged one and the old')
    emit('bands were simply undersampled. If they settle somewhere else, xi_bands')
    emit('is clipping real density.')
    emit('')

    # self-convergence of xi_bands
    emit('--- xi_bands self-convergence (vs its own 800^2) ---')
    for g in GRIDS:
        dE, xk = scans[(True, g)]
        e = np.linalg.norm(dE - xi_best[0]) / np.linalg.norm(xi_best[0])
        ex = np.linalg.norm(xk - xi_best[1]) / np.linalg.norm(xi_best[1])
        emit(f'  grid {g:>4}^2 : rel. diff dE = {e:.5f}   x_kick = {ex:.5f}')
    emit('')
    ref_dE, ref_xk = scans[(False, REF_LADDER[-1])]

    # ---- plots ---------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    for row, (idx, lab) in enumerate(((0, 'dE/dct (MeV/m)'), (1, 'x_kick'))):
        ax = axes[row, 0]
        for g, c in zip(GRIDS, ('#cccccc', '#9999ff', '#ff9999', '#333333')):
            ax.plot(zrange * 1e6, scans[(False, g)][idx], '-o', color=c, ms=3,
                    label=f'{g}$^2$')
        ax.plot(zrange * 1e6, (ref_dE, ref_xk)[idx], 'g-', lw=2,
                label=f'{REF_LADDER[-1]}$^2$')
        ax.set_title(f'OLD bands (sigma_x)\n{lab}')
        ax.set_xlabel(r'z ($\mu$m)')
        ax.set_ylabel(lab)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

        ax = axes[row, 1]
        for g, c in zip(GRIDS, ('#cccccc', '#9999ff', '#ff9999', '#333333')):
            ax.plot(zrange * 1e6, scans[(True, g)][idx], '-o', color=c, ms=3,
                    label=f'{g}$^2$')
        ax.plot(zrange * 1e6, (ref_dE, ref_xk)[idx], 'g-', lw=2,
                label=f'old {REF_LADDER[-1]}$^2$')
        ax.set_title(f'NEW xi_bands\n{lab}')
        ax.set_xlabel(r'z ($\mu$m)')
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

        ax = axes[row, 2]
        gg = np.array(GRIDS, dtype=float)
        for xb, mk, nm in ((False, 'bs--', 'old (sigma_x)'), (True, 'ro-', 'xi_bands')):
            ax.loglog(gg, [roughness(scans[(xb, g)][idx]) for g in GRIDS], mk,
                      label=nm)
        ax.set_xlabel('integration bins per dimension')
        ax.set_ylabel('roughness')
        ax.set_title(f'Noise vs resolution\n{lab}')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, which='both')

    fig.suptitle('Step 4: transverse nodes on the density ribbon vs on a '
                 f'sigma_x rectangle\ns = {s_obs_global:.3f} m, tilt '
                 f'amplification {sig_x/sig_xi:.0f}x', fontsize=13)
    plt.tight_layout()
    p = os.path.join(RESULT_DIR, 'xi_bands_ab.png')
    plt.savefig(p, dpi=130)
    plt.close()
    emit(f'Plot saved: {p}')

    with open(os.path.join(RESULT_DIR, 'xi_bands_log.txt'), 'w') as f:
        f.write('\n'.join(lines))


if __name__ == '__main__':
    main()
