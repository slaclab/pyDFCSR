"""
Convergence test for the xi_bands quadrature, with the axes properly separated.

WHY THIS REPLACES THE EARLIER MARGIN SWEEP
------------------------------------------
The first acceptance test swept `xi_band_margin` at a FIXED node count and
concluded the answer was not domain-converged. That test was confounded: widening
the band at fixed node count also coarsens the cell, so it varied the integration
domain and the resolution at the same time. The observed change could have been
either, and the roughness rising with margin (0.18 -> 2.10) is the signature of
coarsening, not of a domain error.

This test varies one thing at a time. Each axis below is either an INVARIANCE (the
answer must not move at all) or a CONVERGENCE (the answer must settle):

  A. transverse domain, at FIXED cell size   -- INVARIANCE
        margin m with xbins proportional to m, so dx is constant. The extra nodes
        land where the density is zero, so the answer must not move.
  B. transverse resolution, fixed domain     -- CONVERGENCE
  C. longitudinal resolution, fixed domain   -- CONVERGENCE
  D. near-patch radius R2                    -- INVARIANCE
        The Cartesian piece carries weight w(r) and the patch carries 1 - w(r),
        and w + (1-w) = 1 identically, so the total cannot depend on where the
        crossover sits. If it does, either the partition is wrong or one of the
        two pieces is unresolved. This is the sharpest test of the polar patch.
  E. near-patch resolution (nr, nphi)        -- CONVERGENCE
  F. patch on vs off, both well resolved     -- INVARIANCE
        The patch is only a better quadrature of the same integral.

Axis D is the one to read first: it isolates the polar patch from everything else.
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

SHEAR = 20.0
STOP_S = 0.60
DEP_BINS = 200
N_PARTICLE = 200000

# Which deposition method to test. 'bspline_comoving' (poly_degree 1) is the only
# one that takes the two-branch band locator of Step 6; 'bspline_fft' falls back to
# the single-band locator and is kept selectable so the two can be compared.
METHOD = sys.argv[1] if len(sys.argv) > 1 else 'bspline_comoving'
POLY_DEGREE = 1 if METHOD == 'bspline_comoving' else 3

# the setting every axis perturbs away from
BASE = dict(xbins=200, zbins=200, xi_band_margin=2.0,
            near_patch=5.0, near_patch_nr=100, near_patch_nphi=180)


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
    with open(os.path.join(EXAMPLE_DIR, 'input/xibc_beam.yaml'), 'w') as f:
        yaml.dump(beam, f, default_flow_style=False, sort_keys=False)

    config = {
        'input_beam': {'style': 'distgen',
                       'distgen_input_file': 'input/xibc_beam.yaml'},
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
                            'write_name': 'xibc', 'workdir': './output'},
    }
    with open(os.path.join(EXAMPLE_DIR, 'input/xibc_config.yaml'), 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def roughness(w):
    d2 = w[2:] - 2 * w[1:-1] + w[:-2]
    n = np.linalg.norm(w)
    return np.linalg.norm(d2) / n if n > 0 else np.nan


def reldiff(a, b):
    n = np.linalg.norm(b)
    return np.linalg.norm(a - b) / n if n > 0 else np.nan


class Runner:
    def __init__(self, csr):
        self.csr = csr
        csr.get_CSR_mesh()
        self.nz = csr.CSR_params.zbins
        self.ix = csr.CSR_params.xbins // 2
        self.s0 = csr.beam.position
        self.cache = {}

    def __call__(self, **over):
        cfg = dict(BASE, **over)
        key = tuple(sorted(cfg.items()))
        if key in self.cache:
            return self.cache[key]
        ip = self.csr.integration_params
        ip.xi_bands = True
        for k, v in cfg.items():
            setattr(ip, k, v)
        dE = np.zeros(self.nz)
        xk = np.zeros(self.nz)
        for j in range(self.nz):
            k = self.ix * self.nz + j
            dE[j], xk[j] = self.csr.get_CSR_wake(self.s0 + self.csr.CSR_zmesh[k],
                                                 self.csr.CSR_xmesh[k])
        self.cache[key] = (dE, xk)
        return dE, xk


def main():
    os.makedirs(RESULT_DIR, exist_ok=True)
    write_inputs()
    os.chdir(EXAMPLE_DIR)

    csr = CSR2D(input_file='input/xibc_config.yaml')
    csr.run(stop_time=STOP_S)
    run = Runner(csr)

    sig_xi = csr.beam._sigma_x_transform
    lines = []

    def emit(s=''):
        print(s)
        lines.append(s)

    emit('=' * 78)
    emit('xi_bands convergence, one axis at a time')
    emit('=' * 78)
    emit(f'method = {METHOD} (poly_degree {POLY_DEGREE})')
    emit(f's = {run.s0:.4f} m, shear z:x = {SHEAR}, '
         f'|slope| = {abs(csr.beam._slope[0]):.3f}')
    emit(f'sigma_x = {csr.beam._sigma_x*1e6:.2f} um, '
         f'sigma_xi = {sig_xi*1e6:.2f} um, '
         f'amplification = {csr.beam._sigma_x/sig_xi:.1f} x')
    emit(f'base setting: {BASE}')
    emit('')

    axes = {}

    # ---- A. transverse domain at FIXED cell size -- INVARIANCE --------------
    emit('--- A. transverse DOMAIN at fixed cell size (INVARIANCE) ---')
    emit('margin m with xbins = 100*m, so dx is constant and the added nodes fall')
    emit('where the density is zero. Compare with the earlier confounded sweep,')
    emit('which held xbins fixed and therefore coarsened dx as it widened.')
    emit('')
    emit(f"  {'margin':>7} {'xbins':>7} {'rough(dE)':>11} {'dE vs m=1':>11} {'xk vs m=1':>11}")
    ref = run(xi_band_margin=1.0, xbins=100)
    rows = []
    for m in (1.0, 2.0, 4.0, 8.0):
        dE, xk = run(xi_band_margin=m, xbins=int(100 * m))
        rows.append((m, reldiff(dE, ref[0]), reldiff(xk, ref[1])))
        emit(f'  {m:>7.1f} {int(100*m):>7} {roughness(dE):>11.5f} '
             f'{reldiff(dE, ref[0]):>11.5f} {reldiff(xk, ref[1]):>11.5f}')
    axes['A domain (fixed dx)'] = rows
    emit('')
    emit('  For contrast, the SAME margin sweep at fixed xbins=200 (confounded):')
    emit(f"  {'margin':>7} {'xbins':>7} {'rough(dE)':>11} {'dE vs m=1':>11}")
    ref_bad = run(xi_band_margin=1.0)
    for m in (1.0, 2.0, 4.0, 8.0):
        dE, _ = run(xi_band_margin=m)
        emit(f'  {m:>7.1f} {BASE["xbins"]:>7} {roughness(dE):>11.5f} '
             f'{reldiff(dE, ref_bad[0]):>11.5f}')
    emit('')

    # ---- B. transverse resolution -- CONVERGENCE ---------------------------
    emit('--- B. transverse RESOLUTION at fixed domain (CONVERGENCE) ---')
    emit(f"  {'xbins':>7} {'cell/s_xi':>11} {'rough(dE)':>11} {'dE vs 800':>11} {'xk vs 800':>11}")
    ref = run(xbins=800)
    rows = []
    for n in (100, 200, 400, 800):
        dE, xk = run(xbins=n)
        rows.append((n, reldiff(dE, ref[0]), reldiff(xk, ref[1])))
        emit(f'  {n:>7} {2*BASE["xi_band_margin"]*5*sig_xi/n/sig_xi:>11.3f} '
             f'{roughness(dE):>11.5f} {reldiff(dE, ref[0]):>11.5f} '
             f'{reldiff(xk, ref[1]):>11.5f}')
    axes['B transverse res'] = rows
    emit('')

    # ---- C. longitudinal resolution -- CONVERGENCE -------------------------
    emit('--- C. longitudinal RESOLUTION (CONVERGENCE) ---')
    emit(f"  {'zbins':>7} {'rough(dE)':>11} {'dE vs 800':>11} {'xk vs 800':>11}")
    ref = run(zbins=800)
    rows = []
    for n in (100, 200, 400, 800):
        dE, xk = run(zbins=n)
        rows.append((n, reldiff(dE, ref[0]), reldiff(xk, ref[1])))
        emit(f'  {n:>7} {roughness(dE):>11.5f} {reldiff(dE, ref[0]):>11.5f} '
             f'{reldiff(xk, ref[1]):>11.5f}')
    axes['C longitudinal res'] = rows
    emit('')

    # ---- D. near-patch radius -- INVARIANCE --------------------------------
    emit('--- D. near-patch RADIUS (INVARIANCE) ---')
    emit('w(r) + (1 - w(r)) = 1 identically, so the total cannot depend on where')
    emit('the Cartesian/polar crossover sits. Any dependence means the partition')
    emit('is wrong or one of the two pieces is unresolved.')
    emit('')
    emit(f"  {'R2/s_xi':>8} {'rough(dE)':>11} {'dE vs 5 s_xi':>13} {'xk vs 5 s_xi':>13}")
    ref = run(near_patch=5.0)
    rows = []
    for p in (2.0, 3.0, 5.0, 8.0, 12.0):
        dE, xk = run(near_patch=p)
        rows.append((p, reldiff(dE, ref[0]), reldiff(xk, ref[1])))
        emit(f'  {p:>8.1f} {roughness(dE):>11.5f} {reldiff(dE, ref[0]):>13.5f} '
             f'{reldiff(xk, ref[1]):>13.5f}')
    axes['D patch radius'] = rows
    emit('')

    # ---- E. near-patch resolution -- CONVERGENCE ---------------------------
    emit('--- E. near-patch RESOLUTION (CONVERGENCE) ---')
    emit(f"  {'nr':>6} {'nphi':>6} {'rough(dE)':>11} {'dE vs finest':>13} {'xk vs finest':>13}")
    ref = run(near_patch_nr=800, near_patch_nphi=1440)
    rows = []
    for nr, nph in ((50, 90), (100, 180), (200, 360), (400, 720), (800, 1440)):
        dE, xk = run(near_patch_nr=nr, near_patch_nphi=nph)
        rows.append((nr, reldiff(dE, ref[0]), reldiff(xk, ref[1])))
        emit(f'  {nr:>6} {nph:>6} {roughness(dE):>11.5f} '
             f'{reldiff(dE, ref[0]):>13.5f} {reldiff(xk, ref[1]):>13.5f}')
    axes['E patch res'] = rows
    emit('')

    # ---- F. patch on vs off -- INVARIANCE ----------------------------------
    emit('--- F. patch ON vs OFF, both well resolved (INVARIANCE) ---')
    emit('The patch is only a better quadrature of the same integral, so at high')
    emit('resolution the two must agree. If they do not, the patch is double')
    emit('counting or dropping part of the domain.')
    emit('')
    emit(f"  {'xbins':>7} {'dE off vs on':>14} {'xk off vs on':>14} "
         f"{'rough off':>11} {'rough on':>11}")
    for n in (200, 400, 800):
        on = run(xbins=n, near_patch=5.0)
        off = run(xbins=n, near_patch=0.0)
        emit(f'  {n:>7} {reldiff(off[0], on[0]):>14.5f} '
             f'{reldiff(off[1], on[1]):>14.5f} '
             f'{roughness(off[0]):>11.5f} {roughness(on[0]):>11.5f}')
    emit('')

    # ---- verdict ------------------------------------------------------------
    emit('--- verdict ---')
    for name, rows in axes.items():
        worst = max(r[1] for r in rows[:-1]) if len(rows) > 1 else rows[0][1]
        kind = 'INVARIANCE' if 'domain' in name or 'radius' in name else 'CONVERGENCE'
        emit(f'  {name:<22} {kind:<12} worst dE deviation = {worst:.5f}')
    emit('')
    emit('An invariance axis should read ~0. A convergence axis should fall toward 0')
    emit('as the reference is approached.')

    # ---- plot ---------------------------------------------------------------
    fig, axs = plt.subplots(1, 4, figsize=(21, 5))
    for ax, (name, rows) in zip(axs, axes.items()):
        xv = [r[0] for r in rows]
        ax.loglog(xv, [max(r[1], 1e-16) for r in rows], 'ro-', label='dE/dct')
        ax.loglog(xv, [max(r[2], 1e-16) for r in rows], 'bs--', label='x_kick')
        ax.axhline(0.01, color='k', ls=':', alpha=0.6, label='1%')
        ax.set_title(name)
        ax.set_xlabel('axis value')
        ax.set_ylabel('rel. L2 deviation')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, which='both')
    fig.suptitle(f'xi_bands, method = {METHOD}: one axis at a time. Invariance '
                 'axes (A, D) must sit at zero; convergence axes (B, C) must fall.',
                 fontsize=13)
    plt.tight_layout()
    p = os.path.join(RESULT_DIR, f'xi_bands_convergence_{METHOD}.png')
    plt.savefig(p, dpi=130)
    plt.close()
    emit('')
    emit(f'Plot saved: {p}')

    with open(os.path.join(RESULT_DIR,
                           f'xi_bands_convergence_log_{METHOD}.txt'), 'w') as f:
        f.write('\n'.join(lines))


if __name__ == '__main__':
    main()
