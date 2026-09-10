"""
Does two-branch band location actually fix the missing chirp branch (Step 6)?

Axis A of test_xi_bands_converge.py is the acceptance test: widen the transverse
domain at FIXED cell size, so every added node lands where the density is zero and
the answer must not move. It read 1.017 with the single-band locator -- i.e. the
band was missing support that a wider domain kept finding.

That test alone cannot attribute the improvement, because the co-moving interpolant
(Step 5) and the two-branch locator (Step 6) landed together. This script runs the
SAME beam, the SAME interpolant, and toggles ONLY the locator, by monkeypatching
_retarded_xi_bands back to the single-band behaviour. So the difference measured
here is caused by branch location and nothing else.

It also produces the instrumentation §6b asked for before the rad < 0 guard is
trusted: the count of columns where the radicand goes negative, binned by |r - r'|.
The concern is the annulus R2 < r < ~20 sigma_xi, where the polar patch's taper has
already reached w = 1 but the two branches are still closer together than a band
width. A spurious rad < 0 there would silently drop real support, and the wake would
be quietly wrong rather than obviously wrong.
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
    with open(os.path.join(EXAMPLE_DIR, 'input/tbb_beam.yaml'), 'w') as f:
        yaml.dump(beam, f, default_flow_style=False, sort_keys=False)

    config = {
        'input_beam': {'style': 'distgen',
                       'distgen_input_file': 'input/tbb_beam.yaml'},
        'input_lattice': {'lattice_input_file': 'input/dipole_lattice.yaml'},
        'particle_deposition': {'method': 'bspline_comoving',
                                'xbins': DEP_BINS, 'zbins': DEP_BINS,
                                'xlim': 5, 'zlim': 5, 'smoothing_sigma': 3.0,
                                'poly_degree': 1, 'velocity_threhold': 1000},
        'CSR_integration': {'n_formation_length': 1.5, 'zbins': 100, 'xbins': 100},
        'CSR_computation': {'compute_CSR': 1, 'apply_CSR': 0, 'transverse_on': 1,
                            'xbins': 5, 'zbins': 40, 'xlim': 3, 'zlim': 3,
                            'write_beam': [], 'write_wakes': False,
                            'write_name': 'tbb', 'workdir': './output'},
    }
    with open(os.path.join(EXAMPLE_DIR, 'input/tbb_config.yaml'), 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def roughness(w):
    d2 = w[2:] - 2 * w[1:-1] + w[:-2]
    n = np.linalg.norm(w)
    return np.linalg.norm(d2) / n if n > 0 else np.nan


def reldiff(a, b):
    n = np.linalg.norm(b)
    return np.linalg.norm(a - b) / n if n > 0 else np.nan


def z_scan(csr, **over):
    """Wake along z at mid-x, with the integration params overridden."""
    ip = csr.integration_params
    ip.xi_bands = True
    for k, v in dict(BASE, **over).items():
        setattr(ip, k, v)
    nz = csr.CSR_params.zbins
    ix = csr.CSR_params.xbins // 2
    s0 = csr.beam.position
    dE = np.zeros(nz)
    xk = np.zeros(nz)
    for j in range(nz):
        k = ix * nz + j
        dE[j], xk[j] = csr.get_CSR_wake(s0 + csr.CSR_zmesh[k], csr.CSR_xmesh[k])
    return dE, xk


def main():
    os.makedirs(RESULT_DIR, exist_ok=True)
    write_inputs()
    os.chdir(EXAMPLE_DIR)

    csr = CSR2D(input_file='input/tbb_config.yaml')
    csr.run(stop_time=STOP_S)
    csr.get_CSR_mesh()

    sig_xi = csr.beam._sigma_x_transform
    lines = []

    def emit(s=''):
        print(s)
        lines.append(s)

    two_branch = CSR2D._retarded_xi_bands

    def one_branch(self, s, x, t, sp, n_iter=12):
        return [self._retarded_xi_band(s, x, t, sp)]

    emit('=' * 80)
    emit('Two-branch band location (Step 6) vs single-branch, same interpolant')
    emit('=' * 80)
    emit(f's = {csr.beam.position:.4f} m, shear z:x = {SHEAR}, '
         f'slope = {csr.beam._slope[0]:+.4f}')
    emit(f'sigma_x = {csr.beam._sigma_x*1e6:.2f} um, sigma_xi = {sig_xi*1e6:.2f} um, '
         f'amplification = {csr.beam._sigma_x/sig_xi:.1f} x')
    emit('method = bspline_comoving (poly_degree 1) in BOTH columns, so the only')
    emit('difference is how many localization branches the band locator finds.')
    emit('')

    # ---- ACCEPTANCE: axis A, both locators ---------------------------------
    emit('--- Axis A: transverse DOMAIN at fixed cell size (must be INVARIANT) ---')
    emit('margin m with xbins = 100*m, so the cell size is constant and every')
    emit('added node lands where the density is zero. Deviations are vs m = 1.')
    emit('')
    emit(f"  {'':<14} {'margin':>7} {'xbins':>7} {'rough(dE)':>11} "
         f"{'dE vs m=1':>11} {'xk vs m=1':>11}")

    axisA = {}
    for name, fn in (('single-branch', one_branch), ('two-branch', two_branch)):
        CSR2D._retarded_xi_bands = fn
        ref = z_scan(csr, xi_band_margin=1.0, xbins=100)
        rows = []
        for m in (1.0, 2.0, 4.0, 8.0):
            dE, xk = z_scan(csr, xi_band_margin=m, xbins=int(100 * m))
            rows.append((m, reldiff(dE, ref[0]), reldiff(xk, ref[1])))
            emit(f'  {name:<14} {m:>7.1f} {int(100*m):>7} {roughness(dE):>11.5f} '
                 f'{reldiff(dE, ref[0]):>11.5f} {reldiff(xk, ref[1]):>11.5f}')
        axisA[name] = rows
        emit('')

    # ---- how much does the answer itself move? -----------------------------
    emit('--- Absolute effect on the wake at the base setting ---')
    CSR2D._retarded_xi_bands = one_branch
    dE1, xk1 = z_scan(csr)
    CSR2D._retarded_xi_bands = two_branch
    dE2, xk2 = z_scan(csr)
    emit(f'  dE     single -> two branch : rel L2 change {reldiff(dE2, dE1):.5f}')
    emit(f'  x_kick single -> two branch : rel L2 change {reldiff(xk2, xk1):.5f}')
    emit(f'  roughness dE  single {roughness(dE1):.5f}  two {roughness(dE2):.5f}')
    emit('')

    # ---- INSTRUMENTATION: rad < 0 vs |r - r'| ------------------------------
    emit("--- rad < 0 incidence vs |r - r'|, the guard section 6b asked to check ---")
    emit('The worry: inside the polar radius R2 a spurious rad < 0 is harmless')
    emit('(the disc covers it and the taper kills the Cartesian piece), but in the')
    emit('annulus R2 < r < ~20 sigma_xi the taper is already 1 and the branches are')
    emit('still merged, so dropping a branch there would lose real support.')
    emit('')

    nz_mesh = csr.CSR_params.zbins
    ixm = csr.CSR_params.xbins // 2
    s0 = csr.beam.position
    R2_nom = BASE['near_patch'] * sig_xi

    r_all, neg_all, live_all = [], [], []
    for j in range(nz_mesh):
        k = ixm * nz_mesh + j
        s = s0 + csr.CSR_zmesh[k]
        x = csr.CSR_xmesh[k]
        t = csr.beam.position
        sigma_z = csr.beam._sigma_z
        # same sp3 region get_CSR_wake builds: the one straddling s' = s
        sp = np.linspace(s - 20 * sigma_z, s + 5 * sigma_z, BASE['zbins'])
        csr._retarded_xi_bands(s, x, t, sp)
        for br in csr._branch_diag['branches']:
            r_all.append(br['r'])
            neg_all.append(br['rad'] < 0.0)
            live_all.append(br['live'])
    r_all = np.concatenate(r_all)
    neg_all = np.concatenate(neg_all)
    live_all = np.concatenate(live_all)

    edges = np.array([0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 1e9]) * sig_xi
    emit(f'  R2 = {R2_nom*1e6:.1f} um = {BASE["near_patch"]:.0f} sigma_xi')
    emit(f"  {'r / sigma_xi':>16} {'columns':>9} {'rad<0':>8} {'frac':>8} "
         f"{'live':>8} {'zone':>22}")
    for i in range(len(edges) - 1):
        m = (r_all >= edges[i]) & (r_all < edges[i + 1])
        n = int(m.sum())
        if n == 0:
            continue
        hi = edges[i + 1] / sig_xi
        zone = ('inside polar disc' if edges[i + 1] <= R2_nom else
                'ANNULUS (taper = 1)' if edges[i] < 20 * sig_xi else
                'branches separated')
        emit(f'  {edges[i]/sig_xi:>7.0f} - {hi:>6.0f} {n:>9} '
             f'{int(neg_all[m].sum()):>8} {neg_all[m].mean():>8.4f} '
             f'{live_all[m].mean():>8.4f} {zone:>22}')
    emit('')
    ann = (r_all >= R2_nom) & (r_all < 20 * sig_xi)
    emit(f'  ANNULUS R2 < r < 20 sigma_xi : {int(ann.sum())} columns, '
         f'{int(neg_all[ann].sum())} with rad < 0 '
         f'({neg_all[ann].mean()*100:.2f}%)')
    emit('  VERDICT: ' + ('guard is safe in the annulus'
                          if neg_all[ann].sum() == 0 else
                          'rad < 0 DOES occur where the taper is 1 -- investigate'))
    emit('')

    # ---- verdict -----------------------------------------------------------
    emit('--- verdict ---')
    for name, rows in axisA.items():
        worst = max(r[1] for r in rows)
        tail = abs(rows[-1][1] - rows[-2][1])
        emit(f'  axis A, {name:<14} worst dE dev {worst:.5f}   '
             f'change over the last doubling {tail:.5f}')
    emit('')
    emit('An invariance axis must not move. "Change over the last doubling" is the')
    emit('honest number: it says whether the domain has stopped mattering.')

    # ---- plots -------------------------------------------------------------
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.6))

    ax = axs[0]
    for name, mk in (('single-branch', 'rs--'), ('two-branch', 'go-')):
        rows = axisA[name]
        ax.loglog([r[0] for r in rows], [max(r[1], 1e-16) for r in rows], mk,
                  label=name)
    ax.axhline(0.01, color='k', ls=':', alpha=0.6, label='1%')
    ax.set_xlabel('xi_band_margin (cell size held fixed)')
    ax.set_ylabel('rel. L2 deviation of dE/dct vs margin 1')
    ax.set_title('Axis A invariance: does a wider\ndomain still find missing support?')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which='both')

    ax = axs[1]
    zz = csr.CSR_zmesh[ixm * nz_mesh:(ixm + 1) * nz_mesh] * 1e6
    ax.plot(zz, dE1, 'r--', label='single-branch')
    ax.plot(zz, dE2, 'g-', label='two-branch')
    ax.set_xlabel(r'z ($\mu$m)')
    ax.set_ylabel('dE/dct')
    ax.set_title(f'Wake at the base setting\nrel L2 change '
                 f'{reldiff(dE2, dE1):.3f}')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axs[2]
    rr = r_all / sig_xi
    keep = rr > 0
    ax.hist(rr[keep], bins=np.logspace(-2, 3, 60), alpha=0.45,
            label='all columns')
    if neg_all.any():
        ax.hist(rr[keep & neg_all], bins=np.logspace(-2, 3, 60), alpha=0.85,
                color='r', label='rad < 0')
    ax.axvline(BASE['near_patch'], color='b', ls='--',
               label=r'$R_2 = 5\sigma_\xi$')
    ax.axvline(20, color='k', ls=':', label=r'branches separate $\sim 20\sigma_\xi$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r"$|r-r'| / \sigma_\xi$")
    ax.set_ylabel('columns')
    ax.set_title("Where the rad < 0 guard fires")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, which='both')

    fig.suptitle('Step 6: two-branch band location from the corrected Eq 4.24 '
                 '(bspline_comoving, shear z:x = 20)', fontsize=12)
    plt.tight_layout()
    p = os.path.join(RESULT_DIR, 'two_branch_bands.png')
    plt.savefig(p, dpi=130)
    plt.close()
    emit('')
    emit(f'Plot saved: {p}')

    with open(os.path.join(RESULT_DIR, 'two_branch_bands_log.txt'), 'w') as f:
        f.write('\n'.join(lines))


if __name__ == '__main__':
    main()
