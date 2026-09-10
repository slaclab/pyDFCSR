"""
Are the LONGITUDINAL integration extents right? Nobody has ever checked.

get_CSR_wake sets the s' domain from hand-tuned multiples in the lab frame --
500 sigma_z, 200 sigma_z, 20 sigma_z, 5 sigma_z, 3 sigma_z, 10 sigma_x -- plus
s1 = s2 - n_formation_length * L_f. Axis C of test_xi_bands_converge varied
longitudinal RESOLUTION and section 6i varied history sampling, but the extents
themselves have never been swept. Since the transverse extent is now derived from
Eq 4.24 and validated to >= 0.98 union coverage, these multiples are the least
justified part of the integration domain.

Two kinds of boundary, and they must be read differently:

  PARTITION boundaries  s2, s3 -- interior seams between regions that now all run
        the same ribbon-following logic. Moving them repartitions the same domain,
        so the total must be invariant to quadrature error alone. A failure here
        means the implementation double counts or drops a seam.

  DOMAIN boundaries  s1 (far edge), s4 (forward edge), and d (which sets s3 in the
        chirp case) -- true edges of the integrated region. Invariance here tests
        whether the domain is actually big enough.

Thesis 4.4.2, last two paragraphs, gives the physical expectation: the narrow band
extends far in s' (attenuated by 1/|r-r'|) but only ~sigma_xi in x'; the chirp band
extends only ~d = N sigma_x / |tan 2 alpha| in s' but is much wider transversely,
because x2 = x - (s-s')tan(2 alpha) leaves the beam quickly. The condition given
there -- x' within N sigma_x and z_ret within N sigma_z of the beam at t_ret -- is
already what the `live` flag computes per column.
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

RESULT_DIR = os.path.join(os.path.dirname(__file__), 'benchmark_results', 'long_extent')
EXAMPLE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'example'))

SHEAR = 20.0
STOP_S = 0.60
DEP_BINS = 200
N_PARTICLE = 200000
HIST_NFL = 24.0            # deep history so s1 sweeps are never truncation-limited
BASE = dict(xbins=200, zbins=400, xi_band_margin=2.0,
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
    with open(os.path.join(EXAMPLE_DIR, 'input/le_beam.yaml'), 'w') as f:
        yaml.dump(beam, f, default_flow_style=False, sort_keys=False)
    config = {
        'input_beam': {'style': 'distgen',
                       'distgen_input_file': 'input/le_beam.yaml'},
        'input_lattice': {'lattice_input_file': 'input/dipole_lattice.yaml'},
        'particle_deposition': {'method': 'bspline_comoving',
                                'xbins': DEP_BINS, 'zbins': DEP_BINS,
                                'xlim': 5, 'zlim': 5, 'smoothing_sigma': 3.0,
                                'poly_degree': 1, 'velocity_threhold': 1000},
        'CSR_integration': {'n_formation_length': HIST_NFL,
                            'zbins': BASE['zbins'], 'xbins': BASE['xbins']},
        'CSR_computation': {'compute_CSR': 0, 'apply_CSR': 0, 'transverse_on': 1,
                            'xbins': 5, 'zbins': 40, 'xlim': 3, 'zlim': 3,
                            'write_beam': [], 'write_wakes': False,
                            'write_name': 'le', 'workdir': './output'},
    }
    with open(os.path.join(EXAMPLE_DIR, 'input/le_config.yaml'), 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def default_bounds(csr, s, x, n_fl=1.5, fwd=None, d_sig=10.0):
    """
    Reproduce get_CSR_wake's s' decomposition, with the magic numbers exposed.
    fwd is the forward reach in sigma_z (get_CSR_wake uses 5 non-chirp, 3 chirp).
    d_sig is the sigma_x multiple inside d for the chirp case.
    """
    b = csr.beam
    sigma_z, sigma_x, tan_theta = b._sigma_z, b._sigma_x, b._slope[0]
    x0 = (s - b.position) * tan_theta
    xmean = b._mean_x
    if abs(tan_theta) <= 1:
        fwd = 5.0 if fwd is None else fwd
        s2, s3, s4 = s - 500*sigma_z, s - 20*sigma_z, s + fwd*sigma_z
    else:
        fwd = 3.0 if fwd is None else fwd
        if tan_theta > 0:
            tan_alpha = -2*tan_theta/(1 - tan_theta**2)
            d = (d_sig*sigma_x + xmean - x)/tan_alpha
        else:
            tan_alpha = 2*tan_theta/(1 - tan_theta**2)
            d = -(xmean - x - d_sig*sigma_x)/tan_alpha
        s4 = s + fwd*sigma_z
        s3 = max(0.0, s - d)
        s2 = s3 - 200*sigma_z
    s1 = max(0.0, s2 - n_fl*csr.formation_length)
    return [(s1, s2), (s2, s3), (s3, s4)], dict(x0=x0, sigma_z=sigma_z,
                                                sigma_x=sigma_x)


def wake_from_bounds(csr, s, x, bounds, patch=True, cell=None):
    """
    Integrate over an explicit list of s' regions. Mirrors get_CSR_wake's xi_bands
    branch exactly: polar patch on the LAST region only (the one straddling s' = s).

    `cell` is the longitudinal node spacing. It MUST be supplied for any extent
    sweep: with a fixed node count per region, lengthening a region also coarsens
    it, so the sweep would vary domain and resolution together and be uninterpretable
    (this is the same confound that invalidated the Step 4 margin sweep -- see 4c).
    Passing `cell` allocates nodes proportional to each region's length instead, so
    the cell size is held constant and only the domain changes.
    """
    t = csr.beam.position
    nz = csr.integration_params.zbins
    if cell is None:
        sps = [np.linspace(a, b, nz) for a, b in bounds]
    else:
        sps = [np.linspace(a, b, max(3, int(round(abs(b - a)/cell))))
               for a, b in bounds]
    radii = csr._near_patch_radii(s, bounds[-1][0], bounds[-1][1]) if patch else None
    tapers = [None]*(len(bounds) - 1) + [None if radii is None else (*radii, False)]
    parts = [csr._integrate_xi_region(s, x, t, spk, False, tp)
             for spk, tp in zip(sps, tapers)]
    if radii is not None:
        parts.append(csr._integrate_near_patch(s, x, t, *radii, False))
    return sum(p[0] for p in parts), sum(p[1] for p in parts)


def z_scan(csr, bounds_fn, cell=None, **kw):
    nz = csr.CSR_params.zbins
    ix = csr.CSR_params.xbins // 2
    s0 = csr.beam.position
    dE = np.zeros(nz)
    xk = np.zeros(nz)
    for j in range(nz):
        k = ix*nz + j
        s = s0 + csr.CSR_zmesh[k]
        x = csr.CSR_xmesh[k]
        bounds, _ = bounds_fn(csr, s, x, **kw)
        dE[j], xk[j] = wake_from_bounds(csr, s, x, bounds, cell=cell)
    return dE, xk


def default_cell(csr, s, x):
    """The longitudinal spacing get_CSR_wake actually uses in its LAST region."""
    bounds, _ = default_bounds(csr, s, x)
    a, b = bounds[-1]
    return abs(b - a)/(csr.integration_params.zbins - 1)


def reldiff(a, b):
    n = np.linalg.norm(b)
    return np.linalg.norm(a - b)/n if n > 0 else np.nan


def main():
    os.makedirs(RESULT_DIR, exist_ok=True)
    write_inputs()
    os.chdir(EXAMPLE_DIR)
    csr = CSR2D(input_file='input/le_config.yaml')
    csr.run(stop_time=STOP_S, debug=True)
    csr.get_CSR_mesh()
    ip = csr.integration_params
    ip.xi_bands = True
    for k, v in BASE.items():
        setattr(ip, k, v)

    b = csr.beam
    lines = []

    def emit(t=''):
        print(t)
        lines.append(t)

    emit('=' * 82)
    emit('Longitudinal integration extent: invariance sweep')
    emit('=' * 82)
    emit(f's = {b.position:.6f} m, shear z:x = {SHEAR:g}, slope = {b._slope[0]:+.4f}')
    emit(f'sigma_z = {b._sigma_z*1e6:.1f} um, sigma_x = {b._sigma_x*1e6:.1f} um, '
         f'sigma_xi = {b._sigma_x_transform*1e6:.2f} um')
    emit(f'L_f = {csr.formation_length*1e3:.1f} mm, history built with '
         f'n_fl = {HIST_NFL:g} ({csr.DF_tracker.data_density_interp.shape[0]} snapshots)')
    emit('')

    # self-check: does wake_from_bounds reproduce get_CSR_wake?
    ip.n_formation_length = 1.5
    ref_builtin = np.zeros(csr.CSR_params.zbins)
    nz = csr.CSR_params.zbins
    ix = csr.CSR_params.xbins // 2
    for j in range(nz):
        k = ix*nz + j
        ref_builtin[j], _ = csr.get_CSR_wake(b.position + csr.CSR_zmesh[k],
                                             csr.CSR_xmesh[k])
    ref, refx = z_scan(csr, default_bounds, n_fl=1.5)
    emit(f'self-check: wake_from_bounds vs get_CSR_wake = {reldiff(ref, ref_builtin):.2e}'
         '   (must be ~0, else the harness does not reproduce the code)')

    # Fixed-cell reference. Every extent sweep below allocates nodes proportional to
    # region length using this spacing, so the longitudinal cell never changes.
    CELL = default_cell(csr, b.position + csr.CSR_zmesh[ix*nz + nz//2],
                        csr.CSR_xmesh[ix*nz + nz//2])
    refc, refcx = z_scan(csr, default_bounds, cell=CELL, n_fl=1.5)
    emit(f'fixed longitudinal cell = {CELL*1e6:.2f} um '
         f'(= {CELL/b._sigma_z:.4f} sigma_z); fixed-cell vs default-node reference '
         f'differ by {reldiff(refc, ref):.5f}')
    emit('')

    axes = {}

    def sweep(name, kind, values, key, fmt='{:g}'):
        emit(f'--- {name}  [{kind}]   (longitudinal cell held FIXED) ---')
        emit(f"  {'value':>10} {'dE vs default':>14} {'xk vs default':>14}")
        rows = []
        for v in values:
            dE, xk = z_scan(csr, default_bounds, cell=CELL, **{key: v})
            rows.append((v, reldiff(dE, refc), reldiff(xk, refcx)))
            emit(f"  {fmt.format(v):>10} {reldiff(dE, refc):>14.5f} "
                 f"{reldiff(xk, refcx):>14.5f}")
        axes[name] = (kind, rows)
        emit('')

    # DOMAIN: far edge, in units of L_f. history is deep so this is not truncated
    sweep('s1 far edge (n_fl x L_f)', 'DOMAIN', [1.5, 3.0, 6.0, 12.0], 'n_fl')

    # DOMAIN: forward reach past the observation point, in sigma_z
    sweep('s4 forward reach (sigma_z)', 'DOMAIN', [3.0, 5.0, 10.0, 20.0], 'fwd')

    # DOMAIN: d, the chirp-band reach, in sigma_x
    sweep('d chirp reach (sigma_x)', 'DOMAIN', [10.0, 20.0, 40.0, 80.0], 'd_sig')

    # PARTITION: move the interior seams; total must not move at all
    emit('--- interior seams s2, s3  [PARTITION] ---')
    emit('These are seams between regions that all run identical logic, so moving')
    emit('them repartitions the same domain. Any deviation is a seam bug.')
    emit('Longitudinal cell held FIXED, so region count does not change resolution.')
    emit(f"  {'variant':>28} {'dE vs default':>14}")
    bounds0, _ = default_bounds(csr, b.position, csr.CSR_xmesh[ix*nz + nz//2])
    (s1, s2), (_, s3), (_, s4) = bounds0
    variants = {
        'default 3 regions': [(s1, s2), (s2, s3), (s3, s4)],
        'single region (no seams)': [(s1, s4)],
        'seam moved to midpoint': [(s1, 0.5*(s1 + s3)), (0.5*(s1 + s3), s3), (s3, s4)],
        '5 equal regions': [(s1 + i*(s3 - s1)/4, s1 + (i + 1)*(s3 - s1)/4)
                            for i in range(4)] + [(s3, s4)],
    }
    seam_rows = []
    for nm, bnd in variants.items():
        dE = np.zeros(nz)
        for j in range(nz):
            k = ix*nz + j
            s = b.position + csr.CSR_zmesh[k]
            xx = csr.CSR_xmesh[k]
            bb, _ = default_bounds(csr, s, xx)
            # rebuild the same variant shape around this column's own s1..s4
            (a1, a2), (_, a3), (_, a4) = bb
            if nm == 'default 3 regions':
                use = [(a1, a2), (a2, a3), (a3, a4)]
            elif nm == 'single region (no seams)':
                use = [(a1, a4)]
            elif nm == 'seam moved to midpoint':
                use = [(a1, 0.5*(a1 + a3)), (0.5*(a1 + a3), a3), (a3, a4)]
            else:
                use = [(a1 + i*(a3 - a1)/4, a1 + (i + 1)*(a3 - a1)/4)
                       for i in range(4)] + [(a3, a4)]
            dE[j], _ = wake_from_bounds(csr, s, xx, use, cell=CELL)
        seam_rows.append((nm, reldiff(dE, refc)))
        emit(f"  {nm:>28} {reldiff(dE, refc):>14.5f}")
    emit('')

    emit('--- verdict ---')
    for nm, (kind, rows) in axes.items():
        worst = max(r[1] for r in rows)
        emit(f'  {nm:<28} {kind:<10} worst dE deviation {worst:.5f}')
    for nm, d in seam_rows:
        emit(f'  {nm:<28} {"PARTITION":<10} {d:.5f}')
    emit('')
    emit('A DOMAIN axis reading ~0 means the default extent is already big enough.')
    emit('A PARTITION axis must read ~0 or the region bookkeeping is wrong.')

    # ---- plot ----
    fig, axs = plt.subplots(1, len(axes), figsize=(5.0*len(axes), 4.4))
    for ax, (nm, (kind, rows)) in zip(np.atleast_1d(axs), axes.items()):
        ax.semilogy([r[0] for r in rows], [max(r[1], 1e-16) for r in rows],
                    'ro-', label='dE/dct')
        ax.semilogy([r[0] for r in rows], [max(r[2], 1e-16) for r in rows],
                    'bs--', label='x_kick')
        ax.axhline(0.01, color='k', ls=':', alpha=0.6, label='1%')
        ax.set_title(f'{nm}\n[{kind}]', fontsize=10)
        ax.set_xlabel('extent parameter')
        ax.set_ylabel('rel. L2 vs default')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, which='both')
    fig.suptitle('Longitudinal integration extent: does the answer depend on the '
                 'hand-tuned s\' limits?', fontsize=12)
    plt.tight_layout()
    p = os.path.join(RESULT_DIR, 'longitudinal_extent.png')
    plt.savefig(p, dpi=130)
    plt.close()
    emit('')
    emit(f'Plot saved: {p}')
    with open(os.path.join(RESULT_DIR, 'longitudinal_extent_log.txt'), 'w') as f:
        f.write('\n'.join(lines))


if __name__ == '__main__':
    main()
