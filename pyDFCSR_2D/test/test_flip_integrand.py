"""
The chirp band across a full-compression point, at s = 0.1 m into the dipole, shear 20.

Question this answers: the x-z shear changes sign across full compression, so within
ONE near region the chirp band should appear on opposite sides of x'. Is that handled?

The answer turns out to be more interesting than the question. The stored frames
straddling the compression point are

    t = 1.00   tau = +20.00   sigma_z = 50.00 um   tan 2a = -0.1003
    t = 1.05   tau = -16.76   sigma_z =  2.51 um   tan 2a = +0.1198

so tan 2a does flip sign. But tan 2a = 2 tau/(1 - tau^2) has POLES at tau = +-1, and
_comoving_frame_at blends tau LINEARLY, so the interpolated tau traverses
+20 -> +1 -> 0 -> -1 -> -16.76 and the interpolated chirp angle sweeps through both
poles and through 2a = 0. Those orientations are fictitious: the real beam goes from
nearly vertical, through exactly vertical, to nearly vertical the other way -- it is
never the horizontal ribbon that tau = 0 describes.

The principal axis is the quantity that moves smoothly here (87.1 deg -> 90 deg ->
93.4 deg), and moment blending reproduces it because cov and var_z are each smooth;
tau = cov/var_z is a 0/0 ratio at full compression and is the badly conditioned way to
carry the same information.

Uses the LONG upstream drift (1.0 m) so the history is ample. The published s = 0.2
point used a 0.1 m drift and therefore only 3 snapshots, which confounds this
mechanism with history truncation.
"""
import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from pyDFCSR_2D.CSR import CSR2D

EXAMPLE_DIR = os.path.join(os.path.dirname(__file__), '..', 'example')
RESULT_DIR = os.path.join(os.path.dirname(__file__), 'benchmark_results', 'flip')
SHEAR = 20.0
DRIFT = 1.0
STEP = 0.05
S_OBS_DIP = 0.10          # metres into the dipole


def write_inputs(tag):
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
    with open(os.path.join(EXAMPLE_DIR, f'input/fl_beam_{tag}.yaml'), 'w') as f:
        yaml.dump(beam, f, default_flow_style=False, sort_keys=False)

    lat = {'step_size': STEP,
           'element_1': {'type': 'drift', 'L': DRIFT, 'nsep': 1},
           'element_2': {'type': 'dipole', 'L': 1.0, 'angle': 1.0,
                         'E1': 0, 'E2': 0, 'FINT': 0.0, 'FINTX': 0.5,
                         'HGAP': 0.1, 'HGAPX': 0.0, 'FRINGE_AT': 'both_ends',
                         'FRINGE_TYPE': 'linear_edge', 'TILT': 0.0, 'nsep': 1},
           'element_3': {'type': 'drift', 'L': 0.5, 'nsep': 1}}
    lp = f'input/fl_lat_{tag}.yaml'
    with open(os.path.join(EXAMPLE_DIR, lp), 'w') as f:
        yaml.dump(lat, f, default_flow_style=False, sort_keys=False)

    cfg = {
        'input_beam': {'style': 'distgen',
                       'distgen_input_file': f'input/fl_beam_{tag}.yaml'},
        'input_lattice': {'lattice_input_file': lp},
        'particle_deposition': {'method': 'bspline_comoving',
                                'xbins': 200, 'zbins': 200,
                                'xlim': 5, 'zlim': 5, 'smoothing_sigma': 3.0,
                                'poly_degree': 1, 'velocity_threhold': 1000},
        'CSR_integration': {'n_formation_length': 1.5, 'zbins': 200, 'xbins': 200},
        'CSR_computation': {'compute_CSR': 0, 'apply_CSR': 0, 'transverse_on': 1,
                            'xbins': 21, 'zbins': 51, 'xlim': 3, 'zlim': 3,
                            'write_beam': [], 'write_wakes': False,
                            'write_name': f'fl_{tag}', 'workdir': './output'},
    }
    p = f'input/fl_config_{tag}.yaml'
    with open(os.path.join(EXAMPLE_DIR, p), 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    return p


def main():
    os.makedirs(RESULT_DIR, exist_ok=True)
    os.chdir(EXAMPLE_DIR)
    lines = []

    def emit(s=''):
        print(s)
        lines.append(s)

    cfg = write_inputs('a')
    csr = CSR2D(input_file=cfg)
    csr.run(stop_time=DRIFT + S_OBS_DIP, debug=True)
    csr.get_CSR_mesh()

    b = csr.beam
    t = b.position
    s_obs = t                      # observe at the bunch centre, z = 0
    x_obs = 0.0
    tr = csr.DF_tracker

    emit('=' * 88)
    emit('Chirp band across full compression: s = %.4f m (%.3f m into the dipole)'
         % (t, t - DRIFT))
    emit('=' * 88)
    emit(f'shear {SHEAR:g}, drift {DRIFT} m, step_size {STEP} m, '
         f'{len(tr.time_log)} snapshots retained')
    emit(f'tau(s) at the observation point = {csr._frame_tilt():+.4f}')
    emit()

    bnds = csr._layout_bounds(s_obs, x_obs)
    (s1, s2), (_, s3), (_, s4) = bnds
    emit('s-prime regions (path length, and metres into the dipole):')
    for nm, (a, c) in zip(('far  s1-s2', 'mid  s2-s3', 'NEAR s3-s4'), bnds):
        emit(f'  {nm}:  [{a:.5f}, {c:.5f}]   dipole [{a-DRIFT:+.5f}, {c-DRIFT:+.5f}]'
             f'   length {c-a:.5f} m')
    emit()

    # ---- 1. what the interpolant does to tau across the near region -------------
    sp = np.linspace(s3, s4, 4000)
    poly, xi_bar, z_bar, s_xi, s_z = csr._comoving_frame_at(sp)
    tau_i = poly[:, 0]
    den = 1.0 - tau_i ** 2
    tan2a_i = np.where(np.abs(den) > 1e-9, 2.0 * tau_i / den, np.nan)
    sin2a_i = 2.0 * tau_i / (1.0 + tau_i ** 2)

    emit('Interpolated frame across the NEAR region (this is what the code uses;')
    emit('s-prime here is a proxy for the retarded time, exact at x_prime = x):')
    emit(f"  {'s_dip':>8} {'tau':>10} {'alpha/deg':>10} {'2a/deg':>9} "
         f"{'tan2a':>11} {'sig_z/um':>9} {'sig_xi/um':>10}")
    for frac in (0.0, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        i = min(int(frac * (len(sp) - 1)), len(sp) - 1)
        al = np.degrees(np.arctan(tau_i[i]))
        emit(f'  {sp[i]-DRIFT:>8.4f} {tau_i[i]:>10.3f} {al:>10.2f} {2*al:>9.2f} '
             f'{tan2a_i[i]:>11.3f} {s_z[i]*1e6:>9.2f} {s_xi[i]*1e6:>10.2f}')
    emit()
    n_pole = int(np.sum(np.abs(np.diff(np.sign(np.abs(tau_i) - 1.0))) > 0))
    emit(f'  |tau| crosses 1 : {n_pole} times inside the near region  '
         f'-> {n_pole} poles of tan 2a')
    emit(f'  tau range over the near region : {tau_i.min():+.3f} .. {tau_i.max():+.3f}')
    emit(f'  |tan 2a| max                   : {np.nanmax(np.abs(tan2a_i)):.1f}'
         f'   (endpoints are ~0.10)')
    emit()

    # ---- 2. the same thing under moment blending --------------------------------
    # tau = cov/var_z with cov, var_z each blended linearly. Reconstructed from the
    # stored per-snapshot (tau, sigma_z, sigma_xi): var_z = sigma_z^2, cov = tau var_z,
    # var_x = sigma_xi^2 + tau^2 var_z.
    tt = np.asarray(tr.time_log)
    tau_k = np.asarray(tr.poly_coeffs_interp)[:, 0]
    sz_k = np.asarray(tr.sigma_z_arr)
    sxi_k = np.asarray(tr.sigma_xi_arr)
    varz_k = sz_k ** 2
    cov_k = tau_k * varz_k
    varx_k = sxi_k ** 2 + tau_k ** 2 * varz_k

    varz_i = np.interp(sp, tt, varz_k)
    cov_i = np.interp(sp, tt, cov_k)
    varx_i = np.interp(sp, tt, varx_k)
    tau_m = cov_i / varz_i
    sxi_m = np.sqrt(np.maximum(varx_i - cov_i ** 2 / varz_i, 0.0))
    den_m = 1.0 - tau_m ** 2
    tan2a_m = np.where(np.abs(den_m) > 1e-9, 2.0 * tau_m / den_m, np.nan)

    emit('Same region, tau DERIVED from linearly blended second moments:')
    emit(f"  {'s_dip':>8} {'tau_coeff':>10} {'tau_moment':>11} {'a_coeff':>9} "
         f"{'a_moment':>9} {'tan2a_c':>10} {'tan2a_m':>9}")
    for frac in (0.0, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        i = min(int(frac * (len(sp) - 1)), len(sp) - 1)
        emit(f'  {sp[i]-DRIFT:>8.4f} {tau_i[i]:>10.3f} {tau_m[i]:>11.3f} '
             f'{np.degrees(np.arctan(tau_i[i])):>9.2f} '
             f'{np.degrees(np.arctan(tau_m[i])):>9.2f} '
             f'{tan2a_i[i]:>10.3f} {tan2a_m[i]:>9.3f}')
    emit()
    n_pole_m = int(np.sum(np.abs(np.diff(np.sign(np.abs(tau_m) - 1.0))) > 0))
    emit(f'  moment blend: |tau| crosses 1 {n_pole_m} times; '
         f'|tan 2a| max {np.nanmax(np.abs(tan2a_m)):.3f}')
    emit()
    emit('How much of the near region is spent at a fictitious orientation?')
    emit('  (|tau| < 1 means |alpha| < 45 deg, a beam flatter than 45 deg; the true')
    emit('   beam is within 3 deg of vertical everywhere in this region)')
    for nm, tv in (('linear tau  ', tau_i), ('blended mom.', tau_m)):
        f45 = np.mean(np.abs(tv) < 1.0)
        f60 = np.mean(np.abs(tv) < np.tan(np.radians(60.0)))
        emit(f'  {nm}: |alpha| < 45 deg on {f45*100:6.2f}% of the region, '
             f'< 60 deg on {f60*100:6.2f}%')
    emit()
    emit('Is the stored frame at the compression point even meaningful?')
    emit('  r^2 = 1 - sigma_xi^2/sigma_x^2 is the fraction of x-variance the tilt fit')
    emit('  explains. Near vertical the x-on-z regression explains almost nothing, so')
    emit('  tau there is noise-dominated and its SIGN is not reliable.')
    emit(f"  {'t':>8} {'s_dip':>8} {'tau':>10} {'r':>9} {'r^2':>10} "
         f"{'sig_z/um':>9} {'amp':>7}")
    for i, tv in enumerate(tt):
        varx = sxi_k[i] ** 2 + tau_k[i] ** 2 * varz_k[i]
        r2 = 1.0 - sxi_k[i] ** 2 / varx
        r = np.sign(tau_k[i]) * np.sqrt(max(r2, 0.0))
        emit(f'  {tv:>8.4f} {tv-DRIFT:>8.4f} {tau_k[i]:>10.3f} {r:>9.4f} '
             f'{r2:>10.6f} {sz_k[i]*1e6:>9.2f} '
             f'{np.sqrt(varx)/sxi_k[i]:>7.1f}')
    emit()

    # ---- 3. the located bands, and the integrand in the lab plane ---------------
    NS = 260
    spm = np.linspace(s3, s4, NS)
    bands_raw = csr._retarded_xi_bands(s_obs, x_obs, t, spm)
    br = csr._branch_diag['branches']
    c0, c1 = br[0]['centre'], br[1]['centre']
    l0, l1 = br[0]['live'], br[1]['live']
    emit('Located branch centres over the near region (x_prime - x, mm):')
    emit(f"  branch 0: live {l0.mean()*100:5.1f}%   "
         f"range [{np.min(c0[l0]-x_obs)*1e3 if l0.any() else np.nan:+.3f}, "
         f"{np.max(c0[l0]-x_obs)*1e3 if l0.any() else np.nan:+.3f}]")
    emit(f"  branch 1: live {l1.mean()*100:5.1f}%   "
         f"range [{np.min(c1[l1]-x_obs)*1e3 if l1.any() else np.nan:+.3f}, "
         f"{np.max(c1[l1]-x_obs)*1e3 if l1.any() else np.nan:+.3f}]")
    emit()

    # window from the located bands, generously padded, prediction-independent floor
    cand = np.concatenate([c0[l0] - x_obs, c1[l1] - x_obs, [0.0]])
    lo, hi = np.min(cand), np.max(cand)
    pad = 0.30 * max(hi - lo, 4.0 * b._sigma_x)
    xlo, xhi = x_obs + lo - pad, x_obs + hi + pad
    NX = 5000
    xs = np.linspace(xlo, xhi, NX)
    emit(f'lab x_prime window: [{(xlo-x_obs)*1e3:+.2f}, {(xhi-x_obs)*1e3:+.2f}] mm, '
         f'{NX} samples -> {(xhi-xlo)/NX*1e6:.2f} um per sample '
         f'(sigma_xi at s = {b._sigma_x_transform*1e6:.1f} um)')

    G = np.zeros((NX, NS))
    for j, spv in enumerate(spm):
        gz, gx = csr.get_CSR_integrand(s=s_obs, t=t, x=x_obs,
                                       xp=xs.reshape(-1, 1),
                                       sp=np.full((NX, 1), spv))
        G[:, j] = np.abs(gz[:, 0])
    emit(f'integrand sampled: nonzero fraction {np.mean(G > 0):.4f}, '
         f'max {G.max():.4e}')
    emit()

    # ---- figure ----------------------------------------------------------------
    # The thesis 4.4.2 chirp band, x2 = x - (s - s') tan 2a, evaluated with the frame
    # each blend gives. This is "where the chirp band is placed", drawn on top of the
    # integrand that the same frame produced.
    chirp_i = x_obs - (s_obs - sp) * tan2a_i
    chirp_m = x_obs - (s_obs - sp) * tan2a_m

    fig = plt.figure(figsize=(16.0, 12.5))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.5, 1.5, 1.0],
                          hspace=0.34, wspace=0.24)

    axm = fig.add_subplot(gs[0, :])
    pos = G[G > 0]
    vmin = np.percentile(pos, 55) if pos.size else 1e-30
    m = axm.pcolormesh((spm - DRIFT) * 1e3, (xs - x_obs) * 1e3, np.maximum(G, vmin),
                       norm=LogNorm(vmin=vmin, vmax=G.max()),
                       cmap='inferno', shading='auto', rasterized=True)
    fig.colorbar(m, ax=axm, pad=0.01, label=r'$|{\rm integrand}_z|$')
    axm.plot((spm - DRIFT) * 1e3, np.where(l0, (c0 - x_obs) * 1e3, np.nan),
             color='#39d0ff', lw=1.6, label='located branch 0')
    axm.plot((spm - DRIFT) * 1e3, np.where(l1, (c1 - x_obs) * 1e3, np.nan),
             color='#7dff5a', lw=1.6, ls='--', label='located branch 1')
    # where the interpolated |tau| = 1, i.e. the poles of tan 2a
    sgn = np.sign(np.abs(tau_i) - 1.0)
    for i in np.where(np.abs(np.diff(sgn)) > 0)[0]:
        axm.axvline((sp[i] - DRIFT) * 1e3, color='w', lw=1.0, ls=':', alpha=0.8)
    i0 = int(np.argmin(np.abs(tau_i)))
    axm.axvline((sp[i0] - DRIFT) * 1e3, color='#ff2ec4', lw=1.2, alpha=0.9)
    axm.set_xlabel(r"$s'$  (mm into the dipole)")
    axm.set_ylabel(r"$x' - x$  (mm)")
    axm.set_title(r"$|$integrand$_z|$ over the NEAR region at $s=%.3f$ m "
                  r"(%.0f mm into the dipole), shear %g."
                  "\n"
                  r"white dotted: interpolated $|\tau|=1$ (poles of $\tan 2\alpha$);"
                  r"  magenta: interpolated $\tau=0$"
                  % (t, (t - DRIFT) * 1e3, SHEAR), fontsize=10)
    axm.legend(loc='upper left', fontsize=9, framealpha=0.85)

    # ---- panel 2: where the chirp band is placed, both blends -------------------
    axc = fig.add_subplot(gs[1, :])
    axc.plot((sp - DRIFT) * 1e3, (chirp_m - x_obs) * 1e3, color='#d62728', lw=2.2,
             label=r'chirp band $x-(s-s^\prime)\tan2\alpha$, blended moments'
                   '\n(two clean lobes, opposite sign, meeting at compression)')
    axc.plot((sp - DRIFT) * 1e3, (chirp_i - x_obs) * 1e3, color='#1f77b4', lw=1.6,
             label=r'same, linear $\tau$ (what the code does)')
    axc.axhline(0.0, color='0.25', lw=1.0)
    for i in np.where(np.abs(np.diff(sgn0 := np.sign(np.abs(tau_i) - 1.0))) > 0)[0]:
        axc.axvline((sp[i] - DRIFT) * 1e3, color='0.45', lw=1.0, ls=':')
    axc.axvline((tt[np.argmin(sz_k)] - DRIFT) * 1e3, color='#ff2ec4', lw=1.4,
                label='full compression snapshot')
    axc.set_yscale('symlog', linthresh=1.0)
    axc.set_ylim(-3e3, 3e3)
    axc.set_xlabel(r"$s'$  (mm into the dipole)")
    axc.set_ylabel(r"$x' - x$  (mm), symlog")
    axc.set_title('Where the chirp band is placed. The red curve is the geometry you '
                  'described:\ntwo lobes of opposite sign meeting at full '
                  r'compression. Linear $\tau$ (blue) instead runs off to $\pm$metres '
                  r'through the $\tan2\alpha$ poles.', fontsize=10)
    axc.legend(fontsize=8.5, loc='upper right')
    axc.grid(alpha=0.25)

    axt = fig.add_subplot(gs[2, 0])
    axt.plot((sp - DRIFT) * 1e3, tau_i, color='#1f77b4', lw=1.8,
             label=r'$\tau$ blended linearly (code)')
    axt.plot((sp - DRIFT) * 1e3, tau_m, color='#d62728', lw=1.8,
             label=r'$\tau=$cov/var$_z$ from blended moments')
    msk = (tt >= sp[0]) & (tt <= sp[-1])
    axt.plot((tt[msk] - DRIFT) * 1e3, tau_k[msk], 'ko', ms=6,
             label='stored snapshots')
    for y in (1.0, -1.0):
        axt.axhline(y, color='0.6', lw=0.9, ls=':')
    axt.axhline(0.0, color='0.3', lw=0.9)
    axt.set_yscale('symlog', linthresh=1.0)
    axt.set_xlabel(r"$s'$  (mm into the dipole)")
    axt.set_ylabel(r'$\tau$')
    axt.set_title(r'The blended shear. Linear $\tau$ crosses $\pm1$ and $0$;'
                  '\n'
                  r'moment-derived $\tau$ stays near vertical.', fontsize=10)
    axt.legend(fontsize=8.5)
    axt.grid(alpha=0.25)

    axa = fig.add_subplot(gs[2, 1])
    axa.plot((sp - DRIFT) * 1e3, tan2a_i, color='#1f77b4', lw=1.8,
             label=r'$\tan2\alpha$, linear $\tau$')
    axa.plot((sp - DRIFT) * 1e3, tan2a_m, color='#d62728', lw=1.8,
             label=r'$\tan2\alpha$, blended moments')
    axa.axhline(0.0, color='0.3', lw=0.9)
    axa.set_yscale('symlog', linthresh=0.1)
    axa.set_xlabel(r"$s'$  (mm into the dipole)")
    axa.set_ylabel(r'$\tan 2\alpha$')
    axa.set_title(r'Chirp band angle. The chirp offset is $(s-s^\prime)\tan2\alpha$,'
                  '\n'
                  r'so the blue excursion misplaces the band.', fontsize=10)
    axa.legend(fontsize=8.5)
    axa.grid(alpha=0.25)

    out = os.path.join(RESULT_DIR, 'flip_integrand.png')
    fig.savefig(out, dpi=135, bbox_inches='tight')
    emit(f'Plot saved: {out}')

    with open(os.path.join(RESULT_DIR, 'flip_log.txt'), 'w') as f:
        f.write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
