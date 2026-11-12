"""
How large is the cancellation in the longitudinal CSR integrand, what does it cost, and does
integration by parts along each s' column remove it? (progress_cancellation.md)

    python cancellation_effects.py [A B C D E F G]       (default: all sections)

A  anatomy: C = sum|parts|/|sum| for both forms, per case, and their agreement
B  scaling of C with sigma_x and sigma_z (thin-beam prediction C ~ 1/sigma_x)
C  x' quadrature convergence of both forms (xbins per band)
D  shot noise: wake scatter over random seeds, both forms
E  injected noise: rho only, rho_x only, both consistently
F  clipped charge: zero the stored fields beyond |u| > u_c
G  figure of the integrand on the (x', s') nodes, both forms
H  collapse: C at the bunch centre against sigma_z^(2/3)/sigma_xi, every case
I  transverse (u) vs longitudinal (w) grid edges, rms and quantile frames
J  the u edge taper (particle_deposition.u_taper): off vs on, clean, outlier and halo beams
K  full runs with the kicks applied: u_taper x CSR_integration.ibp, with ibp_check on

Writes benchmark_results/cancellation_effects/<section>.txt and figures.
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cancellation_lib as L          # noqa: E402
import cancellation_ibp as I          # noqa: E402

OUT = L.RESULT_DIR
BLUE, ORANGE = '#2a78d6', '#eb6834'   # shipped, integrated by parts
J0 = 0.25


class Log:
    def __init__(self, name):
        self.path = os.path.join(OUT, f'{name}.txt')
        self.lines = []

    def __call__(self, t=''):
        print(t, flush=True)
        self.lines.append(t)

    def close(self):
        with open(self.path, 'w') as f:
            f.write('\n'.join(self.lines) + '\n')


def cut_points(csr, ix=2, step=2):
    nz = csr.CSR_params.zbins
    t = csr.beam.position
    return [(t + csr.CSR_zmesh[ix * nz + j], csr.CSR_xmesh[ix * nz + j],
             csr.CSR_zmesh[ix * nz + j] / csr.beam._sigma_z) for j in range(4, nz - 4, step)]


def both(csr, pts, J0=J0):
    rows = []
    for s, x, zz in pts:
        ps, jm = I.ibp_pieces(csr, s, x, J0=J0)
        T = I.totals(ps)
        rows.append((zz, T['ship'], T['ibp'], T['C_ship'], T['C_ibp'], jm))
    return np.array(rows)


def rel(a, b):
    return np.linalg.norm(a - b) / np.linalg.norm(b)


CASES = {
    'bend':      dict(s_stop=0.5),
    'thin':      dict(s_stop=0.5, sigma_x_um=12.5),
    'exit':      dict(s_stop=1.2),
    'shear0.5':  dict(s_stop=0.5, shear=0.5),
    'shear2':    dict(s_stop=0.5, shear=2.0),
    'shear20':   dict(s_stop=0.5, shear=20.0, extra_dep={'xlim': 8, 'xbins': 320}),
}
LABEL = {'bend': 'in bend, s = 0.5 m', 'thin': 'in bend, sigma_x0 = 12.5 um',
         'exit': 'past exit, s = 1.2 m', 'shear0.5': 'in bend, shear 0.5',
         'shear2': 'in bend, shear 2', 'shear20': 'in bend, shear 20 (xlim 8)'}


def section_A():
    log = Log('A_anatomy')
    log('A. Cancellation factor C = sum|node contributions| / |total|, shipped form vs the')
    log('   column-integrated-by-parts form, on the SAME nodes. Mid-row z cut, 16 points.')
    log(f'   Partition of unity at |J| < {J0} (shipped form kept there).')
    log('')
    log(f'{"case":28s} {"sx um":>6s} {"sz um":>6s} {"tau":>6s} | {"C_ship mid":>10s} {"med":>6s} | '
        f'{"C_ibp mid":>9s} {"med":>5s} {"max":>5s} | {"|ship-ibp|/|ship|":>17s} | {"min|J|":>6s}')
    for name, kw in CASES.items():
        csr = L.build('A_' + name, **kw)
        r = both(csr, cut_points(csr))
        mid = np.argmin(np.abs(r[:, 0]))
        log(f'{LABEL[name]:28s} {csr.beam._sigma_x*1e6:6.1f} {csr.beam._sigma_z*1e6:6.1f} '
            f'{csr.beam._slope[0]:+6.2f} | {r[mid,3]:10.1f} {np.median(r[:,3]):6.1f} | '
            f'{r[mid,4]:9.2f} {np.median(r[:,4]):5.2f} {r[:,4].max():5.1f} | '
            f'{rel(r[:,2], r[:,1]):17.2e} | {r[:,5].min():6.3f}')
    # the transverse integrand, for comparison
    log('')
    log('Transverse integrand, shipped form, same cut (C over nodes):')
    for name in ('bend', 'exit', 'shear20'):
        csr = L.build('A_' + name, **CASES[name])
        cs = []
        for s, x, zz in cut_points(csr):
            ps = L.pieces(csr, s, x)
            cs.append(L.cancel(np.concatenate([p['x'].ravel() for p in ps]))[0])
        log(f'  {LABEL[name]:28s} C_x median {np.median(cs):6.2f}  max {np.max(cs):6.2f}')
    log.close()


def section_B():
    log = Log('B_scaling')
    log('B. C at the bunch centre vs sigma_x (sigma_z0 = 50 um) and vs sigma_z (sigma_x0 = 50 um).')
    log('   Thin-beam estimate: C ~ sqrt(2/pi) * sigma_z / (sigma_x * |R_x|),  |R_x| ~ sin(phi/2),')
    log('   phi ~ (24 sigma_z / R)^(1/3), i.e. C ~ sigma_z^(2/3) R^(1/3) / sigma_x.')
    log('')
    res = {}
    for key, vals in (('sigma_x_um', (10., 25., 50., 100., 200.)),
                      ('sigma_z_um', (25., 50., 100., 200.))):
        log(f'{key:10s} {"sx um":>6s} {"sz um":>6s} {"tau":>6s} {"C_ship":>8s} {"C_ibp":>6s} '
            f'{"ratio":>6s} {"|ship-ibp|/|ship|":>17s}')
        rows = []
        for v in vals:
            csr = L.build(f'B_{key}{v:g}', s_stop=0.5, **{key: v})
            r = both(csr, cut_points(csr, step=4))
            mid = np.argmin(np.abs(r[:, 0]))
            rows.append((csr.beam._sigma_x, csr.beam._sigma_z, r[mid, 3], r[mid, 4]))
            log(f'{v:10.1f} {csr.beam._sigma_x*1e6:6.1f} {csr.beam._sigma_z*1e6:6.1f} '
                f'{csr.beam._slope[0]:+6.2f} {r[mid,3]:8.1f} {r[mid,4]:6.2f} '
                f'{r[mid,3]/r[mid,4]:6.1f} {rel(r[:,2], r[:,1]):17.2e}')
        rows = np.array(rows)
        res[key] = rows
        col = 0 if key == 'sigma_x_um' else 1
        sel = slice(0, 3)
        sl = np.polyfit(np.log(rows[sel, col]), np.log(rows[sel, 2]), 1)[0]
        log(f'   log-log slope of C_ship vs {"sigma_x" if col == 0 else "sigma_z"} over the '
            f'first three points: {sl:+.2f}')
        log('')
    log.close()

    fig, axs = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for ax, key, col, xl in ((axs[0], 'sigma_x_um', 0, r'$\sigma_x$ at s = 0.5 m (µm)'),
                             (axs[1], 'sigma_z_um', 1, r'$\sigma_z$ at s = 0.5 m (µm)')):
        r = res[key]
        ax.loglog(r[:, col] * 1e6, r[:, 2], 'o-', color=BLUE, lw=2, ms=7, label='shipped form')
        ax.loglog(r[:, col] * 1e6, r[:, 3], 's--', color=ORANGE, lw=2, ms=7,
                  label='integrated by parts')
        ax.set_xlabel(xl)
        ax.grid(alpha=0.25, which='both')
    axs[0].set_ylabel('C = Σ|parts| / |Σ parts|  (bunch centre)')
    axs[0].legend(frameon=False)
    fig.suptitle('Cancellation factor of the longitudinal integrand, R = 1 m dipole')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'B_scaling.png'), dpi=140)


def section_C():
    log = Log('C_quadrature')
    log('C. x\' quadrature: relative L2 error of the dE cut vs a 400-node reference (mean of the')
    log('   two forms at 400, which agree to the number in brackets), as xbins (nodes per band) varies.')
    log('   s\' nodes unchanged.')
    log('')
    xbs = (6, 8, 12, 16, 24, 32, 48, 64, 128, 200, 400)
    fig, axs = plt.subplots(1, 4, figsize=(13, 3.4), sharey=True)
    for ax, name in zip(axs, ('bend', 'thin', 'exit', 'shear0.5')):
        csr = L.build('A_' + name, **CASES[name])
        pts = cut_points(csr, step=3)
        res = {}
        for xb in xbs:
            csr.integration_params.xbins = xb
            r = both(csr, pts)
            res[xb] = (r[:, 1], r[:, 2])
        ref = 0.5 * (res[400][0] + res[400][1])
        es = np.array([rel(res[xb][0], ref) for xb in xbs])
        ei = np.array([rel(res[xb][1], ref) for xb in xbs])
        log(f'{LABEL[name]}  [ship400 vs ibp400: {rel(res[400][0], res[400][1]):.1e}]')
        log('   xbins ' + ' '.join(f'{xb:8d}' for xb in xbs))
        log('   ship  ' + ' '.join(f'{e:8.1e}' for e in es))
        log('   ibp   ' + ' '.join(f'{e:8.1e}' for e in ei))
        log('')
        ax.loglog(xbs, np.maximum(es, 1e-8), 'o-', color=BLUE, lw=2, ms=6, label='shipped form')
        ax.loglog(xbs, np.maximum(ei, 1e-8), 's--', color=ORANGE, lw=2, ms=6,
                  label='integrated by parts')
        ax.axhline(1e-2, color='0.6', lw=1, ls=':')
        ax.set_title(LABEL[name], fontsize=10)
        ax.set_xlabel('x\' nodes per band (xbins)')
        ax.grid(alpha=0.25, which='both')
    axs[0].set_ylabel('relative L2 error of the dE cut')
    axs[0].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'C_quadrature.png'), dpi=140)
    log.close()


def section_D():
    log = Log('D_shot_noise')
    log('D. Shot noise: 4 pseudo-random seeds per particle count, in bend (s = 0.5 m).')
    log('   scatter = ||std over seeds|| / ||mean||, over the mid-row dE cut.')
    log('')
    for N in (20000, 200000, 1000000):
        S, B = [], []
        for seed in range(1, 5):
            csr = L.build(f'D_N{N}_{seed}', s_stop=0.5, n_particle=N, seed=seed)
            r = both(csr, cut_points(csr))
            S.append(r[:, 1])
            B.append(r[:, 2])
            log(f'  N = {N:7d} seed {seed}: |ship - ibp| / |ship| = {rel(r[:,2], r[:,1]):.1e}')
        S, B = np.array(S), np.array(B)
        log(f'  N = {N:7d}: scatter shipped {np.linalg.norm(S.std(0))/np.linalg.norm(S.mean(0)):.4f}'
            f'   integrated by parts {np.linalg.norm(B.std(0))/np.linalg.norm(B.mean(0)):.4f}')
        log('')
    log.close()


def _inject(csr, kind, eps, seed):
    """
    Shot-noise-like noise (white, variance proportional to rho), relative to each array's RMS,
    smoothed and differentiated by the deposition's own smooth_and_differentiate. In
    'consistent' mode the perturbation of d rho/du and d rho/dw is then the spectral derivative
    of the perturbation of rho, as for real shot noise.

    Two earlier versions were not consistent, and the cancellation amplified the difference
    (bend, gain shipped vs by parts): uniform noise smoothed by gaussian_filter, derivative by
    np.gradient, 12.7 vs 0.90; uniform noise through smooth_and_differentiate, 3.85 vs 0.92.
    Uniform noise runs up to the grid edge and stops there in a step that d rho/du does not
    contain.
    """
    from pyDFCSR_2D.deposit_smooth import smooth_and_differentiate
    tr = csr.DF_tracker
    sig = float(getattr(tr, 'smoothing_sigma', 3.0))
    rng = np.random.default_rng(seed)
    F, Fu, Fw = tr.data_density_interp, tr.data_density_u_interp, tr.data_density_w_interp
    saved = [a.copy() for a in (F, Fu, Fw)]
    d, du, dw = (np.zeros(F.shape) for _ in range(3))
    for k in range(F.shape[0]):
        # shot-noise-like: variance proportional to the density, so the noise vanishes where
        # the beam does, before the grid edge (uniform noise would end in a step there, and the
        # step is an inconsistency between rho and its derivative, which is not what is tested)
        white = rng.standard_normal(F.shape[1:]) * np.sqrt(np.maximum(saved[0][k], 0.0))
        d[k], du[k], dw[k] = smooth_and_differentiate(white, tr.delta_u, tr.delta_w, sig)
    nd = np.sqrt(np.mean(d ** 2))
    ndu = np.sqrt(np.mean(du ** 2))
    rmsF = np.sqrt(np.mean(saved[0] ** 2))
    rmsFu = np.sqrt(np.mean(saved[1] ** 2))
    if kind == 'rho':
        F[...] = saved[0] + eps * rmsF * d / nd
    elif kind == 'rho_x':
        Fu[...] = saved[1] + eps * rmsFu * du / ndu
    elif kind == 'consistent':
        c = eps * rmsF / nd
        F[...] = saved[0] + c * d
        Fu[...] = saved[1] + c * du
        Fw[...] = saved[2] + c * dw
    return saved


def _restore(csr, saved):
    tr = csr.DF_tracker
    for a, s in zip((tr.data_density_interp, tr.data_density_u_interp,
                     tr.data_density_w_interp), saved):
        a[...] = s


def section_E():
    log = Log('E_injected_noise')
    log('E. Gain = (relative L2 change of the dE cut) / eps, eps = 1e-3 of the array RMS,')
    log('   white noise with variance ~ rho (like shot noise) passed through the deposition\'s')
    log('   smooth_and_differentiate, 3 seeds.')
    log('   rho: noise in rho only. rho_x: in d rho/du only (Step 12\'s test).')
    log('   consistent: the same noise in rho, with its own derivative added to d rho/du, d rho/dw.')
    log('')
    eps = 1e-3
    for name in ('bend', 'shear0.5'):
        csr = L.build('A_' + name, **CASES[name])
        pts = cut_points(csr)
        base = both(csr, pts)
        log(f'{LABEL[name]}: |dE|_L2 = {np.linalg.norm(base[:,1]):.4f}')
        log(f'   {"injected into":14s} {"gain shipped":>16s} {"gain by parts":>16s}')
        for kind in ('rho', 'rho_x', 'consistent'):
            gs, gi = [], []
            for seed in range(3):
                saved = _inject(csr, kind, eps, seed)
                try:
                    r = both(csr, pts)
                finally:
                    _restore(csr, saved)
                gs.append(rel(r[:, 1], base[:, 1]) / eps)
                gi.append(rel(r[:, 2], base[:, 2]) / eps)
            log(f'   {kind:14s} {np.mean(gs):9.2f} +-{np.std(gs):5.2f} '
                f'{np.mean(gi):9.2f} +-{np.std(gi):5.2f}')
        log('')
    log.close()


def section_F():
    log = Log('F_clipped_charge')
    log('F. Clip the stored density fields (rho, d rho/du, d rho/dw) to zero beyond |u| > u_c in')
    log('   every snapshot, as a grid that is too narrow does. q = clipped charge fraction.')
    log('   Change of the dE cut relative to the unclipped wake, both forms.')
    log('')
    for name in ('bend', 'shear0.5'):
        csr = L.build('A_' + name, **CASES[name])
        tr = csr.DF_tracker
        pts = cut_points(csr)
        base = both(csr, pts)
        nu = tr.data_density_interp.shape[1]
        u = tr.u_start + (np.arange(nu) + 0.5) * tr.delta_u
        log(f'{LABEL[name]}')
        log(f'   {"u_c":>4s} {"q":>9s} {"edge/peak":>9s} {"change shipped":>15s} '
            f'{"change by parts":>16s} {"ship vs by parts":>16s}')
        for uc in (4.0, 3.5, 3.0, 2.5):
            saved = [a.copy() for a in (tr.data_density_interp, tr.data_density_u_interp,
                                        tr.data_density_w_interp)]
            m = np.abs(u) > uc
            q = saved[0][-1][m].sum() / saved[0][-1].sum()
            edge = saved[0][-1][~m][[0, -1]].max() / saved[0][-1].max()
            try:
                for a in (tr.data_density_interp, tr.data_density_u_interp,
                          tr.data_density_w_interp):
                    a[:, m, :] = 0.0
                r = both(csr, pts)
            finally:
                _restore(csr, saved)
            log(f'   {uc:4.1f} {q:9.2e} {edge:9.2e} {rel(r[:,1], base[:,1]):15.2e} '
                f'{rel(r[:,2], base[:,2]):16.2e} {rel(r[:,2], r[:,1]):16.2e}')
        log('')
    log.close()


def section_G():
    """The integrand on the far-region nodes, shipped vs by parts, for one observer."""
    csr = L.build('A_bend', **CASES['bend'])
    t = csr.beam.position
    nz = csr.CSR_params.zbins
    k = 2 * nz + nz // 2
    s, x = t + csr.CSR_zmesh[k], csr.CSR_xmesh[k]
    ps, _ = I.ibp_pieces(csr, s, x, J0=J0)
    far = [p for p in ps if p['region'] == 'far' and 'main_ship' in p]
    sxi = csr.DF_tracker.sigma_xi_arr[-1]

    def dens(p, key):
        # node contribution / node area = the integrand itself (MeV/m per m^2)
        wa = L.trapz_weights(p['sp'][0])[None, :] * (p['xp'][1] - p['xp'][0])[None, :]
        return np.where(wa > 0, p[key] / np.where(wa > 0, wa, 1.0), 0.0)

    vs = max(np.abs(dens(p, 'main_ship')).max() for p in far)
    vi = max(np.abs(dens(p, 'main_ibp')).max() for p in far)
    fig, axs = plt.subplots(1, 4, figsize=(17, 4.2),
                            gridspec_kw={'width_ratios': [1, 1, 1, 1.2]})
    for p in far:
        ctr = 0.5 * (p['xp'][0] + p['xp'][-1])
        X = (p['xp'] - ctr[None, :]) / sxi
        Y = (s - p['sp']) * 1e3
        o = np.argsort(Y[0])
        X, Y = X[:, o], Y[:, o]
        im0 = axs[0].pcolormesh(X, Y, dens(p, 'main_ship')[:, o], cmap='RdBu_r', vmin=-vs,
                                vmax=vs, shading='gouraud', rasterized=True)
        axs[1].pcolormesh(X, Y, dens(p, 'main_ibp')[:, o], cmap='RdBu_r', vmin=-vs, vmax=vs,
                          shading='gouraud', rasterized=True)
        im2 = axs[2].pcolormesh(X, Y, dens(p, 'main_ibp')[:, o], cmap='RdBu_r', vmin=-vi,
                                vmax=vi, shading='gouraud', rasterized=True)
        Ys = Y[0]
        axs[3].plot(Ys, np.abs(p['main_ship'][:, o]).sum(0) / L.trapz_weights(Ys / 1e3)
                    .clip(1e-300), color=BLUE, lw=2)
        axs[3].plot(Ys, np.abs(p['main_ibp'][:, o]).sum(0) / L.trapz_weights(Ys / 1e3)
                    .clip(1e-300), color=ORANGE, lw=2, ls='--')
        axs[3].plot(Ys, np.abs(p['main_ship'][:, o].sum(0)) / L.trapz_weights(Ys / 1e3)
                    .clip(1e-300), color='0.15', lw=1.2)
    titles = ('shipped:  G₀ ∂ρ/∂x', 'by parts, same colour scale',
              f'by parts, scale ×{vs/vi:.0f}:  −∂ₓ′(G₀/J) ρ − (G₀/J) D')
    for ax, ttl in zip(axs[:3], titles):
        ax.set_xlim(-6, 6)
        ax.set_ylim(0, 200)
        ax.set_xlabel("x' − band centre  (σξ)")
        ax.set_title(ttl, fontsize=10)
    axs[0].set_ylabel("s − s'  (mm)")
    fig.colorbar(im0, ax=axs[1], label='integrand (MeV/m per m²)', shrink=0.85)
    fig.colorbar(im2, ax=axs[2], shrink=0.85)
    axs[3].set_yscale('log')
    axs[3].set_xlim(0, 200)
    axs[3].set_ylim(1e-2, 3e3)
    axs[3].set_xlabel("s − s'  (mm)")
    axs[3].set_ylabel("per s' column  (MeV/m per m)")
    axs[3].set_title('Σ|nodes|: shipped (blue), by parts (orange dashed)\n'
                     '|Σ nodes| = column integral, same for both (black)', fontsize=9)
    axs[3].grid(alpha=0.25, which='both')
    fig.suptitle(f'Far-region longitudinal integrand (main term), R = 1 m dipole, s = {s:.3f} m, '
                 f'observer at the bunch centre', fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'G_integrand_map.png'), dpi=140, bbox_inches='tight')


def section_H():
    """
    C at the bunch centre for every case, against sigma_z^(2/3) / sigma_xi.

    sigma_xi is the SLICE width (rms across the tilt-removed beam, the co-moving frame's
    sigma_xi at the newest snapshot), which is what the +- lobes of d rho/dx span at fixed z.
    The projected sigma_x is printed too: for a tilted beam (sigma_x0 = 200 um at tau = -1.66,
    shear 20 at tau = -2.7) the two differ by 2x to 100x, and only sigma_xi collapses the data.
    """
    log = Log('H_collapse')
    log('H. C at the bunch centre (shipped form) against the thin-beam law C ~ sigma_z^(2/3)/sigma_xi')
    log('   (R = 1 m throughout; the R^(1/3) factor is not tested). sigma_xi = slice width,')
    log('   sigma_x = projected width, sigma_z = projected bunch length, all at the observation time.')
    log('')
    cases = dict(CASES)
    for v in (10., 25., 100., 200.):
        cases[f'sx{v:g}'] = dict(s_stop=0.5, sigma_x_um=v)
    for v in (25., 100., 200.):
        cases[f'sz{v:g}'] = dict(s_stop=0.5, sigma_z_um=v)
    log(f'{"case":10s} {"sigma_x":>8s} {"sigma_xi":>9s} {"sigma_z":>8s} {"tau":>6s} {"C":>8s} '
        f'{"C/(sz^2/3/sxi)":>15s}   (um)')
    rows = []
    for name, kw in cases.items():
        csr = L.build('H_' + name, **kw)
        nz = csr.CSR_params.zbins
        k = 2 * nz + nz // 2          # z/sigma_z = +0.08, the 'centre' of sections A-C
        s, x = csr.beam.position + csr.CSR_zmesh[k], csr.CSR_xmesh[k]
        ps = L.pieces(csr, s, x)
        C = L.cancel(np.concatenate([p['z'].ravel() for p in ps]))[0]
        sx, sxi, sz = (csr.beam._sigma_x * 1e6, csr.DF_tracker.sigma_xi_arr[-1] * 1e6,
                       csr.beam._sigma_z * 1e6)
        X = sz ** (2 / 3) / sxi
        rows.append((sx, sxi, sz, C, X, name))
        log(f'{name:10s} {sx:8.1f} {sxi:9.1f} {sz:8.1f} {csr.beam._slope[0]:+6.2f} {C:8.1f} '
            f'{C/X:15.1f}')
    r = np.array([q[:5] for q in rows])
    co = np.linalg.lstsq(np.c_[np.log(r[:, 1]), np.log(r[:, 2]), np.ones(len(r))],
                         np.log(r[:, 3]), rcond=None)[0]
    k = np.median(r[:, 3] / r[:, 4])
    log('')
    log(f'free fit: C ~ sigma_xi^{co[0]:+.2f} sigma_z^{co[1]:+.2f}   (law: -1, +2/3)')
    ratio = r[:, 3] / r[:, 4] / k
    log(f'C = {k:.0f} sigma_z^(2/3)/sigma_xi [um, median]: C/law ranges {ratio.min():.2f}..{ratio.max():.2f}')
    co_p = np.linalg.lstsq(np.c_[np.log(r[:, 0]), np.log(r[:, 2]), np.ones(len(r))],
                           np.log(r[:, 3]), rcond=None)[0]
    log(f'the same fit with the PROJECTED sigma_x: C ~ sigma_x^{co_p[0]:+.2f} sigma_z^{co_p[1]:+.2f}')
    log.close()

    fig, axs = plt.subplots(1, 2, figsize=(9.5, 3.8), sharey=True)
    for ax, col, xl in ((axs[0], 0, r'$\sigma_z^{2/3}/\sigma_x$, projected σx  (µm$^{-1/3}$)'),
                        (axs[1], 1, r'$\sigma_z^{2/3}/\sigma_\xi$, slice σξ  (µm$^{-1/3}$)')):
        X = r[:, 2] ** (2 / 3) / r[:, col]
        ax.loglog(X, r[:, 3], 'o', color=BLUE, ms=8)
        xx = np.geomspace(X.min() / 1.5, X.max() * 1.5, 10)
        if col == 1:
            ax.loglog(xx, k * xx, color='0.4', lw=1, ls='--', label=f'C = {k:.0f}·σz^(2/3)/σξ')
            ax.legend(frameon=False)
        for q, xv in zip(rows, X):
            if q[5] in ('shear20', 'sx200', 'shear2', 'exit'):
                ax.annotate(q[5], (xv, q[3]), textcoords='offset points', xytext=(6, -10),
                            fontsize=8, color='0.3')
        ax.set_xlabel(xl)
        ax.grid(alpha=0.25, which='both')
    axs[0].set_ylabel('C at the bunch centre (shipped form)')
    fig.suptitle('Cancellation factor, every case: it is set by the slice width, not the projected one',
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'H_collapse.png'), dpi=140)


def section_I():
    """
    Does a hard edge in w (longitudinal) break the cancellation the way one in u does?

    The quantile/peak frames taper the deposit in z, not in u. A cut in w multiplies rho by a
    function of z only, so d rho/dx of the cut field is exactly (cut) * d rho/dx: still a
    derivative in x, still cancelling. The shipped and by-parts forms should then agree, while
    a cut in u leaves rho with an edge that d rho/dx does not contain.
    """
    log = Log('I_edges')
    log('I. Zero the stored fields beyond |u| > c or |w| > c (every snapshot), in bend s = 0.5 m.')
    log('   q = charge removed. Both forms should move by the same amount if the cut is consistent.')
    log('')

    def edges(tr):
        r = tr.data_density_interp
        eu = max(max(np.abs(a[0]).max(), np.abs(a[-1]).max()) / a.max() for a in r)
        ew = max(max(np.abs(a[:, 0]).max(), np.abs(a[:, -1]).max()) / a.max() for a in r)
        return eu, ew

    for frame, cuts in (('rms', ((1, 3.0), (2, 2.5), (2, 2.0))),
                        ('quantile', ((1, 3.0), (2, 3.5), (2, 3.0)))):
        kw = {} if frame == 'rms' else {'extra_dep': {'frame': 'quantile'}}
        csr = L.build(f'I_{frame}', s_stop=0.5, **kw)
        tr = csr.DF_tracker
        pts = cut_points(csr)
        base = both(csr, pts)
        eu, ew = edges(tr)
        log(f'frame {frame}: grid edge density / peak: u {eu:.1e}, w {ew:.1e}; '
            f'|ship - ibp|/|ship| uncut {rel(base[:,2], base[:,1]):.1e}')
        log(f'   {"cut":>9s} {"q":>9s} {"shipped moves":>14s} {"by parts moves":>15s} '
            f'{"ship vs by parts":>17s}')
        arrs = (tr.data_density_interp, tr.data_density_u_interp, tr.data_density_w_interp)
        for axis, c in cuts:
            saved = [a.copy() for a in arrs]
            n = arrs[0].shape[axis]
            st, d = (tr.u_start, tr.delta_u) if axis == 1 else (tr.w_start, tr.delta_w)
            m = np.abs(st + (np.arange(n) + 0.5) * d) > c
            last = saved[0][-1]
            q = (last[m, :].sum() if axis == 1 else last[:, m].sum()) / last.sum()
            try:
                for a in arrs:
                    if axis == 1:
                        a[:, m, :] = 0.0
                    else:
                        a[:, :, m] = 0.0
                r = both(csr, pts)
            finally:
                _restore(csr, saved)
            log(f'   |{"u" if axis == 1 else "w"}| > {c:3.1f} {q:9.2e} {rel(r[:,1], base[:,1]):14.2e} '
                f'{rel(r[:,2], base[:,2]):15.2e} {rel(r[:,2], r[:,1]):17.1e}')
        log('')
    log.close()


def section_J():
    """particle_deposition.u_taper (deposit_smooth.DF_tracker_comoving.u_weight), off vs on."""
    log = Log('J_u_taper')
    log('J. Transverse edge taper of the deposit, u_taper 0 (hard edge at +-xlim) vs 0.5 (default).')
    log('   edge = max over snapshots of the smoothed density at u = +-xlim / peak; removed = charge')
    log('   outside the grid or in the taper (newest snapshot). In bend, s = 0.5 m.')
    log('')
    log(f'{"beam":24s} {"u_taper":>7s} {"edge":>8s} {"removed":>8s} {"|ship-ibp|/|ship|":>17s}')
    pairs = {}
    for nm, kw in (('Gaussian, Hammersley 200k', {}),
                   ('20k pseudo-random seed 3', dict(n_particle=20000, seed=3)),
                   ('1M pseudo-random seed 1', dict(n_particle=1000000, seed=1))):
        for ut in (0.0, 0.5):
            csr = L.build(f'J_{len(pairs)}_{ut}', s_stop=0.5, extra_dep={'u_taper': ut}, **kw)
            r = both(csr, cut_points(csr))
            pairs[(nm, ut)] = r
            tr = csr.DF_tracker
            log(f'{nm:24s} {ut:7.1f} {tr.edge_u_max:8.1e} {tr.clipped:8.1e} '
                f'{rel(r[:,2], r[:,1]):17.1e}')
        a, b = pairs[(nm, 0.0)], pairs[(nm, 0.5)]
        log(f'{"":24s} wake change on vs off: shipped {rel(b[:,1], a[:,1]):.1e}, '
            f'by parts {rel(b[:,2], a[:,2]):.1e}')
    log('')
    log('Halo beam: 98% Gaussian core + 2% halo 3x wider in x (sigma_xi 46 um at s = 0.5 m).')
    log('Reference: xlim 10, 400 bins (same cell), hard edge; the same observer points for all.')
    h5 = L.halo_beam('J_halo', frac=0.02, halo_factor=3.0)
    rows, pts = {}, None
    for tag, dep in (('xlim 10, hard (ref)', {'xlim': 10, 'xbins': 400, 'u_taper': 0.0}),
                     ('xlim 5, hard', {'u_taper': 0.0}),
                     ('xlim 5, taper 0.5', {'u_taper': 0.5}),
                     ('xlim 7, taper 0.5', {'xlim': 7, 'xbins': 280, 'u_taper': 0.5})):
        csr = L.build('J_halo_' + tag.split(',')[0].replace(' ', '') + str(dep['u_taper']),
                      s_stop=0.5, beam_h5=h5, extra_dep=dep)
        pts = cut_points(csr) if pts is None else pts
        rows[tag] = (both(csr, pts), csr.DF_tracker.edge_u_max, csr.DF_tracker.clipped)
    ref = 0.5 * (rows['xlim 10, hard (ref)'][0][:, 1] + rows['xlim 10, hard (ref)'][0][:, 2])
    log(f'{"grid":20s} {"edge":>8s} {"removed":>8s} {"|ship-ibp|/|ship|":>17s} '
        f'{"shipped vs ref":>15s} {"by parts vs ref":>16s}')
    for tag, (r, e, c) in rows.items():
        log(f'{tag:20s} {e:8.1e} {c:8.1e} {rel(r[:,2], r[:,1]):17.1e} {rel(r[:,1], ref):15.2e} '
            f'{rel(r[:,2], ref):16.2e}')
    log.close()


def section_K():
    """Track through the dipole with CSR applied, shipped vs by-parts form, hard edge vs taper."""
    import time
    log = Log('K_full_runs')
    log('K. dipole_lattice (drift 0.1, R = 1 m bend 1 rad, drift 0.5), CSR applied (step_control auto),')
    log('   200k Hammersley particles, wake mesh 5 x 40, CSR_integration xbins 200, ibp_check 3.')
    log('   <dd> = mean energy change, sd = rms energy spread (both 1e-4 of P0), eps = x-px emittance.')
    log('')
    log(f'{"u_taper":>7s} {"ibp":>5s} {"<dd>":>10s} {"sd":>9s} {"eps nm":>9s} {"max check":>10s} '
        f'{"skipped":>8s} {"wall s":>7s}')
    logs = {}
    off = L.build('K_off', s_stop=None, csr_on=False).beam.particle.pz.copy()
    res = {}
    for ut in (0.0, 0.5):
        for ibp in (False, True):
            t0 = time.time()
            csr = L.build(f'K_{ut}_{int(ibp)}', s_stop=None, csr_on=True,
                          extra_dep={'u_taper': ut},
                          extra_int={'ibp': ibp, 'ibp_check': 3})
            wall = time.time() - t0
            p = csr.beam.particle
            cxx = np.cov(np.vstack([p.x, p.px]))
            eps = float(np.sqrt(np.linalg.det(cxx)))
            lg = getattr(csr, 'ibp_check_log', [])
            chk = np.nanmax([q[1] for q in lg]) if lg else np.nan
            nskip = sum(q[3] for q in lg)
            logs[(ut, ibp)] = lg
            res[(ut, ibp)] = (np.mean(p.pz), np.std(p.pz), eps, p.pz.copy())
            log(f'{ut:7.1f} {str(ibp):>5s} {np.mean(p.pz)*1e4:10.5f} {np.std(p.pz)*1e4:9.5f} '
                f'{eps*1e9:9.5f} {chk:10.1e} {nskip:8d} {wall:7.0f}')
    log('')
    for ut in (0.0, 0.5):
        a, b = res[(ut, False)], res[(ut, True)]
        log(f'u_taper {ut}: by parts vs shipped, per particle: rms d(pz) / rms CSR-induced pz '
            f'change = {np.std(b[3] - a[3]) / np.std(a[3] - off):.1e}')
    a, b = res[(0.0, False)], res[(0.5, False)]
    log(f'shipped form, taper vs hard edge: {np.std(b[3] - a[3]) / np.std(a[3] - off):.1e}')
    log('')
    log('ibp_check per kick (s: relative difference [points used/skipped]), shipped runs:')
    for ut in (0.0, 0.5):
        log(f'  u_taper {ut}: ' + '  '.join(f'{q[0]:.3f}:{q[1]:.1e}[{q[2]}/{q[3]}]'
                                            for q in logs[(ut, False)]))
    log.close()


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    which = sys.argv[1:] or list('ABCDEFGHIJK')
    for w in which:
        globals()[f'section_{w}']()
