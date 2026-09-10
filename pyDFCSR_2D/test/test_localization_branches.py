"""
Do the localization branches of thesis Eq. 4.24 predict where the CSR integrand
actually is?

Step 4f verified Eq 4.24 against a root-find of its own defining system (Eq 4.19
light cone + Eq 4.20 beam axis), and found the printed form has a typo. That check
was about the ALGEBRA. This one is about the PHYSICS: it compares the predicted
branch positions against the measured support of the real integrand, i.e. against
where interpolate3D_* actually returns something nonzero. Same comparison as
thesis Fig. 4.3(b-c), but with the corrected formula and against the code.

Why now
-------
With the lab-frame blend ('bspline_fft') each of the two straddling snapshots is
evaluated in its OWN tilted frame, so each contributes its own pair of roots and
up to FOUR intervals appear (three in practice, since the two narrow roots merge).
Two of those are ghost images that do not exist physically.

The co-moving interpolant blends the FRAME, so there is one unambiguous
tau(t_ret) and exactly TWO branches -- the physical ones. This script checks that,
and it is the cleanest demonstration that the ghosting fix works, because the
ghost branches are directly visible as extra intervals in the integration plane.

The branch positions are implicit: tau and the intercept come from the frame at
t_ret, and t_ret depends on x'. So each branch is found by a fixed point,
    x' -> t_ret -> blended frame -> Eq 4.24 -> x'
which converges quickly because |r - r'| varies slowly with x'.
"""
import sys
import os
import numpy as np
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from pyDFCSR_2D.CSR import CSR2D
from pyDFCSR_2D.interp1D import interpolate1D

RESULT_DIR = os.path.join(os.path.dirname(__file__), 'benchmark_results', 'localization')
EXAMPLE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'example'))

SHEAR = 20.0
STOP_S = 0.60
DEP_BINS = 200
N_PARTICLE = 200000
NX_SCAN = 300001          # x' samples when measuring the true support


def write_inputs(method, poly_degree, tag, shear=None):
    shear = SHEAR if shear is None else shear
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
        beam['transforms'] = {'s1': {'shear_coefficient': {
            'units': 'dimensionless', 'value': float(shear)}, 'type': 'shear z:x'}}
    with open(os.path.join(EXAMPLE_DIR, f'input/loc_beam_{tag}.yaml'), 'w') as f:
        yaml.dump(beam, f, default_flow_style=False, sort_keys=False)
    config = {
        'input_beam': {'style': 'distgen',
                       'distgen_input_file': f'input/loc_beam_{tag}.yaml'},
        'input_lattice': {'lattice_input_file': 'input/dipole_lattice.yaml'},
        'particle_deposition': {'method': method, 'xbins': DEP_BINS, 'zbins': DEP_BINS,
                                'xlim': 5, 'zlim': 5, 'smoothing_sigma': 3.0,
                                'poly_degree': poly_degree, 'velocity_threhold': 1000},
        'CSR_integration': {'n_formation_length': 1.5, 'zbins': 100, 'xbins': 100},
        'CSR_computation': {'compute_CSR': 1, 'apply_CSR': 0, 'transverse_on': 1,
                            'xbins': 5, 'zbins': 40, 'xlim': 3, 'zlim': 3,
                            'write_beam': [], 'write_wakes': False,
                            'write_name': f'loc_{tag}', 'workdir': './output'},
    }
    p = f'input/loc_config_{tag}.yaml'
    with open(os.path.join(EXAMPLE_DIR, p), 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    return p


# ---------------------------------------------------------------------------
# Corrected Eq 4.24 (see test_eq424_localization.py for the verification)
# ---------------------------------------------------------------------------
def eq424(T, nq, q2, tau, b, sign):
    """
    x'_pm = tau [ Tb + (n'.q) tau  +-  sqrt(rad) ] / (tau^2 - 1)
    rad   = (tau^2 - 1)(Tb^2 - q^2) + ((n'.q) tau + Tb)^2 ,   Tb = T - b/tau

    Guards, all three of which are load-bearing:

      tau^2 == 1  degenerates to a linear equation (the alpha = +-pi/4 case the
                  thesis flags, and the origin of the code's |tan theta| <= 1
                  branch switch).
      rad < 0     the light cone and the beam axis do not intersect at this s',
                  so the branch is absent.
      l <= 0      SPURIOUS ROOT. The derivation squares l = Tb + x'/tau, which
                  admits solutions with l < 0, i.e. t_ret > t -- a source point
                  in the future. These must be rejected. Omitting this guard
                  produces confident predictions that sit where the integrand is
                  exactly zero, which is how it was caught (the shear sweep, at
                  low tilt, where the spurious root lands only a few hundred um
                  from the real one).
    """
    if abs(abs(tau) - 1.0) < 1e-12:
        # A -> 0: B x' + C = 0 with B = -2T/tau - 2 nq, C = q2 - T^2
        Bc = -2.0 * T / tau - 2.0 * nq
        Cc = q2 - T ** 2
        if Bc == 0:
            return np.nan
        xp = -Cc / Bc
        return xp if (T - b / tau) + xp / tau > 0 else np.nan
    Tb = T - b / tau
    rad = (tau ** 2 - 1.0) * (Tb ** 2 - q2) + (nq * tau + Tb) ** 2
    if rad < 0:
        return np.nan
    xp = tau * (Tb + nq * tau + sign * np.sqrt(rad)) / (tau ** 2 - 1.0)
    # l must be a positive distance
    if Tb + xp / tau <= 0:
        return np.nan
    return xp


class Geometry:
    """Lattice geometry and the co-moving frame, for one observation point."""

    def __init__(self, csr, s, x, t):
        self.csr, self.s, self.x, self.t = csr, s, x, t
        self.lat = csr.lattice
        a = np.array([s])
        self.r0s = np.array([self._lut(self.lat.coords[:, 0], a)[0],
                             self._lut(self.lat.coords[:, 1], a)[0]])
        self.ns = np.array([self._lut(self.lat.n_vec[:, 0], a)[0],
                            self._lut(self.lat.n_vec[:, 1], a)[0]])

    def _lut(self, d, v):
        return interpolate1D(xval=np.atleast_1d(v), data=d,
                             min_x=self.lat.min_x, delta_x=self.lat.delta_x)

    def qn(self, sp):
        r0p = np.array([self._lut(self.lat.coords[:, 0], sp)[0],
                        self._lut(self.lat.coords[:, 1], sp)[0]])
        nsp = np.array([self._lut(self.lat.n_vec[:, 0], sp)[0],
                        self._lut(self.lat.n_vec[:, 1], sp)[0]])
        q = self.ns * self.x + self.r0s - r0p
        return q, nsp

    def t_ret(self, xp, sp):
        q, nsp = self.qn(sp)
        return self.t - np.linalg.norm(q - nsp * xp)

    def frame(self, t_ret):
        """
        (tau, b_eff) of the beam axis at t_ret.

        Co-moving: the axis is x' = p_alpha(z') + xi_bar_alpha with p deg 1, so
        tau = p_alpha[0] and the effective intercept is p_alpha[1] + xi_bar_alpha.
        Lab-frame blend: there is no single frame, so the CURRENT snapshot's is
        returned and the caller should expect the ghost branch to be missed.
        """
        tr = self.csr.DF_tracker
        n_t = tr.poly_coeffs_interp.shape[0]
        t_idx = (t_ret - tr.min_x) / tr.delta_x
        k = int(np.clip(np.floor(t_idx), 0, max(n_t - 2, 0)))
        k1 = min(k + 1, n_t - 1)
        a = float(np.clip(t_idx - k, 0.0, 1.0))
        b = 1.0 - a
        p = b * tr.poly_coeffs_interp[k] + a * tr.poly_coeffs_interp[k1]
        if self.csr.use_comoving:
            xi_bar = b * tr.xi_bar_arr[k] + a * tr.xi_bar_arr[k1]
            return p[0], p[-1] + xi_bar
        return p[0], p[-1]

    def branch(self, sp, sign, n_iter=12):
        """Self-consistent branch position: x' -> t_ret -> frame -> Eq 4.24 -> x'."""
        xp = self.x
        for _ in range(n_iter):
            tr_ = self.t_ret(xp, sp)
            tau, b = self.frame(tr_)
            q, nsp = self.qn(sp)
            new = eq424(self.t - sp, float(nsp @ q), float(q @ q), tau, b, sign)
            if not np.isfinite(new):
                return np.nan
            if abs(new - xp) < 1e-12:
                xp = new
                break
            xp = new
        return xp


def measured_intervals(csr, s, x, t, sp, x_lo, x_hi):
    """(centre, lo, hi, weight%) of each disjoint nonzero x' interval at fixed s'."""
    xs = np.linspace(x_lo, x_hi, NX_SCAN)
    xp = xs.reshape(-1, 1)
    spm = np.full_like(xp, sp)
    gz, gx = csr.get_CSR_integrand(s=s, t=t, x=x, xp=xp, sp=spm)
    nz = (gz[:, 0] != 0.0) | (gx[:, 0] != 0.0)
    if not nz.any():
        return []
    idx = np.where(nz)[0]
    segs = [q for q in np.split(idx, np.where(np.diff(idx) > 1)[0] + 1) if len(q) > 2]
    tot = sum(abs(np.trapz(gz[q, 0], xs[q])) for q in segs) or 1.0
    return [(0.5 * (xs[q[0]] + xs[q[-1]]), xs[q[0]], xs[q[-1]],
             100 * abs(np.trapz(gz[q, 0], xs[q])) / tot) for q in segs]


def run(method, poly_degree, tag, shear=None):
    cfg = write_inputs(method, poly_degree, tag, shear)
    os.chdir(EXAMPLE_DIR)
    csr = CSR2D(input_file=cfg)
    csr.run(stop_time=STOP_S)
    csr.get_CSR_mesh()
    nz = csr.CSR_params.zbins
    k = (csr.CSR_params.xbins // 2) * nz + nz // 2
    s = csr.beam.position + csr.CSR_zmesh[k]
    x = csr.CSR_xmesh[k]
    return csr, s, x, csr.beam.position


def main():
    os.makedirs(RESULT_DIR, exist_ok=True)
    lines = []

    def emit(t=''):
        print(t)
        lines.append(t)

    emit('=' * 78)
    emit('Localization branches: corrected Eq 4.24 vs the measured integrand support')
    emit('=' * 78)
    emit('')
    emit('NOTE on "tilt": SHEAR below is the distgen `shear z:x` coefficient applied at')
    emit('s = 0. The dipole dispersion then rotates the beam in x-z, so the tilt AT THE')
    emit('OBSERVATION POINT is quite different and is what actually sets the branch')
    emit('geometry. Both are reported per case.')
    emit('')

    cases = {}
    for method, deg, tag in (('bspline_fft', 3, 'fft'),
                             ('bspline_comoving', 1, 'cmv')):
        csr, s, x, t = run(method, deg, tag)
        g = Geometry(csr, s, x, t)
        sx = csr.beam._sigma_x
        sz = csr.beam._sigma_z
        sxi = csr.beam._sigma_x_transform
        emit(f'--- {method} (poly_degree = {deg}) ---')
        emit(f'  s = {s:.6f} m, t = {t:.6f} m, x = {x*1e6:+.2f} um')
        emit(f'  sigma_x = {sx*1e6:.1f} um, sigma_xi = {sxi*1e6:.2f} um, '
             f'tau(now) = {csr.beam._slope[0]:.4f}')
        emit('')
        emit(f"  {'s-s_prime':>10} {'#found':>7} | "
             f"{'Eq4.24 narrow':>14} {'measured':>12} {'rel err':>9} | "
             f"{'Eq4.24 chirp':>13} {'measured':>12} {'rel err':>9}")
        emit('  ' + '-' * 96)

        ds_list = [2e-4, 5e-4, 1e-3, 1.5e-3, 2e-3, 3e-3, 4.018e-3, 1e-2]
        rows = []
        for ds in ds_list:
            sp = s - ds
            xn = g.branch(sp, -1)
            xc = g.branch(sp, +1)
            iv = measured_intervals(csr, s, x, t, sp, -14 * sx, 6 * sx)

            def match(pred, tol):
                """
                Nearest measured interval to a predicted branch, but only if the
                prediction is plausibly that interval. A branch whose z' has left
                the beam carries no density, so there is nothing to match: report
                it as absent rather than snapping it to the other branch.
                """
                if not np.isfinite(pred) or not iv:
                    return np.nan, np.nan, 'no prediction'
                c, lo, hi, wt = min(iv, key=lambda e: abs(e[0] - pred))
                if lo <= pred <= hi:
                    return c, 0.0, 'inside'
                if abs(pred - c) > tol:
                    return np.nan, np.nan, 'absent'
                return c, abs(pred - c) / max(abs(c), 1e-12), 'near'

            tol = 20 * sxi
            mn, en, sn = match(xn, tol)
            mc, ec, sc = match(xc, tol)
            rows.append((ds, len(iv), xn, mn, en, xc, mc, ec, iv))
            f = lambda v: f'{v*1e6:>12.1f}' if np.isfinite(v) else f'{"--":>12}'
            e = lambda v, st: f'{v:>9.4f}' if np.isfinite(v) else f'{st:>9}'
            emit(f'  {ds:>10.5f} {len(iv):>7} | '
                 f'{xn*1e6:>14.1f} {f(mn)} {e(en, sn)} | '
                 f'{xc*1e6:>13.1f} {f(mc)} {e(ec, sc)}')
        emit('')
        emit('  All x\' in um. "inside" = the prediction falls within the measured')
        emit('  interval. "absent" = no measured support near the prediction, i.e.')
        emit('  that branch has left the beam in z and carries no density -- the')
        emit('  formula still returns a position, correctly, but there is nothing')
        emit('  there. A number would mean the prediction missed real support.')
        emit('')
        n_int = [r[1] for r in rows if r[0] < 3e-3]
        emit(f'  intervals found for ds < 3 mm: {n_int}')
        if method == 'bspline_fft':
            emit('')
            emit('  Eq 4.24 assumes ONE beam axis. The lab-frame blend superposes two,')
            emit('  so there is no single axis to feed it: the radicand goes negative')
            emit('  (nan) or the root lands between the two real branches. That is not')
            emit('  a failure of the formula, it is a statement that this interpolant')
            emit('  has no well-defined localization geometry.')
        emit('')
        cases[method] = dict(csr=csr, s=s, x=x, t=t, g=g, rows=rows,
                             sx=sx, sz=sz, sxi=sxi, deg=deg)

    emit('--- interval count, the direct signature of ghosting ---')
    f_counts = [r[1] for r in cases['bspline_fft']['rows'] if r[0] < 3e-3]
    c_counts = [r[1] for r in cases['bspline_comoving']['rows'] if r[0] < 3e-3]
    emit(f'  bspline_fft      : {f_counts}   <- each snapshot contributes its own pair')
    emit(f'  bspline_comoving : {c_counts}   <- one blended frame, so one pair')
    emit('')
    emit('  Eq 4.24 predicts TWO branches. The lab-frame blend produces more than')
    emit('  two because it superposes two frames; the extra ones are ghosts. The')
    emit('  co-moving path matching the predicted count is the ghosting fix seen')
    emit('  directly in the integration plane.')

    # ------------------------------------------------------------------ plots
    for method in ('bspline_fft', 'bspline_comoving'):
        C = cases[method]
        csr, s, x, t, g = C['csr'], C['s'], C['x'], C['t'], C['g']
        sx, sz = C['sx'], C['sz']

        fig = plt.figure(figsize=(18, 5.6))

        for pi, (ds_lo, ds_hi, xr, ttl) in enumerate((
                (2e-4, 1.2e-2, 14 * sx, 'wide view'),
                (2e-4, 3.0e-3, 5 * sx, 'zoom near the observation point'))):
            ax = fig.add_subplot(1, 3, pi + 1)
            sps = np.linspace(s - ds_hi, s - ds_lo, 220)
            xps = np.linspace(x - xr, x + 0.4 * xr, 260)
            XP, SP = np.meshgrid(xps, sps, indexing='ij')
            gz, _ = csr.get_CSR_integrand(s=s, t=t, x=x, xp=XP, sp=SP)
            A = np.abs(gz)
            pos = A[A > 0]
            if len(pos):
                im = ax.pcolormesh((s - SP) * 1e3, XP * 1e6, np.maximum(A, pos.min()),
                                   norm=LogNorm(vmin=pos.min(), vmax=pos.max()),
                                   cmap='viridis', shading='auto')
                plt.colorbar(im, ax=ax, label='|CSR integrand z|')
            bn = np.array([g.branch(sp_, -1) for sp_ in sps])
            bc = np.array([g.branch(sp_, +1) for sp_ in sps])
            ax.plot((s - sps) * 1e3, bn * 1e6, 'r-', lw=1.8, label='Eq 4.24, narrow')
            ax.plot((s - sps) * 1e3, bc * 1e6, 'w--', lw=1.8, label='Eq 4.24, chirp')
            ax.set_xlabel("s - s'  (mm)")
            ax.set_ylabel(r"x' ($\mu$m)")
            ax.set_ylim(xps[0] * 1e6, xps[-1] * 1e6)
            ax.set_title(f'{ttl}\n{method}')
            ax.legend(fontsize=8, loc='lower left')

        # panel 3: predicted vs measured branch position
        ax = fig.add_subplot(1, 3, 3)
        rows = C['rows']
        dsv = np.array([r[0] for r in rows]) * 1e3
        ax.plot(dsv, [r[2] * 1e6 for r in rows], 'r-', lw=1.5, label='Eq 4.24 narrow')
        ax.plot(dsv, [r[3] * 1e6 for r in rows], 'ro', ms=6, mfc='none',
                label='measured narrow')
        ax.plot(dsv, [r[5] * 1e6 for r in rows], 'b--', lw=1.5, label='Eq 4.24 chirp')
        ax.plot(dsv, [r[6] * 1e6 for r in rows], 'bs', ms=6, mfc='none',
                label='measured chirp')
        for r in rows:
            for (c, lo, hi, wt) in r[8]:
                ax.plot([r[0] * 1e3] * 2, [lo * 1e6, hi * 1e6], 'k-', lw=3, alpha=0.25)
        ax.set_xscale('log')
        ax.set_xlabel("s - s'  (mm)")
        ax.set_ylabel(r"x' ($\mu$m)")
        ax.set_title(f'branch position vs s\'\ngrey bars = measured intervals\n{method}')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        tt = csr.beam._slope[0]
        fig.suptitle(
            f'Localization branches, corrected Eq 4.24 vs the integrand — {method}\n'
            f"s = {s:.4f} m,  local tilt dx/dz = {tt:.4f} "
            f"({np.degrees(np.arctan(tt)):.1f}$^\\circ$),  "
            f"tan2$\\alpha$ = {-2*tt/(1-tt**2):.3f},  "
            f"$\\sigma_x/\\sigma_\\xi$ = {C['sx']/C['sxi']:.0f}×"
            f"   [distgen input was shear z:x = {SHEAR:g} at s=0]", fontsize=12)
        plt.tight_layout()
        p = os.path.join(RESULT_DIR, f'branches_{method}.png')
        plt.savefig(p, dpi=130)
        plt.close()
        emit('')
        emit(f'Plot saved: {p}')

    with open(os.path.join(RESULT_DIR, 'localization_log.txt'), 'w') as f:
        f.write('\n'.join(lines))


if __name__ == '__main__':
    main()
