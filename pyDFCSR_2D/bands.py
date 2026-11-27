"""
The retarded beam in lab coordinates: reference-orbit lookups, the time-blended co-moving frame,
and the two-branch band solver (thesis Eq 4.24) that locates the retarded density in x' per s'
column (docs/DFCSR_2D_methods.md §5.3).

Mixed into CSR.CSR2D; the methods use its state (beam, lattice, DF_tracker, integration_params).
"""
import numpy as np

from .interp1D import interpolate1D
from .lookup import lookup_vec

def degenerate_margin_factor(sin2a, factor, lo=0.8, hi=0.95):
    """
    Ribbon-margin multiplier near the band-branch degeneracy: 1 for |sin 2a| <= lo, `factor` for
    |sin 2a| >= hi, a C1 smoothstep in between (a jump would make the wake discontinuous in s).
    """
    if not factor or factor <= 1.0:
        return 1.0
    u = (abs(float(sin2a)) - lo) / (hi - lo)
    u = min(max(u, 0.0), 1.0)
    return 1.0 + (float(factor) - 1.0) * u * u * (3.0 - 2.0 * u)


class BandsMixin:
    def _lattice_vectors(self, sval):
        """
        (X0, Y0, n_x, n_y, tau_x, tau_y) of the reference orbit at arc length(s) sval.

        Past the lattice end the table itself is padded with a straight line (lattice.py). Before
        the start it is not: interpolate1D extrapolates with the first interval for one cell
        below the table and returns ZERO further down -- position at the origin, normal (0, 0) --
        so an observer more than one table cell before s = 0 (the bunch tail at the first kick,
        when the table is fine: a short lattice, or many samples) got a collapsed geometry and a
        transverse integrand of 1e16. Here the orbit is continued backwards along the initial
        tangent, exactly as the physical straight section before the lattice. Queries inside the
        table, and within one cell below it, are untouched (bit-identical).
        """
        lat = self.lattice
        sval = np.asarray(sval, dtype=float)
        out = [interpolate1D(xval=sval, data=d, min_x=lat.min_x, delta_x=lat.delta_x)
               for d in (lat.coords[:, 0], lat.coords[:, 1], lat.n_vec[:, 0], lat.n_vec[:, 1],
                         lat.tau_vec[:, 0], lat.tau_vec[:, 1])]
        before = sval <= lat.min_x - lat.delta_x        # where interpolate1D returns 0
        if before.any():
            ds = sval[before] - lat.min_x
            tx, ty = float(lat.tau_vec[0, 0]), float(lat.tau_vec[0, 1])
            vals = (lat.coords[0, 0] + ds * tx, lat.coords[0, 1] + ds * ty,
                    lat.n_vec[0, 0], lat.n_vec[0, 1], tx, ty)
            for k, v in enumerate(vals):
                out[k] = out[k].copy()
                out[k][before] = v
        return tuple(out)

    def _lattice_at(self, sval):
        """(X0, Y0, n_x, n_y) of the reference orbit at arc length(s) sval."""
        return self._lattice_vectors(sval)[:4]

    @staticmethod
    def _eval_poly_rows(coeffs, z):
        """Evaluate a different polynomial per element of z. coeffs: (n, deg+1)."""
        out = np.zeros_like(z)
        for j in range(coeffs.shape[1]):
            out = out * z + coeffs[:, j]
        return out

    def _comoving_frame_at(self, t_ret):
        """
        The blended co-moving frame at each element of t_ret.

        Mirrors interpolate3D_comoving_fields exactly, in whichever frame_blend mode
        is active. A band located with a different blend than the interpolant uses
        will not sit where the density is.

        Returns (poly, xi_bar, z_bar, sigma_xi, sigma_z); poly is (n, deg+1) with
        the highest power first, matching np.polyfit and _eval_poly_rows. In
        'moment' mode the centroid pivot is folded back into the coefficients, so
        every consumer -- _eval_poly_rows, poly[:, 0] as the slope, and
        poly[:, -1] + xi_bar as the intercept at z = 0 for _eq424 -- keeps working
        unchanged.
        """
        tr = self.DF_tracker
        n_t = tr.poly_coeffs_interp.shape[0]
        k, a = lookup_vec(t_ret, tr.t_arr, tr.is_uniform, tr.min_x, tr.delta_x, n_t)
        k1 = np.minimum(k + 1, n_t - 1)
        b = 1.0 - a

        if tr.frame_blend_code != tr.BLEND_COEFF:
            var_z = b * tr.var_z_arr[k] + a * tr.var_z_arr[k1]
            cov = b * tr.cov_arr[k] + a * tr.cov_arr[k1]
            z_bar = b * tr.z_bar_arr[k] + a * tr.z_bar_arr[k1]
            x_bar = b * tr.x_bar_arr[k] + a * tr.x_bar_arr[k1]

            tau = cov / var_z
            poly = np.stack([tau, x_bar - tau * z_bar], axis=1)
            if tr.frame_blend_code == tr.BLEND_MOMENT:
                var_x = b * tr.var_x_arr[k] + a * tr.var_x_arr[k1]
                s_xi = np.sqrt(np.maximum(var_x - cov * cov / var_z, 0.0))
                s_z = np.sqrt(var_z)
            else:
                s_xi = np.exp(b * np.log(tr.sigma_xi_arr[k])
                              + a * np.log(tr.sigma_xi_arr[k1]))
                s_z = np.exp(b * np.log(tr.sigma_z_arr[k])
                             + a * np.log(tr.sigma_z_arr[k1]))
            return poly, np.zeros_like(tau), z_bar, s_xi, s_z

        poly = (b[:, None] * tr.poly_coeffs_interp[k]
                + a[:, None] * tr.poly_coeffs_interp[k1])
        return (poly,
                b * tr.xi_bar_arr[k] + a * tr.xi_bar_arr[k1],
                b * tr.z_bar_arr[k] + a * tr.z_bar_arr[k1],
                np.exp(b * np.log(tr.sigma_xi_arr[k])
                       + a * np.log(tr.sigma_xi_arr[k1])),
                np.exp(b * np.log(tr.sigma_z_arr[k])
                       + a * np.log(tr.sigma_z_arr[k1])))

    def _curv_shift_at(self, t_ret, z_ret):
        """
        sigma_xi * g(w) at the retarded frame: the lab-x' offset of the beam ridge from
        the straight localization axis (DF_tracker_comoving.curvature). 0 when off.
        Blends the curvature coefficients exactly as interpolate3D_comoving_fields does.
        """
        tr = self.DF_tracker
        if not tr.curvature:
            return 0.0
        n_t = tr.poly_coeffs_interp.shape[0]
        k, a = lookup_vec(t_ret, tr.t_arr, tr.is_uniform, tr.min_x, tr.delta_x, n_t)
        k1 = np.minimum(k + 1, n_t - 1)
        b = 1.0 - a
        c2 = b * tr.curv_c2_arr[k] + a * tr.curv_c2_arr[k1]
        c3 = b * tr.curv_c3_arr[k] + a * tr.curv_c3_arr[k1]
        _, _, z_bar, s_xi, s_z = self._comoving_frame_at(t_ret)
        w = (z_ret - z_bar) / s_z
        wc = np.clip(w, -tr.curv_zlim, tr.curv_zlim)
        g = c2 * (wc * wc - 1.0) + c3 * wc ** 3 + (2.0 * c2 * wc + 3.0 * c3 * wc * wc) * (w - wc)
        return s_xi * g

    @staticmethod
    def _eq424(T, nq, q2, tau, b, sign, tangent=False):
        """
        Vectorized thesis Eq 4.24, in the form corrected in Step 4f.

            Tb  = T - b/tau
            rad = (tau^2 - 1)(Tb^2 - q2) + (nq*tau + Tb)^2
            x'  = tau [ Tb + nq*tau +- sqrt(rad) ] / (tau^2 - 1)

        where the beam axis is x' = tau*z' + b and l = |r - r'| = Tb + x'/tau.

        Returns (x', rad); x' is nan where the branch does not exist. Every guard
        below is load-bearing:

          |tau| ~ 0   the axis is the horizontal line x' = b, independent of l, so
                      there is a single root and no squaring was involved.
          |tau| ~ 1   (tau^2 - 1) -> 0. This is the alpha = +-pi/4 degeneracy the
                      thesis flags and the origin of the code's |tan theta| <= 1
                      branch switch; the quadratic collapses to a linear equation.
          rad < 0     the light cone and the beam axis do not meet at this s', so
                      the branch is genuinely absent.
          tangent     with tangent=True, rad < 0 returns the column's closest
                      approach to the axis instead of nan. The retarded offset from
                      the axis is xi_ret(x') = -tau (l(x') - Tb - x'/tau), extremal
                      where dl/dx' = 1/tau, i.e. at
                          x' = nq + sign(tau) sqrt((q2 - nq^2) / (tau^2 - 1)),
                      which exists for |tau| > 1. (The vertex of the squared quadratic
                      agrees only to leading order in 1/tau.) See
                      Integration_params.tangent_bands for why the beam is still
                      there although its axis is not.
          l <= 0      SPURIOUS ROOT. The derivation squares l = Tb + x'/tau, which
                      admits l < 0, i.e. t_ret > t -- a source point in the future.
                      Omitting this guard puts quadrature nodes where the integrand
                      is identically zero (Step 5d found this the hard way).
        """
        flat = np.abs(tau) < 1e-9
        tau_s = np.where(flat, 1.0, tau)
        deg = (~flat) & (np.abs(np.abs(tau_s) - 1.0) < 1e-9)
        Tb = T - b / tau_s

        rad = (tau_s ** 2 - 1.0) * (Tb ** 2 - q2) + (nq * tau_s + Tb) ** 2
        gen = (~flat) & (~deg) & (rad >= 0.0)
        xp = np.where(gen,
                      tau_s * (Tb + nq * tau_s
                               + sign * np.sqrt(np.maximum(rad, 0.0)))
                      / np.where(gen, tau_s ** 2 - 1.0, 1.0),
                      np.nan)

        if tangent:
            tan = (~flat) & (~deg) & (rad < 0.0) & (np.abs(tau_s) > 1.0)
            p2 = np.maximum(q2 - nq * nq, 0.0)
            xp = np.where(tan, nq + np.sign(tau_s) * np.sqrt(
                p2 / np.where(tan, tau_s ** 2 - 1.0, 1.0)), xp)

        Bc = -2.0 * T / tau_s - 2.0 * nq
        lin = deg & (Bc != 0.0)
        xp = np.where(lin, -(q2 - T ** 2) / np.where(lin, Bc, 1.0), xp)
        xp = np.where(flat, b, xp)

        # the l > 0 guard does not apply where no squaring took place
        l = np.where(flat, 1.0, Tb + xp / tau_s)
        return np.where(np.isfinite(xp) & (l > 0.0), xp, np.nan), rad

    def _retarded_xi_bands(self, s, x, t, sp, n_iter=12, scale_bands=True):
        """
        Locate EVERY branch of the retarded density support in lab x': one (lo, hi)
        interval per branch per s' column.

        BOTH roots of the corrected Eq 4.24 are taken, each refined by a fixed point (the
        retarded frame depends on x' through t_ret), each given its own sub-band; a single
        fixed point from x' = x finds one root and missed up to 67% of a column (Step 4e).

        The roots are deliberately NOT labelled narrow/chirp. Which sign is which
        flips as tau^2 crosses 1, and at low tilt they merge into a single ridge
        (Step 5d), so they are treated symmetrically and any overlap is removed by
        _disjoint_bands.

        Eq 4.24 assumes one straight beam axis: poly_degree 1 (checked in parse_input).
        """
        tr = self.DF_tracker

        X0_s, Y0_s, n_s_x, n_s_y = (v[0] for v in self._lattice_at(np.array([s])))
        X0_sp, Y0_sp, n_sp_x, n_sp_y = self._lattice_at(sp)

        qx = X0_s - X0_sp + x * n_s_x
        qy = Y0_s - Y0_sp + x * n_s_y
        nq = n_sp_x * qx + n_sp_y * qy
        q2 = qx * qx + qy * qy
        T = t - sp
        margin = self.integration_params.xi_band_margin
        tau0 = self._frame_tilt()
        margin = margin * degenerate_margin_factor(
            2.0 * tau0 / (1.0 + tau0 * tau0),
            self.integration_params.xi_band_margin_degenerate)

        # BOTH roots are always taken. The branch degeneracy of thesis 4.4.2 is a
        # property of the tilt at the RETARDED time a(s'), which varies column to
        # column -- the retarded tilt sweeps +19.4 -> -0.22 -> -19.8 across one near
        # region -- so it cannot be decided from a(s) at the observation point. Doing
        # so discarded 28-31% of the wake wherever it fired.
        # The per-column decision needs no new logic: on a degenerate column the
        # second root either fails the _eq424 guards (rad < 0, l <= 0) and is parked
        # dead, or lands on top of the first and is collapsed to zero width by
        # _disjoint_bands.
        bands, diag = [], []
        for sign in (-1.0, +1.0):
            xp = np.full(sp.shape, float(x))
            live = np.ones(sp.shape, dtype=bool)
            rad = np.zeros(sp.shape)
            for _ in range(n_iter):
                dx = qx - xp * n_sp_x
                dy = qy - xp * n_sp_y
                poly, xi_bar, _, _, _ = self._comoving_frame_at(
                    t - np.sqrt(dx * dx + dy * dy))
                new, rad = self._eq424(T, nq, q2, poly[:, 0],
                                       poly[:, -1] + xi_bar, sign,
                                       tangent=self.integration_params.tangent_bands)
                ok = np.isfinite(new)
                xp = np.where(ok, new, xp)
                live &= ok

            # the width and the z support come from the frame at the converged root
            dx = qx - xp * n_sp_x
            dy = qy - xp * n_sp_y
            r_phys = np.sqrt(dx * dx + dy * dy)
            z_ret = sp - (t - r_phys)
            poly, xi_bar, z_bar, s_xi, s_z = self._comoving_frame_at(t - r_phys)
            w = (z_ret - z_bar) / s_z
            live &= (w >= tr.w_start) & (w < tr.w_end)

            centre = self._eval_poly_rows(poly, z_ret) + xi_bar
            # with the curvature shear the beam ridge sits sigma_xi*g(w) off the straight
            # axis Eq 4.24 solved on; centre the band on the ridge (the root error from
            # solving on the axis is inside the band, as for the frame blend)
            # du/dx' along the column (see band_jacobian_min below); the ridge offset is
            # sigma_xi*g in u, i.e. sigma_xi*g/J in x'
            drdx = -(dx * n_sp_x + dy * n_sp_y) / np.maximum(r_phys, 1e-300)
            J = 1.0 - poly[:, 0] * drdx
            Jc = np.where(J < 0, -1.0, 1.0) * np.maximum(np.abs(J), 0.2)
            centre = centre + self._curv_shift_at(t - r_phys, z_ret) / Jc
            half = 0.5 * (tr.u_end - tr.u_start) * s_xi * margin
            jmin = self.integration_params.band_jacobian_min
            if jmin and scale_bands:
                # The band is sized in lab x', but the beam is sized in u at the
                # retarded time, and z_ret moves with x' through |r - r'|:
                #     du/dx' = (1 - tau * d|r-r'|/dx') / sigma_xi
                # (the frame's own change across one band is second order). Where the
                # line of sight is oblique to a tilted beam -- columns in a bend seen
                # from the drift after it -- |J| falls to ~0.25, so +-10 sigma_xi of x'
                # covers only +-2.5 sigma_xi of the beam (18% of the D3 far integral
                # missed). Scale the band by 1/|J| so it covers the same +-xlim*margin
                # in u everywhere, and so the same nodes per beam width.
                half = half / np.maximum(np.abs(J), jmin)
            # A dead branch carries no density whatever band it is given, so it gets
            # zero width and _integrate_xi_region drops its contribution outright.
            # Its position still has to be finite and sane: on a dead column the
            # fixed point had no root to converge to and can wander (45 m has been
            # observed), which contributes nothing but produces absurd node
            # coordinates in debug output and risks landing on the 1/|r-r'| pole.
            # Park it a few band widths off the observation point instead: finite,
            # in a zero-density region, and never exactly at r = 0.
            parked = x + 4.0 * half
            bands.append((np.where(live, centre - half, parked),
                          np.where(live, centre + half, parked)))
            diag.append({'rad': rad, 'live': live, 'xp': xp, 'r': r_phys,
                         'centre': centre, 'half': half})

        self._branch_diag = {'sp': sp, 'branches': diag}
        return bands

    @staticmethod
    def _disjoint_bands(bands):
        """
        Make per-column intervals mutually disjoint by SUBTRACTION.

        The bands all have the same width, so two of them can only overlap
        one-sidedly: whichever starts first covers [lo1, hi1], and the part of the
        other not already covered is [hi1, hi2]. Clipping to that is exact, and a
        band fully inside another collapses to zero width.

        Deliberately not a union. Replacing two overlapping bands by their union is
        also exact, but it spreads the same xbins nodes over up to twice the width,
        doubling the transverse cell size precisely where the integrand peaks --
        which would undo Step 4. Subtraction never widens a band.

        NOTE: the returned bands are ordered by position PER COLUMN, so band j here
        is not branch j of _retarded_xi_bands -- the two swap wherever lo_B < lo_A.
        Aliveness must therefore be tested as (hi > lo) on the returned bands, never
        by pairing them with _branch_diag['branches'][j]['live'], which is recorded
        before this reordering.
        """
        if len(bands) < 2:
            return bands
        if len(bands) > 2:
            raise NotImplementedError('subtraction is exact for two bands only')
        (loA, hiA), (loB, hiB) = bands
        a_first = loA <= loB
        lo1 = np.where(a_first, loA, loB)
        hi1 = np.where(a_first, hiA, hiB)
        lo2 = np.where(a_first, loB, loA)
        hi2 = np.where(a_first, hiB, hiA)
        return [(lo1, hi1), (np.minimum(np.maximum(lo2, hi1), hi2), hi2)]

    def _frame_tilt(self):
        """
        The tilt the DENSITY frame uses, i.e. the core fit stored by the tracker.

        beams.slope fits ALL particles (beams.py); DF_tracker_comoving fits the core
        only (fit_zlim). They disagree maximally where the x-z correlation vanishes,
        and the integration layout must agree with the frame the density actually
        lives in, or the bands sit beside the ribbon.
        """
        tr = self.DF_tracker
        pc = getattr(tr, 'poly_coeffs', None)
        if pc is None:
            return float(self.beam._slope[0])
        return float(pc[0])
