"""
The CSR wake at one observer: the band-localised integral over every s' region with the
error-based x' refinement, the polar patch around the pole, and the integrand from the co-moving
history (docs/DFCSR_2D_methods.md §5.1, §5.4, §5.5).

Mixed into CSR.CSR2D; the methods use its state (beam, lattice, DF_tracker, integration_params).
"""
import numpy as np

from .integrand_kernel import csr_integrand_kernel
from .interp3D import interpolate3D_comoving_fields, interpolate3D_comoving_ibp


class WakeMixin:

    def get_CSR_wake(self, s, x, debug=False):
        """
        (dE/dct, x kick) at the observer (s, x): the band-localised integral over every s'
        region, plus the polar patch around the 1/|r - r'| pole (methods §5.1). debug=True
        returns the regions' node meshes and integrands instead.
        """
        lay = self.wake_layout(s, x)
        t = lay['t']
        ip = self.integration_params
        self._ibp_ship_acc = 0.0
        self._ibp_gross_acc = 0.0
        self._ibp_zone_acc = 0.0
        parts = [self._integrate_xi_region(s, x, t, reg['sp'], reg['taper'], adapt_x=reg['adapt_x'],
                                           bands=reg['bands'])
                 for reg in lay['regions']]
        radii = lay['patch']
        if radii is not None:
            parts.append(self._integrate_near_patch(s, x, t, *radii))
        sps = [reg['sp'] for reg in lay['regions'][:2]]

        if debug:
            return {'mode': 'xi_bands',
                    'near_patch_radii': radii,
                    'sp': list(sps),
                    'xp_mesh': [p[2] for p in parts],
                    'sp_mesh': [p[3] for p in parts],
                    'integrand_z': [p[4] for p in parts],
                    'integrand_x': [p[5] for p in parts]}
        if ip.ibp:
            # the shipped form's total for the consistency check: the regions' shipped sums plus
            # the patch, which both forms share
            self._last_wake_shipped = self._ibp_ship_acc + (parts[-1][0] if radii is not None else 0.0)
        return sum(p[0] for p in parts), sum(p[1] for p in parts)

    def wake_layout(self, s, x):
        """
        The quadrature layout of get_CSR_wake at the observer (s, x), without any integrand
        evaluation: per s' region its nodes, the two disjoint x' bands per column, the radial taper
        and whether the x' refinement applies; and the polar patch radii (None: no patch). The
        torch backend evaluates the same layout (torch_backend.evaluate).
        """
        t = self.beam.position
        ip = self.integration_params
        bnds = self._layout_bounds(s, x)
        (s1, s3), (_, s4) = bnds
        self._region_node_counts(bnds)      # sets _last_region_nodes for diagnostics
        sps = [self._far_region_nodes(s, x, t, s1, s3),
               self._near_region_nodes(s, s3, s4, x=x, t=t)]
        # only the near region straddles s' = s, so only it owns the pole and gets the patch
        radii = self._near_patch_radii(s, s3, s4)
        tapers = [None, None if radii is None else (*radii, False)]
        # band_jacobian_min widens the far region's oblique columns only (methods §5.3)
        regions = [dict(sp=spk, taper=tp, scale_bands=(k == 0),
                        adapt_x=(k == 1) or (k == 0 and ip.far_adapt_x))
                   for k, (spk, tp) in enumerate(zip(sps, tapers))]
        for sp_extra in self._extra_region_nodes(s, x, t, s1):
            regions.append(dict(sp=sp_extra, taper=None, scale_bands=True, adapt_x=ip.far_adapt_x))
        for reg in regions:
            reg['bands'] = self._disjoint_bands(self._retarded_xi_bands(s, x, t, reg['sp'],
                                                                        scale_bands=reg['scale_bands']))
        return dict(t=t, regions=regions, patch=radii)

    def _integrate_xi_region(self, s, x, t, sp, taper=None, scale_bands=True, adapt_x=False, bands=None):
        """
        Integrate one s' region with the transverse nodes riding the density ribbon.

        Each localization branch gets its own set of x' nodes per s' column, so the
        inner trapezoid rule needs a per-column spacing. For a uniform grid
        trapz(y, dx=h) = h * trapz(y, dx=1), so the unit-spacing result is simply
        scaled by each column's dx.
        """
        if bands is None:
            bands = self._disjoint_bands(self._retarded_xi_bands(s, x, t, sp, scale_bands=scale_bands))
        nx = self.integration_params.xbins
        frac = np.linspace(0.0, 1.0, nx)
        ibp = self.integration_params.ibp

        dE = xk = 0.0
        meshes = []
        for lo, hi in bands:
            width = hi - lo
            alive = width > 0.0
            if not alive.any():
                continue
            xp_mesh = lo[None, :] + frac[:, None] * width[None, :]
            sp_mesh = np.broadcast_to(sp[None, :], xp_mesh.shape).copy()
            dxp = width / (nx - 1)

            ip = self.integration_params
            # a column refined in x' keeps the shipped form on its own nodes (with the error-based
            # refinement that is every live column, so ibp acts only where refinement is off)
            do_adapt_x = bool(adapt_x and ip.near_adapt and ip.near_adapt_x)
            ex = {} if ibp else None
            gz, gx = self.get_CSR_integrand(s=s, t=t, x=x, xp=xp_mesh, sp=sp_mesh, taper=taper, extras=ex)
            refined = None
            if do_adapt_x:
                dxp_col, refined = self._adapt_x_error(s, x, t, sp, lo, width, gz, gx, taper)
            # A zero-width column has dxp = 0 and contributes nothing. Select it out
            # rather than multiplying by zero, so a degenerate node that landed on
            # the 1/|r-r'| pole cannot turn the whole region into a nan.
            iz = np.where(alive, np.trapz(y=gz, axis=0) * dxp, 0.0)
            ix = np.where(alive, np.trapz(y=gx, axis=0) * dxp, 0.0)
            iz_ship = iz
            if ibp:
                g_ibp, bnd, am, zone = self._ibp_integrand(gz, ex, t, taper, xp_mesh.shape)
                iz = np.where(alive, np.trapz(y=g_ibp, axis=0) * dxp + bnd, 0.0)
                wcol = np.where(alive, dxp, 0.0) * np.gradient(sp)
                self._ibp_gross_acc += float(np.sum(am.sum(axis=0) * wcol))
                self._ibp_zone_acc += float(np.sum((am * zone).sum(axis=0) * wcol))
            if refined is not None and refined.any():
                # refined columns: (iz, ix) already integrated over their own x' nodes, in BOTH
                # forms (the shipped total of the consistency check must see the same refinement)
                iz = np.where(alive & refined, dxp_col[0], iz)
                iz_ship = np.where(alive & refined, dxp_col[0], iz_ship)
                ix = np.where(alive & refined, dxp_col[1], ix)
            if ibp:
                # the shipped total is kept for the consistency check (get_CSR_wake)
                self._ibp_ship_acc += -self.CSR_scaling * np.trapz(y=iz_ship, x=sp)
            dE += -self.CSR_scaling * np.trapz(y=iz, x=sp)
            xk += self.CSR_scaling * np.trapz(y=ix, x=sp)
            meshes.append((xp_mesh, sp_mesh, gz, gx))

        if not meshes:
            z = np.zeros((nx, len(sp)))
            return (0.0, 0.0, z, np.broadcast_to(sp[None, :], z.shape).copy(), z, z)
        return (dE, xk) + tuple(np.concatenate(a, axis=0) for a in zip(*meshes))

    def _adapt_x_error(self, s, x, t, sp, lo, width, gz, gx, taper):
        """
        Per-column x' quadrature from the integrand's own error estimate (methods §5.4).

        gz, gx are the integrands on the xbins uniform nodes of every column (lo + k h). Each
        round adds the midpoints of the current intervals of the columns still flagged, so the
        trapezoid sum is refined in place, T_2m = T_m / 2 + (h / 2) sum f(mid), and the error of the
        finer sum is estimated as |T_2m - T_m| / 3. The first round covers every live column; after
        it, the columns are ranked by their s'-weighted error w_j e_j and the smallest set whose
        removal leaves at most x_adapt_tol * sum_j w_j |T_j| (W and F separately) is refined again,
        up to x_adapt_rounds rounds in all. Returns ((iz, ix), refined) with refined = every live
        column, since all of them now carry at least the 2n-node value.
        """
        ip = self.integration_params
        nx, ncol = gz.shape
        alive = width > 0.0
        h = np.where(alive, width / (nx - 1), 0.0)
        Tz = np.trapz(gz, axis=0) * h
        Tx = np.trapz(gx, axis=0) * h
        m = nx - 1                                          # intervals per column (same for all live ones)
        wcol = np.zeros(ncol)
        if ncol > 1:
            d = np.diff(sp); wcol[:-1] += 0.5 * d; wcol[1:] += 0.5 * d
        else:
            wcol[:] = 1.0
        wcol = np.abs(wcol)
        cols = np.flatnonzero(alive)
        h_c = h.copy()
        m_c = np.full(ncol, m)
        n_eval = 0
        for rnd in range(max(ip.x_adapt_rounds, 1)):
            if cols.size == 0:
                break
            mm = int(m_c[cols].max())                       # columns refined together have the same count
            k = np.arange(mm) + 0.5
            xn = lo[None, cols] + k[:, None] * h_c[None, cols]
            sn = np.broadcast_to(sp[cols][None, :], xn.shape).copy()
            fz, fx = self.get_CSR_integrand(s=s, t=t, x=x, xp=xn, sp=sn, taper=taper)
            n_eval += xn.size
            Tz_new = 0.5 * Tz[cols] + 0.5 * h_c[cols] * fz.sum(axis=0)
            Tx_new = 0.5 * Tx[cols] + 0.5 * h_c[cols] * fx.sum(axis=0)
            ez = np.abs(Tz_new - Tz[cols]) / 3.0
            ex_ = np.abs(Tx_new - Tx[cols]) / 3.0
            Tz[cols], Tx[cols] = Tz_new, Tx_new
            h_c[cols] *= 0.5
            m_c[cols] *= 2
            # which columns still carry too much of the error: per component, the largest weighted errors
            # until what is left unrefined is within tolerance of the region's gross sum
            keep = np.zeros(cols.size, dtype=bool)
            for e, T in ((ez, Tz), (ex_, Tx)):
                we = wcol[cols] * e
                budget = ip.x_adapt_tol * float(np.sum(wcol * np.abs(T)))
                if not we.sum() > budget:
                    continue
                order = np.argsort(-we)
                rest = we.sum() - np.cumsum(we[order])
                nk = int(np.searchsorted(-rest, -budget)) + 1
                keep[order[:nk]] = True
            cols = cols[keep]
        self._last_adapt_x = dict(cols=int(alive.sum()), evals=n_eval, unconverged=int(cols.size))
        return (Tz, Tx), alive

    def _integrate_near_patch(self, s, x, t, R1, R2):
        """
        Integrate the near field on a polar mesh centred on x' = x, s' = s.

        The integrand diverges as 1/|r - r'| there. Verified numerically: along rays
        out of that point, integrand * |r - r'| is constant over four decades. The
        polar area element r dr dphi cancels that factor exactly, leaving a bounded
        smooth integrand -- which is what the trapezoid rule needs. A uniform
        Cartesian mesh instead accumulates equal contributions per decade of r and
        never converges.

        Radial nodes sit at cell midpoints so r = 0 is never evaluated. phi is
        periodic, so a plain sum is already the trapezoid rule.
        """
        nr = self.integration_params.near_patch_nr
        nphi = self.integration_params.near_patch_nphi

        dr = R2 / nr
        r = (np.arange(nr) + 0.5) * dr
        dphi = 2.0 * np.pi / nphi
        phi = np.arange(nphi) * dphi

        xp_mesh = x + r[:, None] * np.cos(phi)[None, :]
        sp_mesh = s + r[:, None] * np.sin(phi)[None, :]

        gz, gx = self.get_CSR_integrand(s=s, t=t, x=x, xp=xp_mesh, sp=sp_mesh, taper=(R1, R2, True))
        jac = r[:, None] * dr * dphi
        dE = -self.CSR_scaling * np.sum(gz * jac)
        xk = self.CSR_scaling * np.sum(gx * jac)
        return dE, xk, xp_mesh, sp_mesh, gz, gx

    def get_CSR_integrand(self, s, x, t, sp, xp, taper=None, extras=None):
        """
        Longitudinal and transverse integrands at the source points (xp, sp) for the observer
        (s, x) at time t (thesis Eqs 4.4 and 4.8; the arithmetic is integrand_kernel).

        taper: (R1, R2, keep_inner) -- multiply by a radial partition of |r - r'|: with keep_inner
        False the weight rises 0 -> 1 across [R1, R2] (removing the pole's neighbourhood), with True
        it is the complement; the two pieces sum to the untapered integral exactly.
        extras: a dict to fill with what the integrated-by-parts form needs (CSR_integration.ibp);
        the fields then come from interpolate3D_comoving_ibp, bit-identical for the five fields.
        """
        self._assert_history()
        ip = self.integration_params
        sp_flat = sp.ravel()
        xp_flat = xp.ravel()

        vx = self._comoving_fields(np.array([x]), np.array([s - t]), np.array([t]))[3][0]
        if self._windowed_frame():
            vx = self._observer_vx(x, s - t, vx)

        (X0_s, Y0_s, n_vec_s_x, n_vec_s_y,
         tau_vec_s_x, tau_vec_s_y) = (v[0] for v in self._lattice_vectors(np.array([s])))
        (X0_sp, Y0_sp, n_vec_sp_x, n_vec_sp_y,
         tau_vec_sp_x, tau_vec_sp_y) = self._lattice_vectors(sp_flat)
        r_minus_rp_x = X0_s - X0_sp + x * n_vec_s_x - xp_flat * n_vec_sp_x
        r_minus_rp_y = Y0_s - Y0_sp + x * n_vec_s_y - xp_flat * n_vec_sp_y
        r_minus_rp = np.sqrt(r_minus_rp_x**2 + r_minus_rp_y**2)

        # curvature of the element each s' falls in (0 past the lattice end, and on the synthetic
        # straight pre-history before s' = 0 even when the lattice starts with a dipole)
        dist = np.asarray(self.lattice.distance)
        rho_el = np.asarray(self.lattice.rho)
        k_el = np.searchsorted(dist, sp_flat, side='right')
        rho_sp = np.where(k_el < dist.size, rho_el[np.minimum(k_el, dist.size - 1)], 0.0)
        if sp_flat.min() < 0.0:
            rho_sp = np.where(sp_flat < 0.0, 0.0, rho_sp)

        t_ret = t - r_minus_rp
        z_ret = sp_flat - t_ret
        if extras is None or not ip.ibp:
            # the five fields share the frame construction and stencil, so one pass returns all
            (density_ret, density_x_ret, density_z_ret,
             vx_ret, vx_x_ret) = self._comoving_fields(xp_flat, z_ret, t_ret)
            A_ibp = B_ibp = None
        else:
            (density_ret, density_x_ret, density_z_ret,
             vx_ret, vx_x_ret, A_ibp, B_ibp) = self._comoving_fields_ibp(xp_flat, z_ret, t_ret)

        n = xp_flat.size
        CSR_integrand_z = np.empty(n)
        CSR_integrand_x = np.empty(n)
        vn = np.empty(n)
        as_arr = lambda v: (np.broadcast_to(np.asarray(v, dtype=float), (n,)) if np.ndim(v) == 0
                            else np.ascontiguousarray(v, dtype=float))
        R1, R2, keep_inner = taper if taper is not None else (0.0, 1.0, False)
        csr_integrand_kernel(np.ascontiguousarray(xp_flat, dtype=float), as_arr(rho_sp), as_arr(r_minus_rp_x),
                             as_arr(r_minus_rp_y), as_arr(r_minus_rp), as_arr(n_vec_sp_x), as_arr(n_vec_sp_y),
                             as_arr(tau_vec_sp_x), as_arr(tau_vec_sp_y),
                             as_arr(density_ret), as_arr(density_x_ret), as_arr(density_z_ret),
                             as_arr(vx_ret), as_arr(vx_x_ret),
                             float(vx), float(n_vec_s_x), float(n_vec_s_y), float(tau_vec_s_x), float(tau_vec_s_y),
                             ip.transverse_terms == 'W1', taper is not None,
                             float(R1), float(R2), bool(keep_inner), CSR_integrand_z, CSR_integrand_x, vn)
        if extras is not None:
            extras.update(rx=r_minus_rp_x, ry=r_minus_rp_y, nx=n_vec_sp_x, ny=n_vec_sp_y,
                          xp=xp_flat, sp=sp_flat, kappa=rho_sp, vn=vn,
                          rho=density_ret, rho_x=density_x_ret, A=A_ibp, B=B_ibp)
        return CSR_integrand_z.reshape(xp.shape), CSR_integrand_x.reshape(xp.shape)

    def _comoving_fields(self, xval, zval, tval):
        """
        (rho, drho/dx, drho/dz|_x, vx, dvx/dx) from the co-moving history, in
        physical units. Thin binding of the tracker's metadata onto the numba
        kernel; see interp3D.interpolate3D_comoving_fields.
        """
        tr = self.DF_tracker
        return interpolate3D_comoving_fields(
            xval, zval, tval,
            tr.data_density_interp, tr.data_density_u_interp,
            tr.data_density_w_interp, tr.data_vx_interp, tr.data_vx_u_interp,
            tr.poly_coeffs_interp, tr.xi_bar_arr, tr.sigma_xi_arr,
            tr.z_bar_arr, tr.sigma_z_arr,
            tr.u_start, tr.delta_u, tr.w_start, tr.delta_w,
            tr.min_x, tr.delta_x,
            tr.frame_blend_code, tr.var_z_arr, tr.cov_arr, tr.var_x_arr,
            tr.x_bar_arr,
            tr.t_arr, tr.bucket, tr.bucket_inv_h, tr.bucket_t0, tr.bucket_M,
            tr.is_uniform, tr.curv_c2_arr, tr.curv_c3_arr, float(tr.curv_zlim))

    def _comoving_fields_ibp(self, xval, zval, tval, want_fields=True):
        """
        _comoving_fields plus the by-parts coefficients (A, B); see
        interp3D.interpolate3D_comoving_ibp. want_fields False returns only A (the rest 0).
        """
        tr = self.DF_tracker
        return interpolate3D_comoving_ibp(
            xval, zval, tval,
            tr.data_density_interp, tr.data_density_u_interp,
            tr.data_density_w_interp, tr.data_vx_interp, tr.data_vx_u_interp,
            tr.poly_coeffs_interp, tr.xi_bar_arr, tr.sigma_xi_arr,
            tr.z_bar_arr, tr.sigma_z_arr,
            tr.u_start, tr.delta_u, tr.w_start, tr.delta_w,
            tr.min_x, tr.delta_x,
            tr.frame_blend_code, tr.var_z_arr, tr.cov_arr, tr.var_x_arr,
            tr.x_bar_arr,
            tr.t_arr, tr.bucket, tr.bucket_inv_h, tr.bucket_t0, tr.bucket_M,
            tr.is_uniform, tr.curv_c2_arr, tr.curv_c3_arr, float(tr.curv_zlim),
            bool(want_fields))

    @staticmethod
    def _radial_taper(R, taper):
        """The near-patch partition weight of get_CSR_integrand (1 if taper is None)."""
        if taper is None:
            return np.ones_like(R)
        R1, R2, keep_inner = taper
        u = np.clip((R - R1) / (R2 - R1), 0.0, 1.0)
        w = u * u * u * (u * (6.0 * u - 15.0) + 10.0)
        return 1.0 - w if keep_inner else w

    def _assert_history(self):
        """
        Refuse to evaluate a wake without a history to interpolate.

        `compute_CSR: 0` gates get_DF/append_DF, so the run produces a beam with NO density
        history. Anything calling get_CSR_integrand afterwards then got near-zeros with no error
        raised -- it cost a full set of wrong figures in §6f, and bit again while benchmarking
        the B-spline hoist in Step 10, where debug=False gave a one-snapshot history and
        data_rho[k+1] read out of bounds. The name suggests a pure output switch and it is not.
        """
        n = len(self.DF_tracker.time_log)
        if n < 2:
            raise RuntimeError(
                f'the density history holds {n} snapshot(s); the wake needs at least 2 to '
                f'interpolate between. Set CSR_computation.compute_CSR = 1, or pass '
                f'debug=True to run(), so that snapshots are recorded.')
