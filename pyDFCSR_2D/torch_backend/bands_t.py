"""
The band solve of the integration layout (BandsMixin._retarded_xi_bands) in torch, for the batched
layout of many mesh points (LayoutMixin._drive_many) on the device. Line by line the numpy solver's
arithmetic: the two Eq 4.24 roots, each refined by a fixed point through the retarded frame, the
width and z support from the frame at the converged root, the curvature shift of the ridge, the
oblique-band widening. Inputs and outputs are numpy (the layout around it stays numpy); the
results agree with the numpy solver to round-off, so the layout's decisions (node counts, causal
edges) can differ only where a threshold is met to round-off.

Selected by CSR_computation.torch_bands (with backend: torch).
"""
import numpy as np
import torch

from .history import F64

TINY = 1e-300


def _lookup(times, q, is_uniform, min_t, delta_t):
    """lookup.lookup_vec in torch: (k, a) with a one-snapshot history handled (k = 0, a = 0)."""
    n = times.shape[0]
    if is_uniform:
        t_idx = (q - min_t) / delta_t
        k = torch.floor(t_idx).long().clamp(0, max(n - 2, 0))
        return k, (t_idx - k).clamp(0.0, 1.0)
    k = (torch.searchsorted(times, q, right=True) - 1).clamp(0, max(n - 2, 0))
    k1 = (k + 1).clamp(max=n - 1)
    den = times[k1] - times[k]
    a = torch.where(den > 0.0, (q - times[k]) / torch.where(den > 0.0, den, torch.ones_like(den)),
                    torch.zeros_like(q))
    return k, a.clamp(0.0, 1.0)


class BandSolver:
    """The tracker's frame history and the lattice tables on the device, and the solve itself."""

    def __init__(self, sim, device):
        from .history import Orbit
        tr = sim.DF_tracker
        dev = torch.device(device)
        self.dev = dev
        self.sim = sim
        T = lambda a: torch.as_tensor(np.ascontiguousarray(a, dtype=np.float64), dtype=F64, device=dev)
        self.n_t = int(np.asarray(tr.poly_coeffs_interp).shape[0])
        self.t_arr, self.is_uniform = T(tr.t_arr), bool(tr.is_uniform)
        self.min_t, self.delta_t = float(tr.min_x), float(tr.delta_x)
        self.poly = T(tr.poly_coeffs_interp)
        self.xi_bar, self.z_bar = T(tr.xi_bar_arr), T(tr.z_bar_arr)
        self.sigma_xi, self.sigma_z = T(tr.sigma_xi_arr), T(tr.sigma_z_arr)
        self.var_z, self.cov, self.var_x, self.x_bar = T(tr.var_z_arr), T(tr.cov_arr), T(tr.var_x_arr), T(tr.x_bar_arr)
        self.c2, self.c3 = T(tr.curv_c2_arr), T(tr.curv_c3_arr)
        self.blend = int(tr.frame_blend_code)
        self.BLEND_COEFF, self.BLEND_MOMENT = tr.BLEND_COEFF, tr.BLEND_MOMENT
        self.curvature, self.curv_zlim = bool(tr.curvature), float(tr.curv_zlim)
        self.w_start, self.w_end = float(tr.w_start), float(tr.w_end)
        self.u_span = float(tr.u_end - tr.u_start)
        self.orbit = Orbit.from_lattice(sim.lattice, dev)          # the lattice file's geometry, as numpy's

    # --- BandsMixin._comoving_frame_at
    def frame_at(self, t_ret):
        k, a = _lookup(self.t_arr, t_ret, self.is_uniform, self.min_t, self.delta_t)
        k1 = (k + 1).clamp(max=self.n_t - 1)
        b = 1.0 - a
        lin = lambda arr: b * arr[k] + a * arr[k1]
        loglin = lambda arr: torch.exp(b * torch.log(arr[k]) + a * torch.log(arr[k1]))
        if self.blend != self.BLEND_COEFF:
            var_z, cov, z_bar, x_bar = lin(self.var_z), lin(self.cov), lin(self.z_bar), lin(self.x_bar)
            tau = cov / var_z
            poly = torch.stack([tau, x_bar - tau * z_bar], dim=1)
            if self.blend == self.BLEND_MOMENT:
                s_xi = torch.sqrt((lin(self.var_x) - cov * cov / var_z).clamp(min=0.0))
                s_z = torch.sqrt(var_z)
            else:
                s_xi, s_z = loglin(self.sigma_xi), loglin(self.sigma_z)
            return poly, torch.zeros_like(tau), z_bar, s_xi, s_z, k, a
        poly = b[:, None] * self.poly[k] + a[:, None] * self.poly[k1]
        return poly, lin(self.xi_bar), lin(self.z_bar), loglin(self.sigma_xi), loglin(self.sigma_z), k, a

    def curv_shift_at(self, t_ret, z_ret):
        if not self.curvature:
            return torch.zeros_like(t_ret)
        _, _, z_bar, s_xi, s_z, k, a = self.frame_at(t_ret)
        k1 = (k + 1).clamp(max=self.n_t - 1)
        b = 1.0 - a
        c2 = b * self.c2[k] + a * self.c2[k1]
        c3 = b * self.c3[k] + a * self.c3[k1]
        w = (z_ret - z_bar) / s_z
        wc = w.clamp(-self.curv_zlim, self.curv_zlim)
        g = c2 * (wc * wc - 1.0) + c3 * wc ** 3 + (2.0 * c2 * wc + 3.0 * c3 * wc * wc) * (w - wc)
        return s_xi * g

    @staticmethod
    def eq424(T, nq, q2, tau, b, sign, tangent):
        """BandsMixin._eq424 (see its docstring for every guard)."""
        flat = tau.abs() < 1e-9
        tau_s = torch.where(flat, torch.ones_like(tau), tau)
        deg = (~flat) & ((tau_s.abs() - 1.0).abs() < 1e-9)
        Tb = T - b / tau_s
        rad = (tau_s ** 2 - 1.0) * (Tb ** 2 - q2) + (nq * tau_s + Tb) ** 2
        gen = (~flat) & (~deg) & (rad >= 0.0)
        one = torch.ones_like(tau_s)
        nan = torch.full_like(tau_s, float('nan'))
        xp = torch.where(gen, tau_s * (Tb + nq * tau_s + sign * torch.sqrt(rad.clamp(min=0.0)))
                         / torch.where(gen, tau_s ** 2 - 1.0, one), nan)
        if tangent:
            tan = (~flat) & (~deg) & (rad < 0.0) & (tau_s.abs() > 1.0)
            p2 = (q2 - nq * nq).clamp(min=0.0)
            xp = torch.where(tan, nq + torch.sign(tau_s) * torch.sqrt(p2 / torch.where(tan, tau_s ** 2 - 1.0, one)), xp)
        Bc = -2.0 * T / tau_s - 2.0 * nq
        lin = deg & (Bc != 0.0)
        xp = torch.where(lin, -(q2 - T ** 2) / torch.where(lin, Bc, one), xp)
        xp = torch.where(flat, b, xp)
        l = torch.where(flat, one, Tb + xp / tau_s)
        return torch.where(torch.isfinite(xp) & (l > 0.0), xp, nan), rad

    def solve(self, s, x, t, sp, n_iter=12, scale_bands=True):
        """BandsMixin._retarded_xi_bands for per-column observers s, x (numpy arrays like sp)."""
        sim = self.sim
        ip = sim.integration_params
        from ..bands import degenerate_margin_factor
        T_ = lambda a: torch.as_tensor(np.asarray(a, dtype=np.float64), dtype=F64, device=self.dev)
        sp_t, s_t, x_t = T_(sp), T_(s), T_(x)
        X0s, Y0s, nsx, nsy, _, _ = self.orbit.vectors(s_t)
        X0p, Y0p, npx, npy, _, _ = self.orbit.vectors(sp_t)
        qx = X0s - X0p + x_t * nsx
        qy = Y0s - Y0p + x_t * nsy
        nq = npx * qx + npy * qy
        q2 = qx * qx + qy * qy
        Tt = t - sp_t
        tau0 = sim._frame_tilt()
        margin = ip.xi_band_margin * degenerate_margin_factor(2.0 * tau0 / (1.0 + tau0 * tau0),
                                                              ip.xi_band_margin_degenerate)
        bands, diag = [], []
        for sign in (-1.0, +1.0):
            xp = x_t.clone()
            live = torch.ones_like(sp_t, dtype=torch.bool)
            rad = torch.zeros_like(sp_t)
            for _ in range(n_iter):
                dx = qx - xp * npx
                dy = qy - xp * npy
                poly, xi_bar = self.frame_at(t - torch.sqrt(dx * dx + dy * dy))[:2]
                new, rad = self.eq424(Tt, nq, q2, poly[:, 0], poly[:, -1] + xi_bar, sign, ip.tangent_bands)
                ok = torch.isfinite(new)
                xp = torch.where(ok, new, xp)
                live = live & ok
            dx = qx - xp * npx
            dy = qy - xp * npy
            r = torch.sqrt(dx * dx + dy * dy)
            z_ret = sp_t - (t - r)
            poly, xi_bar, z_bar, s_xi, s_z, _, _ = self.frame_at(t - r)
            w = (z_ret - z_bar) / s_z
            live = live & (w >= self.w_start) & (w < self.w_end)
            centre = poly[:, 0] * z_ret + poly[:, 1] + xi_bar
            drdx = -(dx * npx + dy * npy) / r.clamp(min=TINY)
            J = 1.0 - poly[:, 0] * drdx
            Jc = torch.where(J < 0, -torch.ones_like(J), torch.ones_like(J)) * J.abs().clamp(min=0.2)
            centre = centre + self.curv_shift_at(t - r, z_ret) / Jc
            half = 0.5 * self.u_span * s_xi * margin
            jmin = ip.band_jacobian_min
            if jmin and scale_bands:
                half = half / J.abs().clamp(min=jmin)
            parked = x_t + 4.0 * half
            lo = torch.where(live, centre - half, parked)
            hi = torch.where(live, centre + half, parked)
            N = lambda v: v.detach().cpu().numpy()
            bands.append((N(lo), N(hi)))
            diag.append({'rad': N(rad), 'live': N(live), 'xp': N(xp), 'r': N(r), 'centre': N(centre), 'half': N(half)})
        return bands, diag
