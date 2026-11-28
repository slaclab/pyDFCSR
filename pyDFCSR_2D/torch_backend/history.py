"""
The density history and the reference-orbit tables as device tensors, for one kick.

History: the five field stacks of DF_tracker_comoving.build_interpolant (n_t, nu, nw each) packed as
one (n_t, nu, nw, 5) tensor, so a stencil tap gathers all five fields at once, and the per-snapshot
frame arrays. Orbit: the lattice tables used by BandsMixin._lattice_vectors and the element
curvatures used by get_CSR_integrand.
"""
from dataclasses import dataclass

import numpy as np
import torch

F64 = torch.float64


def _t(a, device):
    return torch.as_tensor(np.ascontiguousarray(a, dtype=np.float64), dtype=F64, device=device)


@dataclass
class History:
    fields: torch.Tensor          # (n_t, nu, nw, 5): rho, rho_u, rho_w, vx, vx_u
    poly: torch.Tensor            # (n_t, deg+1), polyfit order
    xi_bar: torch.Tensor
    sigma_xi: torch.Tensor
    z_bar: torch.Tensor
    sigma_z: torch.Tensor
    var_z: torch.Tensor
    cov: torch.Tensor
    var_x: torch.Tensor
    x_bar: torch.Tensor
    curv_c2: torch.Tensor
    curv_c3: torch.Tensor
    t_arr: torch.Tensor
    is_uniform: bool
    min_t: float
    delta_t: float
    u_start: float
    delta_u: float
    w_start: float
    delta_w: float
    blend_code: int
    curv_wl: float

    @classmethod
    def from_tracker(cls, tr, device):
        fields = np.stack([tr.data_density_interp, tr.data_density_u_interp, tr.data_density_w_interp,
                           tr.data_vx_interp, tr.data_vx_u_interp], axis=-1)
        return cls(fields=_t(fields, device), poly=_t(tr.poly_coeffs_interp, device),
                   xi_bar=_t(tr.xi_bar_arr, device), sigma_xi=_t(tr.sigma_xi_arr, device),
                   z_bar=_t(tr.z_bar_arr, device), sigma_z=_t(tr.sigma_z_arr, device),
                   var_z=_t(tr.var_z_arr, device), cov=_t(tr.cov_arr, device),
                   var_x=_t(tr.var_x_arr, device), x_bar=_t(tr.x_bar_arr, device),
                   curv_c2=_t(tr.curv_c2_arr, device), curv_c3=_t(tr.curv_c3_arr, device),
                   t_arr=_t(tr.t_arr, device), is_uniform=bool(tr.is_uniform),
                   min_t=float(tr.min_x), delta_t=float(tr.delta_x),
                   u_start=float(tr.u_start), delta_u=float(tr.delta_u),
                   w_start=float(tr.w_start), delta_w=float(tr.delta_w),
                   blend_code=int(tr.frame_blend_code), curv_wl=float(tr.curv_zlim))


def history_from_tensors(tr, times, fields, frames, device):
    """
    A History from per-snapshot tensors (the differentiable run): fields[k] = (rho, rho_u, rho_w,
    vx, vx_u), frames[k] = the frame dict of torch_backend.deposit.get_DF_rms. Grid constants and
    the time-grid bookkeeping are the tracker's (DF_tracker_comoving.build_interpolant,
    _build_moment_arrays).
    """
    from ..lookup import is_uniform_times
    n_t = len(times)
    t_np = np.asarray(times, dtype=np.float64)
    fstack = torch.stack([torch.stack(f, dim=-1) for f in fields])
    st = lambda k: torch.stack([torch.as_tensor(fr[k], dtype=F64, device=device) for fr in frames])
    poly = st('poly_coeffs')
    sigma_xi, sigma_z, z_bar, xi_bar = st('sigma_xi'), st('sigma_z'), st('z_bar'), st('xi_bar')
    tau = poly[:, 0]
    var_z = sigma_z ** 2
    return History(fields=fstack, poly=poly, xi_bar=xi_bar, sigma_xi=sigma_xi, z_bar=z_bar,
                   sigma_z=sigma_z, var_z=var_z, cov=tau * var_z, var_x=sigma_xi ** 2 + tau ** 2 * var_z,
                   x_bar=tau * z_bar + poly[:, -1] + xi_bar, curv_c2=st('curv_c2'), curv_c3=st('curv_c3'),
                   t_arr=torch.as_tensor(t_np, dtype=F64, device=device),
                   is_uniform=bool(is_uniform_times(t_np)), min_t=float(t_np[0]),
                   delta_t=float((t_np[-1] - t_np[0]) / (n_t - 1)) if n_t > 1 else 1.0,
                   u_start=float(tr.u_start), delta_u=float(tr.delta_u),
                   w_start=float(tr.w_start), delta_w=float(tr.delta_w),
                   blend_code=int(tr.frame_blend_code), curv_wl=float(tr.curv_zlim))


@dataclass
class Orbit:
    table: torch.Tensor           # (n, 6): X0, Y0, n_x, n_y, tau_x, tau_y on the uniform s table
    min_s: float
    delta_s: float
    dist: torch.Tensor            # element ends (lattice.distance)
    rho_el: torch.Tensor          # element curvature angle/L (lattice.rho)

    @classmethod
    def from_lattice(cls, lat, device):
        table = np.column_stack([lat.coords[:, 0], lat.coords[:, 1], lat.n_vec[:, 0], lat.n_vec[:, 1],
                                 lat.tau_vec[:, 0], lat.tau_vec[:, 1]])
        return cls(table=_t(table, device), min_s=float(lat.min_x), delta_s=float(lat.delta_x),
                   dist=_t(lat.distance, device), rho_el=_t(lat.rho, device))

    @classmethod
    def from_lattice_params(cls, lat, params, device):
        """
        The orbit table of lattice.get_reference_traj rebuilt in torch, with bend angles that may
        be tensors (params {(element, 'angle'): tensor}), so the CSR geometry is differentiable in
        them. Same samples and the same element-per-sample rule as the numpy table (the sample that
        crosses an element boundary takes the next element's rule), the same straight pad.
        """
        cfg = lat.lattice_config
        names = [k for k in cfg if k != 'step_size']
        s = np.asarray(lat.s, dtype=float)
        n_lat = int(np.searchsorted(s, lat.distance[-1], side='right'))     # samples up to the end
        n_lat = min(max(n_lat, 2), s.size)
        ds = torch.as_tensor(np.diff(s[:n_lat]), dtype=F64, device=device)
        # element of each step k (1..n_lat-1): the first element whose end is >= s_k
        e_k = np.minimum(np.searchsorted(np.asarray(lat.distance), s[1:n_lat], side='left'), len(names) - 1)
        angle, length = [], []
        for nm in names:
            c = cfg[nm]
            a = params.get((nm, 'angle'), c.get('angle', 0.0)) if c['type'] == 'dipole' else 0.0
            angle.append(torch.as_tensor(a, dtype=F64, device=device))
            length.append(float(c['L']))
        angle = torch.stack(angle)
        length = torch.as_tensor(length, dtype=F64, device=device)
        is_bend = torch.as_tensor([cfg[nm]['type'] == 'dipole' for nm in names], device=device)
        e_t = torch.as_tensor(e_k, device=device)
        bend = is_bend[e_t]
        safe_angle = torch.where(is_bend, angle, torch.ones_like(angle))
        phi = torch.where(bend, ds * angle[e_t] / length[e_t], torch.zeros_like(ds))
        theta = torch.cat([torch.zeros(1, dtype=F64, device=device), torch.cumsum(phi, 0)])
        th0 = theta[:-1]
        r = length[e_t] / safe_angle[e_t]
        dx = torch.where(bend, r * (torch.sin(th0 + phi) - torch.sin(th0)), ds * torch.cos(th0))
        dy = torch.where(bend, r * (torch.cos(th0) - torch.cos(th0 + phi)), ds * torch.sin(th0))
        zero = torch.zeros(1, dtype=F64, device=device)
        X = torch.cat([zero, torch.cumsum(dx, 0)])
        Y = torch.cat([zero, torch.cumsum(dy, 0)])
        # straight pad past the end, as in get_reference_traj
        n_pad = s.size - n_lat
        if n_pad:
            arc = torch.as_tensor(s[n_lat:] - s[n_lat - 1], dtype=F64, device=device)
            th = theta[-1]
            X = torch.cat([X, X[-1] + arc * torch.cos(th)])
            Y = torch.cat([Y, Y[-1] + arc * torch.sin(th)])
            theta = torch.cat([theta, th.expand(n_pad)])
        table = torch.stack([X, Y, torch.sin(theta), -torch.cos(theta), torch.cos(theta), torch.sin(theta)], dim=1)
        rho_el = torch.where(is_bend, angle / length, torch.zeros_like(angle))
        return cls(table=table, min_s=float(lat.min_x), delta_s=float(lat.delta_x),
                   dist=torch.as_tensor(np.asarray(lat.distance, dtype=float), dtype=F64, device=device),
                   rho_el=rho_el)

    def vectors(self, s):
        """
        (X0, Y0, n_x, n_y, tau_x, tau_y) at s: interp1D.interpolate1D (int() truncation, the last
        node held, zero outside) plus BandsMixin._lattice_vectors' straight continuation before the
        table start.
        """
        n = self.table.shape[0]
        q = (s - self.min_s) / self.delta_s
        x0 = torch.trunc(q)
        x0i = x0.long()
        x1i = torch.where(x0i == n - 1, x0i, x0i + 1)
        valid = (x0i >= 0) & (x1i < n)
        i0 = x0i.clamp(0, n - 1)
        i1 = x1i.clamp(0, n - 1)
        xd = (q - x0).unsqueeze(-1)
        v = self.table[i0] * (1 - xd) + self.table[i1] * xd
        v = torch.where(valid.unsqueeze(-1), v, torch.zeros_like(v))
        # computed unconditionally (no data-dependent branch, so torch.compile sees one graph)
        before = s <= self.min_s - self.delta_s
        t0 = self.table[0]
        ds = s - self.min_s
        ext = torch.stack([t0[0] + ds * t0[4], t0[1] + ds * t0[5], t0[2].expand_as(ds),
                           t0[3].expand_as(ds), t0[4].expand_as(ds), t0[5].expand_as(ds)], dim=-1)
        v = torch.where(before.unsqueeze(-1), ext, v)
        return v.unbind(-1)

    def curvature(self, sp):
        """Curvature of the element each s' falls in; 0 past the end and before s' = 0."""
        nd = self.dist.shape[0]
        k = torch.searchsorted(self.dist, sp, right=True)
        r = self.rho_el[k.clamp(max=nd - 1)]
        r = torch.where(k < nd, r, torch.zeros_like(r))
        return torch.where(sp < 0.0, torch.zeros_like(r), r)
