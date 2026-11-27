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
        before = s <= self.min_s - self.delta_s
        if bool(before.any()):
            t0 = self.table[0]
            ds = (s - self.min_s).unsqueeze(-1)
            ext = torch.stack([t0[0] + ds[:, 0] * t0[4], t0[1] + ds[:, 0] * t0[5],
                               t0[2].expand_as(ds[:, 0]), t0[3].expand_as(ds[:, 0]),
                               t0[4].expand_as(ds[:, 0]), t0[5].expand_as(ds[:, 0])], dim=-1)
            v = torch.where(before.unsqueeze(-1), ext, v)
        return v.unbind(-1)

    def curvature(self, sp):
        """Curvature of the element each s' falls in; 0 past the end and before s' = 0."""
        nd = self.dist.shape[0]
        k = torch.searchsorted(self.dist, sp, right=True)
        r = self.rho_el[k.clamp(max=nd - 1)]
        r = torch.where(k < nd, r, torch.zeros_like(r))
        return torch.where(sp < 0.0, torch.zeros_like(r), r)
