"""
Batched torch port of interp3D.interpolate3D_comoving_fields: the co-moving frame blended in time,
the curvature shear, a 4 x 4 cubic B-spline stencil on the two bracketing snapshots, and the chain
rule back to physical units. Same arithmetic per query; the sums run in a different order, so the
results agree with the numba kernel to round-off, not bit for bit.
"""
import torch


def bspline3(x):
    """Cubic B-spline B3(x), support [-2, 2] (interp3D.cubic_bspline_w)."""
    ax = x.abs()
    inner = (4.0 - 6.0 * ax * ax + 3.0 * ax * ax * ax) / 6.0
    outer = (2.0 - ax).clamp(min=0.0) ** 3 / 6.0
    return torch.where(ax < 1.0, inner, outer)


def lookup(H, q):
    """(k, a) of lookup.lookup_hybrid: bracketing snapshot k in [0, n-2] and weight a in [0, 1]."""
    n = H.t_arr.shape[0]
    if H.is_uniform:
        t_idx = (q - H.min_t) / H.delta_t
        k = torch.floor(t_idx).long().clamp(0, n - 2)
        a = (t_idx - k).clamp(0.0, 1.0)
        return k, a
    k = (torch.searchsorted(H.t_arr, q, right=True) - 1).clamp(0, n - 2)
    a = ((q - H.t_arr[k]) / (H.t_arr[k + 1] - H.t_arr[k])).clamp(0.0, 1.0)
    return k, a


def _poly(coeffs, z):
    out = coeffs[:, 0]
    for j in range(1, coeffs.shape[1]):
        out = out * z + coeffs[:, j]
    return out


def _poly_deriv(coeffs, z):
    n = coeffs.shape[1]
    if n <= 1:
        return torch.zeros_like(z)
    out = coeffs[:, 0] * (n - 1)
    for j in range(1, n - 1):
        out = out * z + coeffs[:, j] * (n - 1 - j)
    return out


def curv_g(c2, c3, w, wl):
    """interp3D.curv_g: the cubic shear inside |w| <= wl, continued linearly beyond."""
    wc = w.clamp(-wl, wl)
    g_in = c2 * (wc * wc - 1.0) + c3 * wc * wc * wc
    d_in = 2.0 * c2 * wc + 3.0 * c3 * wc * wc
    return g_in + d_in * (w - wc), d_in


def frame(H, z, t):
    """The blended frame at (z, t): (k, a, p_a, dp_a, xi_bar, z_bar, s_xi, s_z, c2, c3)."""
    k, a = lookup(H, t)
    b = 1.0 - a
    k1 = k + 1

    def lin(arr):
        return b * arr[k] + a * arr[k1]

    def loglin(arr):
        return torch.exp(b * torch.log(arr[k]) + a * torch.log(arr[k1]))

    z_bar = lin(H.z_bar)
    if H.blend_code == 0:
        p_a = b * _poly(H.poly[k], z) + a * _poly(H.poly[k1], z)
        dp_a = b * _poly_deriv(H.poly[k], z) + a * _poly_deriv(H.poly[k1], z)
        xi_bar = lin(H.xi_bar)
        s_xi, s_z = loglin(H.sigma_xi), loglin(H.sigma_z)
    else:
        var_z = lin(H.var_z)
        cov = lin(H.cov)
        tau = cov / var_z
        dp_a = tau
        p_a = tau * (z - z_bar) + lin(H.x_bar)
        xi_bar = torch.zeros_like(z)
        if H.blend_code == 1:
            s_z = torch.sqrt(var_z)
            s_xi = torch.sqrt((lin(H.var_x) - cov * cov / var_z).clamp(min=0.0))
        else:
            s_xi, s_z = loglin(H.sigma_xi), loglin(H.sigma_z)
    return k, a, p_a, dp_a, xi_bar, z_bar, s_xi, s_z, lin(H.curv_c2), lin(H.curv_c3)


def comoving_fields(H, x, z, t):
    """(rho, rho_x, rho_z, vx, vx_x) at the points (x, z, t), all 1D tensors of one length."""
    k, a, p_a, dp_a, xi_bar, z_bar, s_xi, s_z, c2, c3 = frame(H, z, t)
    b = 1.0 - a
    u = (x - p_a - xi_bar) / s_xi
    w = (z - z_bar) / s_z
    g_w, dg_w = curv_g(c2, c3, w, H.curv_wl)
    u = u - g_w
    u_cell = (u - H.u_start) / H.delta_u - 0.5
    w_cell = (w - H.w_start) / H.delta_w - 0.5

    n_t, n_u, n_w, nf = H.fields.shape
    flat = H.fields.reshape(-1, nf)
    i0 = torch.floor(u_cell).long() - 1
    j0 = torch.floor(w_cell).long() - 1
    base0 = k * (n_u * n_w)
    base1 = base0 + n_u * n_w
    acc0 = torch.zeros(x.shape[0], nf, dtype=x.dtype, device=x.device)
    acc1 = torch.zeros_like(acc0)
    for di in range(4):
        ii = i0 + di
        wi = bspline3(u_cell - ii) * ((ii >= 0) & (ii < n_u))
        iic = ii.clamp(0, n_u - 1)
        for dj in range(4):
            jj = j0 + dj
            wgt = (wi * bspline3(w_cell - jj) * ((jj >= 0) & (jj < n_w))).unsqueeze(-1)
            idx = iic * n_w + jj.clamp(0, n_w - 1)
            acc0 = acc0 + flat[base0 + idx] * wgt
            acc1 = acc1 + flat[base1 + idx] * wgt
    hat = b.unsqueeze(-1) * acc0 + a.unsqueeze(-1) * acc1
    rho_h, rho_hu, rho_hw, vx_h, vx_hu = hat.unbind(-1)
    jac = s_xi * s_z
    rho = rho_h / jac
    rho_x = rho_hu / (s_xi * jac)
    rho_z = (rho_hw / s_z - (dp_a / s_xi + dg_w / s_z) * rho_hu) / jac
    return rho, rho_x, rho_z, vx_h, vx_hu / s_xi
