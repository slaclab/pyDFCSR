"""
Differentiable deposition in torch: the cubic B-spline deposit, the FFT Gaussian smoothing with
spectral derivatives, and DF_tracker_comoving.get_DF (tilt and curvature fits, the affine vx control
variate, the velocity floor and mask) for the 'rms' frame and the windowed frames 'quantile' and
'peak'. In the windowed frames the window (lo, hi, margins), the frame it implies (z_bar, sigma_z),
the z smoothing width and the adaptive-smoothing blend weights are structure, taken from the numpy
tracker's deposit of the same particles (or replayed); the window weight T(z) stays differentiable
in z.

Gradients flow to the particle coordinates through the deposit weights, the frame moments and the
fits. The core masks (|w| inside a fit window) are structure: membership is decided in the forward
pass and not differentiated (a particle crossing the window edge moves the fit by a jump of
O(1/N)).

The deposit is a custom autograd Function: its backward is the B-spline gather of the adjoint
grid with B3', so it saves only the particle coordinates and weights, not 16 weights per particle.
"""
import math

import numpy as np
import torch

F64 = torch.float64


def _b3(x):
    ax = x.abs()
    return torch.where(ax < 1.0, (4.0 - 6.0 * ax * ax + 3.0 * ax * ax * ax) / 6.0,
                       (2.0 - ax).clamp(min=0.0) ** 3 / 6.0)


def _db3(x):
    ax = x.abs()
    inner = -2.0 * x + 1.5 * x * ax
    outer = -0.5 * torch.sign(x) * (2.0 - ax).clamp(min=0.0) ** 2
    return torch.where(ax < 1.0, inner, outer)


def _stencil(q, start, end, n):
    inv = n / (end - start)
    c = (q - start) * inv - 0.5
    i0 = torch.floor(c).long() - 1
    return c, i0, inv


class BSplineDeposit(torch.autograd.Function):
    """
    deposit_smooth.histogram_bspline_2d: sum_p w_p B3(q1_p - i) B3(q2_p - j) on bin-centred cells,
    entries outside the grid dropped.
    """

    @staticmethod
    def forward(ctx, q1, q2, w, n1, s1, e1, n2, s2, e2):
        c1, i1, _ = _stencil(q1, s1, e1, n1)
        c2, i2, _ = _stencil(q2, s2, e2, n2)
        grid = torch.zeros(n1 * n2, dtype=q1.dtype, device=q1.device)
        for a in range(4):
            ia = i1 + a
            wa = _b3(c1 - ia) * ((ia >= 0) & (ia < n1))
            for b in range(4):
                jb = i2 + b
                wb = _b3(c2 - jb) * ((jb >= 0) & (jb < n2))
                idx = ia.clamp(0, n1 - 1) * n2 + jb.clamp(0, n2 - 1)
                grid.index_add_(0, idx, w * wa * wb)
        ctx.save_for_backward(q1, q2, w)
        ctx.geom = (n1, s1, e1, n2, s2, e2)
        return grid.view(n1, n2)

    @staticmethod
    def backward(ctx, G):
        q1, q2, w = ctx.saved_tensors
        n1, s1, e1, n2, s2, e2 = ctx.geom
        c1, i1, inv1 = _stencil(q1, s1, e1, n1)
        c2, i2, inv2 = _stencil(q2, s2, e2, n2)
        Gf = G.reshape(-1)
        g1 = torch.zeros_like(q1)
        g2 = torch.zeros_like(q1)
        gw = torch.zeros_like(q1)
        for a in range(4):
            ia = i1 + a
            ma = (ia >= 0) & (ia < n1)
            wa, da = _b3(c1 - ia) * ma, _db3(c1 - ia) * ma
            for b in range(4):
                jb = i2 + b
                mb = (jb >= 0) & (jb < n2)
                wb, db = _b3(c2 - jb) * mb, _db3(c2 - jb) * mb
                gv = Gf[ia.clamp(0, n1 - 1) * n2 + jb.clamp(0, n2 - 1)]
                gw = gw + gv * wa * wb
                g1 = g1 + gv * da * wb
                g2 = g2 + gv * wa * db
        return (g1 * w * inv1 if ctx.needs_input_grad[0] else None,
                g2 * w * inv2 if ctx.needs_input_grad[1] else None,
                gw if ctx.needs_input_grad[2] else None,
                None, None, None, None, None, None)


def deposit(q1, q2, w, n1, s1, e1, n2, s2, e2):
    w = torch.broadcast_to(torch.as_tensor(w, dtype=q1.dtype, device=q1.device), q1.shape)
    return BSplineDeposit.apply(q1, q2, w, int(n1), float(s1), float(e1), int(n2), float(s2), float(e2))


def _fftfreq(n, d, device):
    return torch.as_tensor(np.fft.fftfreq(n, d=d) * 2 * np.pi, dtype=F64, device=device)


def smooth_and_differentiate(field, dx, dz, sigma_smooth, sigma_smooth_z=None, want=(True, True, True)):
    """deposit_smooth.smooth_and_differentiate (zero-padded FFT Gaussian, spectral d/dx, d/dz)."""
    nxi, nz = field.shape
    px, pz = nxi // 2, nz // 2
    fp = torch.nn.functional.pad(field, (pz, pz, px, px))
    kx = _fftfreq(fp.shape[0], dx, field.device)[:, None]
    kz = _fftfreq(fp.shape[1], dz, field.device)[None, :]
    sx = sigma_smooth * dx
    sz = (sigma_smooth if sigma_smooth_z is None else sigma_smooth_z) * dz
    G = torch.exp(-0.5 * (kx ** 2 * sx ** 2 + kz ** 2 * sz ** 2))
    F = torch.fft.fft2(fp) * G
    crop = lambda a: torch.real(a)[px:px + nxi, pz:pz + nz]
    out = [crop(torch.fft.ifft2(F)) if want[0] else None,
           crop(torch.fft.ifft2(F * (1j * kx))) if want[1] else None,
           crop(torch.fft.ifft2(F * (1j * kz))) if want[2] else None]
    return tuple(out)


def _smootherstep(t):
    t = t.clamp(0.0, 1.0)
    return t * t * t * (t * (6.0 * t - 15.0) + 10.0)


def window_weight(z, lo, hi, margin, margin_hi=None):
    """peakfind.window_weight for margins > 0."""
    m_hi = margin if margin_hi is None else margin_hi
    return _smootherstep((z - (lo - margin)) / margin) * _smootherstep(((hi + m_hi) - z) / m_hi)


def smooth_levels(field, dx, dz, sigma_x, sigma_z_levels, weights, want=(True, True, True)):
    """deposit_smooth.smooth_levels: the field smoothed at each z width, blended with `weights`
    (n_levels, nx, nz, constants), spectral derivatives of the blend."""
    nxi, nz = field.shape
    px, pz = nxi // 2, nz // 2
    pad = lambda a: torch.nn.functional.pad(a, (pz, pz, px, px))
    F = torch.fft.fft2(pad(field))
    kx = _fftfreq(F.shape[0], dx, field.device)[:, None]
    kz = _fftfreq(F.shape[1], dz, field.device)[None, :]
    gx = torch.exp(-0.5 * kx ** 2 * (sigma_x * dx) ** 2)
    crop = lambda a: torch.real(a)[px:px + nxi, pz:pz + nz]
    W = torch.as_tensor(weights, dtype=F64, device=field.device)
    blend = torch.zeros_like(field)
    for k, szk in enumerate(sigma_z_levels):
        if bool(torch.any(W[k] != 0)):
            blend = blend + W[k] * crop(torch.fft.ifft2(F * gx * torch.exp(-0.5 * kz ** 2 * (szk * dz) ** 2)))
    out = [blend if want[0] else None, None, None]
    if want[1] or want[2]:
        B = torch.fft.fft2(pad(blend))
        if want[1]:
            out[1] = crop(torch.fft.ifft2(B * (1j * kx)))
        if want[2]:
            out[2] = crop(torch.fft.ifft2(B * (1j * kz)))
    return tuple(out)


def _lstsq(A, b, wt=None):
    """
    (Weighted) least squares by the normal equations (differentiable; 2-5 columns). wt are the
    weights of the squared residuals, i.e. the square of numpy's polyfit/lstsq row weights; using
    them directly keeps the derivative finite where a weight is 0 (sqrt is not).
    """
    if wt is None:
        return torch.linalg.solve(A.T @ A, A.T @ b)
    Aw = A * wt[:, None]
    return torch.linalg.solve(Aw.T @ A, Aw.T @ b)


def window_structure(tr):
    """The structure of a windowed deposit, from the numpy tracker that just made it."""
    lo, hi, m_lo, m_hi = tr.window
    return dict(window=(lo, hi, m_lo, m_hi), z_bar=float(tr.z_bar), sigma_z=float(tr.sigma_z),
                sz=float(tr.sz_used), adaptive=getattr(tr, 'adaptive_levels', None))


def curv_g(c2, c3, w, wl):
    wc = w.clamp(-wl, wl)
    g = c2 * (wc * wc - 1.0) + c3 * wc ** 3
    d = 2.0 * c2 * wc + 3.0 * c3 * wc * wc
    return g + d * (w - wc)


def get_DF_rms(tr, x, z, px):
    """
    DF_tracker_comoving.get_DF for frame 'rms' with torch tensors x, z, px. Returns
    (fields, frame): fields (density, density_u, density_w, vx, vx_u) on the (u, w) grid, frame
    dict (poly_coeffs, xi_bar, z_bar, sigma_xi, sigma_z, curv_c2, curv_c3, vx_coef) as tensors.
    """
    if tr.frame != 'rms':
        raise ValueError(f"get_DF_rms: frame {tr.frame!r}; use get_DF_windowed")
    if tr.poly_degree != 1:
        raise NotImplementedError('torch get_DF: poly_degree 1 only')
    zmean = z.mean()
    zstd = z.std(correction=0)
    with torch.no_grad():
        core = (z - zmean).abs() < tr.fit_zlim * zstd
        if int(core.sum()) < 10 * (tr.poly_degree + 1):
            core = torch.ones_like(core)
    zc, xc = z[core], x[core]
    A = torch.stack([zc, torch.ones_like(zc)], dim=1)
    poly = _lstsq(A, xc)                                    # (slope, intercept), polyfit order
    xi = x - (poly[0] * z + poly[1])
    xi_bar = xi.mean()
    sigma_xi = xi.std(correction=0)
    z_bar, sigma_z = zmean, zstd
    u = (xi - xi_bar) / sigma_xi
    w = (z - z_bar) / sigma_z

    c2 = c3 = torch.zeros((), dtype=F64, device=x.device)
    if tr.curvature:
        with torch.no_grad():
            cw = w.abs() < tr.curv_zlim
        wc, uc = w[cw], u[cw]
        Ac = torch.stack([wc * wc - 1.0, wc ** 3, torch.ones_like(wc), wc], dim=1)
        coef = _lstsq(Ac, uc)
        c2, c3 = coef[0], coef[1]
        u = u - curv_g(c2, c3, w, tr.curv_zlim)

    npart = u.shape[0]
    if tr.u_taper <= 0.0:
        Tu = torch.ones_like(u)
    else:
        flat = tr.u_flat()
        Tu = window_weight(u, -flat, flat, tr.u_taper)
    g = (tr.xbins, tr.u_start, tr.u_end, tr.zbins, tr.w_start, tr.w_end)
    density = deposit(u, w, Tu, *g)
    if tr.vx_model == 'affine':
        wcore = w[core]
        Av = torch.stack([torch.ones_like(wcore), u[core], wcore, wcore ** 2, wcore ** 3], dim=1)
        vx_coef = _lstsq(Av, px[core])
    else:
        vx_coef = torch.zeros(5, dtype=F64, device=x.device)
    wq = w.clamp(-tr.fit_zlim, tr.fit_zlim)
    vx_fit = vx_coef[0] + vx_coef[1] * u + vx_coef[2] * wq + vx_coef[3] * wq ** 2 + vx_coef[4] * wq ** 3
    vx_weighted = deposit(u, w, Tu * (px - vx_fit if tr.vx_model == 'affine' else px), *g)

    norm = npart * tr.delta_u * tr.delta_w
    density = density / norm
    vx_weighted = vx_weighted / norm

    du, dw, ss, ssz = tr.delta_u, tr.delta_w, tr.smoothing_sigma, tr.smoothing_sigma_z
    density_smooth, density_du, density_dw = smooth_and_differentiate(density, du, dw, ss, ssz)
    rho_vx_smooth = smooth_and_differentiate(vx_weighted, du, dw, ss, ssz, want=(True, False, False))[0]
    epsilon = (1.0 / (2.0 * np.pi)) / tr.velocity_threshold
    rho_c = density_smooth.clamp(min=0.0)
    vx_smooth = rho_vx_smooth / (rho_c + epsilon)
    vx_du = smooth_and_differentiate(vx_smooth, du, dw, ss, ssz, want=(False, True, False))[1]
    vx_du = vx_du * (rho_c / (rho_c + epsilon)) ** 3
    if tr.vx_model == 'affine':
        ug = torch.as_tensor(tr.u_grids, dtype=F64, device=x.device)[:, None]
        wg = torch.as_tensor(np.clip(tr.w_grids, -tr.fit_zlim, tr.fit_zlim), dtype=F64, device=x.device)[None, :]
        vx_smooth = vx_smooth + (vx_coef[0] + vx_coef[1] * ug + vx_coef[2] * wg
                                 + vx_coef[3] * wg ** 2 + vx_coef[4] * wg ** 3)
        vx_du = vx_du + vx_coef[1]
    fields = (density_smooth, density_du, density_dw, vx_smooth, vx_du)
    frame = dict(poly_coeffs=poly, xi_bar=xi_bar, z_bar=z_bar, sigma_xi=sigma_xi, sigma_z=sigma_z,
                 curv_c2=c2, curv_c3=c3, vx_coef=vx_coef)
    return fields, frame


def get_DF_windowed(tr, x, z, px, st):
    """
    DF_tracker_comoving._get_DF_quantile (frames 'quantile' and 'peak') with torch tensors; st is
    window_structure(tr) of the numpy deposit of the same particles (or a replayed one). Same
    returns as get_DF_rms; the frame dict also carries f_window.
    """
    if tr.poly_degree != 1:
        raise NotImplementedError('torch get_DF: poly_degree 1 only')
    dev = x.device
    npart = z.shape[0]
    lo, hi, m_lo, m_hi = st['window']
    z_bar, sigma_z = st['z_bar'], st['sigma_z']
    T = window_weight(z, lo, hi, m_lo, m_hi)
    with torch.no_grad():
        sel = T > 0.0
    Ts, zs, xs = T[sel], z[sel], x[sel]
    poly = _lstsq(torch.stack([zs, torch.ones_like(zs)], dim=1), xs, Ts)
    xi = x - (poly[0] * z + poly[1])
    sT = Ts.sum()
    xi_bar = (xi[sel] * Ts).sum() / sT
    sigma_xi = torch.sqrt(((xi[sel] - xi_bar) ** 2 * Ts).sum() / sT)
    u = (xi - xi_bar) / sigma_xi
    w = (z - z_bar) / sigma_z
    w_lo, w_hi = (lo - z_bar) / sigma_z, (hi - z_bar) / sigma_z

    c2 = c3 = torch.zeros((), dtype=F64, device=dev)
    if tr.curvature:
        with torch.no_grad():
            cw = sel & (w.abs() < tr.curv_zlim)
        wc, uc = w[cw], u[cw]
        Ac = torch.stack([wc * wc - 1.0, wc ** 3, torch.ones_like(wc), wc], dim=1)
        coef = _lstsq(Ac, uc, T[cw])
        c2, c3 = coef[0], coef[1]
        u = u - curv_g(c2, c3, w, tr.curv_zlim)

    if tr.u_taper <= 0.0:
        Tu = torch.ones_like(u)
    else:
        flat = tr.u_flat()
        Tu = window_weight(u, -flat, flat, tr.u_taper)
    g = (tr.xbins, tr.u_start, tr.u_end, tr.zbins, tr.w_start, tr.w_end)
    density = deposit(u, w, T * Tu, *g)
    wq = w.clamp(w_lo, w_hi)
    if tr.vx_model == 'affine':
        wcq = wq[sel]
        Av = torch.stack([torch.ones_like(wcq), u[sel], wcq, wcq ** 2, wcq ** 3], dim=1)
        vx_coef = _lstsq(Av, px[sel], Ts)
    else:
        vx_coef = torch.zeros(5, dtype=F64, device=dev)
    vx_fit = vx_coef[0] + vx_coef[1] * u + vx_coef[2] * wq + vx_coef[3] * wq ** 2 + vx_coef[4] * wq ** 3
    vx_weighted = deposit(u, w, T * Tu * (px - vx_fit if tr.vx_model == 'affine' else px), *g)
    norm = npart * tr.delta_u * tr.delta_w
    density = density / norm
    vx_weighted = vx_weighted / norm
    f_window = T.sum() / npart

    du, dw, sx, sz = tr.delta_u, tr.delta_w, tr.smoothing_sigma, st['sz']
    epsilon = f_window * (1.0 / (2.0 * np.pi)) / tr.velocity_threshold
    if st['adaptive'] is not None:
        levels, W = st['adaptive']
        density_smooth, density_du, density_dw = smooth_levels(density, du, dw, sx, levels, W)
        rho_vx_smooth = smooth_levels(vx_weighted, du, dw, sx, levels, W, want=(True, False, False))[0]
        vx_smooth = rho_vx_smooth / (density_smooth.clamp(min=0.0) + epsilon)
        vx_du = smooth_levels(vx_smooth, du, dw, sx, levels, W, want=(False, True, False))[1]
    else:
        density_smooth, density_du, density_dw = smooth_and_differentiate(density, du, dw, sx, sz)
        rho_vx_smooth = smooth_and_differentiate(vx_weighted, du, dw, sx, sz, want=(True, False, False))[0]
        vx_smooth = rho_vx_smooth / (density_smooth.clamp(min=0.0) + epsilon)
        vx_du = smooth_and_differentiate(vx_smooth, du, dw, sx, sz, want=(False, True, False))[1]
    rho_c = density_smooth.clamp(min=0.0)
    vx_du = vx_du * (rho_c / (rho_c + epsilon)) ** 3
    if tr.vx_model == 'affine':
        ug = torch.as_tensor(tr.u_grids, dtype=F64, device=dev)[:, None]
        wg = torch.as_tensor(np.clip(tr.w_grids, w_lo, w_hi), dtype=F64, device=dev)[None, :]
        vx_smooth = vx_smooth + (vx_coef[0] + vx_coef[1] * ug + vx_coef[2] * wg
                                 + vx_coef[3] * wg ** 2 + vx_coef[4] * wg ** 3)
        vx_du = vx_du + vx_coef[1]
    fields = (density_smooth, density_du, density_dw, vx_smooth, vx_du)
    frame = dict(poly_coeffs=poly, xi_bar=xi_bar, z_bar=torch.tensor(z_bar, dtype=F64, device=dev),
                 sigma_xi=sigma_xi, sigma_z=torch.tensor(sigma_z, dtype=F64, device=dev),
                 curv_c2=c2, curv_c3=c3, vx_coef=vx_coef, f_window=f_window)
    return fields, frame
