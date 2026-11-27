"""
Torch port of integrand_kernel.csr_integrand_kernel and of the geometry in
WakeMixin.get_CSR_integrand: separation vector, retarded time, fields at the retarded point, and
the longitudinal (gz) and transverse (gx) integrands, for a batch of source points that may belong
to different observers.
"""
import torch

from .interp import comoving_fields


def integrand(H, orbit, obs, xp, sp, t, taper, w1_only):
    """
    gz, gx at the source points (xp, sp) (1D tensors). obs: per-point observer tensors of the same
    length (s, x, vx, X0, Y0, n_x, n_y, tau_x, tau_y). taper: None or per-point (R1, R2, keep_inner)
    tensors with keep_inner as 0/1 floats. t: the common observation time (float).
    """
    s, x, vx, X0s, Y0s, nsx, nsy, tsx, tsy = obs
    X0p, Y0p, nspx, nspy, tspx, tspy = orbit.vectors(sp)
    rx = X0s - X0p + x * nsx - xp * nspx
    ry = Y0s - Y0p + x * nsy - xp * nspy
    r = torch.sqrt(rx * rx + ry * ry)
    kappa = orbit.curvature(sp)
    t_ret = t - r
    rho, rho_x, rho_z, vx_ret, vx_x_ret = comoving_fields(H, xp, sp - t_ret, t_ret)

    velocity_x = tsx + vx * nsx
    velocity_y = tsy + vx * nsy
    scale = 1.0 + xp * kappa
    vrx = tspx + vx_ret * nspx
    vry = tspy + vx_ret * nspy
    nabla_x = rho_x * nspx + rho_z / scale * tspx
    nabla_y = rho_x * nspy + rho_z / scale * tspy
    part1 = velocity_x * vrx + velocity_y * vry
    num1 = scale * ((velocity_x - part1 * vrx) * nabla_x + (velocity_y - part1 * vry) * nabla_y)
    num2 = -scale * part1 * rho * vx_x_ret
    gz = num1 / r + num2 / r
    p1 = rx * (nsx - nspx) + ry * (nsy - nspy)
    p2 = nsx * tspx + nsy * tspy
    dens_t = -(vrx * nabla_x + vry * nabla_y) - rho * vx_x_ret
    gx = scale * p1 / (r * r * r) * rho
    if not w1_only:
        gx = gx + scale * p1 / (r * r) * dens_t - scale * p2 / r * dens_t
    if taper is not None:
        R1, R2, keep_inner = taper
        u = ((r - R1) / (R2 - R1)).clamp(0.0, 1.0)
        w = u * u * u * (u * (6.0 * u - 15.0) + 10.0)
        w = keep_inner * (1.0 - w) + (1.0 - keep_inner) * w
        gz = gz * w
        gx = gx * w
    return gz, gx
