"""
Fused evaluation of the CSR integrand's arithmetic (CSR.get_CSR_integrand), one pass per
(x', s') point in numba instead of ~40 whole-array numpy operations.

Profiling the BC20 loss beam (progress_facet_bc20.md 11g): get_CSR_integrand is 66% of a wake
point's time, half of it the numba field interpolation and half the numpy arithmetic that
follows it -- some 30 temporaries of 80k doubles per call, i.e. memory traffic, which is also
what stops the MPI ranks from scaling past ~3 on one machine. The kernel keeps every operation
in the order numpy evaluated it (no fastmath, no reassociation), so the results are meant to
be bit-identical to the numpy arithmetic it replaced; test_integrand_kernel checks that against
the numpy reference kept in the test.
"""
from numba import njit


def _csr_integrand_kernel_impl(xp, kappa, rx, ry, r, nspx, nspy, tspx, tspy,
                         rho, rho_x, rho_z, vx_ret, vx_x_ret,
                         vx, nsx, nsy, tsx, tsy, w1_only, use_taper, R1, R2, keep_inner,
                         gz, gx, vn):
    """
    Longitudinal (gz) and transverse (gx) integrands at n points; vn = v . n' (for the ibp
    extras). Scalars: the observer's vx and lattice vectors (n_s, tau_s); flags as ints.
    vs = vs_ret = 1, vs_s_ret = 0 as in the numpy path (ultra-relativistic, no longitudinal
    velocity gradient).
    """
    velocity_x = tsx + vx * nsx
    velocity_y = tsy + vx * nsy
    for i in range(xp.shape[0]):
        scale_term = 1.0 + xp[i] * kappa[i]
        velocity_ret_x = tspx[i] + vx_ret[i] * nspx[i]
        velocity_ret_y = tspy[i] + vx_ret[i] * nspy[i]
        nabla_x = rho_x[i] * nspx[i] + rho_z[i] / scale_term * tspx[i]
        nabla_y = rho_x[i] * nspy[i] + rho_z[i] / scale_term * tspy[i]
        div_velocity = vx_x_ret[i]
        part1 = velocity_x * velocity_ret_x + velocity_y * velocity_ret_y
        num1 = scale_term * ((velocity_x - part1 * velocity_ret_x) * nabla_x
                             + (velocity_y - part1 * velocity_ret_y) * nabla_y)
        num2 = -scale_term * part1 * rho[i] * div_velocity
        giz = num1 / r[i] + num2 / r[i]
        nmx = nsx - nspx[i]
        nmy = nsy - nspy[i]
        p1 = rx[i] * nmx + ry[i] * nmy
        p2 = nsx * tspx[i] + nsy * tspy[i]
        partial_density = -(velocity_ret_x * nabla_x + velocity_ret_y * nabla_y) - rho[i] * div_velocity
        W1 = scale_term * p1 / (r[i] * r[i] * r[i]) * rho[i]
        if w1_only:
            gix = W1
        else:
            W2 = scale_term * p1 / (r[i] * r[i]) * partial_density
            W3 = -scale_term * p2 / r[i] * partial_density
            gix = W1 + W2 + W3
        if use_taper:
            u = (r[i] - R1) / (R2 - R1)
            if u < 0.0:
                u = 0.0
            elif u > 1.0:
                u = 1.0
            w = u * u * u * (u * (6.0 * u - 15.0) + 10.0)
            if keep_inner:
                w = 1.0 - w
            giz = giz * w
            gix = gix * w
        gz[i] = giz
        gx[i] = gix
        vn[i] = velocity_x * nspx[i] + velocity_y * nspy[i]


csr_integrand_kernel = njit(cache=True, fastmath=False)(_csr_integrand_kernel_impl)
