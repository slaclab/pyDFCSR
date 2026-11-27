"""
csr_integrand_kernel against the whole-array numpy arithmetic it replaced (CSR.get_CSR_integrand before
the 2026-11 cleanup), on random inputs: the kernel keeps numpy's operation order, so the results must agree
to the last bit.
"""
import numpy as np
import pytest

from pyDFCSR_2D.integrand_kernel import csr_integrand_kernel


def numpy_reference(xp, kappa, rx, ry, r, nspx, nspy, tspx, tspy, rho, rho_x, rho_z, vx_ret, vx_x_ret,
                    vx, nsx, nsy, tsx, tsy, w1_only, taper):
    vs = vs_ret = 1
    vs_s_ret = 0
    scale_term = 1 + xp * kappa
    velocity_x = vs * tsx + vx * nsx
    velocity_y = vs * tsy + vx * nsy
    velocity_ret_x = vs_ret * tspx + vx_ret * nspx
    velocity_ret_y = vs_ret * tspy + vx_ret * nspy
    nabla_x = rho_x * nspx + rho_z / scale_term * tspx
    nabla_y = rho_x * nspy + rho_z / scale_term * tspy
    div_velocity = vs_s_ret + vx_x_ret
    part1 = velocity_x * velocity_ret_x + velocity_y * velocity_ret_y
    num1 = scale_term * ((velocity_x - part1 * velocity_ret_x) * nabla_x + (velocity_y - part1 * velocity_ret_y) * nabla_y)
    num2 = -scale_term * part1 * rho * div_velocity
    gz = num1 / r + num2 / r
    p1 = rx * (nsx - nspx) + ry * (nsy - nspy)
    p2 = nsx * tspx + nsy * tspy
    partial_density = -(velocity_ret_x * nabla_x + velocity_ret_y * nabla_y) - rho * div_velocity
    W1 = scale_term * p1 / (r * r * r) * rho
    W2 = scale_term * p1 / (r * r) * partial_density
    W3 = -scale_term * p2 / r * partial_density
    gx = W1 if w1_only else W1 + W2 + W3
    if taper is not None:
        R1, R2, keep_inner = taper
        u = np.clip((r - R1) / (R2 - R1), 0.0, 1.0)
        w = u * u * u * (u * (6.0 * u - 15.0) + 10.0)
        if keep_inner:
            w = 1.0 - w
        gz, gx = gz * w, gx * w
    vn = velocity_x * nspx + velocity_y * nspy
    return gz, gx, vn


@pytest.mark.parametrize('w1_only, taper', [(False, None), (True, None), (False, (1e-5, 2e-5, False)),
                                            (False, (1e-5, 2e-5, True))])
def test_kernel_matches_numpy(w1_only, taper):
    rng = np.random.default_rng(11)
    n = 4000
    a = lambda scale=1.0: rng.normal(0, scale, n)
    ang = rng.uniform(0, 2 * np.pi, n)
    args = dict(xp=a(1e-4), kappa=a(0.1), rx=a(1e-3), ry=a(1e-3), nspx=np.cos(ang), nspy=np.sin(ang),
                tspx=-np.sin(ang), tspy=np.cos(ang), rho=np.abs(a(1e8)), rho_x=a(1e12), rho_z=a(1e12),
                vx_ret=a(1e-4), vx_x_ret=a(1.0))
    args['r'] = np.sqrt(args['rx'] ** 2 + args['ry'] ** 2) + 1e-6 * np.abs(a())
    obs = dict(vx=2e-5, nsx=0.3, nsy=np.sqrt(1 - 0.09), tsx=-np.sqrt(1 - 0.09), tsy=0.3)
    gz, gx, vn = np.empty(n), np.empty(n), np.empty(n)
    R1, R2, keep = taper if taper is not None else (0.0, 1.0, False)
    csr_integrand_kernel(args['xp'], args['kappa'], args['rx'], args['ry'], args['r'], args['nspx'], args['nspy'],
                         args['tspx'], args['tspy'], args['rho'], args['rho_x'], args['rho_z'], args['vx_ret'],
                         args['vx_x_ret'], obs['vx'], obs['nsx'], obs['nsy'], obs['tsx'], obs['tsy'],
                         w1_only, taper is not None, R1, R2, keep, gz, gx, vn)
    rz, rx_, rvn = numpy_reference(**args, **obs, w1_only=w1_only, taper=taper)
    assert np.array_equal(gz, rz)
    assert np.array_equal(gx, rx_)
    assert np.array_equal(vn, rvn)
