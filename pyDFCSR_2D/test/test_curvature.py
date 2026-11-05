"""
particle_deposition.curvature: the normalized frame is sheared, u' = u - g(w), g = c2 (w^2 - 1) + c3 w^3 (linear
beyond fit_zlim), so a curved ("banana") beam sits on the grid instead of falling off it.
  - default off; then c2 = c3 = 0 and the fields are bit-identical to the affine frame
  - on a banana beam whose tails leave the +-5 grid, the clipped charge drops and the line density follows the
    particles out to the tails
  - the kernel returns the density where the particles are (rho peak on the curved ridge), and d rho/dz|x
    matches a finite difference of rho in z
"""
import numpy as np

from pyDFCSR_2D.deposit_smooth import DF_tracker_comoving, _curv_g_vec
from pyDFCSR_2D.interp3D import interpolate3D_comoving_fields, curv_g

CFG = dict(method='bspline_comoving', xbins=200, zbins=200, xlim=5, zlim=5, smoothing_sigma=3.0,
           velocity_threhold=1000, frame_blend='orient')


def banana(n=400_000, tau=-12.0, sxi=35e-6, sz=150e-6, bend=1.0, seed=2):
    rng = np.random.default_rng(seed)
    w = rng.normal(0, 1, n); z = w * sz
    u_true = rng.normal(0, 0.5, n) - bend * (w * w - 1.0)        # ridge u = -bend (w^2 - 1), slice width 0.5
    x = tau * z + sxi * u_true
    return x, z, -0.2 * x


def deposit(curv, **kw):
    x, z, px = banana(**kw)
    tr = DF_tracker_comoving({**CFG, 'curvature': curv}); tr.get_DF(x, z, px, 0.0); tr.append_DF()
    tr.append_DF(); tr.time_log[-1] = 1.0                       # two identical snapshots for the kernel
    tr.build_interpolant()
    return tr, x, z


def kernel(tr, xs, zs):
    return interpolate3D_comoving_fields(
        xs, zs, np.full_like(xs, 0.5),
        tr.data_density_interp, tr.data_density_u_interp, tr.data_density_w_interp, tr.data_vx_interp, tr.data_vx_u_interp,
        tr.poly_coeffs_interp, tr.xi_bar_arr, tr.sigma_xi_arr, tr.z_bar_arr, tr.sigma_z_arr,
        tr.u_start, tr.delta_u, tr.w_start, tr.delta_w, tr.min_x, tr.delta_x,
        tr.frame_blend_code, tr.var_z_arr, tr.cov_arr, tr.var_x_arr, tr.x_bar_arr,
        tr.t_arr, tr.bucket, tr.bucket_inv_h, tr.bucket_t0, tr.bucket_M, tr.is_uniform,
        tr.curv_c2_arr, tr.curv_c3_arr, float(tr.fit_zlim))


def test_default_off_is_affine():
    x, z, px = banana(100_000)
    a = DF_tracker_comoving(CFG); a.get_DF(x, z, px, 0.0)
    b = DF_tracker_comoving({**CFG, 'curvature': False}); b.get_DF(x, z, px, 0.0)
    assert a.curvature is False and a.curv_c2 == 0.0 and a.curv_c3 == 0.0
    for k in ('density', 'density_u', 'density_w', 'vx', 'vx_u'):
        assert np.array_equal(getattr(a, k), getattr(b, k))


def test_curvature_keeps_the_tails_on_the_grid():
    off, _, _ = deposit(False); on, _, _ = deposit(True)
    assert off.clipped > 3 * on.clipped and on.clipped < 5e-4
    assert on.curv_c2_arr[0] < -0.3                              # the banana u = -bend (w^2 - 1) is found
    # line density at |w| = 3 relative to the Gaussian SMOOTHED like the deposit (sigma = smoothing_sigma cells),
    # which is ~9% above the bare Gaussian there
    w = on.w_grids; v = 1.0 + (on.smoothing_sigma * on.delta_w) ** 2
    G = np.exp(-w ** 2 / (2 * v)) / np.sqrt(2 * np.pi * v)
    m = np.abs(np.abs(w) - 3.0) < 0.1
    lam_on = on.density.sum(0) * on.delta_u; lam_off = off.density.sum(0) * off.delta_u
    assert np.all(np.abs(lam_on[m] / G[m] - 1) < 0.08)          # follows the particles (counting noise ~5%)
    assert np.mean(lam_off[m] / G[m]) < 0.9                      # the affine frame loses the tails


def test_g_vec_matches_kernel_g():
    for w in (-6.0, -3.0, -1.3, 0.0, 0.7, 3.0, 4.5):
        assert abs(_curv_g_vec(0.4, -0.05, np.array([w]), 3.0)[0] - curv_g(0.4, -0.05, w, 3.0)[0]) < 1e-14


def test_kernel_density_on_the_ridge_and_rho_z():
    tr, x, z = deposit(True)
    sz, sxi = tr.sigma_z, tr.sigma_xi
    for wq in (-2.5, 0.0, 2.5):
        zq = tr.z_bar + wq * sz
        xs = np.polyval(tr.poly_coeffs, zq) + tr.xi_bar + np.linspace(-6, 6, 2401) * sxi
        rho = kernel(tr, xs, np.full_like(xs, zq))[0]
        m = np.abs((z - zq) / sz) < 0.05
        xi_true = np.median(x[m] - np.polyval(tr.poly_coeffs, z[m]) - tr.xi_bar)
        assert abs(xs[np.argmax(rho)] - np.polyval(tr.poly_coeffs, zq) - tr.xi_bar - xi_true) < 0.3 * sxi
    # d rho/dz at fixed x vs a finite difference
    xq = np.polyval(tr.poly_coeffs, tr.z_bar + 1.5 * sz) + tr.xi_bar
    h = 1e-3 * sz
    zq = np.array([tr.z_bar + 1.5 * sz]); xq = np.array([xq])
    rz = kernel(tr, xq, zq)[2][0]
    fd = (kernel(tr, xq, zq + h)[0][0] - kernel(tr, xq, zq - h)[0][0]) / (2 * h)
    assert abs(rz - fd) < 0.02 * abs(fd) + 1e-6 * abs(kernel(tr, xq, zq)[0][0]) / sz
