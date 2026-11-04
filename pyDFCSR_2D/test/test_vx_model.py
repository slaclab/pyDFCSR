"""
particle_deposition.vx_model: the affine control variate for the deposited v_x (DF_tracker_comoving).

  - default 'none' is the old path, unchanged
  - 'affine' recovers d vx/dx exactly when px = b x + g(z), on a tilted and curved ("banana") beam,
    which is the case where depositing px itself gives a z-trend in d vx/dx (chicane D3, tau = -49)
  - an unknown value is rejected
"""
import numpy as np
import pytest

from pyDFCSR_2D.deposit_smooth import DF_tracker_comoving

CFG = dict(method='bspline_comoving', xbins=200, zbins=200, xlim=5, zlim=5, smoothing_sigma=3.0,
           velocity_threhold=1000, frame_blend='orient')


def banana_beam(n=400_000, tau=-49.0, b=-0.28, seed=1, g='cubic'):
    rng = np.random.default_rng(seed)
    sz, sxi = 20e-6, 150e-6
    z = rng.normal(0, sz, n)
    w = z / sz
    x = tau * z + sxi * (0.7 * (1 - w ** 2) * 0.5 + rng.normal(0, 1, n))   # curved in (xi, z)
    gz = 1e-3 * (0.8 * w - 0.1 * w ** 3) if g == 'cubic' else 1e-3 * np.tanh(w)
    px = b * x + gz                                                        # px = b x + g(z)
    return x, z, px, b


def slope_at_particles(tr, x, z):
    from scipy.interpolate import RegularGridInterpolator
    f = RegularGridInterpolator((tr.u_grids, tr.w_grids), tr.vx_u / tr.sigma_xi, bounds_error=False, fill_value=np.nan)
    xi = x - np.polyval(tr.poly_coeffs, z)
    return f(np.c_[(xi - tr.xi_bar) / tr.sigma_xi, (z - tr.z_bar) / tr.sigma_z])


def test_default_is_none_and_unchanged():
    x, z, px, _ = banana_beam(100_000)
    a = DF_tracker_comoving(CFG); a.get_DF(x, z, px, 0.0)
    b = DF_tracker_comoving({**CFG, 'vx_model': 'none'}); b.get_DF(x, z, px, 0.0)
    assert a.vx_model == 'none'
    for k in ('density', 'density_u', 'density_w', 'vx', 'vx_u'):
        assert np.array_equal(getattr(a, k), getattr(b, k))


def _slope_err(g, model):
    x, z, px, bslope = banana_beam(g=g)
    tr = DF_tracker_comoving({**CFG, 'vx_model': model}); tr.get_DF(x, z, px, 0.0)
    u = (x - np.polyval(tr.poly_coeffs, z) - tr.xi_bar) / tr.sigma_xi
    w = (z - tr.z_bar) / tr.sigma_z; core = (np.abs(w) < 2) & (np.abs(u) < 2)
    return np.nanmax(np.abs(slope_at_particles(tr, x[core], z[core]) - bslope))


def test_affine_exact_when_px_is_in_the_model():
    # g(z) cubic: the deposited residual is identically zero, so d vx/dx = b to round-off,
    # while depositing px itself is off by O(1) /m in the core of this beam
    assert _slope_err('cubic', 'none') > 1.0
    assert _slope_err('cubic', 'affine') < 1e-6


def test_affine_reduces_error_when_px_is_outside_the_model():
    # g(z) = tanh: the cubic misses part of g, the residual is deposited and smoothed as before
    assert _slope_err('tanh', 'affine') < 0.1 * _slope_err('tanh', 'none')


def test_affine_vx_matches_particles():
    x, z, px, _ = banana_beam()
    tr = DF_tracker_comoving({**CFG, 'vx_model': 'affine'}); tr.get_DF(x, z, px, 0.0)
    from scipy.interpolate import RegularGridInterpolator
    f = RegularGridInterpolator((tr.u_grids, tr.w_grids), tr.vx, bounds_error=False, fill_value=np.nan)
    xi = x - np.polyval(tr.poly_coeffs, z)
    w = (z - tr.z_bar) / tr.sigma_z; core = np.abs(w) < 2
    v = f(np.c_[(xi[core] - tr.xi_bar) / tr.sigma_xi, w[core]])
    assert np.nanmax(np.abs(v - px[core])) < 1e-6     # px spans ~ 1e-3; cubic g is in the model, error = bilinear lookup


def test_bad_value():
    with pytest.raises(ValueError):
        DF_tracker_comoving({**CFG, 'vx_model': 'quadratic'})
