"""
Integration_params.band_jacobian_min: each x' band scaled by 1/|du/dx'| so it covers the same +-xlim*margin of the
beam in u whatever the angle of the line of sight. Checked here on the analytic quantity alone (no tracking):
  - default 0 = off
  - J = 1 - tau * d|r-r'|/dx' reproduces a finite difference of u_ret(x') for a straight, tilted beam
"""
import numpy as np

from pyDFCSR_2D.params import Integration_params


def test_default_off():
    assert Integration_params().band_jacobian_min == 0.0


def test_jacobian_formula_straight_beam():
    # straight line lattice: n = (1, 0), source at distance D behind, observer offset x. Beam axis at the
    # retarded time: x' = tau z' + b with z' = s' - (t - r) = s' - t + r, r = |r - r'|.
    rng = np.random.default_rng(0)
    for tau in (-90.0, -12.0, 2.0):
        D, x, sp, t, b = 0.5, 1e-2, 1.0, 1.5, 3e-5            # oblique line of sight: |tau d r/dx'| ~ 1.8 at tau -90
        def u(xp):
            r = np.hypot(x - xp, D)
            return xp - (tau * (sp - t + r) + b)
        xp = 2e-5; h = 1e-7
        fd = (u(xp + h) - u(xp - h)) / (2 * h)
        r = np.hypot(x - xp, D); drdx = -(x - xp) / r
        assert abs((1 - tau * drdx) - fd) < 1e-5 * max(1, abs(fd))
