"""
Integration_params.tangent_bands: what _eq424 returns where the light cone misses the beam AXIS (rad < 0).

  - tangent=False: nan (the column is dropped), the pre-2026-11-04 behaviour
  - tangent=True: identical roots where rad >= 0; where rad < 0, the extremum over x' of the retarded
    offset from the axis, i.e. the column's closest approach to the beam axis
"""
import numpy as np

from pyDFCSR_2D.CSR import CSR2D
from pyDFCSR_2D.params import Integration_params


def _cases():
    # straight lattice (n = (1, 0)), observer at (0, x), source columns at distance T = s - s' behind
    rng = np.random.default_rng(3)
    T = rng.uniform(1e-5, 5e-3, 400)
    x = rng.uniform(-2e-4, 2e-4, 400)
    tau = rng.choice([-49.0, -6.9, 3.0, 20.0], 400)
    b = rng.uniform(-1e-4, 1e-4, 400)
    # q = (r0(s) - r0(s')) + x n(s): along tau_s by T, along n by x
    qx, qy = x, T          # n_sp = (1, 0): lab x along n, lab y along the orbit
    nq, q2 = qx, qx * qx + qy * qy
    return T, nq, q2, tau, b


def test_default_unchanged_and_tangent_adds_only_missing_roots():
    T, nq, q2, tau, b = _cases()
    for sign in (-1.0, 1.0):
        x0, rad0 = CSR2D._eq424(T, nq, q2, tau, b, sign)
        x1, rad1 = CSR2D._eq424(T, nq, q2, tau, b, sign, tangent=True)
        assert np.array_equal(rad0, rad1)
        hit = rad0 >= 0
        assert np.array_equal(x0[hit & np.isfinite(x0)], x1[hit & np.isfinite(x0)])
        assert np.all(np.isnan(x0[~hit]))
        assert (np.isfinite(x1[~hit])).sum() > 0          # some missing columns now have a band


def test_double_root_is_closest_approach():
    # For rad < 0 the returned x' minimises |(retarded axis position) - x'| over x', i.e. g(x') = l(x') - (x' - b)/tau
    # (distance along z between the light cone and the axis) has its extremum there.
    T, nq, q2, tau, b = _cases()
    x1, rad = CSR2D._eq424(T, nq, q2, tau, b, 1.0, tangent=True)
    m = (rad < 0) & np.isfinite(x1)
    assert m.sum() > 10
    for k in np.where(m)[0][:50]:
        def g(xp):
            l = np.sqrt((nq[k] - xp) ** 2 + (q2[k] - nq[k] ** 2))
            return l - (T[k] - b[k] / tau[k]) - xp / tau[k]   # l - (Tb + x'/tau): zero on the axis
        h = 1e-9 * max(1.0, abs(x1[k]))
        d = (g(x1[k] + h) - g(x1[k] - h)) / (2 * h)
        assert abs(d) < 1e-5


def test_param_default_true():
    assert Integration_params().tangent_bands is True
    assert Integration_params({'tangent_bands': False}).tangent_bands is False
