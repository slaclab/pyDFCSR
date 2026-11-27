"""
The torch backend (torch_backend) against the numba path: the interpolation and integrand kernels
on random queries, and whole wake meshes of the golden case with and without the x' refinement.
Skipped when torch is not installed.
"""
import os
import sys

import numpy as np
import pytest

torch = pytest.importorskip('torch')

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'golden'))


def _run(extra_integ, backend):
    import make_golden_2d
    return make_golden_2d.run_case({}, {'backend': backend}, extra_integ=extra_integ)


@pytest.mark.parametrize('integ', [{}, {'near_adapt': True, 'near_adapt_x': True, 'xbins': 40}])
def test_torch_wakes_match_numba(integ):
    """Whole runs: every kick's mesh to 1e-8 of its largest value. Each mesh alone agrees to ~1e-13
    (same state, summation order differs); the kicks feed back into the beam, and the integrand's
    +- d rho/dx cancellation amplifies the round-off of the later kicks."""
    a, b = _run(integ, 'numba'), _run(integ, 'torch')
    assert int(a['nkicks']) == int(b['nkicks'])
    for i in range(int(a['nkicks'])):
        for k in ('dE', 'xk'):
            ref = a[f'{k}_{i}']
            assert np.abs(b[f'{k}_{i}'] - ref).max() <= 1e-8 * np.abs(ref).max(), (k, i)
    assert np.allclose(a['beam'], b['beam'], rtol=0, atol=1e-10)


def test_interp_and_integrand_kernels():
    """torch_backend.interp/integrand against interp3D and integrand_kernel on one golden state."""
    import contextlib
    import io
    import tempfile
    import make_golden_2d
    from pyDFCSR_2D import CSR2D
    from pyDFCSR_2D.torch_backend.history import History, Orbit
    from pyDFCSR_2D.torch_backend.interp import comoving_fields
    from pyDFCSR_2D.torch_backend.integrand import integrand
    with tempfile.TemporaryDirectory() as tmp:
        sim = CSR2D(input_file=make_golden_2d.config(tmp, None, None, None))
        with contextlib.redirect_stdout(io.StringIO()):
            sim.run()
    tr = sim.DF_tracker
    rng = np.random.default_rng(3)
    n = 20000
    t = sim.beam.position
    tq = rng.uniform(tr.t_arr[0] - 0.01, t, n)
    zq = rng.normal(tr.z_bar_arr[-1], 4 * tr.sigma_z_arr[-1], n)
    xq = rng.normal(0.0, 5 * tr.sigma_xi_arr[-1], n) + np.polyval(tr.poly_coeffs_interp[-1], zq)
    ref = sim._comoving_fields(xq, zq, tq)
    H = History.from_tracker(tr, 'cpu')
    got = comoving_fields(H, *(torch.as_tensor(v) for v in (xq, zq, tq)))
    for r, g in zip(ref, got):
        assert np.abs(g.numpy() - r).max() <= 1e-12 * np.abs(r).max()
    # integrand at one observer
    s, x = t + 0.3 * tr.sigma_z_arr[-1], 0.5 * tr.sigma_xi_arr[-1]
    sp = rng.uniform(t - 0.2, t + 1e-4, n)
    xp = rng.normal(x, 5 * tr.sigma_xi_arr[-1], n)
    gz, gx = sim.get_CSR_integrand(s=s, x=x, t=t, sp=sp, xp=xp)
    orbit = Orbit.from_lattice(sim.lattice, 'cpu')
    vx = sim._comoving_fields(np.array([x]), np.array([s - t]), np.array([t]))[3][0]
    vec = orbit.vectors(torch.tensor([s], dtype=torch.float64))
    obs = [torch.full((n,), float(v)) for v in (s, x, vx)] + [v.expand(n) for v in vec]
    tz, tx = integrand(H, orbit, obs, torch.as_tensor(xp), torch.as_tensor(sp), t, None, False)
    # per point the integrand is a difference of terms up to ~1e7 x larger than itself (the
    # +- d rho/dx cancellation at random points), so the fields' round-off (1e-15) shows up at up to
    # ~1e-8 of the integrand; a wrong formula would be O(1)
    assert np.abs(tz.numpy() - gz).max() <= 1e-6 * np.abs(gz).max()
    assert np.abs(tx.numpy() - gx).max() <= 1e-6 * np.abs(gx).max()
