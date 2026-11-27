"""
The differentiable run (torch_backend.diff.CSR2DDiff) and its pieces: the torch deposition against
DF_tracker_comoving.get_DF, its gradients, the torch orbit table against lattice.get_reference_traj,
the forward run against CSR2D, and end-to-end gradients against central finite differences with the
structure replayed. Skipped when torch is not installed.
"""
import contextlib
import io
import os
import sys
import tempfile

import numpy as np
import pytest
import yaml

torch = pytest.importorskip('torch')

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'golden'))
import make_golden_2d as G  # noqa: E402


def test_deposit_matches_numpy_and_gradcheck():
    from pyDFCSR_2D.deposit_smooth import DF_tracker_comoving
    from pyDFCSR_2D.torch_backend.deposit import deposit, get_DF_rms
    c = G.beam_coords()
    x, px, z = c[:, 0], c[:, 1], c[:, 4]
    tr = DF_tracker_comoving(dict(xbins=64, zbins=64, smoothing_sigma=2.0, velocity_threshold=1000,
                                  curvature=True))
    tr.get_DF(x=x, z=z, px=px, t=0.0)
    fields, frame = get_DF_rms(tr, *(torch.as_tensor(v) for v in (x, z, px)))
    for a, b in zip(fields, (tr.density, tr.density_u, tr.density_w, tr.vx, tr.vx_u)):
        assert np.abs(a.numpy() - b).max() <= 1e-12 * np.abs(b).max()
    for k in ('xi_bar', 'z_bar', 'sigma_xi', 'sigma_z', 'curv_c2', 'curv_c3'):
        assert abs(float(frame[k]) - float(getattr(tr, k))) <= 1e-12 * max(abs(float(getattr(tr, k))), 1e-30)
    # windowed frames on a spike + pedestal (window, z smoothing, adaptive weights from the numpy deposit)
    from pyDFCSR_2D.torch_backend.deposit import get_DF_windowed, window_structure
    rng = np.random.default_rng(2)
    zz = np.r_[rng.normal(0, 30e-6, 20000), rng.normal(10e-6, 2e-6, 20000)]
    xx = 1e-5 * rng.normal(size=zz.size) + 0.3 * zz
    pp = 1e-5 * rng.normal(size=zz.size) + 2 * xx
    for kw in (dict(frame='quantile'), dict(frame='peak', smoothing_res_fwhm=0.1),
               dict(frame='quantile', smoothing_adaptive=True)):
        tr = DF_tracker_comoving(dict(dict(xbins=64, zbins=400, smoothing_sigma=2.0, velocity_threshold=1000,
                                           curvature=True), **kw))
        tr.get_DF(x=xx, z=zz, px=pp, t=0.0)
        fields, _ = get_DF_windowed(tr, *(torch.as_tensor(v) for v in (xx, zz, pp)), window_structure(tr))
        for a, b in zip(fields, (tr.density, tr.density_u, tr.density_w, tr.vx, tr.vx_u)):
            assert np.abs(a.numpy() - b).max() <= 1e-12 * np.abs(b).max(), kw
    q = [torch.tensor(rng.normal(0, 1.5, 200), requires_grad=True) for _ in range(2)]
    w = torch.tensor(rng.uniform(0.5, 1.5, 200), requires_grad=True)
    assert torch.autograd.gradcheck(lambda a, b, c: deposit(a, b, c, 16, -5, 5, 12, -5, 5), (*q, w),
                                    eps=1e-6, atol=1e-7)


def test_orbit_table_matches_numpy():
    from pyDFCSR_2D import CSR2D
    from pyDFCSR_2D.torch_backend.history import Orbit
    with tempfile.TemporaryDirectory() as tmp:
        cfg = yaml.safe_load(open(G.config(tmp, None, None, None)))
        cfg['input_lattice']['lattice_input_file'] = os.path.join(HERE, '..', 'example', 'input', 'chicane_lattice.yaml')
        cfg['CSR_computation']['compute_CSR'] = 0
        f = os.path.join(tmp, 'c.yaml')
        yaml.safe_dump(cfg, open(f, 'w'))
        with contextlib.redirect_stdout(io.StringIO()):
            lat = CSR2D(input_file=f).lattice
    o = Orbit.from_lattice_params(lat, {}, 'cpu')
    ref = np.column_stack([lat.coords, lat.n_vec, lat.tau_vec])
    assert np.abs(o.table.numpy() - ref).max() < 1e-13
    assert np.array_equal(o.rho_el.numpy(), np.asarray(lat.rho))


def _lattice(tmp):
    lat = {'step_size': 0.05, 'D0': {'type': 'drift', 'L': 0.1, 'nsep': 1},
           'Q1': {'type': 'quad', 'L': 0.1, 'K1': 3.0, 'nsep': 1},
           'B1': {'type': 'dipole', 'L': 0.3, 'angle': 0.05, 'E1': 0.0, 'E2': 0.0, 'nsep': 1},
           'D1': {'type': 'drift', 'L': 0.1, 'nsep': 1}}
    yaml.safe_dump(lat, open(os.path.join(tmp, 'lat.yaml'), 'w'), sort_keys=False)


@pytest.mark.parametrize('dep, csr', [
    (None, None),
    ({'frame': 'quantile', 'zbins': 128, 'smoothing_sigma_z': 1.5},
     {'mesh_mode': 'charge', 'mesh_x_quantile': 0.01, 'mesh_z_quantile': 0.001}),
])
def test_forward_matches_csr2d(dep, csr):
    from pyDFCSR_2D import CSR2D
    from pyDFCSR_2D.torch_backend.diff import CSR2DDiff
    with tempfile.TemporaryDirectory() as tmp:
        f = G.config(tmp, dep, csr, None)
        with contextlib.redirect_stdout(io.StringIO()):
            a = CSR2D(input_file=f)
            a.run()
            b = CSR2DDiff(input_file=f)
            b.run()
    for k in ('x', 'px', 'z', 'pz'):
        assert np.abs(getattr(a.beam.particle, k) - getattr(b.tp, k).detach().numpy()).max() < 1e-13


def test_end_to_end_gradients():
    """
    d loss / d (bend angle, quad K1, initial coordinates) on a quad + bend + drift line (5000 of the
    golden particles) against central differences of the same forward map with the structure
    replayed: one combined direction in (angle, K1, coordinates), so two extra runs.
    """
    from pyDFCSR_2D.torch_backend.diff import CSR2DDiff
    c0 = torch.as_tensor(G.beam_coords().T[:, :5000].copy())
    loss = lambda tp: tp.pz.mean() * 1e6 + tp.px.std() * 1e3
    rng = np.random.default_rng(5)
    v = torch.as_tensor(rng.normal(size=c0.shape)) * c0.std(dim=1, keepdim=True)
    with tempfile.TemporaryDirectory() as tmp:
        f = G.config(tmp, None, None, None)
        _lattice(tmp)

        def run(c, ang, k1, structure=None):
            with contextlib.redirect_stdout(io.StringIO()):
                d = CSR2DDiff(input_file=f, coords=c, params={('B1', 'angle'): ang, ('Q1', 'K1'): k1},
                              structure=structure, checkpoint=True)
                d.run()
            return d
        c = c0.clone().requires_grad_(True)
        ang = torch.tensor(0.05, dtype=torch.float64, requires_grad=True)
        k1 = torch.tensor(3.0, dtype=torch.float64, requires_grad=True)
        d = run(c, ang, k1)
        loss(d.tp).backward()
        ad = float((c.grad * v).sum()) + float(ang.grad) * 0.05 + float(k1.grad) * 3.0
        h = 1e-5
        T = lambda x: torch.tensor(x, dtype=torch.float64)
        lp = float(loss(run(c0 + h * v, T(0.05 * (1 + h)), T(3.0 * (1 + h)), d.record).tp))
        lm = float(loss(run(c0 - h * v, T(0.05 * (1 - h)), T(3.0 * (1 - h)), d.record).tp))
    fd = (lp - lm) / (2 * h)
    assert abs(fd - ad) <= 1e-4 * abs(fd), (ad, fd)


def test_checkpoint_gives_the_same_gradient():
    """Checkpointed tracking steps and wake chunks: the same gradient, to round-off."""
    from pyDFCSR_2D.torch_backend.diff import CSR2DDiff
    c0 = torch.as_tensor(G.beam_coords().T[:, :3000].copy())
    out = []
    with tempfile.TemporaryDirectory() as tmp:
        f = G.config(tmp, None, {'torch_chunk': 1 << 19}, None)
        for ck in (False, True):
            c = c0.clone().requires_grad_(True)
            ang = torch.tensor(0.05, dtype=torch.float64, requires_grad=True)
            with contextlib.redirect_stdout(io.StringIO()):
                d = CSR2DDiff(input_file=f, coords=c, params={('B1', 'angle'): ang}, checkpoint=ck)
                d.run(stop_time=0.15)
            (d.tp.pz.mean() * 1e6 + d.tp.px.std() * 1e3).backward()
            out.append((float(ang.grad), c.grad.clone()))
    assert abs(out[0][0] - out[1][0]) <= 1e-12 * abs(out[0][0])
    assert torch.allclose(out[0][1], out[1][1], rtol=1e-12, atol=1e-12 * float(out[0][1].abs().max()))
