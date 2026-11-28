"""
Parameterised initial beams (torch_backend.beam_param): identities, the statistics they produce,
gradcheck of each generator, and one end-to-end gradient through CSR2DDiff with respect to
BeamTransform knobs against central differences (structure replayed). Skipped without torch.
"""
import contextlib
import io
import os
import sys
import tempfile

import numpy as np
import pytest

torch = pytest.importorskip('torch')

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'golden'))
import make_golden_2d as G  # noqa: E402
from pyDFCSR_2D.torch_backend.beam_param import BeamFromParams, BeamTransform, NeuralBeam  # noqa: E402


def _base(n=4000):
    return torch.as_tensor(G.beam_coords().T[:, :n].copy())


def test_transform_identity_and_knobs():
    c = _base()
    t = BeamTransform(c)
    b = t.beam()                                    # identity to round-off (the profile map is tabulated)
    assert bool(((b - c).abs().max(dim=1).values <= 1e-12 * c.abs().max(dim=1).values).all())
    z0, sz = float(c[4].mean()), float(c[4].std(correction=0))
    T = lambda v: torch.tensor(v, dtype=torch.float64)
    b = t.beam({'z_scale': T(0.5)})
    assert abs(float(b[4].std(correction=0)) - 0.5 * sz) < 1e-12 * sz
    b = t.beam({'chirp': T(10.0)})
    dz = c[4] - z0
    assert torch.allclose(b[5] - c[5], 10.0 * dz, rtol=0, atol=1e-15)
    # the profile map is monotonic and keeps the order of the particles
    q = torch.tensor(np.random.default_rng(1).normal(0, 0.5, t.knot_w.numel()))
    b = t.beam({'profile_knots': q})
    order = torch.argsort(c[4])
    assert bool((torch.diff(b[4][order]) >= 0).all())
    # a positive knot in the core stretches it: lower peak current, longer core
    k = torch.zeros_like(q); k[t.knot_w.numel() // 2] = 1.0
    assert float(t.beam({'profile_knots': k})[4].std()) > float(c[4].std())


def test_from_params_statistics():
    b = BeamFromParams(200000, eps_x=2e-9, beta_x=5.0, alpha_x=1.5, sigma_z=30e-6, sigma_pz=2e-3, chirp=20.0)
    x, px, y, py, z, pz = b.beam().detach().unbind(0)
    eps = float(torch.sqrt(x.var() * px.var() - ((x - x.mean()) * (px - px.mean())).mean() ** 2))
    assert abs(eps / 2e-9 - 1) < 0.01
    assert abs(float(x.var()) / eps / 5.0 - 1) < 0.01                          # beta
    assert abs(-float(((x - x.mean()) * (px - px.mean())).mean()) / eps / 1.5 - 1) < 0.02   # alpha
    assert abs(float(z.std()) / 30e-6 - 1) < 0.01
    assert abs(float(((z - z.mean()) * (pz - pz.mean())).mean() / z.var()) / 20.0 - 1) < 0.01
    s = BeamFromParams(200000, sigma_z=30e-6, profile='spline')                # starts Gaussian
    zs = s.beam().detach()[4]
    assert abs(float(zs.std()) / 30e-6 - 1) < 0.02


def test_generators_gradcheck():
    c = _base(300)
    t = BeamTransform(c)
    q0 = torch.tensor(np.random.default_rng(2).normal(0, 0.3, t.knot_w.numel()), requires_grad=True)
    f = lambda s, h, q: (t.beam({'z_scale': s, 'chirp': h, 'profile_knots': q})[[4, 5]] * 1e3).sum(0)
    assert torch.autograd.gradcheck(f, (torch.tensor(1.1, dtype=torch.float64, requires_grad=True),
                                        torch.tensor(5.0, dtype=torch.float64, requires_grad=True), q0),
                                    eps=1e-6, atol=1e-6)
    b = BeamFromParams(300, profile='spline')
    k0 = b.params['profile_knots'].detach().clone().requires_grad_(True)
    assert torch.autograd.gradcheck(lambda q: b.beam({'profile_knots': q})[4] * 1e5, (k0,), eps=1e-6, atol=1e-6)
    nb = NeuralBeam(500, scale=[1e-4, 1e-5, 1e-4, 1e-5, 2e-5, 1e-3])
    out = nb.beam()
    assert torch.allclose(out, (nb.scale * nb.xi).T)                           # identity at start
    out[5].mean().backward()
    assert nb.net[-1].bias.grad is not None and float(nb.net[-1].bias.grad.abs().sum()) > 0


def test_end_to_end_gradient_wrt_beam_knobs():
    """d loss / d (chirp, z_scale, a profile knot) of the golden run against central differences."""
    from pyDFCSR_2D.torch_backend.diff import CSR2DDiff
    loss = lambda tp: tp.pz.mean() * 1e6 + tp.px.std() * 1e3
    t = BeamTransform(_base(5000))
    k = t.knot_w.numel() // 2
    with tempfile.TemporaryDirectory() as tmp:
        f = G.config(tmp, None, None, None)

        def run(p, structure=None):
            with contextlib.redirect_stdout(io.StringIO()):
                d = CSR2DDiff(input_file=f, coords=t.beam(p), structure=structure, checkpoint=True)
                d.run(stop_time=0.3)
            return d
        d = run(None)
        loss(d.tp).backward()
        P = t.params
        v = {'chirp': 5.0, 'z_scale': 0.02, 'knot': 0.05}              # a direction in the knobs
        ad = (float(P['chirp'].grad) * v['chirp'] + float(P['z_scale'].grad) * v['z_scale']
              + float(P['profile_knots'].grad[k]) * v['knot'])
        h = 1e-5
        T = lambda x: torch.tensor(x, dtype=torch.float64)

        def at(sgn):
            q = torch.zeros(t.knot_w.numel(), dtype=torch.float64)
            q[k] = sgn * h * v['knot']
            return float(loss(run({'chirp': T(sgn * h * v['chirp']), 'z_scale': T(1 + sgn * h * v['z_scale']),
                                   'profile_knots': q}, d.record).tp))
        fd = (at(+1) - at(-1)) / (2 * h)
    assert abs(fd - ad) <= 1e-4 * abs(fd), (ad, fd)
