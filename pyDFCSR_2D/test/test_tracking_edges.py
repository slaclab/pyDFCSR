"""
Tracking-only regressions found while porting the FACET-II W chicane (progress.md, FACET-II BC20).

1. Bend edge kicks. The run loop applied a bend's exit edge (E2) only in the split step that
   crosses into the next element, and skipped that piece when it was < 1e-6 m -- so whenever a
   node lay ON the boundary (every boundary in 'manual'/'auto', and in 'legacy' when L is a
   multiple of step_size) E2 was dropped. The first element's E1 was never applied at all.
2. Sextupole. bmadx.Sextupole drifts with x += h px and divides the kick by (1 + pz); for
   canonical px both are wrong. tracking.SextupoleKDK is Bmad's kick-drift-kick.

    pytest pyDFCSR_2D/test/test_tracking_edges.py -v
"""
import contextlib
import io

import numpy as np
import pytest
import yaml
from bmadx import Particle, SBend, Drift, track_element

from pyDFCSR_2D.CSR import CSR2D
from pyDFCSR_2D.tracking import SextupoleKDK, track

P0C = 1.0e9
M = 510998.95069


def _beam(n=500, seed=0):
    rng = np.random.default_rng(seed)
    return np.column_stack([rng.normal(0, 1e-4, n), rng.normal(0, 1e-5, n), rng.normal(0, 1e-4, n),
                            rng.normal(0, 1e-5, n), rng.normal(0, 1e-4, n), rng.normal(0, 1e-3, n)])


def _run(tmp_path, lattice, mode, coords):
    np.savetxt(tmp_path / 'beam.txt', coords)
    (tmp_path / 'lat.yaml').write_text(yaml.safe_dump(lattice, sort_keys=False))
    cfg = dict(input_beam=dict(style='from_file', beamfile=str(tmp_path / 'beam.txt'),
                               charge=1.0e-9, energy=P0C),
               input_lattice=dict(lattice_input_file=str(tmp_path / 'lat.yaml')),
               step_control=dict(mode=mode),
               particle_deposition=dict(method='bspline_comoving', xbins=32, zbins=32),
               CSR_computation=dict(compute_CSR=0, apply_CSR=0, write_name='t', workdir=str(tmp_path)))
    (tmp_path / 'cfg.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
    with contextlib.redirect_stdout(io.StringIO()):
        csr = CSR2D(input_file=str(tmp_path / 'cfg.yaml'))
        csr.run()
    return csr.beam.particle


def _arr(p):
    return np.array([p.x, p.px, p.y, p.py, p.z, p.pz])


@pytest.mark.parametrize('mode', ['legacy', 'manual'])
def test_bend_edges_applied(tmp_path, mode):
    """Both edges of every bend, including E1 of the first element and E2 on a node boundary."""
    lat = {'step_size': 0.05,
           'B0': dict(type='dipole', L=0.5, angle=0.05, E1=0.03, E2=0.0, nsep=1),
           'D1': dict(type='drift', L=0.5, nsep=1),
           'B1': dict(type='dipole', L=0.5, angle=0.05, E1=0.0, E2=0.04, nsep=1),
           'D2': dict(type='drift', L=0.5, nsep=1)}
    c = _beam()
    got = _run(tmp_path, lat, mode, c)
    ref = Particle(*c.T, 0.0, P0C, M)
    for el in [SBend(L=0.5, P0C=P0C, G=0.1, E1=0.03), Drift(L=0.5),
               SBend(L=0.5, P0C=P0C, G=0.1, E2=0.04), Drift(L=0.5)]:
        ref = track_element(ref, el)
    # the edge kicks are ~1e-6 here, so a dropped edge fails by 6 orders of magnitude
    assert np.abs(_arr(got) - _arr(ref)).max() < 1e-13


def test_sextupole_kdk_matches_bmad_formulas():
    """One step = kick(K2 L/2), exact drift, kick: independent of pz in the kick."""
    c = _beam(seed=1) * np.array([30, 30, 30, 30, 1, 10])[None, :]
    p = Particle(*c.T, 0.0, P0C, M)
    L, K2, xo = 0.381, -12.4, -6.4e-4
    got = _arr(track(p, SextupoleKDK(L=L, K2=K2, X_OFFSET=xo, NUM_STEPS=1)))

    x, px, y, py, z, pz = c.T.copy()
    x = x - xo
    k = K2 * L / 2
    px = px - k * (x ** 2 - y ** 2) / 2
    py = py + k * x * y
    q = track_element(Particle(x, px, y, py, z, pz, 0.0, P0C, M), Drift(L=L))
    x, px, y, py, z, pz = _arr(q)
    px = px - k * (x ** 2 - y ** 2) / 2
    py = py + k * x * y
    assert np.abs(got - np.array([x + xo, px, y, py, z, pz])).max() < 1e-18


def test_sextupole_kdk_converges_to_thick():
    """NUM_STEPS = n and n pieces of NUM_STEPS = 1 are the same integrator."""
    p = Particle(*(_beam(seed=2) * 30).T, 0.0, P0C, M)
    a = track(p, SextupoleKDK(L=0.25, K2=16.6, NUM_STEPS=5))
    b = p
    for _ in range(5):
        b = track(b, SextupoleKDK(L=0.05, K2=16.6, NUM_STEPS=1))
    assert np.abs(_arr(a) - _arr(b)).max() < 1e-18
