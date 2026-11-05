"""
Integration_params.drift_reach / drift_snap_thin and DF_tracker_comoving.thin_previous.

  - defaults since 2026-11-04: drift_reach True, drift_snap_thin 0.01, far_fine_max 500
  - thin_previous drops a snapshot exactly when linear interpolation reproduces it, keeps the ring and every
    per-snapshot log in lockstep, and never drops the newest
"""
import numpy as np

from pyDFCSR_2D.deposit_smooth import DF_tracker_comoving
from pyDFCSR_2D.params import Integration_params

CFG = dict(method='bspline_comoving', xbins=64, zbins=64, xlim=5, zlim=5, smoothing_sigma=3.0,
           velocity_threhold=1000, frame_blend='orient')


def beam(t, n=40_000, sxi=50e-6, sz=20e-6, tau=-10.0, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.normal(0, sz, n); xi = rng.normal(0, sxi, n)
    return tau * z + xi, z, -0.2 * (tau * z + xi)


def push(tr, t, **kw):
    x, z, px = beam(t, **kw)
    tr.get_DF(x, z, px, t); tr.append_DF()


def test_defaults():
    p = Integration_params()
    assert p.drift_reach is True and p.drift_snap_thin == 0.01 and p.far_fine_max == 500


def test_thin_drops_linear_middle_and_keeps_lockstep():
    tr = DF_tracker_comoving(CFG)
    for i, t in enumerate((0.0, 0.1, 0.2)):
        push(tr, t, sxi=50e-6 * (1 + 0.01 * i))           # frame changes linearly, same particles
    dropped, err = tr.thin_previous(tol=0.05)
    assert dropped and err < 0.05
    assert list(tr.time_log) == [0.0, 0.2]
    assert tr._ring_head - tr._ring_tail == len(tr.DF_log) == len(tr.sigma_xi_log) == 2
    tr.build_interpolant()                                 # the lockstep assert inside must pass
    assert np.array_equal(tr.data_density_interp[-1], tr.density)


def test_thin_keeps_when_frame_jumps():
    tr = DF_tracker_comoving(CFG)
    push(tr, 0.0); push(tr, 0.1, sxi=80e-6); push(tr, 0.2)
    dropped, err = tr.thin_previous(tol=0.05)
    assert not dropped and err > 0.05
    assert len(tr.time_log) == 3


def test_thin_respects_max_gap():
    tr = DF_tracker_comoving(CFG)
    for t in (0.0, 0.1, 0.2):
        push(tr, t)
    dropped, _ = tr.thin_previous(tol=1.0, max_gap=0.15)
    assert not dropped
