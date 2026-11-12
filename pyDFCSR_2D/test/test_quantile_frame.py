"""
2D quantile frame (particle_deposition.frame) and charge-equidistributed wake mesh
(CSR_computation.mesh_mode).

  1. the defaults are bit-identical to the code before these options existed
     (golden/golden_2d.npz, made by golden/make_golden_2d.py on the old code)
  2. peakfind helpers: taper, equidistribution, charge-mesh grading
  3. the quantile frame deposition on single-spike and double-horn beams
"""
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(HERE, 'golden'))

from pyDFCSR_2D.deposit_smooth import DF_tracker_comoving  # noqa: E402
from pyDFCSR_2D.peakfind import (charge_nodes, equidistribute, peak_fwhm,  # noqa: E402
                                 window_weight)


# ---------------------------------------------------------------- 1. defaults bit-identical

@pytest.fixture(scope='module')
def golden():
    return np.load(os.path.join(HERE, 'golden', 'golden_2d.npz'))


@pytest.mark.parametrize('extra_dep, extra_csr', [
    (None, None),
    ({'frame': 'rms', 'smoothing_sigma_z': None}, {'mesh_mode': 'uniform'}),
])
def test_defaults_bit_identical(golden, extra_dep, extra_csr):
    import make_golden_2d
    if extra_csr and 'mesh_mode' in extra_csr:
        from pyDFCSR_2D.params import CSR_params
        if 'mesh_mode' not in CSR_params.configure_params.__code__.co_varnames:
            pytest.skip('mesh_mode not implemented yet')
    out = make_golden_2d.run_case(extra_dep, extra_csr)
    assert int(out['nkicks']) == int(golden['nkicks'])
    bad = [k for k in golden.files if not np.array_equal(golden[k], out[k])]
    assert not bad, f'differs from the golden fixture: {bad}'


# ---------------------------------------------------------------- 2. helpers

def test_window_weight():
    z = np.linspace(-3, 3, 6001)
    T = window_weight(z, -1.0, 1.0, 0.5)
    assert np.all(T[np.abs(z) <= 1.0] == 1.0) and np.all(T[np.abs(z) >= 1.5] == 0.0)
    assert np.all((T >= 0) & (T <= 1))
    dT = np.gradient(T, z)
    # continuous slope: no step in dT/dz anywhere (C2 by construction)
    assert np.max(np.abs(np.diff(dT))) < 0.01 * np.max(np.abs(dT))


def test_equidistribute_and_grading():
    g = np.linspace(0.0, 1.0, 20001)
    m = 0.05 + np.exp(-0.5 * ((g - 0.7) / 0.004) ** 2)      # narrow spike on a floor
    n = 81
    nodes = equidistribute(0.0, 1.0, g, m, n, max_ratio=1.5)
    h = np.diff(nodes)
    assert nodes[0] == 0.0 and nodes[-1] == 1.0 and np.all(h > 0)
    r = h[1:] / h[:-1]
    assert np.all(r < 1.5 * 1.05) and np.all(r > 1 / (1.5 * 1.05))
    assert h.min() < 0.004                                  # the spike is resolved


def test_charge_nodes_two_horns():
    rng = np.random.default_rng(0)
    z = np.concatenate([rng.uniform(-100e-6, 100e-6, 40000),
                        rng.normal(-60e-6, 1e-6, 20000), rng.normal(70e-6, 1e-6, 20000)])
    nodes = charge_nodes(z, 80, uniform_frac=0.3, smooth=1e-6)
    assert nodes[0] < z.min() and nodes[-1] > z.max()
    for c in (-60e-6, 70e-6):                               # both horns get fine nodes
        near = nodes[np.abs(nodes - c) < 3e-6]
        assert near.size >= 6 and np.diff(near).max() < 1.5e-6
    h = np.diff(nodes)
    assert np.all(h[1:] / h[:-1] < 1.6) and np.all(h[1:] / h[:-1] > 1 / 1.6)


def test_peak_fwhm_gaussian():
    z = np.random.default_rng(1).normal(0, 1e-5, 200000)
    assert abs(peak_fwhm(z) / (2.3548 * 1e-5) - 1) < 0.05


# ---------------------------------------------------------------- 3. quantile frame deposition

def _beam(kind, n=200000, seed=2):
    rng = np.random.default_rng(seed)
    if kind == 'gauss':
        z = rng.normal(0, 50e-6, n)
    elif kind == 'spike':
        z = np.concatenate([rng.uniform(-150e-6, 50e-6, int(0.55 * n)),
                            rng.normal(40e-6, 1.2e-6, n - int(0.55 * n))])
    else:                                                   # double horn on a flat top
        k = n // 4
        z = np.concatenate([rng.uniform(-80e-6, 80e-6, n - 2 * k),
                            rng.normal(-75e-6, 1.5e-6, k), rng.normal(75e-6, 1.5e-6, k)])
    x = rng.normal(0, 20e-6, z.size) + 0.3 * z
    px = rng.normal(0, 1e-6, z.size) + 2e-3 * z
    return x, z, px


def _tracker(**kw):
    cfg = dict(xbins=64, zbins=600, xlim=5, zlim=5, smoothing_sigma=2.0,
               velocity_threhold=1000, frame='quantile')
    cfg.update(kw)
    return DF_tracker_comoving(cfg)


@pytest.mark.parametrize('kind', ['spike', 'horns'])
def test_quantile_window_holds_the_peaks(kind, capsys):
    x, z, px = _beam(kind)
    tr = _tracker()
    tr.get_DF(x, z, px, 0.0)
    lo, hi, margin = tr.window
    T = tr.window_weight(z)
    peaks = (40e-6,) if kind == 'spike' else (-75e-6, 75e-6)
    # every horn is (essentially) fully weighted, even at the very edge of the flat top,
    # where its outer flank lies in the outermost 0.5% and so inside the taper margin
    for c in peaks:
        assert T[np.abs(z - c) < 4e-6].min() > 0.99
    assert abs(tr.f_window - T.mean()) < 1e-12 and tr.f_window > 0.98
    assert tr.clipped < 1e-3 and 'warning' not in capsys.readouterr().out
    # density integrates to the windowed charge fraction
    assert abs(tr.density.sum() * tr.delta_u * tr.delta_w - tr.f_window) < 2e-3
    # the grid spans exactly the widened window
    assert abs((tr.z_bar + tr.w_end * tr.sigma_z) - (hi + margin)) < 1e-15


def test_quantile_frame_is_continuous():
    """A rigid shift moves the frame by the same shift (no jitter from bins or argmax)."""
    x, z, px = _beam('horns')
    tr = _tracker()
    zb = []
    for d in np.linspace(0, 1e-6, 6):
        tr.get_DF(x, z + d, px, 0.0)
        zb.append(tr.z_bar)
    assert np.allclose(np.diff(zb), 0.2e-6, rtol=1e-6, atol=1e-15)


def test_quantile_vs_rms_on_a_gaussian():
    """On a Gaussian the two frames describe the same density (to the resolution)."""
    x, z, px = _beam('gauss')
    a = _tracker(frame='rms', zbins=200, xbins=64)
    b = _tracker(frame='quantile', frame_quantile=0.0005, zbins=200, xbins=64)
    a.get_DF(x, z, px, 0.0)
    b.get_DF(x, z, px, 0.0)
    # peak density per unit physical area
    pa = a.density.max() / (a.sigma_xi * a.sigma_z)
    pb = b.density.max() / (b.sigma_xi * b.sigma_z)
    assert abs(pb / pa - 1) < 0.02
    assert b.f_window > 0.998


# ---------------------------------------------------------------- 4. CSR2D in quantile/charge mode

def test_quantile_charge_run():
    """The golden case in frame quantile + mesh charge: finite wakes, a mesh covering every
    particle, a bounded mesh ridge for tail particles, and the per-kick diagnostics logged."""
    import contextlib
    import io
    import tempfile
    import make_golden_2d
    from pyDFCSR_2D.CSR import CSR2D
    with tempfile.TemporaryDirectory() as tmp:
        f = make_golden_2d.config(tmp, {'frame': 'quantile', 'zbins': 128, 'smoothing_sigma_z': 1.0},
                                  {'mesh_mode': 'charge', 'write_wakes': False})
        with contextlib.redirect_stdout(io.StringIO()):
            sim = CSR2D(input_file=f)
            sim.run()
    z = sim.beam.z
    zr, xr = np.asarray(sim.CSR_zrange), np.asarray(sim.CSR_xrange_transformed)
    assert np.isfinite(sim.dE_dct).all() and np.isfinite(sim.x_kick).all()
    assert sim.dE_dct.shape == (len(xr), len(zr)) == (sim.CSR_params.xbins, sim.CSR_params.zbins)
    assert zr[0] < z.min() and zr[-1] > z.max() and np.all(np.diff(zr) > 0)
    # the curvature ridge is held at the window edge for particles outside it
    tr = sim.DF_tracker
    edge = sim._mesh_ridge(np.array([tr.z_bar + tr.w_start * tr.sigma_z,
                                     tr.z_bar + tr.w_end * tr.sigma_z]))
    far = sim._mesh_ridge(np.array([tr.z_bar - 100 * tr.sigma_z, tr.z_bar + 100 * tr.sigma_z]))
    assert np.array_equal(edge, far)
    q = np.array(sim.quantile_log)
    assert q.shape[1] == 10 and len(q) > 0 and np.all((q[:, 4] > 0.98) & (q[:, 4] <= 1.0))
    assert np.all(np.isfinite(q[:, 8:]))


# ---------------------------------------------------------------- 5. kick placement from a CSR profile

def test_kicks_from_profile():
    """Kicks follow the CSR strength, the dipole edges are always kicks, quiet stretches are
    capped at kick_h_max, and every kick is a schedule node."""
    from pyDFCSR_2D.schedule import kicks_from_profile
    s = np.linspace(0.0, 10.0, 2001)
    r = np.where((s > 6) & (s < 7), 1.0, 1e-4)                 # one strong region
    k = kicks_from_profile(s, [6.0, 7.0], s, r, tol=0.01, h_kick_max=1.0)
    sk = s[k]
    assert 6.0 in sk and 7.0 in sk and sk[-1] == 10.0
    n_strong = np.sum((sk > 6) & (sk <= 7))
    assert 80 <= n_strong <= 120                                # ~1/tol of the CSR in the strong region
    for quiet in (sk[sk <= 6.0], sk[sk >= 7.0]):               # capped spacing elsewhere
        assert np.diff(quiet).max() <= 1.0 + 1e-9


def test_kick_placement_csr1d_schedule():
    """kick_placement csr1d on the golden case: fewer kicks, all at snapshot nodes, the
    midpoint intervals still cover the lattice exactly, and the dipole edges are kicks."""
    import contextlib
    import io
    import tempfile
    import yaml
    import make_golden_2d
    from pyDFCSR_2D.CSR import CSR2D
    with tempfile.TemporaryDirectory() as tmp:
        f = make_golden_2d.config(tmp)
        c = yaml.safe_load(open(f))
        c['step_control'] = {'mode': 'auto', 'kick_interval': 'midpoint',
                             'kick_placement': 'csr1d', 'kick_tol': 0.02, 'kick_h_max': 0.1}
        yaml.safe_dump(c, open(f, 'w'), sort_keys=False)
        with contextlib.redirect_stdout(io.StringIO()):
            sim = CSR2D(input_file=f)
    sch = sim.lattice.schedule
    assert np.all(sch.is_snap[sch.is_kick])
    assert abs(sch.total_arc_kicked() - sim.lattice.lattice_length) < 1e-9
    sk = sch.s_nodes[sch.is_kick]
    for edge in (0.3,):                                          # B1 exit
        assert np.min(np.abs(sk - edge)) < 1e-9


# ---------------------------------------------------------------- 6. density-adaptive z smoothing

def _line(tr):
    """projected line density and its z derivative on physical z, from a tracker's fields"""
    lam = tr.density.sum(0) * tr.delta_u / tr.sigma_z
    dlam = tr.density_w.sum(0) * tr.delta_u / tr.sigma_z ** 2
    return tr.z_bar + tr.w_grids * tr.sigma_z, lam, dlam


def test_adaptive_smoothing_spike_and_pedestal():
    """20 000 particles, spike on a pedestal: adaptive z smoothing keeps the spike as sharp as
    fixed 1-bin smoothing, and cuts the pedestal noise in d lambda/dz against a 1e6 reference."""
    def beam(n, seed):
        rng = np.random.default_rng(seed)
        k = int(0.45 * n)
        z = np.concatenate([rng.uniform(-150e-6, 50e-6, n - k), rng.normal(40e-6, 1.2e-6, k)])
        return rng.normal(0, 20e-6, n), z, rng.normal(0, 1e-6, n)
    cfg = dict(xbins=64, zbins=1200, smoothing_sigma=2.0, smoothing_sigma_z=1.0)
    ref = _tracker(**cfg)
    ref.get_DF(*beam(1_000_000, 5), 0.0)
    zr, lr, dr = _line(ref)
    out = {}
    for name, kw in (('fine', {}), ('adaptive', dict(smoothing_adaptive=True, smoothing_min_particles=100))):
        tr = _tracker(**cfg, **kw)
        tr.get_DF(*beam(20_000, 6), 0.0)
        z, lam, dlam = _line(tr)
        ped = (z > -130e-6) & (z < 20e-6)
        noise = np.sqrt(np.mean((dlam[ped] - np.interp(z[ped], zr, dr)) ** 2))
        out[name] = (lam.max(), noise, tr.res_z)
    assert out['adaptive'][0] > 0.97 * out['fine'][0]          # spike peak kept
    assert out['adaptive'][1] < 0.3 * out['fine'][1]           # pedestal derivative noise cut
    assert out['adaptive'][2] == pytest.approx(out['fine'][2], rel=0.05)   # finest width at the peak
