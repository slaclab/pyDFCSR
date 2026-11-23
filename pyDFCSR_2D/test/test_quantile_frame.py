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


# u_taper (default 0.5 since 2026-11-12, progress_cancellation.md §4e) is the one default that
# deliberately changes the deposit; with it off, the old path must still be bit-identical.
@pytest.mark.parametrize('extra_dep, extra_csr', [
    ({'u_taper': 0.0}, None),
    ({'frame': 'rms', 'smoothing_sigma_z': None, 'u_taper': 0.0}, {'mesh_mode': 'uniform'}),
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
    lo, hi, margin, margin_hi = tr.window
    assert margin_hi == margin
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
    assert q.shape[1] == 11 and len(q) > 0 and np.all((q[:, 5] > 0.98) & (q[:, 5] <= 1.0))
    assert np.all(np.isfinite(q[:, 9:]))


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


# ---------------------------------------------------------------- 7. peak frame

def test_peak_frame_long_bunch_is_the_quantile_frame():
    """With no spike, the peak window covers more than the quantile window: identical frame."""
    x, z, px = _beam('gauss')
    a = _tracker(frame='quantile', zbins=300)
    b = _tracker(frame='peak', zbins=300)
    a.get_DF(x, z, px, 0.0)
    b.get_DF(x, z, px, 0.0)
    assert np.allclose(a.window[:3], b.window[:3], rtol=0, atol=1e-15) and b.window[3] == b.window[2]
    assert abs(a.z_bar - b.z_bar) < 1e-15 and abs(a.sigma_z / b.sigma_z - 1) < 1e-12
    assert np.allclose(a.density, b.density, rtol=1e-9, atol=1e-12)


def test_peak_frame_keeps_the_spike_drops_the_pedestal():
    x, z, px = _beam('spike')                      # 1.2 um spike at +40 um, flat pedestal behind
    tr = _tracker(frame='peak', zbins=300)
    tr.get_DF(x, z, px, 0.0)
    lo, hi, m_lo, m_hi = tr.window
    T = tr.window_weight(z)
    assert np.all(T[np.abs(z - 40e-6) < 3e-6] == 1.0)              # the spike: full weight
    assert np.all(T[z < 40e-6 - 15e-6] == 0.0)                     # the pedestal far behind: dropped
    fw = 2.3548 * 1.2e-6
    assert abs((40e-6 - lo) / (3 * fw) - 1) < 0.15 and abs(m_lo / (2 * fw) - 1) < 0.15
    # the spike's 45% plus the pedestal under the window (its +-~14 um of the 200 um flat top)
    assert 0.44 < tr.f_window < 0.53
    assert tr.delta_z_phys < 0.2e-6                                # 300 bins resolve the spike


def test_peak_frame_rate_limited_and_continuous():
    """The window half-width changes by at most exp(peak_rate) per snapshot, and the centre
    follows a rigid shift continuously."""
    rng = np.random.default_rng(4)
    n = 100_000
    x, px = rng.normal(0, 20e-6, n), rng.normal(0, 1e-6, n)
    ped = rng.uniform(-150e-6, 50e-6, n // 2)
    tr = _tracker(frame='peak', zbins=300, peak_rate=0.2)
    Hs = []
    for sig in (40e-6, 10e-6, 3e-6, 1.2e-6, 1.2e-6, 1.2e-6):   # a spike sharpening fast
        z = np.concatenate([ped, rng.normal(40e-6, sig, n - n // 2)])
        tr.get_DF(x, z, px, 0.0)
        Hs.append(tr._peak_prev[1])
    r = np.array(Hs[1:]) / np.array(Hs[:-1])
    assert np.all(r >= np.exp(-0.2) - 1e-12) and np.all(r <= np.exp(0.2) + 1e-12)
    zs = np.concatenate([ped, rng.normal(40e-6, 1.2e-6, n - n // 2)])
    tr = _tracker(frame='peak', zbins=300)
    cs = []
    for d in np.linspace(0, 0.5e-6, 6):
        tr.get_DF(x, zs + d, px, 0.0)
        cs.append(tr._peak_prev[0])
    assert np.allclose(np.diff(cs), 0.1e-6, atol=0.02e-6)


def test_peak_frame_follows_one_of_two_equal_horns():
    """Two horns of equal height: the window must stay on the one it started with, not drift."""
    rng = np.random.default_rng(5)
    n = 200_000
    x, px = rng.normal(0, 20e-6, n), rng.normal(0, 1e-6, n)
    tr = _tracker(frame='peak', zbins=300)
    cs = []
    for _ in range(12):                                     # a fresh sample every snapshot
        k = n // 4
        z = np.concatenate([rng.uniform(-80e-6, 80e-6, n - 2 * k),
                            rng.normal(-75e-6, 1.5e-6, k), rng.normal(75e-6, 1.5e-6, k)])
        tr.get_DF(x, z, px, 0.0)
        cs.append(tr._peak_prev[0])
    cs = np.array(cs)
    assert np.all(np.abs(np.abs(cs) - 75e-6) < 1.0e-6) and np.all(np.sign(cs) == np.sign(cs[0]))


# ---------------------------------------------------------------- 8. physical z-smoothing cap, clipped x mesh

@pytest.mark.parametrize('frame', ['quantile', 'peak'])
def test_smoothing_res_fwhm_caps_the_spike_only(frame):
    """smoothing_res_fwhm caps the z smoothing in physical length on a narrow spike, and leaves a
    long Gaussian bunch (whose FWHM is large) at the configured width."""
    x, z, px = _beam('spike')
    tr = _tracker(frame=frame, smoothing_sigma_z=6.5, smoothing_res_fwhm=0.1)
    tr.get_DF(x, z, px, 0.0)
    fw = tr._peak_fw if frame == 'peak' else peak_fwhm(z[tr.window_weight(z) > 0])
    floor = tr.delta_z_phys * np.sqrt(tr.smoothing_min_bins ** 2 + 2.0 / 3.0)   # the smoothing_min_bins floor (1 bin)
    assert tr.sz_used < 6.5 and tr.res_z <= max(0.1 * fw, floor) * (1 + 1e-9)
    xg = np.random.default_rng(3).normal(0, 20e-6, 200000)
    zg = np.random.default_rng(4).normal(0, 50e-6, 200000)
    tg = _tracker(frame=frame, smoothing_sigma_z=6.5, smoothing_res_fwhm=0.1)
    tg.get_DF(xg, zg, np.zeros_like(zg), 0.0)
    assert tg.sz_used == 6.5
    # no cap: unchanged fields
    t0, t1 = _tracker(frame=frame, smoothing_sigma_z=2.0), _tracker(frame=frame, smoothing_sigma_z=2.0,
                                                                     smoothing_res_fwhm=None)
    t0.get_DF(x, z, px, 0.0)
    t1.get_DF(x, z, px, 0.0)
    assert np.array_equal(t0.density, t1.density) and t1.sz_used == 2.0


def test_velocity_mask_clamps_negative_density():
    """The FFT Gaussian can leave the smoothed density slightly negative at a sharp edge; the
    velocity floor and the d vx/dx mask clamp it at zero, so neither passes through a zero
    denominator (Indranil's loss beam, progress_facet_bc20.md 11b). Non-negative input: unchanged."""
    from pyDFCSR_2D.deposit_smooth import velocity_mask
    eps = 1.6e-4
    rho = np.array([0.5, 1e-3, 0.0, -0.5 * eps, -eps, -1.4 * eps, -8.9e-5])
    m = velocity_mask(rho, eps)
    assert np.all(np.isfinite(m)) and np.all(m >= 0) and np.all(m <= 1) and np.all(m[3:] == 0)
    pos = np.abs(rho)
    assert np.array_equal(velocity_mask(pos, eps), (pos / (pos + eps)) ** 3)
    # a sharp 100x step deposited and smoothed with a sub-bin kernel goes negative; the tracker's
    # vx and d vx/du stay bounded by the particles' px
    x = np.random.default_rng(5).normal(0, 20e-6, 400000)
    z = np.concatenate([np.random.default_rng(6).uniform(-1.0e-6, 0.0, 300000),
                        np.random.default_rng(7).uniform(0.0, 60e-6, 100000)])
    px = np.random.default_rng(8).normal(0, 5e-6, 400000)
    tr = _tracker(frame='quantile', smoothing_sigma_z=6.5, smoothing_res_fwhm=0.1, smoothing_min_bins=0.5)
    tr.velocity_threhold = 1000
    tr.get_DF(x, z, px, 0.0)
    assert tr.density.min() < 0, 'the test needs a negative overshoot to be meaningful'
    assert np.abs(tr.vx).max() < 10 * np.abs(px).max() and np.all(np.isfinite(tr.vx_u))
    assert np.abs(tr.vx_u).max() < 10 * np.abs(px).max() / tr.delta_u


def test_auto_frame_decision():
    """frame 'auto': a spike on a weak pedestal -> 'peak'; a double horn -> 'quantile' (the dropped
    horn carries half of the 1D wake); no bend -> 'peak' with zero fractions; the resolved tracker
    deposits exactly as the frame it chose; an unresolved one refuses to deposit."""
    from pyDFCSR_2D.deposit_smooth import choose_auto_frame
    geom = dict(case='AB', R=6.0, phi=0.05, phim=0.05, x_R=0.0, bend='B1')
    tr = _tracker(frame='auto')
    assert tr.frame == 'auto' and tr.frame_requested == 'auto'
    x, z, px = _beam('spike')
    with pytest.raises(RuntimeError):
        tr.get_DF(x, z, px, 0.0)
    choice, f_peak, fl, ff = choose_auto_frame(tr, z, geom)
    assert choice == 'peak' and 0.3 < f_peak < 0.7 and max(abs(fl), ff) <= tr.auto_pedestal_max
    assert tr._peak_prev is None            # the preview left the rate limiter untouched
    xh, zh, pxh = _beam('horns')
    choice_h, f_h, fl_h, ff_h = choose_auto_frame(tr, zh, geom)
    assert choice_h == 'quantile' and f_h < 0.5 and max(abs(fl_h), ff_h) > 0.2
    none = dict(case='none', R=np.inf, phi=0.0, phim=0.0, x_R=0.0, bend=None)
    assert choose_auto_frame(tr, zh, none) == ('peak', pytest.approx(f_h), 0.0, 0.0)
    # resolved 'peak' == frame 'peak', bit for bit
    tr.resolve_frame('peak', (choice, f_peak, fl, ff, 'B1'))
    tr.get_DF(x, z, px, 0.0)
    ref = _tracker(frame='peak')
    ref.get_DF(x, z, px, 0.0)
    assert tr.frame == 'peak' and np.array_equal(tr.density, ref.density) and tr.window == ref.window
    with pytest.raises(ValueError):
        _tracker(frame='auto').resolve_frame('rms')


def test_auto_frame_in_csr2d_matches_peak_run():
    """A CSR2D run with frame 'auto' on the golden beam (a Gaussian: the peak window is the whole
    bunch) chooses 'peak', prints the decision, records it in statistics.h5 and reproduces the
    frame 'peak' run bit for bit."""
    import contextlib
    import glob
    import io
    import tempfile
    import h5py
    import make_golden_2d
    from pyDFCSR_2D.CSR import CSR2D
    out = {}
    for frame in ('peak', 'auto'):
        with tempfile.TemporaryDirectory() as tmp:
            f = make_golden_2d.config(tmp, {'frame': frame, 'zbins': 128, 'smoothing_sigma_z': 1.0},
                                      {'mesh_mode': 'charge', 'write_wakes': False})
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                sim = CSR2D(input_file=f)
                sim.run()
            attrs = {}
            st = glob.glob(os.path.join(tmp, '*statistics.h5'))
            if st:
                with h5py.File(st[0], 'r') as hf:
                    attrs = dict(hf['quantile_frame'].attrs) if 'quantile_frame' in hf else {}
            out[frame] = (np.array(sim.beam.z), np.array(sim.beam.pz), sim.DF_tracker.frame, buf.getvalue(), attrs)
    assert out['auto'][2] == 'peak' and "[frame auto] 'peak'" in out['auto'][3]
    assert out['auto'][4].get('frame') == 'peak' and out['auto'][4].get('frame_requested') == 'auto'
    assert 'auto_decision' in out['auto'][4] and 'frame_decision' not in out['peak'][4]
    assert np.array_equal(out['auto'][0], out['peak'][0]) and np.array_equal(out['auto'][1], out['peak'][1])


def test_mesh_x_quantile_clips_axis_and_clamps_kick():
    """mesh_x_quantile: the charge mesh's x axis stays within the quantile span (so the core gets
    the rows), and particles beyond it get the edge row's kick instead of 0."""
    import contextlib
    import io
    import tempfile
    import make_golden_2d
    from pyDFCSR_2D.CSR import CSR2D
    runs = {}
    for q in (None, 0.01):
        with tempfile.TemporaryDirectory() as tmp:
            f = make_golden_2d.config(tmp, {'frame': 'quantile', 'zbins': 128, 'smoothing_sigma_z': 1.0},
                                      {'mesh_mode': 'charge', 'write_wakes': False, 'mesh_x_quantile': q})
            with contextlib.redirect_stdout(io.StringIO()):
                sim = CSR2D(input_file=f)
                sim.run()
        runs[q] = sim
    full, clip = runs[None], runs[0.01]
    xr_full, xr = np.asarray(full.CSR_xrange_transformed), np.asarray(clip.CSR_xrange_transformed)
    assert np.ptp(xr) < np.ptp(xr_full)
    assert np.diff(xr).min() < np.diff(xr_full).min()
    # Beam.apply_wakes with a uniform unit wake: with clamp_x every particle gets the same kick,
    # without it the particles beyond the clipped x axis get none
    b = clip.beam
    xt = b.x_transform - clip._mesh_ridge(b.z)
    outside = (xt < xr[0]) | (xt > xr[-1])
    assert outside.any()
    ones = np.ones((len(xr), len(clip.CSR_zrange)))
    dpz = {}
    for clamp in (False, True):
        pz0 = b.particle.pz.copy()
        b.apply_wakes(ones, ones, xr, clip.CSR_zrange, 1.0, 0, x_shift=clip._mesh_ridge(b.z), clamp_x=clamp)
        dpz[clamp] = b.particle.pz - pz0
    unit = 1e6 / b.init_energy
    assert np.allclose(dpz[True], unit, rtol=1e-12)
    assert np.all(dpz[False][outside] == 0.0) and np.allclose(dpz[False][~outside], unit, rtol=1e-12)


# ---------------------------------------------------------------- 9. adaptive near-region nodes

def test_near_adapt_gaussian_close_to_fixed():
    """CSR_integration.near_adapt on the golden Gaussian case: the run completes, the adaptive node
    set is at least as fine as the fixed layout, well formed (increasing, ends and observer included),
    and the wake barely changes, because the fixed layout already resolves a Gaussian source."""
    import contextlib
    import io
    import tempfile
    import yaml
    import make_golden_2d
    from pyDFCSR_2D.CSR import CSR2D
    res = {}
    for adapt, adapt_x in ((False, False), (True, False), (True, True)):
        with tempfile.TemporaryDirectory() as tmp:
            f = make_golden_2d.config(tmp, None, {'write_wakes': False})
            cfg = yaml.safe_load(open(f))
            cfg['CSR_integration'].update(near_adapt=adapt, near_adapt_x=adapt_x)
            yaml.safe_dump(cfg, open(f, 'w'), sort_keys=False)
            with contextlib.redirect_stdout(io.StringIO()):
                sim = CSR2D(input_file=f)
                sim.run()
            res[adapt, adapt_x] = sim
    s0, s1 = res[False, False], res[True, False]
    for sx in (s1, res[True, True]):
        assert np.isfinite(sx.dE_dct).all() and np.isfinite(sx.x_kick).all()
        assert np.linalg.norm(sx.dE_dct - s0.dE_dct) < 2e-2 * np.linalg.norm(s0.dE_dct)
        assert np.linalg.norm(sx.x_kick - s0.x_kick) < 5e-2 * np.linalg.norm(s0.x_kick)
    # the node sets at a central mesh point of the final kick
    t = s1.beam.position
    i = s1.CSR_zmesh.size // 2
    s, x = t + s1.CSR_zmesh[i], s1.CSR_xmesh[i]
    (_, s3), (_, s4) = s1._layout_bounds(s, x)
    s1.integration_params.near_adapt = False
    base = s1._near_region_nodes(s, s3, s4, x=x, t=t)
    s1.integration_params.near_adapt = True
    nodes = s1._near_region_nodes(s, s3, s4, x=x, t=t)
    assert nodes.size >= base.size and np.all(np.diff(nodes) > 0)
    assert nodes[0] == s3 and nodes[-1] == s4 and np.any(np.isclose(nodes, s, rtol=0, atol=1e-12))
    assert s1._last_adapt['scan'] >= base.size


def test_kicks_from_profile_refinement_terms():
    """The waist/edge refinement and exit-transient terms add kicks only where they apply, and the
    default call (both off) is unchanged."""
    from pyDFCSR_2D.schedule import kicks_from_profile
    s = np.linspace(0.0, 10.0, 2001)
    r = np.where((s > 6) & (s < 7), 1.0, 1e-4)
    k0 = kicks_from_profile(s, [6.0, 7.0], s, r, tol=0.01, h_kick_max=1.0)
    k0b = kicks_from_profile(s, [6.0, 7.0], s, r, tol=0.01, h_kick_max=1.0, h_eff_nodes=np.full(s.size, 0.05),
                             h_max=0.05, kappa_refine=None, exits=None, h_exit=None)
    assert np.array_equal(k0, k0b)
    # the step profile refines to 12.5 mm in [2, 2.5] m: a kick per 4 steps = every 5 cm there
    h_eff = np.where((s > 2.0) & (s < 2.5), 0.0125, 0.05)
    k1 = kicks_from_profile(s, [6.0, 7.0], s, r, tol=0.01, h_kick_max=1.0, h_eff_nodes=h_eff, h_max=0.05,
                            kappa_refine=4)
    sk = s[k1]
    assert 8 <= np.sum((sk > 2.0) & (sk < 2.5)) <= 12
    # downstream the density is unchanged; the running integral's phase shifts, so compare counts
    assert abs(np.sum(k1[s > 3.0]) - np.sum(k0[s > 3.0])) <= 1
    # the exit of the bend at 7 m with R*phi = 1.5 m: spacing <= 0.1 m up to 8.5 m, unchanged beyond
    k2 = kicks_from_profile(s, [6.0, 7.0], s, r, tol=0.01, h_kick_max=1.0, exits=[(7.0, 1.5)], h_exit=0.1)
    sk = s[k2]
    assert np.diff(sk[(sk >= 7.0) & (sk <= 8.5)]).max() <= 0.1 + 0.005 + 1e-9    # + one 5 mm node
    assert np.sum((sk > 7.0) & (sk <= 8.5)) >= 14


def test_degenerate_margin_factor():
    """1 away from the degeneracy, the full factor at it, monotone and continuous in between."""
    from pyDFCSR_2D.CSR import degenerate_margin_factor as f
    assert f(0.0, 3.0) == 1.0 and f(0.8, 3.0) == 1.0 and f(1.0, 3.0) == 3.0 and f(-1.0, 3.0) == 3.0
    v = [f(x, 3.0) for x in np.linspace(0.8, 0.95, 50)]
    assert np.all(np.diff(v) >= 0) and abs(v[0] - 1.0) < 1e-12 and abs(v[-1] - 3.0) < 1e-12
    assert f(1.0, 1.0) == 1.0 and f(1.0, 0.0) == 1.0
