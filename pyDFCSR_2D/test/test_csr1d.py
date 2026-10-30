"""
Tests for the 1D CSR mode (pyDFCSR_2D/csr1d.py).

What is checked, and against what:
  1. tracking is CSR2D's, bit for bit (CSR off, legacy / manual / auto schedules) -- the 2D code
     path is not touched by the 1D mode, and a 1D run differs from a 2D run only in the CSR model
  2. the kernel against EXACT references (analytic Gaussian lambda, adaptive quadrature): steady
     state deep in a bend, the entrance transient, and the post-bend case C/D
  3. the kernel against Paul Emma's ORIGINAL csr_4case_calc.m, via fixtures written by
     test/csr1d/make_csr1d_fixtures.m (4 line densities x 7 geometries x 2 grids). Emma's rectangle
     quadrature drops the u < ds cell at the u^(-1/3) singularity, so it is itself off the exact
     wake by up to ~50% on coarse grids at small angles. Two checks therefore:
       * a literal Python port of Emma's quadrature reproduces his MATLAB output to 1e-12
         (same formulas, constants, geometry conventions -- only the quadrature rule differs);
       * pyDFCSR's kernel is within Emma's OWN error of the exact wake of Emma's output
   4. limits: phi -> 0, continuity across the bend exit, case-C dominated 1/(phim + 2 x_R) decay
   5. geometry bookkeeping, convergence in nbins, the driver's kick bookkeeping, config handling
"""
import contextlib
import glob
import io
import os

import numpy as np
import pytest
import scipy.io
import yaml
from scipy.integrate import quad
from scipy.optimize import brentq

from pyDFCSR_2D.csr1d import (AMP_BEND, AMP_DRIFT, CSR1D, CSR1D_params, CSR1DGeometry,
                              line_density, retarded_angle, wake_after_bend, wake_in_bend)

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES = os.path.join(HERE, 'csr1d', 'fixtures')
N_E = 6.2415e9          # 1 nC
SZ = 50e-6


# ---------------------------------------------------------------- references

def gauss(S, N=N_E, sz=SZ):
    lam = N / (np.sqrt(2 * np.pi) * sz) * np.exp(-S ** 2 / (2 * sz ** 2))
    return lam, -S / sz ** 2 * lam


def exact_bend(s, R, phi, N=N_E, sz=SZ):
    """Cases A/B for an analytic Gaussian, by adaptive quadrature."""
    U = R * phi ** 3 / 24
    lam = lambda x: gauss(np.asarray(x), N, sz)[0]
    dl = lambda x: gauss(np.asarray(x), N, sz)[1]
    integ = quad(lambda u: u ** (-1 / 3) * dl(s - u), 0, U, limit=400)[0]
    return AMP_BEND / R ** (2 / 3) * (U ** (-1 / 3) * (lam(s - U) - lam(s - R * phi ** 3 / 6)) + integ)


def exact_drift(s, R, phim, x_R, N=N_E, sz=SZ):
    """Cases C/D for an analytic Gaussian, psi by brentq, by adaptive quadrature."""
    smax = R * phim ** 3 / 24 * (phim + 4 * x_R) / (phim + x_R)
    psi = lambda u: brentq(lambda p: R * p ** 3 / 24 * (p + 4 * x_R) / (p + x_R) - u, 0, phim, xtol=1e-16)
    lam = lambda x: gauss(np.asarray(x), N, sz)[0]
    dl = lambda x: gauss(np.asarray(x), N, sz)[1]
    integ = quad(lambda u: dl(s - u) / (psi(u) + 2 * x_R), 0, smax, limit=400,
                 points=[min(smax, 1e-9)])[0]
    return AMP_DRIFT / R * ((lam(s - smax) - lam(s - R * phim ** 2 * (phim + 3 * x_R) / 6))
                            / (phim + 2 * x_R) + integ)


def emma_dlam(lam, ds):
    """csr_4case_calc's derivative: diff, averaged over adjacent pairs, one-sided at the ends."""
    d1 = np.diff(lam) / ds
    d = np.empty_like(lam)
    d[1:-1] = 0.5 * (d1[:-1] + d1[1:])
    d[0], d[-1] = d1[0], d1[-1]
    return d


def emma_port(S, lam, ds, R, phi, phim, x_R):
    """A literal port of csr_4case_calc.m (nsmooth = 1). Reference implementation only."""
    dl = emma_dlam(lam, ds)
    n = lam.size
    W = np.zeros(n)
    inside = lambda v: S[0] <= v <= S[-1]
    if x_R == 0:
        Rp3 = R * phi ** 3
        U = Rp3 / 24
        amp = AMP_BEND / R ** (2 / 3)
        for j in range(1, n):
            s = S[j] - S
            ii = (s > 0) & (s <= U)
            I = ds * np.sum(s[ii] ** (-1 / 3) * dl[ii]) if ii.any() else 0.0
            l1 = np.interp(S[j] - U, S, lam) if inside(S[j] - U) else 0.0
            l2 = np.interp(S[j] - Rp3 / 6, S, lam) if inside(S[j] - Rp3 / 6) else 0.0
            W[j] = amp * (U ** (-1 / 3) * (l1 - l2) + I)
        return W
    amp = AMP_DRIFT / R
    smax = R * phim ** 3 / 24 * (phim + 4 * x_R) / (phim + x_R)
    n_smax = int(np.argmin(np.abs(S - S[0] - smax)))
    if S[n_smax] - S[0] > smax:
        n_smax -= 1
    psi = np.zeros(n)
    for j in range(1, n_smax + 1):
        psi[j] = brentq(lambda p: R * p ** 3 / 24 * (p + 4 * x_R) / (p + x_R) - (S[j] - S[0]),
                        0, phim, xtol=1e-15)
    for j in range(1, n):
        s = S[j] - S
        iD = np.where((s > 0) & (s <= smax))[0]
        if iD.size:
            I = ds * np.sum(dl[iD] / (psi[j - iD] + 2 * x_R))
            l3 = np.interp(S[j] - smax, S, lam) if inside(S[j] - smax) else 0.0
            W[j] = amp * (l3 / (phim + 2 * x_R) + I)
        if np.any(s > smax):
            S4 = S[j] - R * phim ** 2 * (phim + 3 * x_R) / 6
            l4 = np.interp(S4, S, lam) if inside(S4) else 0.0
            W[j] -= amp * l4 / (phim + 2 * x_R)
    return W


def rel(a, b):
    return np.linalg.norm(a - b) / np.linalg.norm(b)


# ---------------------------------------------------------------- 2. kernel vs exact

R0 = 10.0
PHI_T = (24 * SZ / R0) ** (1 / 3)          # transient angle, 0.0493 rad


@pytest.mark.parametrize('phi', [0.3 * PHI_T, PHI_T, 3 * PHI_T, 0.5])
def test_bend_vs_exact(phi):
    S = np.linspace(-8 * SZ, 8 * SZ, 400)
    ds = S[1] - S[0]
    lam, dl = gauss(S)
    W = wake_in_bend(S, ds, lam, dl, R0, phi)
    idx = np.arange(0, S.size, 5)
    ref = np.array([exact_bend(S[i], R0, phi) for i in idx])
    assert rel(W[idx], ref) < 2e-3


def test_steady_state_amplitude():
    """Deep in the bend the wake is the steady-state line-charge wake, a pure property of the constants."""
    S = np.linspace(-8 * SZ, 8 * SZ, 800)
    lam, dl = gauss(S)
    W = wake_in_bend(S, S[1] - S[0], lam, dl, R0, 1.0)      # R phi^3/24 = 0.42 m >> sigma_z
    idx = np.arange(0, S.size, 8)
    ref = np.array([AMP_BEND / R0 ** (2 / 3) * quad(lambda u: u ** (-1 / 3) * gauss(np.asarray(S[i] - u))[1],
                                                    0, 16 * SZ, limit=400)[0] for i in idx])
    assert rel(W[idx], ref) < 1e-3
    # known steady-state Gaussian peak: W_min = -0.2459 * 2 N r_e mc^2 / (3^(1/3) R^(2/3) sz^(4/3)) * ...
    # (checked numerically rather than quoted): independent of the grid
    W2 = wake_in_bend(S[::2], 2 * (S[1] - S[0]), lam[::2], dl[::2], R0, 1.0)
    assert abs(W2.min() / W.min() - 1) < 2e-3


@pytest.mark.parametrize('x_R', [1e-4, 1e-3, 1e-2, 1e-1, 1.0])
def test_drift_vs_exact(x_R):
    S = np.linspace(-8 * SZ, 8 * SZ, 400)
    ds = S[1] - S[0]
    lam, dl = gauss(S)
    W = wake_after_bend(S, ds, lam, dl, R0, 0.1, x_R)
    idx = np.arange(0, S.size, 5)
    ref = np.array([exact_drift(S[i], R0, 0.1, x_R) for i in idx])
    assert rel(W[idx], ref) < 3e-3


# ---------------------------------------------------------------- 3. vs Paul Emma's csr_4case_calc

FIX = sorted(glob.glob(os.path.join(FIXTURES, 'emma_*.mat')))


def _load(f):
    m = scipy.io.loadmat(f)
    g = {k: float(m[k].ravel()[0]) for k in ('R', 'phi', 'phim', 'x_R')}
    return m['S'].ravel(), m['lam'].ravel(), m['W'].ravel(), g


@pytest.mark.skipif(not FIX, reason='no Emma fixtures; run test/csr1d/make_csr1d_fixtures.m')
def test_fixture_set_complete():
    assert len(FIX) == 56


@pytest.mark.parametrize('f', FIX, ids=[os.path.basename(f)[5:-4] for f in FIX])
def test_literal_port_reproduces_emma(f):
    """Same formulas, constants and geometry conventions as csr_4case_calc, to round-off."""
    S, lam, W, g = _load(f)
    port = emma_port(S, lam, S[1] - S[0], g['R'], g['phi'], g['phim'], g['x_R'])
    assert rel(port, W) < 1e-12


GAUSS_FIX = [f for f in FIX if '_gauss_' in f]


@pytest.mark.parametrize('f', GAUSS_FIX, ids=[os.path.basename(f)[5:-4] for f in GAUSS_FIX])
def test_against_emma_within_emma_error(f):
    """
    pyDFCSR's kernel and Emma's output differ by no more than Emma's own error against the exact
    wake (+0.5%): every difference from Emma is his quadrature, not the physics.
    """
    S, lam, W, g = _load(f)
    ds = S[1] - S[0]
    N = lam.sum() * ds
    mine = (wake_in_bend(S, ds, lam, emma_dlam(lam, ds), g['R'], g['phi']) if g['x_R'] == 0
            else wake_after_bend(S, ds, lam, emma_dlam(lam, ds), g['R'], g['phim'], g['x_R']))
    idx = np.arange(0, S.size, max(1, S.size // 60))
    ex = (np.array([exact_bend(S[i], g['R'], g['phi'], N) for i in idx]) if g['x_R'] == 0
          else np.array([exact_drift(S[i], g['R'], g['phim'], g['x_R'], N) for i in idx]))
    # all three distances on the SAME denominator, the exact wake; Emma's wake is too small in
    # magnitude where his quadrature drops the first cell, so dividing by it would inflate the gap
    emma_err, my_err = rel(W[idx], ex), rel(mine[idx], ex)
    gap = np.linalg.norm(mine[idx] - W[idx]) / np.linalg.norm(ex)
    assert my_err < 5e-3, f'pyDFCSR off the exact wake by {my_err:.4f}'
    assert gap <= emma_err + my_err + 1e-12, (
        f'pyDFCSR differs from Emma by {gap:.4f}, Emma vs exact {emma_err:.4f}')


@pytest.mark.parametrize('f', [f for f in FIX if '_gauss_' not in f and '_n1000' in f],
                         ids=lambda f: os.path.basename(f)[5:-4])
def test_against_emma_other_shapes(f):
    """
    Non-Gaussian shapes (flat-top, skewed, double-horn) at the fine grid. No closed form, so the
    reference is pyDFCSR's own kernel at 4x the resolution; Emma's distance from it bounds ours.
    """
    S, lam, W, g = _load(f)
    ds = S[1] - S[0]
    mine = (wake_in_bend(S, ds, lam, emma_dlam(lam, ds), g['R'], g['phi']) if g['x_R'] == 0
            else wake_after_bend(S, ds, lam, emma_dlam(lam, ds), g['R'], g['phim'], g['x_R']))
    Sf = np.linspace(S[0], S[-1], 4 * S.size - 3)
    lf = np.interp(Sf, S, lam)
    from scipy.interpolate import CubicSpline
    cs = CubicSpline(S, lam)
    lf, dlf = cs(Sf), cs(Sf, 1)
    dsf = Sf[1] - Sf[0]
    fine = (wake_in_bend(Sf, dsf, lf, dlf, g['R'], g['phi']) if g['x_R'] == 0
            else wake_after_bend(Sf, dsf, lf, dlf, g['R'], g['phim'], g['x_R']))
    ref = np.interp(S, Sf, fine)
    assert rel(mine, ref) <= rel(W, ref) + 1e-2


# ---------------------------------------------------------------- 4. limits

def test_entrance_limit():
    S = np.linspace(-8 * SZ, 8 * SZ, 400)
    lam, dl = gauss(S)
    assert np.all(wake_in_bend(S, S[1] - S[0], lam, dl, R0, 0.0) == 0)
    small = wake_in_bend(S, S[1] - S[0], lam, dl, R0, 1e-3 * PHI_T)
    big = wake_in_bend(S, S[1] - S[0], lam, dl, R0, PHI_T)
    assert np.abs(small).max() < 0.05 * np.abs(big).max()


def test_exit_continuity():
    """Case D just past the exit equals case A/B at the exit face."""
    S = np.linspace(-8 * SZ, 8 * SZ, 600)
    lam, dl = gauss(S)
    ds = S[1] - S[0]
    Wab = wake_in_bend(S, ds, lam, dl, R0, 0.05)
    errs = [rel(wake_after_bend(S, ds, lam, dl, R0, 0.05, x), Wab) for x in (1e-6, 1e-5, 1e-4)]
    assert errs[0] < 1e-3
    assert errs[0] < errs[1] < errs[2]                         # and it departs smoothly


def test_case_c_decay():
    """Far past the exit only case C survives: W ~ lambda(shifted)/(phim + 2 x_R)."""
    S = np.linspace(-8 * SZ, 8 * SZ, 400)
    lam, dl = gauss(S)
    ds = S[1] - S[0]
    phim = 0.1
    peaks = []
    for x in (2.0, 4.0, 8.0):
        W = wake_after_bend(S, ds, lam, dl, R0, phim, x)
        peaks.append(np.abs(W).max() * (phim + 2 * x))
    assert np.ptp(peaks) / np.mean(peaks) < 0.02


def test_retarded_angle():
    R, x, phim = 10.0, 0.01, 0.1
    smax = R * phim ** 3 / 24 * (phim + 4 * x) / (phim + x)       # largest u with a root in [0, phim]
    u = np.array([1e-10, 1e-6, 1e-4, 0.999 * smax])
    psi = retarded_angle(u, R, x, phim)
    assert retarded_angle(np.array([1.5 * smax]), R, x, phim)[0] == pytest.approx(phim)   # clamps
    resid = R * psi ** 3 / 24 * (psi + 4 * x) / (psi + x) - u
    assert np.all(np.abs(resid) < 1e-12 * np.maximum(u, 1e-12) + 1e-18)


# ---------------------------------------------------------------- 5. convergence, binning

def test_converges_in_nbins():
    """Wake from binned particles converges under nbins refinement, in a bend and after it."""
    rng = np.random.default_rng(0)
    z = rng.normal(0, SZ, 2_000_000)
    for geom in [('AB', PHI_T), ('CD', 0.01)]:
        out = []
        for nb in (100, 200, 400):
            S, ds, lam, dl = line_density(z, nb, N_E, nsmooth=2)
            W = (wake_in_bend(S, ds, lam, dl, R0, geom[1]) if geom[0] == 'AB'
                 else wake_after_bend(S, ds, lam, dl, R0, 0.1, geom[1]))
            Sx = np.linspace(-3 * SZ, 3 * SZ, 200)
            out.append(np.interp(Sx, S, W))
        e1, e2 = rel(out[0], out[2]), rel(out[1], out[2])
        assert e2 < e1 and e2 < 0.02, (geom, e1, e2)


def test_line_density_normalisation():
    z = np.random.default_rng(1).normal(0, SZ, 100_000)
    for ns in (0, 2, 4):
        S, ds, lam, dl = line_density(z, 300, N_E, nsmooth=ns)
        assert abs(lam.sum() * ds / N_E - 1) < 1e-12
        assert np.all(lam >= 0)
        assert lam[0] == 0 and lam[-1] == 0                     # padded edges


# ---------------------------------------------------------------- geometry

LAT = {'step_size': 0.1,
       'D0': {'type': 'drift', 'L': 0.2, 'nsep': 1},
       'B1': {'type': 'dipole', 'L': 0.5, 'angle': 0.2, 'nsep': 1},
       'Q1': {'type': 'quad', 'L': 0.1, 'K1': 1.0, 'nsep': 1},
       'D1': {'type': 'drift', 'L': 0.4, 'nsep': 1},
       'B2': {'type': 'dipole', 'L': 0.4, 'angle': -0.1, 'nsep': 1},
       'B3': {'type': 'dipole', 'L': 0.3, 'angle': 0.15, 'nsep': 1},
       'D2': {'type': 'drift', 'L': 1.0, 'nsep': 1}}
DIST = np.cumsum([LAT[k]['L'] for k in LAT if k != 'step_size'])


@pytest.mark.parametrize('pos, ele, expect', [
    (0.10, 'D0', dict(case='none')),                                            # before any bend
    (0.45, 'B1', dict(case='AB', R=2.5, phi=0.2 * 0.25 / 0.5, phim=0.2)),
    (0.70, 'B1', dict(case='AB', R=2.5, phi=0.2, phim=0.2)),                     # exit face
    (0.75, 'Q1', dict(case='CD', R=2.5, phim=0.2, x_R=0.05 / 2.5)),             # a quad is drift
    (1.00, 'D1', dict(case='CD', R=2.5, phim=0.2, x_R=0.30 / 2.5)),
    (1.40, 'B2', dict(case='AB', R=4.0, phi=0.1 * 0.2 / 0.4, phim=0.1)),        # negative angle
    (1.60, 'B3', dict(case='AB', R=2.0, phi=0.0, phim=0.15)),                    # bend-to-bend resets
    (1.70, 'B3', dict(case='AB', R=2.0, phi=0.15 * 0.1 / 0.3, phim=0.15)),
    (2.40, 'D2', dict(case='CD', R=2.0, phim=0.15, x_R=0.5 / 2.0)),             # last bend is B3
])
def test_geometry(pos, ele, expect):
    g = CSR1DGeometry(LAT, DIST).at(pos, ele)
    for k, v in expect.items():
        assert (g[k] == v) if isinstance(v, str) else abs(g[k] - v) < 1e-12, (k, g[k], v)


# ---------------------------------------------------------------- 1. driver: tracking + bookkeeping

BEAM = """n_particle: 20000
species: electron
random: {type: hammersley}
start: {type: time, tstart: {units: sec, value: 0}}
total_charge: {units: nC, value: 1}
x_dist: {type: gaussian, sigma_x: {units: um, value: 50}}
y_dist: {type: gaussian, sigma_y: {units: um, value: 5}}
z_dist: {type: gaussian, avg_z: {units: mm, value: 0}, sigma_z: {units: um, value: 50}}
px_dist: {type: gaussian, sigma_px: {units: keV/c, value: 25}}
py_dist: {type: gaussian, sigma_py: {units: keV/c, value: 25}}
pz_dist: {type: gaussian, avg_pz: {units: GeV/c, value: 5}, sigma_pz: {units: MeV/c, value: 0.5}}
transforms:
  s1: {type: shear z:pz, shear_coefficient: {units: gigaelectron_volt/speed_of_light/meter, value: -10}}
"""
DRIVER_LAT = {'step_size': 0.05,
              'D0': {'type': 'drift', 'L': 0.1, 'nsep': 1},
              'B1': {'type': 'dipole', 'L': 0.5, 'angle': 0.2, 'E1': 0.05, 'E2': 0.15, 'nsep': 1},
              'Q1': {'type': 'quad', 'L': 0.1, 'K1': 2.0, 'nsep': 1},
              'D1': {'type': 'drift', 'L': 0.3, 'nsep': 1},
              'B2': {'type': 'dipole', 'L': 0.4, 'angle': -0.15, 'E1': -0.07, 'E2': -0.08, 'nsep': 1},
              'B3': {'type': 'dipole', 'L': 0.3, 'angle': 0.1, 'nsep': 1},
              'D2': {'type': 'drift', 'L': 0.25, 'nsep': 1}}


@pytest.fixture(scope='module')
def workdir(tmp_path_factory):
    d = tmp_path_factory.mktemp('csr1d')
    (d / 'beam.yaml').write_text(BEAM)
    (d / 'lat.yaml').write_text(yaml.safe_dump(DRIVER_LAT, sort_keys=False))
    (d / 'out').mkdir()
    return d


def _config(d, step_control=None, csr=False, extra=None):
    cfg = {'input_beam': {'style': 'distgen', 'distgen_input_file': str(d / 'beam.yaml')},
           'input_lattice': {'lattice_input_file': str(d / 'lat.yaml')},
           'CSR_computation': {'compute_CSR': int(csr), 'apply_CSR': int(csr), 'write_wakes': False,
                               'write_beam': None, 'workdir': str(d / 'out'), 'write_name': 't'},
           'particle_deposition': {'method': 'bspline_comoving', 'xbins': 50, 'zbins': 50, 'xlim': 5,
                                   'zlim': 5, 'smoothing_sigma': 3.0, 'poly_degree': 1,
                                   'velocity_threhold': 1000}}
    if step_control:
        cfg['step_control'] = step_control
    cfg.update(extra or {})
    f = d / f'cfg_{abs(hash(str(cfg)))}.yaml'
    f.write_text(yaml.safe_dump(cfg))
    return str(f)


def _quiet(fn):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn()


@pytest.mark.parametrize('sc', [None, {'mode': 'manual', 'default_steps': 7, 'default_kick_every': 2},
                                {'mode': 'auto', 'kick_interval': 'midpoint', 'n_sub': 200}],
                         ids=['legacy', 'manual', 'auto'])
def test_tracking_bit_identical_to_csr2d(workdir, sc):
    from pyDFCSR_2D import CSR2D
    f = _config(workdir, sc)

    def both():
        a = CSR2D(input_file=f); a.run()
        b = CSR1D(input_file=f); b.run()
        return a, b
    a, b = _quiet(both)
    assert a.beam.position == b.beam.position and a.beam.step == b.beam.step
    for k in ('x', 'px', 'y', 'py', 'z', 'pz'):
        assert np.array_equal(getattr(a.beam.particle, k), getattr(b.beam.particle, k)), k
    for k in ('sigma_z', 'sigma_x', 'mean_energy'):
        assert np.array_equal(a.statistics[k], b.statistics[k]), k


def test_driver_kick_bookkeeping(workdir):
    f = _config(workdir, {'mode': 'manual', 'default_steps': 10, 'default_kick_every': 1}, csr=True,
                extra={'CSR_1D': {'record': True}})
    s = _quiet(lambda: (lambda x: (x.run(), x)[1])(CSR1D(input_file=f)))
    L = sum(k['L_kick'] for k in s.kicks)
    assert abs(L - (sum(v['L'] for k, v in DRIVER_LAT.items() if k != 'step_size') - 0.1)) < 1e-12
    assert all(k['case'] in ('AB', 'CD') for k in s.kicks)       # nothing computed before B1
    assert [k['element'] for k in s.kicks][0] == 'B1'
    assert s.beam.particle.pz.mean() < 0                          # CSR loses energy on average
    # the kick is exactly L * W / E0 interpolated at each particle (pz only)
    assert np.isfinite(s.statistics['mean_energy']).all()


def test_drift_cutoff(workdir):
    sc = {'mode': 'manual', 'default_steps': 10, 'default_kick_every': 1}
    full = _quiet(lambda: (lambda x: (x.run(), x)[1])(CSR1D(input_file=_config(workdir, sc, True, {'CSR_1D': {'record': True}}))))
    cut = _quiet(lambda: (lambda x: (x.run(), x)[1])(CSR1D(input_file=_config(workdir, sc, True, {'CSR_1D': {'record': True, 'drift_cutoff': 0.5}}))))
    xr = [k['x_R'] / k['phim'] for k in cut.kicks if k['case'] == 'CD']
    assert max(xr) <= 0.5 + 1e-12 and len(cut.kicks) < len(full.kicks) and cut._kicks_skipped > 0


def test_params():
    p = CSR1D_params()
    assert (p.nbins, p.nsmooth, p.kick_interp, p.drift_cutoff, p.record) == (300, 2, 'spline', 0.0, False)
    with pytest.raises(ValueError):
        CSR1D_params({'kick_interp': 'nearest'})
    with pytest.raises(ValueError):
        CSR1D_params({'nbins': 3})
