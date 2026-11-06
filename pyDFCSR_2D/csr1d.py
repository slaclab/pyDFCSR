"""
1D (line-charge) CSR on top of pyDFCSR's bmad-x tracking.

The wake model is the one in P. Emma's csr_calc / csr_4case_calc (MATLAB):
  * in a bend, cases A and B of Saldin, Schneidmiller & Yurkov (TESLA-FEL-96-14), i.e. the
    entrance transient that grows into the steady-state line-charge wake;
  * after a bend, cases C and D in the gamma -> infinity limit of G. Stupakov.
The beam is reduced to its projected line density lambda(z), so every particle at the same z gets
the same energy kick and there is no transverse force.

Only the CSR model is new. Beam, lattice, step schedule, tracking, statistics and output are
pyDFCSR's own: CSR1D reuses CSR2D's methods for them unchanged (called as plain functions on this
object), so a 1D and a 2D run of the same YAML differ only in the CSR model. CSR2D itself is not
modified, and test_csr1d.py checks that with CSR off the two give bit-identical particles.

The numerics follow the physics of Emma's code but not its implementation, where a better choice
was cheap:
  * the singular integral int u^(-1/3) lambda'(s-u) du is product-integrated (u^(-1/3) exactly
    against a piecewise-linear lambda'), instead of a rectangle sum that skips the first cell;
  * case D's 1/(psi(u) + 2x/R) weight is integrated per cell with Gauss-Legendre after the
    substitution u = t^3 in the first cell, which removes its u^(-1/3)-like peak at small x;
  * smoothing is a centred quadratic least-squares fit (Savitzky-Golay), the same estimator as
    Emma's smooth.m but without its one-bin offset;
  * psi is found by vectorised bisection of a monotone function instead of fzero per bin.

YAML: the same file as for CSR2D, plus an optional block

    CSR_1D:
      nbins: 300            # bins across the beam's z extent
      nsmooth: 2            # Savitzky-Golay half-window in bins; <= 1 disables smoothing
      kick_interp: spline   # 'spline' (not-a-knot cubic, as MATLAB interp1 'spline') or 'linear'
      drift_cutoff: 0.0     # stop computing CSR this many |R*phi| past a bend exit; 0 = never
      record: false         # keep every kick's line density, geometry and wake in self.kicks

`particle_deposition` and `CSR_integration` are accepted and ignored (they configure 2D only).
"""
import os
import time

import h5py
import numpy as np
from scipy.interpolate import CubicSpline

from .beams import Beam
from .CSR import CSR2D
from .lattice import Lattice
from .params import CSR_params
from .physical_constants import MC2, e as E_CHARGE
from .tools import full_path, isotime
from .waist import sigma_from_coords
from .yaml_parser import parse_yaml

# eV*m^(1/3) per electron, = -Z0 e c / (2 pi 3^(1/3)); divided by R^(2/3) (cases A/B, csr_4case_calc)
AMP_BEND = -1.998214552326886e-09
# eV per electron, = -Z0 e c / pi; divided by R (cases C/D, csr_4case_calc)
AMP_DRIFT = -5.763848158950925e-09


# ---------------------------------------------------------------------------------------------
# kernel: pure functions on a uniform grid S with spacing ds
# ---------------------------------------------------------------------------------------------

def line_density(z, nbins, n_electrons, nsmooth=2):
    """
    Bin z into a line density lambda(S) [electrons/m] and its derivative.

    `nbins` bins span [min z, max z]; nsmooth + 2 empty bins are added on each side so that the
    smoothing window and the derivative see lambda fall to zero instead of a grid edge.
    Returns S (bin centres), ds, lambda, dlambda/ds.
    """
    z = np.asarray(z, dtype=np.float64)
    zmin, zmax = z.min(), z.max()
    if not zmax > zmin:
        raise ValueError('line_density: all particles at the same z')
    ds = (zmax - zmin) / nbins
    pad = max(int(nsmooth), 0) + 2
    edges = zmin + ds * np.arange(-pad, nbins + pad + 1)
    counts, _ = np.histogram(z, bins=edges)
    S = 0.5 * (edges[1:] + edges[:-1])
    lam = n_electrons * counts / (z.size * ds)
    if nsmooth > 1:
        from scipy.signal import savgol_filter
        total = lam.sum()
        lam = np.clip(savgol_filter(lam, 2 * int(nsmooth) + 1, 2, mode='constant'), 0.0, None)
        if lam.sum() > 0:
            lam *= total / lam.sum()
    dlam = np.gradient(lam, ds)
    return S, ds, lam, dlam


def _bend_weights(U, ds):
    """
    Weights w_k with  int_0^U u^(-1/3) g(u) du = sum_k w_k g(k ds)  for g linear on each cell.
    Exact for piecewise-linear g, including the integrable singularity at u = 0.
    """
    K = int(np.ceil(U / ds - 1e-12))
    w = np.zeros(K + 1)
    for k in range(K):
        a, b = k * ds, min((k + 1) * ds, U)
        if b <= a:
            continue
        m0 = 1.5 * (b ** (2 / 3) - a ** (2 / 3))        # int u^(-1/3)
        m1 = 0.6 * (b ** (5 / 3) - a ** (5 / 3))        # int u^(+2/3)
        w[k] += ((k + 1) * ds * m0 - m1) / ds
        w[k + 1] += (m1 - k * ds * m0) / ds
    return w


def wake_in_bend(S, ds, lam, dlam, R, phi):
    """
    Cases A/B, csr_4case_calc: longitudinal wake dE/d(ct) [eV/m] at angle phi into a bend of
    radius R (both taken as absolute values).
        W(s) = A R^(-2/3) [ (R phi^3/24)^(-1/3) (lam(s - R phi^3/24) - lam(s - R phi^3/6))
                            + int_0^(R phi^3/24) u^(-1/3) lam'(s - u) du ]
    """
    R, phi = abs(R), abs(phi)
    if phi <= 0:
        return np.zeros_like(lam)
    U = R * phi ** 3 / 24
    w = _bend_weights(U, ds)
    integral = np.convolve(dlam, w)[:lam.size]                   # sum_k w_k lam'(s_i - k ds)
    lam1 = np.interp(S - U, S, lam, left=0.0, right=0.0)
    lam2 = np.interp(S - R * phi ** 3 / 6, S, lam, left=0.0, right=0.0)
    return AMP_BEND / R ** (2 / 3) * (U ** (-1 / 3) * (lam1 - lam2) + integral)


def retarded_angle(u, R, x_R, phim, iters=80):
    """
    psi(u) for the post-bend case D: the root in [0, phim] of
        R psi^3 (psi + 4 x_R) / (24 (psi + x_R)) = u,
    monotone in psi, solved by vectorised bisection (as csr_cased_fun / fzero in Emma's code).
    """
    u = np.asarray(u, dtype=np.float64)
    lo, hi = np.zeros_like(u), np.full_like(u, phim)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        f = R * mid ** 3 / 24 * (mid + 4 * x_R) / (mid + x_R) - u
        lo = np.where(f < 0, mid, lo)
        hi = np.where(f < 0, hi, mid)
    return 0.5 * (lo + hi)


def _drift_weights(smax, ds, n, R, x_R, phim, ngauss=12):
    """
    Weights w_k with  int_0^smax g(u)/(psi(u) + 2 x_R) du = sum_k w_k g(k ds)  for g linear per
    cell, the weight 1/(psi + 2 x_R) integrated by Gauss-Legendre per cell. The first cell uses
    u = t^3: psi ~ (24 u/R)^(1/3) there, so at small x_R the weight peaks like u^(-1/3).
    Only cells that can reach a source inside the n-bin grid are built. All cells at once: psi is
    one vectorised bisection over every Gauss point.
    """
    K = int(min(np.ceil(smax / ds - 1e-12), n))
    w = np.zeros(K + 1)
    if K == 0:
        return w
    xg, wg = np.polynomial.legendre.leggauss(ngauss)
    k = np.arange(K)[:, None]
    a = k * ds
    b = np.minimum((k + 1) * ds, smax)
    u = a + 0.5 * (xg + 1) * (b - a)                             # (K, ngauss)
    jac = 0.5 * (b - a) * wg
    t = 0.5 * (xg + 1) * b[0, 0] ** (1 / 3)                      # first cell: u = t^3
    u[0] = t ** 3
    jac[0] = 3 * t ** 2 * 0.5 * b[0, 0] ** (1 / 3) * wg
    f = jac / (retarded_angle(u, R, x_R, phim) + 2 * x_R)
    np.add.at(w, np.arange(K), np.sum(f * ((k + 1) * ds - u) / ds, axis=1))
    np.add.at(w, np.arange(1, K + 1), np.sum(f * (u - k * ds) / ds, axis=1))
    return w


def wake_after_bend(S, ds, lam, dlam, R, phim, x_R):
    """
    Cases C/D, csr_4case_calc: wake [eV/m] a distance x = R x_R past the exit of a bend of radius
    R and total angle phim (absolute values).
        smax = R phim^3/24 (phim + 4 x_R)/(phim + x_R)
        D:  A/R [ lam(s - smax)/(phim + 2 x_R) + int_0^smax lam'(s - u)/(psi(u) + 2 x_R) du ]
        C: -A/R lam(s - R phim^2 (phim + 3 x_R)/6)/(phim + 2 x_R)
    """
    R, phim = abs(R), abs(phim)
    if x_R <= 0:
        raise ValueError('wake_after_bend needs x_R > 0; use wake_in_bend at the exit face')
    smax = R * phim ** 3 / 24 * (phim + 4 * x_R) / (phim + x_R)
    w = _drift_weights(smax, ds, lam.size, R, x_R, phim)
    integral = np.convolve(dlam, w)[:lam.size]
    lam3 = np.interp(S - smax, S, lam, left=0.0, right=0.0)
    lam4 = np.interp(S - R * phim ** 2 * (phim + 3 * x_R) / 6, S, lam, left=0.0, right=0.0)
    # the integral already carries its 1/(psi + 2 x_R) weight; only the two lambda terms take
    # the 1/(phim + 2 x_R) of the retarded points at the bend entrance
    return AMP_DRIFT / R * ((lam3 - lam4) / (phim + 2 * x_R) + integral)


def wake_1d(S, ds, lam, dlam, geom):
    """Dispatch on the geometry returned by CSR1DGeometry.at(), like csr_4case_calc on x_R."""
    if geom['case'] == 'AB':
        return wake_in_bend(S, ds, lam, dlam, geom['R'], geom['phi'])
    if geom['case'] == 'CD':
        return wake_after_bend(S, ds, lam, dlam, geom['R'], geom['phim'], geom['x_R'])
    return np.zeros_like(lam)


# ---------------------------------------------------------------------------------------------
# geometry: where the bunch is relative to the bends
# ---------------------------------------------------------------------------------------------

class CSR1DGeometry:
    """
    The (R, phi, phim, x_R) that csr_calc carries along the beamline, computed from the beam's
    arc position rather than accumulated, so it does not depend on the step schedule.

    * before the first bend: no CSR (csr_calc's `firstbend`)
    * in a bend: case A/B with phi = |angle| * (distance into the bend) / L
    * after a bend, in ANY non-bend element (drift, quad, sextupole): case C/D with
      x_R = (distance past that bend's exit) / |R|, and R, phim of that bend
    * a new bend restarts phi from 0, whatever came before it
    """

    def __init__(self, lattice_config, distance):
        keys = [k for k in lattice_config if k != 'step_size']
        self.elements = []
        start = 0.0
        for key, end in zip(keys, distance):
            cfg = lattice_config[key]
            bend = cfg['type'] == 'dipole' and float(cfg.get('angle', 0.0)) != 0.0
            self.elements.append(dict(name=key, start=start, end=float(end), bend=bend,
                                      L=float(cfg['L']),
                                      angle=abs(float(cfg['angle'])) if bend else 0.0))
            start = float(end)

    def at(self, position, element_name):
        """Geometry at arc position `position`, inside lattice element `element_name`."""
        idx = next(i for i, el in enumerate(self.elements) if el['name'] == element_name)
        el = self.elements[idx]
        if el['bend']:
            R = el['L'] / el['angle']
            phi = el['angle'] * min(max(position - el['start'], 0.0), el['L']) / el['L']
            return dict(case='AB', R=R, phi=phi, phim=el['angle'], x_R=0.0, bend=el['name'])
        prev = next((p for p in reversed(self.elements[:idx]) if p['bend']), None)
        if prev is None:
            return dict(case='none', R=np.inf, phi=0.0, phim=0.0, x_R=0.0, bend=None)
        R = prev['L'] / prev['angle']
        x = position - prev['end']
        if x <= 0:                      # at the exit face itself
            return dict(case='AB', R=R, phi=prev['angle'], phim=prev['angle'], x_R=0.0,
                        bend=prev['name'])
        return dict(case='CD', R=R, phi=0.0, phim=prev['angle'], x_R=x / R, bend=prev['name'])


# ---------------------------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------------------------

class CSR1D_params:
    def __init__(self, input_dic=None):
        self.configure_params(**(input_dic or {}))

    def configure_params(self, nbins=300, nsmooth=2, kick_interp='spline', drift_cutoff=0.0,
                         record=False):
        if kick_interp not in ('spline', 'linear'):
            raise ValueError(f"CSR_1D.kick_interp must be 'spline' or 'linear', got {kick_interp!r}")
        if int(nbins) < 10:
            raise ValueError(f'CSR_1D.nbins must be >= 10, got {nbins}')
        self.nbins = int(nbins)
        self.nsmooth = int(nsmooth)
        self.kick_interp = kick_interp
        self.drift_cutoff = float(drift_cutoff)
        self.record = bool(record)


class CSR1D:
    """
    1D CSR driver. Same YAML, beam, lattice, schedule, tracking and output files as CSR2D.

        sim = CSR1D(input_file='input/chicane_1d.yaml')
        sim.run()
    """
    ALLOWED = ['input_beam', 'input_lattice', 'particle_deposition', 'distribution_interpolation',
               'CSR_integration', 'CSR_computation', 'step_control', 'CSR_1D']

    # reused from CSR2D unchanged: they only touch self.lattice / self.beam / self.CSR_params /
    # self.statistics / self.prefix / self.parallel, which this class provides
    get_bmadx_element = CSR2D.get_bmadx_element
    _ends_element = CSR2D._ends_element
    init_statistics = CSR2D.init_statistics
    update_statistics = CSR2D.update_statistics
    write_statistics = CSR2D.write_statistics
    _write_schedule = CSR2D._write_schedule
    dump_beam = CSR2D.dump_beam

    def __init__(self, input_file=None):
        self.timestamp = isotime()
        self.parallel = False
        self.input_file = input_file
        self.parse_input(input_file)
        self.prefix = f'{self.CSR_params.write_name}-{self.timestamp}'
        self.init_statistics()
        self.kicks = []

    def parse_input(self, input_file):
        cfg = parse_yaml(input_file)
        for k in cfg:
            assert k in self.ALLOWED, f'Incorrect param given to CSR1D: {k}\nAllowed params: {self.ALLOWED}'
        for k in ('input_beam', 'input_lattice'):
            assert k in cfg, f'Required input parameter {k} to CSR1D was not found.'
        self.input = cfg
        self.beam = Beam(cfg['input_beam'])
        # exactly as CSR2D.parse_input, including the default mode 'auto' + midpoint kicks,
        # so CSR1D and CSR2D keep sharing one schedule for the same config
        sc = dict(cfg.get('step_control') or {})
        sc.setdefault('mode', 'auto')
        if sc['mode'] == 'auto':
            sc.setdefault('kick_interval', 'midpoint')
        sigma0 = None
        if sc.get('mode') == 'auto':
            b = self.beam
            sigma0 = sigma_from_coords(b.x, b.px, b.particle.y, b.particle.py, b.z, b.pz)
        self.lattice = Lattice(cfg['input_lattice'], step_control=sc, sigma0=sigma0)
        self.CSR_params = CSR_params(cfg['CSR_computation']) if 'CSR_computation' in cfg else CSR_params()
        self.params = CSR1D_params(cfg.get('CSR_1D'))
        self.geometry = CSR1DGeometry(self.lattice.lattice_config, self.lattice.distance)
        self.n_electrons = abs(self.beam.charge) / E_CHARGE

    # ---- one CSR evaluation ----
    def calculate_1D_CSR(self, element_name):
        p = self.params
        geom = self.geometry.at(self.beam.position, element_name)
        S, ds, lam, dlam = line_density(self.beam.z, p.nbins, self.n_electrons, p.nsmooth)
        W = wake_1d(S, ds, lam, dlam, geom)
        if geom['case'] != 'none' and not getattr(self, '_warned_ds', False):
            gam = self.beam.init_energy / MC2
            if gam ** 3 * ds / abs(geom['R']) < 10:
                self._warned_ds = True
                print(f'  [CSR1D] WARNING: bin size {ds:.3e} m violates ds >> R/gamma^3 = '
                      f'{abs(geom["R"]) / gam ** 3:.3e} m (csr_4case_calc Eq. 11)')
        if not getattr(self, '_warned_bins', False) and self.beam.sigma_z / ds < 2:
            self._warned_bins = True
            print(f'  [CSR1D] WARNING: only {self.beam.sigma_z / ds:.2f} bins per sigma_z '
                  f'at s = {self.beam.position:.4f} m; raise CSR_1D.nbins')
        self.wake_S, self.wake_ds, self.wake_lam, self.wake_W, self.wake_geom = S, ds, lam, W, geom
        return S, W, geom

    def apply_wake_1d(self, S, W, L_kick):
        """delta += L_kick W(z) / E0, interpolated to each particle's z. pz only."""
        dE_E = L_kick * W / self.beam.init_energy
        if self.params.kick_interp == 'spline':
            k = CubicSpline(S, dE_E)(self.beam.z)
        else:
            k = np.interp(self.beam.z, S, dE_E)
        from bmadx import Particle
        q = self.beam.particle
        self.beam.particle = Particle(q.x, q.px, q.y, q.py, q.z, q.pz + k, q.s, q.p0c, q.mc2)
        # no beam.update_status(): it caches only x/z moments (sigma_x, sigma_z, slope, means), which
        # a pz-only kick leaves exactly unchanged, and it costs a 1e6-particle polyfit per call
        return k

    def write_wakes(self):
        """Buffer this kick's wake; flush_wakes() writes them all at the end of run()."""
        if not hasattr(self, '_wake_buffer'):
            self._wake_buffer = []
        attrs = dict(step=self.beam.step, position=self.beam.position,
                     element=self.lattice.current_element, beam_energy=self.beam.init_energy,
                     charge=self.beam.charge, **{k: self.wake_geom[k] for k in ('case', 'R', 'phi', 'phim', 'x_R')})
        self._wake_buffer.append((attrs, self.wake_S, self.wake_W, self.wake_lam))

    def flush_wakes(self):
        """
        Write every buffered kick to <prefix>-wakes1d.h5 in ONE open. Reopening the file in append
        mode per kick (thousands of times in a chicane) failed on a cloud-synced directory: the sync
        client took the file lock mid-run (BlockingIOError errno 35).
        """
        buf = getattr(self, '_wake_buffer', None)
        if not buf:
            return
        filename = os.path.join(full_path(self.CSR_params.workdir), f'{self.prefix}-wakes1d.h5')
        if os.path.isfile(filename):
            os.remove(filename)
        with h5py.File(filename, 'w') as hf:
            for attrs, S, W, lam in buf:
                g = hf.create_group(f"step_{attrs['step']}")
                for k, v in attrs.items():
                    g.attrs[k] = v
                g1 = g.create_group('longitudinal_1d')
                g1.attrs['unit'] = 'eV/m'
                g1.create_dataset('z_grid', data=S)
                g1.create_dataset('dE_dct', data=W)
                g1.create_dataset('lam', data=lam)
        print('1D wakes written to ', filename)
        self._wake_buffer = []

    # ---- the run loop: tracking mirrors CSR2D.run step for step ----
    def run(self, stop_time=None):
        if getattr(self, '_has_run', False):
            raise RuntimeError('run() has already been called on this CSR1D; construct a new one.')
        self._has_run = True
        print('Starting the 1D CSR run')
        print(self.lattice.schedule.summary())   # CSR2D.report_schedule is about 2D history memory

        sched = self.lattice.schedule
        step_count = 1
        self._s_last_kick = self.beam.position
        ele_count = 0
        skip_ele = False
        cutoff = self.params.drift_cutoff
        self._kicks_skipped = 0

        for ele in list(self.lattice.lattice_config.keys())[1:]:
            self.lattice.update(ele)
            L = self.lattice.lattice_config[ele]['L']
            steps = self.lattice.steps_per_element[ele_count]

            if (not skip_ele) and ele_count > 0:
                DL_1 = self.lattice.distance[ele_count - 1] - self.beam.position
                if DL_1 > 1.0e-6:
                    element = self.get_bmadx_element(ele=ele_prev, DL=DL_1, exit=True)
                    self.beam.track(element, DL_1, update_step=False)
            if steps == 0:
                skip_ele = True
                element = self.get_bmadx_element(ele=ele, DL=L, exit=True, entrance=True)
                self.beam.track(element, L, update_step=False)

            for step in range(steps):
                time0 = time.time()
                if (step == 0) and (ele_count > 0):
                    DL_2 = self.lattice._positions_record[step_count] - self.lattice.distance[ele_count - 1]
                    element = self.get_bmadx_element(
                        ele=ele, DL=DL_2, entrance=True,
                        exit=self._ends_element(self.beam.position + DL_2, ele_count))
                    self.beam.track(element, DL_2)
                    skip_ele = False
                else:
                    DL_i = sched.dl[step_count]
                    element = self.get_bmadx_element(
                        ele=ele, DL=DL_i, entrance=(step == 0),
                        exit=self._ends_element(self.beam.position + DL_i, ele_count))
                    self.beam.track(element, DL_i)

                if self.CSR_params.compute_CSR and sched.is_kick[step_count]:
                    geom = self.geometry.at(self.beam.position, ele)
                    blocked = bool(cutoff and geom['case'] == 'CD'
                                   and geom['x_R'] > cutoff * geom['phim'])
                    if blocked:
                        self._kicks_skipped += 1
                    elif geom['case'] == 'none':
                        # before the first bend there is no CSR (csr_calc's `firstbend`), but the
                        # kick interval must still start here, not at s = 0
                        self._s_last_kick = self.beam.position
                    else:
                        S, W, geom = self.calculate_1D_CSR(ele)
                        L_kick = None
                        if self.CSR_params.apply_CSR:
                            if sched.kick_interval == 'midpoint':
                                L_kick = sched.kick_hi[step_count] - sched.kick_lo[step_count]
                            else:
                                L_kick = self.beam.position - self._s_last_kick
                            self.apply_wake_1d(S, W, L_kick)
                            self._s_last_kick = self.beam.position
                        if self.params.record:
                            self.kicks.append(dict(position=self.beam.position, element=ele,
                                                   L_kick=L_kick, S=S, ds=self.wake_ds,
                                                   lam=self.wake_lam, W=W, **geom))
                        if self.CSR_params.write_wakes:
                            self.write_wakes()
                    if (self.CSR_params.write_beam == 'all' or
                            (isinstance(self.CSR_params.write_beam, list)
                             and step_count in self.CSR_params.write_beam)):
                        self.dump_beam(label=step_count)

                self.update_statistics(step=step_count)
                print('Finish step {}, s = {},  in {} seconds'.format(
                    step_count, self.beam.position, time.time() - time0))
                step_count += 1
                if stop_time and self.beam.position >= stop_time - 1e-12 * max(abs(stop_time), 1.0):
                    self.flush_wakes()
                    return

            ele_prev = ele
            ele_count += 1

        if self._kicks_skipped:
            print(f'  [CSR1D] drift_cutoff = {cutoff:g}: {self._kicks_skipped} kicks skipped')
        self.flush_wakes()
        self.dump_beam(label='end')
        self.write_statistics()
