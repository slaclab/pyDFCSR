"""
The differentiable 1D CSR run: csr1d.CSR1D with a torch shadow of the beam, as CSR2DDiff is for 2D.

The numpy CSR1D keeps every structural decision (schedule, element pieces, the bin grid S and ds of
each kick, the geometry case, the kick lengths) on the detached values of the torch beam; the torch
side recomputes the values with gradients:

  tracking        bmadx on torch particles, element parameters as tensors (CSR2DDiff's code)
  line density    a deposit on the kick's bin grid (see `deposit` below), the Savitzky-Golay filter
                  as a fixed convolution (scipy's coefficients, zero padding as mode='constant'),
                  the clip at 0 and the renormalisation, lambda' as np.gradient
  wake            csr1d.wake_in_bend / wake_after_bend in torch: the product-integration weights of
                  the u^(-1/3) integral (exact for piecewise-linear lambda', so differentiable in
                  U = R phi^3/24), case D's Gauss-Legendre weights with psi(u) from a detached
                  bisection plus one Newton step (the implicit-function derivative), the shifted
                  lambda terms by linear interpolation
  geometry        with a bend-angle parameter: R = L/|theta|, phi, phim and x_R of that bend
  kick            pz += L_kick W(z) / E0 by LINEAR interpolation of W at the particles

Two deliberate differences from CSR1D:
  deposit  'cic' (default): linear (cloud-in-cell) weights on the bins. CSR1D's histogram has zero
           derivative with respect to the particle positions, so it cannot carry a gradient from a
           chirp or a bunch-length change into the wake. 'histogram' reproduces CSR1D exactly (for
           checks; its gradient misses that path).
  kick     linear interpolation of the wake (CSR_1D.kick_interp: linear in CSR1D); a differentiable
           not-a-knot cubic spline would need a banded solve per kick.

Segment checkpointing (segment=k, segment_gradient) and replay (structure=) work as in CSR2DDiff.
"""
import numpy as np
import torch
from bmadx import Particle

from ..csr1d import AMP_BEND, AMP_DRIFT, CSR1D
from .diff import CSR2DDiff
from .history import F64


def _interp_uniform(xq, x0, dx, f, left=0.0, right=0.0, clamp=False):
    """np.interp on the uniform grid x0 + k dx: zero (left/right) outside, or held at the ends."""
    n = f.shape[0]
    c = (xq - x0) / dx
    if clamp:
        c = c.clamp(0.0, n - 1.0)
    i = torch.floor(c).long().clamp(0, n - 2)
    t = c - i
    v = f[i] * (1 - t) + f[i + 1] * t
    if clamp:
        return v
    return torch.where(c < 0, torch.full_like(v, left), torch.where(c > n - 1, torch.full_like(v, right), v))


def _causal_conv(dlam, w):
    """np.convolve(dlam, w)[:n]: sum_k w_k dlam[i - k]."""
    n, K = dlam.shape[0], w.shape[0]
    x = torch.nn.functional.pad(dlam, (K - 1, 0))
    return torch.nn.functional.conv1d(x.view(1, 1, -1), w.flip(0).view(1, 1, -1)).view(-1)[:n]


def _bend_weights(U, ds):
    """csr1d._bend_weights in torch: differentiable in U (the cell count is structure)."""
    K = int(np.ceil(float(U.detach()) / ds - 1e-12))
    k = torch.arange(K, dtype=F64, device=U.device)
    a = k * ds
    b = torch.minimum((k + 1) * ds, U)
    m0 = 1.5 * (b ** (2 / 3) - a ** (2 / 3))
    m1 = 0.6 * (b ** (5 / 3) - a ** (5 / 3))
    w = torch.zeros(K + 1, dtype=F64, device=U.device)
    w = w.index_add(0, torch.arange(K, device=U.device), ((k + 1) * ds * m0 - m1) / ds)
    return w.index_add(0, torch.arange(1, K + 1, device=U.device), (m1 - k * ds * m0) / ds)


def _retarded_angle(u, R, x_R, phim, iters=80):
    """csr1d.retarded_angle: the bisection detached, then one Newton step for the derivative."""
    with torch.no_grad():
        lo, hi = torch.zeros_like(u), torch.full_like(u, float(phim))
        Rd, xd = float(R), float(x_R)
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            f = Rd * mid ** 3 / 24 * (mid + 4 * xd) / (mid + xd) - u
            lo = torch.where(f < 0, mid, lo)
            hi = torch.where(f < 0, hi, mid)
        p0 = 0.5 * (lo + hi)
        fp = Rd / 24 * (3 * p0 ** 2 * (p0 + 4 * xd) / (p0 + xd) + p0 ** 3 / (p0 + xd)
                        - p0 ** 3 * (p0 + 4 * xd) / (p0 + xd) ** 2)
    f = R * p0 ** 3 / 24 * (p0 + 4 * x_R) / (p0 + x_R) - u
    return p0 - f / fp


def _drift_weights(smax, ds, n, R, x_R, phim, ngauss=12):
    """csr1d._drift_weights in torch (differentiable in smax, R, x_R; the cell count is structure)."""
    dev = smax.device
    K = int(min(np.ceil(float(smax.detach()) / ds - 1e-12), n))
    if K == 0:
        return torch.zeros(1, dtype=F64, device=dev)
    xg, wg = (torch.as_tensor(v, dtype=F64, device=dev) for v in np.polynomial.legendre.leggauss(ngauss))
    k = torch.arange(K, dtype=F64, device=dev)[:, None]
    a = k * ds
    b = torch.minimum((k + 1) * ds, smax)
    u = a + 0.5 * (xg + 1) * (b - a)
    jac = 0.5 * (b - a) * wg
    b0 = b[0, 0]
    t = 0.5 * (xg + 1) * b0 ** (1 / 3)                     # first cell: u = t^3
    u = torch.cat([(t ** 3)[None, :], u[1:]], dim=0)
    jac = torch.cat([(3 * t ** 2 * 0.5 * b0 ** (1 / 3) * wg)[None, :], jac[1:]], dim=0)
    f = jac / (_retarded_angle(u, R, x_R, phim) + 2 * x_R)
    w = torch.zeros(K + 1, dtype=F64, device=dev)
    w = w.index_add(0, torch.arange(K, device=dev), (f * ((k + 1) * ds - u) / ds).sum(1))
    return w.index_add(0, torch.arange(1, K + 1, device=dev), (f * (u - k * ds) / ds).sum(1))


def wake_1d_t(S0, ds, lam, dlam, geom):
    """csr1d.wake_1d in torch; geom with tensor R, phi, phim, x_R (cases 'AB', 'CD', 'none')."""
    n = lam.shape[0]
    if geom['case'] == 'AB':
        R, phi = geom['R'].abs(), geom['phi'].abs()
        if not float(phi) > 0:
            return torch.zeros_like(lam)
        U = R * phi ** 3 / 24
        integral = _causal_conv(dlam, _bend_weights(U, ds))
        Sg = S0 + ds * torch.arange(n, dtype=F64, device=lam.device)
        lam1 = _interp_uniform(Sg - U, S0, ds, lam)
        lam2 = _interp_uniform(Sg - R * phi ** 3 / 6, S0, ds, lam)
        return AMP_BEND / R ** (2 / 3) * (U ** (-1 / 3) * (lam1 - lam2) + integral)
    if geom['case'] == 'CD':
        R, phim, x_R = geom['R'].abs(), geom['phim'].abs(), geom['x_R']
        smax = R * phim ** 3 / 24 * (phim + 4 * x_R) / (phim + x_R)
        integral = _causal_conv(dlam, _drift_weights(smax, ds, n, R, x_R, phim))
        Sg = S0 + ds * torch.arange(n, dtype=F64, device=lam.device)
        lam3 = _interp_uniform(Sg - smax, S0, ds, lam)
        lam4 = _interp_uniform(Sg - R * phim ** 2 * (phim + 3 * x_R) / 6, S0, ds, lam)
        return AMP_DRIFT / R * ((lam3 - lam4) / (phim + 2 * x_R) + integral)
    return torch.zeros_like(lam)


class CSR1DDiff(CSR1D):
    """
    CSR1DDiff(input_file, coords=None, params=None, device='cpu', structure=None, checkpoint=False,
              segment=None, deposit='cic')

    The arguments are CSR2DDiff's (coords, params {(element, key): tensor}, structure = a previous
    run's record, checkpoint, segment), plus `deposit` ('cic' or 'histogram', see the module doc).
    After run(): self.tp is the final torch beam, self.record the structure record; with segment=k,
    segment_gradient(loss_fn, coords=) gives the gradient.
    """

    # CSR2DDiff's tracking, element and segment machinery: it touches only the lattice, the
    # parameters, the beam and the tape, which this class provides
    _torch_element = CSR2DDiff._torch_element
    op_track = CSR2DDiff.op_track
    _track = CSR2DDiff._track
    _coords = staticmethod(CSR2DDiff._coords)
    _sync = CSR2DDiff._sync
    segment_gradient = CSR2DDiff.segment_gradient

    def __init__(self, input_file, coords=None, params=None, device='cpu', structure=None,
                 checkpoint=False, segment=None, deposit='cic'):
        if deposit not in ('cic', 'histogram'):
            raise ValueError(f"deposit must be 'cic' or 'histogram', got {deposit!r}")
        self.device = torch.device(device)
        self.checkpoint = bool(checkpoint)
        self.segment = int(segment) if segment else None
        self.deposit = deposit
        self.tape, self._bounds, self._n_track = [], {}, 0
        self._replay = structure
        self.structure = []
        self.record = dict(kicks=self.structure, snap={})
        self.lattice_params = dict(params or {})        # self.params is CSR1D's CSR_1D options
        self._ele = None
        super().__init__(input_file=input_file)
        p = self.beam.particle
        c = coords
        if c is None:
            c = torch.as_tensor(np.stack([p.x, p.px, p.y, p.py, p.z, p.pz]), dtype=F64)
        c = c.to(self.device)
        ten = lambda v: torch.tensor(float(v), dtype=F64, device=self.device)
        self.tp = Particle(*c.unbind(0), ten(p.s), ten(p.p0c), ten(p.mc2))
        self._sync()
        self.beam.track = self._track

    def get_bmadx_element(self, ele, DL, entrance=False, exit=False):
        self._ele = (ele, entrance, exit)
        return CSR1D.get_bmadx_element(self, ele, DL, entrance=entrance, exit=exit)

    def run(self, stop_time=None):
        if self.segment:
            with torch.no_grad():
                return CSR1D.run(self, stop_time)
        return CSR1D.run(self, stop_time)

    # ---- the structure of a kick (numpy) and the kick itself (torch)
    def calculate_1D_CSR(self, element_name):
        k = len(self.structure)
        rec = self._replay['kicks'][k] if self._replay is not None else None
        if rec is None:
            S, W, geom = CSR1D.calculate_1D_CSR(self, element_name)
            # the bin edges exactly as csr1d.line_density makes them (a histogram's bins are
            # decided on these, so rebuilding them from S would move particles at an edge)
            z = self.beam.z
            pad = max(int(self.params.nsmooth), 0) + 2
            nb = S.size - 2 * pad
            edges = z.min() + self.wake_ds * np.arange(-pad, nb + pad + 1)
            rec = dict(S0=float(S[0]), n=int(S.size), ds=float(self.wake_ds), edges=edges, geom=dict(geom), times=[])
        else:
            n, ds = rec['n'], rec['ds']
            S = rec['S0'] + ds * np.arange(n)
            W, geom = np.zeros(n), rec['geom']
            self.wake_S, self.wake_ds, self.wake_lam, self.wake_W, self.wake_geom = S, ds, W, W, geom
        self._cur = dict(rec, L_kick=None)
        return S, W, geom

    def apply_wake_1d(self, S, W, L_kick):
        rec = self._cur
        rec['L_kick'] = L_kick
        idx = len(self.structure)
        self.structure.append(rec)
        if self.segment:
            self.tape.append(('kick', idx, L_kick))
        self.tp, Wt = self.op_kick(self.tp, None, rec, L_kick, self.lattice_params)
        self.wake_W = Wt.detach().cpu().numpy()
        self._sync()

    def _geometry_t(self, geom, params):
        """The kick's geometry as tensors, following a bend-angle parameter of its bend if given."""
        T = lambda v: torch.tensor(float(v), dtype=F64, device=self.device)
        g = dict(case=geom['case'])
        if geom['case'] == 'none':
            return g
        theta = params.get((geom['bend'], 'angle')) if geom.get('bend') else None
        R0, phim0 = abs(float(geom['R'])), abs(float(geom['phim']))
        if theta is None:
            g.update(R=T(R0), phi=T(geom['phi']), phim=T(phim0), x_R=T(geom['x_R']))
            return g
        th = theta.abs()
        L = float(self.lattice.lattice_config[geom['bend']]['L'])
        R = L / th
        g.update(R=R, phim=th, phi=th * (float(geom['phi']) / phim0),          # the same fraction of the bend
                 x_R=float(geom['x_R']) * R0 / R)                            # the same distance past its exit
        return g

    def line_density(self, z, S0, n, ds, edges=None):
        """lambda and lambda' on the kick's grid (csr1d.line_density with the given deposit)."""
        p = self.params                                    # CSR_1D options (nsmooth)
        c = (z - S0) / ds
        if self.deposit == 'histogram':
            # np.histogram: bins [e_i, e_i+1), the last one closed
            e = torch.as_tensor(edges, dtype=F64, device=z.device)
            i = torch.searchsorted(e, z.detach(), right=True) - 1
            i = torch.where(z.detach() == e[-1], torch.full_like(i, n - 1), i)
            ok = (i >= 0) & (i < n)
            cnt = torch.zeros(n, dtype=F64, device=z.device).index_add(0, i.clamp(0, n - 1), ok.to(F64))
        else:
            i0 = torch.floor(c).long()
            f = c - i0
            cnt = torch.zeros(n, dtype=F64, device=z.device)
            for j, wgt in ((i0, 1.0 - f), (i0 + 1, f)):
                ok = (j >= 0) & (j < n)
                cnt = cnt.index_add(0, j.clamp(0, n - 1), wgt * ok)
        lam = self.n_electrons * cnt / (z.shape[0] * ds)
        if p.nsmooth > 1:
            from scipy.signal import savgol_coeffs
            h = torch.as_tensor(savgol_coeffs(2 * p.nsmooth + 1, 2), dtype=F64, device=z.device)
            total = lam.sum()
            lam_s = torch.nn.functional.conv1d(lam.view(1, 1, -1), h.flip(0).view(1, 1, -1),
                                               padding=p.nsmooth).view(-1).clamp(min=0.0)
            lam = lam_s * (total / lam_s.sum()) if float(lam_s.sum()) > 0 else lam_s
        dlam = torch.cat([((lam[1] - lam[0]) / ds)[None], (lam[2:] - lam[:-2]) / (2 * ds), ((lam[-1] - lam[-2]) / ds)[None]])
        return lam, dlam

    def op_kick(self, tp, snaps, rec, L_kick, params):
        """One 1D kick in torch: the line density, the wake on the kick's grid, the kick. -> (beam, W)."""
        S0, n, ds = rec['S0'], rec['n'], rec['ds']
        lam, dlam = self.line_density(tp.z, S0, n, ds, rec.get('edges'))
        W = wake_1d_t(S0, ds, lam, dlam, self._geometry_t(rec['geom'], params))
        if L_kick is not None:
            dE = _interp_uniform(tp.z, S0, ds, L_kick * W / self.beam.init_energy, clamp=True)
            tp = Particle(tp.x, tp.px, tp.y, tp.py, tp.z, tp.pz + dE, tp.s, tp.p0c, tp.mc2)
        return tp, W
