"""
The differentiable 2D CSR run: CSR2D with a torch shadow of the beam.

The numpy CSR2D keeps making every structural decision (schedule, element pieces, history
retention and thinning, the deposition masks, the wake mesh, the integration layout and the x'
refinement), on the detached values of the torch beam, which it is re-synchronised to after every
step. The torch side recomputes the VALUES with gradients:

  tracking     bmadx on torch particles, with element parameters that may be tensors
  deposition   torch_backend.deposit.get_DF_rms at every snapshot (fields and frame)
  wake         torch_backend.evaluate on the torch history
  kick         bilinear interpolation of the wake mesh at the particles (as Beam.apply_wakes)

so d(anything computed from the final beam) / d(initial coordinates, lattice parameters) is the
derivative of the forward map at fixed structure. Node positions (wake mesh, s' and x' nodes) are
fixed in absolute coordinates; the kick lookup maps the particles into the mesh with the detached
tilt fit and ridge, which is the consistent derivative of 'kick = W at the particle'.

A bend angle parameter enters the tracking (G = angle/L) and the CSR geometry (the reference-orbit
table and the element curvature, Orbit.from_lattice_params); the s' layout and the schedule stay
those of the lattice file. Not differentiable yet: element lengths, frames other than 'rms', MPI.
"""
import numpy as np
import torch
from bmadx import Particle

from ..CSR import CSR2D
from ..tracking import track
from .deposit import get_DF_rms
from .evaluate import TorchWake
from .history import F64, Orbit, history_from_tensors
from .interp import comoving_fields


def bilinear(xr, zr, grid, xq, zq, clamp_x=False, clamp_z=False):
    """scipy RegularGridInterpolator((xr, zr), grid, linear, fill_value=0) at (xq, zq)."""
    if clamp_x:
        xq = xq.clamp(xr[0], xr[-1])
    if clamp_z:
        zq = zq.clamp(zr[0], zr[-1])
    i = (torch.searchsorted(xr, xq.detach()) - 1).clamp(0, xr.numel() - 2)
    j = (torch.searchsorted(zr, zq.detach()) - 1).clamp(0, zr.numel() - 2)
    tx = (xq - xr[i]) / (xr[i + 1] - xr[i])
    tz = (zq - zr[j]) / (zr[j + 1] - zr[j])
    v = ((1 - tx) * (1 - tz) * grid[i, j] + tx * (1 - tz) * grid[i + 1, j]
         + (1 - tx) * tz * grid[i, j + 1] + tx * tz * grid[i + 1, j + 1])
    inside = (xq >= xr[0]) & (xq <= xr[-1]) & (zq >= zr[0]) & (zq <= zr[-1])
    return torch.where(inside, v, torch.zeros_like(v))


class CSR2DDiff(CSR2D):
    """
    CSR2DDiff(input_file, coords=None, params=None, device='cpu')

    coords: (6, N) tensor of the initial (x, px, y, py, z, pz), e.g. a leaf with requires_grad;
            default: the beam of the input file.
    params: {(element name, key): tensor} lattice parameters as tensors, key in 'K1', 'K2',
            'angle' (tracking and CSR geometry), 'E1', 'E2'.
    structure: the `structure` list of a previous run on the same input, to replay its per-kick
            structure (wake mesh, kick lookup frame, integration layout) instead of recomputing it
            from this run's beam. With it, a finite difference sees exactly the map the gradient
            differentiates; without, moving mesh and quadrature nodes add discretisation-level
            terms that the gradient (correctly, at fixed structure) does not contain.
    After run(), self.tp is the final torch beam (bmadx Particle of tensors), self.structure the
    per-kick record.
    """

    def __init__(self, input_file, coords=None, params=None, device='cpu', structure=None):
        self.device = torch.device(device)
        self._replay = structure
        self.structure = []
        self._coords0 = coords
        self.params = dict(params or {})
        self._thist = {}
        self._ele = None
        super().__init__(input_file=input_file, parallel=False)
        self.beam.track = self._track

    # ------------------------------------------------------------------ beam
    def _sync(self):
        """The numpy beam = the torch beam's values."""
        tp = self.tp
        arr = [a.detach().cpu().numpy().astype(np.float64) for a in (tp.x, tp.px, tp.y, tp.py, tp.z, tp.pz)]
        p = self.beam.particle
        self.beam.particle = Particle(*arr, p.s, p.p0c, p.mc2)
        self.beam.update_status()

    def initialization(self):
        p = self.beam.particle
        c = self._coords0
        if c is None:
            c = torch.as_tensor(np.stack([p.x, p.px, p.y, p.py, p.z, p.pz]), dtype=F64)
        c = c.to(self.device)
        # p0c and mc2 as tensors too: some bmadx routines (the quadrupole's z correction) take
        # lib.sqrt of them
        ten = lambda v: torch.tensor(float(v), dtype=F64, device=self.device)
        self.tp = Particle(*c.unbind(0), ten(p.s), ten(p.p0c), ten(p.mc2))
        self._sync()
        super().initialization()
        self._torch_snapshot()

    # ------------------------------------------------------------------ tracking
    def get_bmadx_element(self, ele, DL, entrance=False, exit=False):
        self._ele = (ele, entrance, exit)
        return super().get_bmadx_element(ele, DL, entrance=entrance, exit=exit)

    def _torch_element(self, element):
        name, entrance, exit_ = self._ele
        cfg = self.lattice.lattice_config[name]
        f = element._asdict()
        for (ename, key), val in self.params.items():
            if ename != name:
                continue
            if key == 'angle':
                f['G'] = val / cfg['L']
            elif key == 'E1':
                if entrance:
                    f['E1'] = val
            elif key == 'E2':
                if exit_:
                    f['E2'] = val
            else:
                f[key] = val
        for k, v in f.items():
            if isinstance(v, float):
                f[k] = torch.tensor(v, dtype=F64, device=self.device)
        return type(element)(**f)

    def _track(self, element, step_size, update_step=True):
        """Beam.track: the torch beam through the element, the numpy beam synchronised to it."""
        self.tp = track(self.tp, self._torch_element(element))
        self._sync()
        self.beam.position += step_size
        if update_step:
            self.beam.step += 1

    # ------------------------------------------------------------------ history
    def _torch_snapshot(self):
        tr = self.DF_tracker
        fields, frame = get_DF_rms(tr, self.tp.x, self.tp.z, self.tp.px)
        self._thist[float(tr.t)] = (fields, frame)
        live = set(float(t) for t in tr.time_log)
        for t in list(self._thist):
            if t not in live:
                del self._thist[t]

    def _record_snapshot(self):
        super()._record_snapshot()
        self._torch_snapshot()

    def _history(self):
        tr = self.DF_tracker
        times = [float(t) for t in tr.time_log]
        return history_from_tensors(tr, times, [self._thist[t][0] for t in times],
                                    [self._thist[t][1] for t in times], self.device)

    # ------------------------------------------------------------------ kicks
    def _kick(self, step_count):
        if self._windowed_frame():
            raise NotImplementedError('CSR2DDiff: windowed frames (quantile/peak) are not differentiable yet')
        sched = self.lattice.schedule
        k = len(self.structure)
        rec = self._replay[k] if self._replay is not None else None
        if rec is None:
            self.get_CSR_mesh()
            ridge = self._ridge_coeffs()
            slope = np.polyfit(self.beam.z, self.beam.x, deg=1)
        else:
            (self.CSR_xmesh, self.CSR_zmesh, self.CSR_xrange_transformed, self.CSR_zrange,
             slope, ridge) = rec['mesh']
        H = self._history()
        if getattr(self, '_tw', None) is None:
            self._tw = TorchWake(self, device=self.device, chunk=self.CSR_params.torch_chunk)
            if any(k == 'angle' for _, k in self.params):
                # the CSR geometry follows the bend angles too
                self._tw.orbit = Orbit.from_lattice_params(self.lattice, self.params, self.device)
        nxm, nzm = self.CSR_params.xbins, self.CSR_params.zbins
        t = float(self.beam.position)
        s_arr = t + np.asarray(self.CSR_zmesh)[:nxm * nzm]
        x_arr = np.asarray(self.CSR_xmesh)[:nxm * nzm]
        st = torch.as_tensor(s_arr, dtype=F64, device=self.device)
        xt = torch.as_tensor(x_arr, dtype=F64, device=self.device)
        vx_obs = comoving_fields(H, xt, st - t, torch.full_like(st, t))[3]
        dE, xk = self._tw.compute(s_arr, x_arr, H=H, vx_obs=vx_obs, as_tensor=True,
                                  layout=None if rec is None else rec['layout'])
        self.structure.append(dict(mesh=(np.asarray(self.CSR_xmesh), np.asarray(self.CSR_zmesh),
                                          np.asarray(self.CSR_xrange_transformed), np.asarray(self.CSR_zrange),
                                          slope, ridge),
                                   layout=self._tw.last_layout))
        dE, xk = dE.view(nxm, nzm), xk.view(nxm, nzm)
        self.dE_dct = dE.detach().cpu().numpy()
        self.x_kick = xk.detach().cpu().numpy()
        self.t_wakes = (dE, xk)
        if self.CSR_params.apply_CSR:
            if sched.kick_interval == 'midpoint':
                L_kick = sched.kick_hi[step_count] - sched.kick_lo[step_count]
            else:
                L_kick = self.beam.position - self._s_last_kick
            self._apply_torch_kick(dE, xk, L_kick, slope, ridge)
            self._s_last_kick = self.beam.position
        if self.CSR_params.write_wakes:
            self.write_wakes()

    def _ridge_coeffs(self):
        """The newest snapshot's ridge sigma_xi * g(w) as constants (CSR2D._mesh_ridge), or None."""
        tr = self.DF_tracker
        if not (tr.curvature and tr.sigma_z):
            return None
        return (float(tr.z_bar), float(tr.sigma_z), float(tr.sigma_xi), float(tr.curv_c2),
                float(tr.curv_c3), float(tr.curv_zlim))

    def _apply_torch_kick(self, dE, xk, L_kick, slope, ridge):
        """
        Beam.apply_wakes in torch. The particles are mapped into the mesh with the detached tilt fit
        and ridge (the mesh's own), evaluated at the differentiable coordinates.
        """
        tp = self.tp
        dev = self.device
        x_rel = tp.x - (slope[0] * tp.z + slope[1])
        if ridge is not None:
            from .deposit import curv_g
            z_bar, s_z, s_xi, c2, c3, wl = ridge
            x_rel = x_rel - s_xi * curv_g(c2, c3, (tp.z - z_bar) / s_z, wl)
        xr = torch.as_tensor(np.asarray(self.CSR_xrange_transformed), dtype=F64, device=dev)
        zr = torch.as_tensor(np.asarray(self.CSR_zrange), dtype=F64, device=dev)
        cx = self.CSR_params.mesh_x_quantile is not None
        cz = self.CSR_params.mesh_z_quantile is not None
        E0 = self.beam.init_energy
        pz = tp.pz + bilinear(xr, zr, L_kick * dE * 1e6 / E0, x_rel, tp.z, cx, cz)
        px = tp.px
        if self.CSR_params.transverse_on:
            px = px + bilinear(xr, zr, L_kick * xk * 1e6 / E0, x_rel, tp.z, cx, cz)
        self.tp = Particle(tp.x, px, tp.y, tp.py, tp.z, pz, tp.s, tp.p0c, tp.mc2)
        self._sync()
