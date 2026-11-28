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
those of the lattice file. Windowed frames (quantile, peak): the window, the frame it implies, the
z smoothing width and the adaptive-smoothing weights are per-snapshot structure (recorded and
replayed like the per-kick structure); T(z) and everything else is differentiated. Not
differentiable yet: element lengths, MPI.
"""
import numpy as np
import torch
from bmadx import Particle

from ..CSR import CSR2D
from ..tracking import track
from .deposit import get_DF_rms, get_DF_windowed, window_structure, window_weight
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
    structure: the `record` of a previous run on the same input, to replay its per-kick
            structure (wake mesh, kick lookup frame, integration layout) and per-snapshot window
            structure (windowed frames) instead of recomputing it
            from this run's beam. With it, a finite difference sees exactly the map the gradient
            differentiates; without, moving mesh and quadrature nodes add discretisation-level
            terms that the gradient (correctly, at fixed structure) does not contain.
    checkpoint: recompute each tracking step and each wake chunk in the backward pass instead of
            keeping their graphs (memory ~ particles x steps instead of the whole run's graph).
    After run(), self.tp is the final torch beam (bmadx Particle of tensors), self.record the
    structure record (per kick and per snapshot).
    """

    def __init__(self, input_file, coords=None, params=None, device='cpu', structure=None,
                 checkpoint=False, segment=None):
        self.device = torch.device(device)
        self.checkpoint = bool(checkpoint)
        # segment = k: record the run on a tape and keep the beam only every k tracking steps (and
        # every snapshot), for segment_gradient(); the run itself then records no graph
        self.segment = int(segment) if segment else None
        self.tape = []
        self._bounds = {}                   # tape index -> beam (6 detached tensors) before that op
        self._allsnaps = {}                 # snapshot time -> (fields, frame), detached
        self._n_track = 0
        self._replay = structure            # a previous run's self.record
        self.structure = []                 # per kick
        self.snap_structure = {}            # per snapshot time (windowed frames)
        self.record = dict(kicks=self.structure, snap=self.snap_structure)
        # the snapshots a replayed run must keep although its own retention or drift thinning
        # (decided on its own beam) would drop them
        self._keep = (set(t for r in structure['kicks'] for t in r['times'])
                      if structure is not None else set())
        self._coords0 = coords
        self.lattice_params = dict(params or {})
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

    def _torch_element(self, element, ele=None, params=None):
        name, entrance, exit_ = self._ele if ele is None else ele
        params = self.lattice_params if params is None else params
        cfg = self.lattice.lattice_config[name]
        f = element._asdict()
        for (ename, key), val in params.items():
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
            # every numeric parameter but the step count (an int such as E2: 0 in a YAML file too)
            if isinstance(v, (int, float)) and not isinstance(v, bool) and k != 'NUM_STEPS':
                f[k] = torch.tensor(float(v), dtype=F64, device=self.device)
        return type(element)(**f)

    def _track(self, element, step_size, update_step=True):
        """Beam.track: the torch beam through the element, the numpy beam synchronised to it."""
        if self.segment:
            if self._n_track % self.segment == 0:
                self._bounds[len(self.tape)] = tuple(a.detach() for a in self._coords(self.tp))
            self._n_track += 1
            self.tape.append(('track', element, self._ele))
        self.tp = self.op_track(self.tp, element, self._ele, self.lattice_params)
        self._sync()
        self.beam.position += step_size
        if update_step:
            self.beam.step += 1

    @staticmethod
    def _coords(tp):
        return (tp.x, tp.px, tp.y, tp.py, tp.z, tp.pz)

    # the frame entries the history reads (history_from_tensors), with the five fields the tensors
    # a snapshot passes on to later kicks
    FRAME_KEYS = ('poly_coeffs', 'xi_bar', 'sigma_xi', 'z_bar', 'sigma_z', 'curv_c2', 'curv_c3')

    @classmethod
    def _snap_tensors(cls, snap):
        fields, frame = snap
        return list(fields) + [frame[k] for k in cls.FRAME_KEYS]

    def run(self, *a, **kw):
        if self.segment:
            # the forward pass of segment_gradient: no graph, the tape and the boundary beams
            with torch.no_grad():
                return super().run(*a, **kw)
        return super().run(*a, **kw)

    def segment_gradient(self, loss_fn, coords=None):
        """
        The gradient of loss_fn(final beam) after a run with segment=k, by a reverse sweep over the
        tape: each segment of k tracking steps is recomputed with gradients from its stored start
        beam, and the adjoints are carried backwards (the beam's, the snapshots' that later kicks
        read, and the parameters'). Memory: the beam every k steps, every snapshot, and one
        segment's graph; with checkpoint=True the segment's own steps and wake chunks are
        checkpointed too. Cost: about one forward pass more than checkpointing every step.

        loss_fn(tp) -> scalar tensor of the final beam (a bmadx Particle of tensors).
        coords: the (6, N) tensor the run started from, if it carries a graph to generator
        parameters (torch_backend.beam_param); the start beam's adjoint is passed into it.
        Accumulates .grad of the leaves behind self.lattice_params (a parameter may be derived from a leaf)
        and behind coords. Returns the loss as a float.
        """
        if not self.segment:
            raise RuntimeError('segment_gradient needs a run with segment=k')
        tp_end = self.tp
        aux = (tp_end.s, tp_end.p0c, tp_end.mc2)
        with torch.enable_grad():
            leaves = [a.detach().clone().requires_grad_(True) for a in self._coords(tp_end)]
            L = loss_fn(Particle(*leaves, *aux))
            adj = [g if g is not None else torch.zeros_like(a)
                   for g, a in zip(torch.autograd.grad(L, leaves, allow_unused=True), leaves)]
        P = [p for p in dict.fromkeys(self.lattice_params.values()) if torch.is_tensor(p) and p.requires_grad]
        gP = [torch.zeros_like(p) for p in P]
        adj_snap = {}
        starts = sorted(self._bounds)
        for b, e in reversed(list(zip(starts, starts[1:] + [len(self.tape)]))):
            ops = self.tape[b:e]
            made = [op[1] for op in ops if op[0] == 'snap']
            read = {t for op in ops if op[0] == 'kick' for t in self.structure[op[1]]['times']}
            with torch.enable_grad():
                S = [a.clone().requires_grad_(True) for a in self._bounds[b]]
                tp = Particle(*S, *aux)
                local, inputs = {}, {}
                for t in sorted(read - set(made)):
                    fields, frame = self._allsnaps[t]
                    fl = tuple(f.clone().requires_grad_(True) for f in fields)
                    fr = {k: (v.clone().requires_grad_(True) if torch.is_tensor(v) and v.is_floating_point() else v)
                          for k, v in frame.items()}
                    local[t] = (fl, fr)
                    inputs[t] = self._snap_tensors(local[t])
                for op in ops:
                    if op[0] == 'track':
                        tp = self.op_track(tp, op[1], op[2], self.lattice_params)
                    elif op[0] == 'snap':
                        local[op[1]] = self.op_snap(tp, op[2])
                    else:
                        tp, _ = self.op_kick(tp, local, self.structure[op[1]], op[2], self.lattice_params)
                outs, gouts = [], []
                for a, g in zip(self._coords(tp), adj):
                    if a.requires_grad:
                        outs.append(a)
                        gouts.append(g)
                for t in made:
                    for a, g in zip(self._snap_tensors(local[t]), adj_snap.pop(t, [])):
                        if g is not None and torch.is_tensor(a) and a.requires_grad:
                            outs.append(a)
                            gouts.append(g)
                flat_in = [(t, i) for t in inputs for i in range(len(inputs[t]))]
                ins = S + [inputs[t][i] for t, i in flat_in] + P
                grads = torch.autograd.grad(outs, ins, gouts, allow_unused=True) if outs else [None] * len(ins)
            adj = [g if g is not None else torch.zeros_like(a) for g, a in zip(grads[:6], S)]
            for (t, i), g in zip(flat_in, grads[6:6 + len(flat_in)]):
                lst = adj_snap.setdefault(t, [None] * len(inputs[t]))
                if g is not None:
                    lst[i] = g if lst[i] is None else lst[i] + g
            for j, g in enumerate(grads[6 + len(flat_in):]):
                if g is not None:
                    gP[j] = gP[j] + g
        # into the parameters' own graphs: a parameter may be derived from a leaf (e.g. -theta for the
        # second bend), and the start beam from a generator's parameters
        outs, gouts = list(P), list(gP)
        if coords is not None and coords.requires_grad:
            outs.append(coords)
            gouts.append(torch.stack(adj))
        if outs:
            torch.autograd.backward(outs, gouts)
        return float(L.detach())

    def op_track(self, tp, element, ele, params):
        """One tracking piece in torch (a pure function of the beam and the parameters)."""
        el = self._torch_element(element, ele, params)
        if self.checkpoint and torch.is_grad_enabled():
            from torch.utils.checkpoint import checkpoint

            def step(*c):
                q = track(Particle(*c, tp.s, tp.p0c, tp.mc2), el)
                return q.x, q.px, q.y, q.py, q.z, q.pz
            c = checkpoint(step, tp.x, tp.px, tp.y, tp.py, tp.z, tp.pz, use_reentrant=False)
            return Particle(*c, tp.s, tp.p0c, tp.mc2)
        return track(tp, el)

    # ------------------------------------------------------------------ history
    def _torch_snapshot(self):
        tr = self.DF_tracker
        t = float(tr.t)
        st = None
        if tr.frame != 'rms':
            rep = self._replay['snap'] if self._replay is not None else {}
            st = rep.get(t) or window_structure(tr)
            self.snap_structure[t] = st
        fields, frame = self.op_snap(self.tp, st)
        self._thist[t] = (fields, frame)
        if self.segment:
            if not self.tape:                          # the initial deposit opens the first segment
                self._bounds[0] = tuple(a.detach() for a in self._coords(self.tp))
            self.tape.append(('snap', t, st))
            self._allsnaps[t] = (tuple(f.detach() for f in fields),
                                 {k: (v.detach() if torch.is_tensor(v) else v) for k, v in frame.items()})
        live = set(float(t) for t in tr.time_log)
        for t in list(self._thist):
            if t not in live and t not in self._keep:
                del self._thist[t]

    def op_snap(self, tp, st):
        """The deposit of the beam tp in torch: (fields, frame); st the window structure (windowed frames)."""
        tr = self.DF_tracker

        def dep(x, z, px):
            return get_DF_rms(tr, x, z, px) if st is None else get_DF_windowed(tr, x, z, px, st)
        if not (self.checkpoint and torch.is_grad_enabled()):
            return dep(tp.x, tp.z, tp.px)
        # checkpointed: the fits keep dozens of particle-sized intermediates per snapshot (masks, u, w,
        # weights; ~0.4 GB at 1e6 particles), so recompute the deposit in the backward pass instead
        from torch.utils.checkpoint import checkpoint
        keys = []

        def flat(x, z, px):
            fields, frame = dep(x, z, px)
            keys[:] = [k for k, v in frame.items() if torch.is_tensor(v)]
            return tuple(fields) + tuple(frame[k] for k in keys)
        out = checkpoint(flat, tp.x, tp.z, tp.px, use_reentrant=False)
        frame = dict(zip(keys, out[5:]))
        return tuple(out[:5]), frame

    def _record_snapshot(self):
        super()._record_snapshot()
        self._torch_snapshot()

    def _history(self, times, snaps=None):
        tr = self.DF_tracker
        snaps = self._thist if snaps is None else snaps
        return history_from_tensors(tr, times, [snaps[t][0] for t in times],
                                    [snaps[t][1] for t in times], self.device)

    # ------------------------------------------------------------------ kicks
    def _kick(self, step_count):
        sched = self.lattice.schedule
        k = len(self.structure)
        rec = self._replay['kicks'][k] if self._replay is not None else None
        if rec is None:
            self.get_CSR_mesh()
            ridge = self._ridge_coeffs()
            slope = np.polyfit(self.beam.z, self.beam.x, deg=1)
        else:
            (self.CSR_xmesh, self.CSR_zmesh, self.CSR_xrange_transformed, self.CSR_zrange,
             slope, ridge) = rec['mesh']
        # the history's snapshots are structure too: retention and drift thinning are decided on the
        # beam, so a replayed run uses the recorded set
        times = [float(t) for t in self.DF_tracker.time_log] if rec is None else rec['times']
        t = float(self.beam.position)
        rec_new = dict(mesh=(np.asarray(self.CSR_xmesh), np.asarray(self.CSR_zmesh),
                             np.asarray(self.CSR_xrange_transformed), np.asarray(self.CSR_zrange), slope, ridge),
                       times=times, t=t,
                       window=(self.snap_structure[times[-1]]['window'] if self._windowed_frame() else None),
                       layout=None if rec is None else rec['layout'], refine=None if rec is None else rec['refine'])
        L_kick = None
        if self.CSR_params.apply_CSR:
            if sched.kick_interval == 'midpoint':
                L_kick = sched.kick_hi[step_count] - sched.kick_lo[step_count]
            else:
                L_kick = self.beam.position - self._s_last_kick
        if self.segment:
            self.tape.append(('kick', len(self.structure), L_kick))
        self.tp, (dE, xk) = self.op_kick(self.tp, self._thist, rec_new, L_kick, self.lattice_params)
        rec_new['layout'], rec_new['refine'] = self._tw.last_layout, self._tw.last_refine
        self.structure.append(rec_new)
        self.dE_dct = dE.detach().cpu().numpy()
        self.x_kick = xk.detach().cpu().numpy()
        self.t_wakes = (dE, xk)
        if L_kick is not None:
            self._sync()
            self._s_last_kick = self.beam.position
        if self.CSR_params.write_wakes:
            self.write_wakes()

    def op_kick(self, tp, snaps, rec, L_kick, params):
        """
        One kick in torch, a pure function of the beam, the snapshots and the parameters given the
        structure record rec (mesh, history times, time, window, layout and refinement or None to
        build them): the wake mesh on the history, then the kick over L_kick (None: not applied).
        Returns (beam, (dE, xk)).
        """
        if getattr(self, '_tw', None) is None:
            self._tw = TorchWake(self, device=self.device, chunk=self.CSR_params.torch_chunk,
                                 checkpoint=self.checkpoint, compile=self.CSR_params.torch_compile,
                                 bands=self.CSR_params.torch_bands)
        if any(k == 'angle' for _, k in params):
            # the CSR geometry follows the bend angles too (rebuilt from the parameters, so a
            # replayed segment differentiates it as well)
            self._tw.orbit = Orbit.from_lattice_params(self.lattice, params, self.device)
        xmesh, zmesh, xrange_t, zrange, slope, ridge = rec['mesh']
        H = self._history(rec['times'], snaps)
        nxm, nzm = self.CSR_params.xbins, self.CSR_params.zbins
        t = rec['t']
        s_arr = t + np.asarray(zmesh)[:nxm * nzm]
        x_arr = np.asarray(xmesh)[:nxm * nzm]
        st = torch.as_tensor(s_arr, dtype=F64, device=self.device)
        xt = torch.as_tensor(x_arr, dtype=F64, device=self.device)
        vx_obs = comoving_fields(H, xt, st - t, torch.full_like(st, t))[3]
        if rec['window'] is not None:
            vx_obs = self._observer_vx_t(xt, st - t, vx_obs, tp, rec['window'])
        dE, xk = self._tw.compute(s_arr, x_arr, H=H, vx_obs=vx_obs, as_tensor=True,
                                  layout=rec['layout'], refine=rec['refine'], t=t)
        dE, xk = dE.view(nxm, nzm), xk.view(nxm, nzm)
        if L_kick is not None:
            tp = self._apply_torch_kick(tp, dE, xk, L_kick, slope, ridge, xrange_t, zrange)
        return tp, (dE, xk)

    def _observer_vx_t(self, x, zobs, vx_src, tp, window):
        """
        CSR2D._observer_vx in torch: T * (windowed source vx) + (1 - T) * (beam-wide fit), T the
        newest window at the (fixed) observer z, the fit (_fit_beam_vx) on the beam tp.
        """
        lo, hi, m_lo, m_hi = window
        T = window_weight(zobs, lo, hi, m_lo, m_hi)
        mz, sz = tp.z.mean(), tp.z.std(correction=0)
        zc = tp.z - mz
        slope = (zc * (tp.x - tp.x.mean())).sum() / (zc * zc).sum()
        p1 = tp.x.mean() - slope * mz
        wq = ((tp.z - mz) / sz).clamp(-3.0, 3.0)
        A = torch.stack([torch.ones_like(wq), tp.x - (slope * tp.z + p1), wq, wq ** 2, wq ** 3], dim=1)
        c = torch.linalg.solve(A.T @ A, A.T @ tp.px)
        wo = ((zobs - mz) / sz).clamp(-3.0, 3.0)
        vb = c[0] + c[1] * (x - (slope * zobs + p1)) + c[2] * wo + c[3] * wo ** 2 + c[4] * wo ** 3
        return torch.where(T >= 1.0, vx_src, T * vx_src + (1.0 - T) * vb)

    def _ridge_coeffs(self):
        """The newest snapshot's ridge sigma_xi * g(w) as constants (CSR2D._mesh_ridge), or None."""
        tr = self.DF_tracker
        if not (tr.curvature and tr.sigma_z):
            return None
        # windowed frames hold w at the grid edge for tail particles (CSR2D._mesh_ridge)
        wclip = (tr.w_start, tr.w_end) if tr.frame in ('quantile', 'peak') else None
        return (float(tr.z_bar), float(tr.sigma_z), float(tr.sigma_xi), float(tr.curv_c2),
                float(tr.curv_c3), float(tr.curv_zlim), wclip)

    def _apply_torch_kick(self, tp, dE, xk, L_kick, slope, ridge, xrange_t, zrange):
        """
        Beam.apply_wakes in torch. The particles are mapped into the mesh with the detached tilt fit
        and ridge (the mesh's own), evaluated at the differentiable coordinates. Returns the beam.
        """
        dev = self.device
        x_rel = tp.x - (slope[0] * tp.z + slope[1])
        if ridge is not None:
            from .deposit import curv_g
            z_bar, s_z, s_xi, c2, c3, wl, wclip = ridge
            w = (tp.z - z_bar) / s_z
            if wclip is not None:
                w = w.clamp(*wclip)
            x_rel = x_rel - s_xi * curv_g(c2, c3, w, wl)
        xr = torch.as_tensor(np.asarray(xrange_t), dtype=F64, device=dev)
        zr = torch.as_tensor(np.asarray(zrange), dtype=F64, device=dev)
        cx = self.CSR_params.mesh_x_quantile is not None
        cz = self.CSR_params.mesh_z_quantile is not None
        E0 = self.beam.init_energy
        pz = tp.pz + bilinear(xr, zr, L_kick * dE * 1e6 / E0, x_rel, tp.z, cx, cz)
        px = tp.px
        if self.CSR_params.transverse_on:
            px = px + bilinear(xr, zr, L_kick * xk * 1e6 / E0, x_rel, tp.z, cx, cz)
        return Particle(tp.x, px, tp.y, tp.py, tp.z, pz, tp.s, tp.p0c, tp.mc2)
