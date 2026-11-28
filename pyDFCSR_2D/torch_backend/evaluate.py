"""
The wake mesh of one kick with the torch backend: the numpy layout of every mesh point
(WakeMixin.wake_layout), then one batched evaluation of all their quadrature nodes on the device.

Per mesh point the quadrature is the numba path's (wake.py):
  every s' region, every disjoint x' band: per live column a uniform x' trapezoid on xbins nodes,
  optionally refined by the error-based x' refinement (_adapt_x_error, same rounds and selection);
  then the trapezoid over the region's s' nodes; plus the polar patch around the pole.
Here the columns of all points are packed into flat tensors (point id, group id, lo, width, s'),
expanded to nodes chunk by chunk, and the per-column sums are combined with the s' trapezoid
weights by index_add. The selection of the refinement rounds runs in numpy on the per-column error
estimates (vectorised over all groups); a group is one (point, region, band).
"""
import time

import numpy as np
import torch

from .history import F64, History, Orbit
from .integrand import integrand

NO_TAPER = (-2.0, -1.0, 0.0)      # r > R2 everywhere: weight exactly 1


def _trapz_weights(sp):
    """np.trapz(y, x=sp) == sum(w * y)."""
    w = np.zeros(sp.size)
    if sp.size > 1:
        d = np.diff(sp)
        w[:-1] += 0.5 * d
        w[1:] += 0.5 * d
    return w


def _select(group, we, budget):
    """
    The refinement selection of _adapt_x_error for many groups at once: per group, the columns with
    the largest we until the unselected rest is <= budget[group] (nothing if the group's total is).
    """
    n = we.size
    if n == 0:
        return np.zeros(0, dtype=bool)
    order = np.lexsort((-we, group))
    gs, ws = group[order], we[order]
    total = np.bincount(group, we, minlength=budget.size)
    cs = np.cumsum(ws)
    first = np.r_[True, gs[1:] != gs[:-1]]
    start = np.maximum.accumulate(np.where(first, np.arange(n), 0))
    before = cs - ws - (cs[start] - ws[start])
    keep_sorted = (total[gs] > budget[gs]) & (total[gs] - before > budget[gs])
    keep = np.zeros(n, dtype=bool)
    keep[order] = keep_sorted
    return keep


class TorchWake:

    def __init__(self, sim, device='cpu', chunk=1 << 21, checkpoint=False, compile=False):
        """
        checkpoint: when gradients are recorded, recompute each chunk's integrand in the backward
        pass instead of keeping its graph (torch.utils.checkpoint). A chunk keeps only its column
        range, so a kick's memory is the column tables, not ~50 doubles per integrand node.
        compile: torch.compile the integrand (history interpolation + kernel) into fused kernels.
        Unfused, every elementwise op writes a chunk-sized array (~300 per node); on an RTX 5000 Ada
        that bound the evaluation at ~23M nodes/s, and fusing gave 10x (progress_gpu.md §6). The
        first calls compile (~25 s); it needs a C compiler and the CUDA headers (triton).
        """
        self.sim = sim
        self.checkpoint = bool(checkpoint)
        self._integrand = torch.compile(integrand, dynamic=True) if compile else integrand
        self.device = torch.device(device)
        self.chunk = int(chunk)
        self.orbit = Orbit.from_lattice(sim.lattice, self.device)
        self.stats = {}

    def _t(self, a, dtype=F64):
        return torch.as_tensor(np.asarray(a), dtype=dtype, device=self.device)

    # ------------------------------------------------------------------ layout (numpy, per point)
    def _layouts(self, s_arr, x_arr, want_vx=True):
        sim = self.sim
        ip = sim.integration_params
        t = sim.beam.position
        nx = ip.xbins
        cols = {k: [] for k in ('pt', 'grp', 'lo', 'w', 'sp', 'ws', 'wc', 'R1', 'R2', 'ki', 'adapt')}
        patch, vxs = [], []
        g = 0
        lays = sim.wake_layouts(s_arr, x_arr)
        for i, (s, x, lay) in enumerate(zip(s_arr, x_arr, lays)):
            if want_vx:
                vx = sim._comoving_fields(np.array([x]), np.array([s - t]), np.array([t]))[3][0]
                if sim._windowed_frame():
                    vx = sim._observer_vx(x, s - t, vx)
                vxs.append(vx)
            for reg in lay['regions']:
                sp = np.asarray(reg['sp'], dtype=float)
                ws = _trapz_weights(sp)
                wc = np.abs(ws) if sp.size > 1 else np.ones(sp.size)
                adapt = bool(reg['adapt_x'] and ip.near_adapt and ip.near_adapt_x)
                R1, R2, ki = (NO_TAPER if reg['taper'] is None
                              else (reg['taper'][0], reg['taper'][1], float(reg['taper'][2])))
                for lo, hi in reg['bands']:
                    width = hi - lo
                    alive = width > 0.0
                    m = int(alive.sum())
                    if m:
                        cols['pt'].append(np.full(m, i)); cols['grp'].append(np.full(m, g))
                        cols['lo'].append(lo[alive]); cols['w'].append(width[alive]); cols['sp'].append(sp[alive])
                        cols['ws'].append(ws[alive]); cols['wc'].append(wc[alive])
                        cols['R1'].append(np.full(m, R1)); cols['R2'].append(np.full(m, R2))
                        cols['ki'].append(np.full(m, ki)); cols['adapt'].append(np.full(m, adapt))
                    g += 1
            if lay['patch'] is not None:
                patch.append((i, s, x) + tuple(lay['patch']))
        cat = {k: (np.concatenate(v) if v else np.zeros(0)) for k, v in cols.items()}
        cat['pt'] = cat['pt'].astype(np.int64); cat['grp'] = cat['grp'].astype(np.int64)
        cat['adapt'] = cat['adapt'].astype(bool)
        return cat, g, patch, np.array(vxs), nx

    # ------------------------------------------------------------------ evaluation (torch)
    def _obs(self, s_arr, x_arr, vxs):
        s = self._t(s_arr); x = self._t(x_arr)
        vec = self.orbit.vectors(s)
        vx = vxs if torch.is_tensor(vxs) else self._t(vxs)
        return torch.stack([s, x, vx] + list(vec), dim=-1)                # (P, 9)

    def _chunked(self, fn, a):
        """fn(a) -> (Sz, Sx) for one chunk, checkpointed when gradients are being recorded."""
        if self.checkpoint and torch.is_grad_enabled():
            from torch.utils.checkpoint import checkpoint
            return checkpoint(fn, a, use_reentrant=False)
        return fn(a)

    def _eval(self, H, obs_pt, pt, xp, sp, taper, t):
        """
        gz, gx at nodes at time t; pt: (n,) point index into obs_pt; taper: (R1, R2, ki) per node.
        t is passed, not read from the beam: a checkpointed chunk is re-evaluated in the backward
        pass, when the beam has moved on.
        """
        ip = self.sim.integration_params
        obs = obs_pt[pt].unbind(-1)
        return self._integrand(H, self.orbit, obs, xp, sp, t, taper, ip.transverse_terms == 'W1')

    def _columns(self, H, obs_pt, C, idx, offs, h):
        """sum_k f(lo + offs_k * h) over the nodes of the columns idx, for W and F."""
        t = float(self.sim.beam.position)
        n_off = offs.numel()
        per = max(1, self.chunk // n_off)
        Sz, Sx = [], []

        def chunk(a):
            j = idx[a:a + per]
            xp = (C['lo'][j, None] + offs[None, :] * h[a:a + per, None]).reshape(-1)
            rep = lambda v: v[j, None].expand(-1, n_off).reshape(-1)
            gz, gx = self._eval(H, obs_pt, rep(C['pt']), xp, rep(C['sp']),
                                (rep(C['R1']), rep(C['R2']), rep(C['ki'])), t)
            return gz.view(-1, n_off).sum(1), gx.view(-1, n_off).sum(1)
        for a in range(0, idx.numel(), per):
            z, x = self._chunked(chunk, a)
            Sz.append(z)
            Sx.append(x)
        return torch.cat(Sz), torch.cat(Sx)

    def compute(self, s_arr, x_arr, H=None, vx_obs=None, as_tensor=False, layout=None, refine=None):
        """
        (dE/dct, x kick) at the observers (s_arr, x_arr). H: the history as tensors (default: the
        tracker's, as constants); vx_obs: the observers' vx as a tensor (default: from the numba
        interpolant). as_tensor: return tensors (with their graph) instead of numpy arrays.
        layout, refine: a previous call's self.last_layout / self.last_refine (the refinement's
        active columns per round) to reuse (frozen structure); the new ones are kept there.
        """
        sim = self.sim
        ip = sim.integration_params
        t0 = time.time()
        if H is None:
            H = History.from_tracker(sim.DF_tracker, self.device)
        if layout is None:
            cat, n_grp, patch, vxs, nx = self._layouts(s_arr, x_arr, want_vx=vx_obs is None)
        else:
            cat, n_grp, patch, vxs, nx = layout
        self.last_layout = (cat, n_grp, patch, vxs, nx)
        if vx_obs is not None:
            vxs = vx_obs
        t_layout = time.time() - t0
        P = len(s_arr)
        obs_pt = self._obs(s_arr, x_arr, vxs)
        C = {k: self._t(cat[k]) for k in ('lo', 'w', 'sp', 'R1', 'R2', 'ki')}
        C['pt'] = self._t(cat['pt'], torch.long)
        ncol = cat['lo'].size
        Tz = torch.zeros(ncol, dtype=F64, device=self.device)
        Tx = torch.zeros_like(Tz)
        if ncol:
            # base pass: uniform trapezoid on xbins nodes, T = h * (sum f - (f_0 + f_last)/2)
            h = C['w'] / (nx - 1)
            frac = torch.linspace(0.0, 1.0, nx, dtype=F64, device=self.device)
            trap = torch.ones(nx, dtype=F64, device=self.device)
            trap[0] = trap[-1] = 0.5
            all_idx = torch.arange(ncol, device=self.device)
            Tz, Tx = self._weighted_columns(H, obs_pt, C, all_idx, frac, trap, C['w'])
            Tz, Tx = Tz * h, Tx * h
            Tz, Tx = self._refine(H, obs_pt, C, cat, n_grp, Tz, Tx, h, nx, refine)
        ws = self._t(cat['ws'])
        pt = C['pt']
        zero = torch.zeros(P, dtype=F64, device=self.device)
        dE = zero.index_add(0, pt, ws * Tz) if ncol else zero
        xk = zero.index_add(0, pt, ws * Tx) if ncol else zero
        pz, px = self._patch(H, obs_pt, patch, P)
        dE = -sim.CSR_scaling * (dE + pz)
        xk = sim.CSR_scaling * (xk + px)
        out = (dE, xk) if as_tensor else (dE.detach().cpu().numpy(), xk.detach().cpu().numpy())
        self.stats = dict(columns=ncol, points=P, t_layout=t_layout, t_eval=time.time() - t0 - t_layout)
        return out

    def _weighted_columns(self, H, obs_pt, C, idx, frac, trap, width):
        """sum_k trap_k f(lo + frac_k * width) per column (the base pass)."""
        t = float(self.sim.beam.position)
        n_off = frac.numel()
        per = max(1, self.chunk // n_off)
        Sz, Sx = [], []

        def chunk(a):
            j = idx[a:a + per]
            xp = (C['lo'][j, None] + frac[None, :] * width[j, None]).reshape(-1)
            rep = lambda v: v[j, None].expand(-1, n_off).reshape(-1)
            gz, gx = self._eval(H, obs_pt, rep(C['pt']), xp, rep(C['sp']),
                                (rep(C['R1']), rep(C['R2']), rep(C['ki'])), t)
            return (gz.view(-1, n_off) * trap).sum(1), (gx.view(-1, n_off) * trap).sum(1)
        for a in range(0, idx.numel(), per):
            z, x = self._chunked(chunk, a)
            Sz.append(z)
            Sx.append(x)
        return torch.cat(Sz), torch.cat(Sx)

    def _refine(self, H, obs_pt, C, cat, n_grp, Tz, Tx, h, nx, replay=None):
        """
        _adapt_x_error for every refined group at once. replay: the active columns of each round
        from a previous call (self.last_refine), used instead of the selection.
        """
        ip = self.sim.integration_params
        active = np.flatnonzero(cat['adapt'])
        self.last_refine = []
        if active.size == 0:
            return Tz, Tx
        grp = cat['grp']
        wc = cat['wc']
        h_c = h.clone()
        m = nx - 1
        for rnd in range(max(ip.x_adapt_rounds, 1)):
            if replay is not None:
                active = replay[rnd] if rnd < len(replay) else np.zeros(0, dtype=np.int64)
            if active.size == 0:
                break
            self.last_refine.append(active)
            mm = m * 2 ** rnd
            offs = torch.arange(mm, dtype=F64, device=self.device) + 0.5
            idx = self._t(active, torch.long)
            Sz, Sx = self._columns(H, obs_pt, C, idx, offs, h_c[idx])
            Tz_new = 0.5 * Tz[idx] + 0.5 * h_c[idx] * Sz
            Tx_new = 0.5 * Tx[idx] + 0.5 * h_c[idx] * Sx
            ez = ((Tz_new - Tz[idx]).abs() / 3.0).detach().cpu().numpy()
            ex = ((Tx_new - Tx[idx]).abs() / 3.0).detach().cpu().numpy()
            Tz = Tz.index_put((idx,), Tz_new)
            Tx = Tx.index_put((idx,), Tx_new)
            h_c = h_c.index_put((idx,), h_c[idx] * 0.5)
            g = grp[active]
            keep = np.zeros(active.size, dtype=bool)
            for e, T in ((ez, Tz), (ex, Tx)):
                budget = ip.x_adapt_tol * np.bincount(grp, wc * np.abs(T.detach().cpu().numpy()), minlength=n_grp)
                keep |= _select(g, wc[active] * e, budget)
            active = active[keep]
        return Tz, Tx

    def _patch(self, H, obs_pt, patch, P):
        """The polar patch of every point that has one (_integrate_near_patch)."""
        z = torch.zeros(P, dtype=F64, device=self.device)
        zx = torch.zeros(P, dtype=F64, device=self.device)
        if not patch:
            return z, zx
        ip = self.sim.integration_params
        t = float(self.sim.beam.position)
        nr, nphi = ip.near_patch_nr, ip.near_patch_nphi
        arr = np.array(patch)
        pid = arr[:, 0].astype(np.int64)
        s, x, R1, R2 = (self._t(arr[:, k]) for k in (1, 2, 3, 4))
        rr = (torch.arange(nr, dtype=F64, device=self.device) + 0.5)
        dphi = 2.0 * np.pi / nphi
        phi = torch.arange(nphi, dtype=F64, device=self.device) * dphi
        cphi, sphi = torch.cos(phi), torch.sin(phi)
        per = max(1, self.chunk // (nr * nphi))
        pid_t = self._t(pid, torch.long)

        def chunk(a):
            sl = slice(a, a + per)
            dr = R2[sl] / nr
            r = rr[None, :] * dr[:, None]                                       # (q, nr)
            xp = (x[sl, None, None] + r[:, :, None] * cphi[None, None, :]).reshape(-1)
            sp = (s[sl, None, None] + r[:, :, None] * sphi[None, None, :]).reshape(-1)
            q = r.shape[0]
            rep = lambda v: v[:, None].expand(-1, nr * nphi).reshape(-1)
            gz, gx = self._eval(H, obs_pt, rep(pid_t[sl]), xp, sp,
                                (rep(R1[sl]), rep(R2[sl]), torch.ones(q * nr * nphi, dtype=F64, device=self.device)), t)
            jac = (r * dr[:, None] * dphi)[:, :, None].expand(-1, -1, nphi).reshape(q, -1)
            return (gz.view(q, -1) * jac).sum(1), (gx.view(q, -1) * jac).sum(1)
        for a in range(0, pid.size, per):
            pz_, px_ = self._chunked(chunk, a)
            z = z.index_add(0, pid_t[a:a + per], pz_)
            zx = zx.index_add(0, pid_t[a:a + per], px_)
        return z, zx
