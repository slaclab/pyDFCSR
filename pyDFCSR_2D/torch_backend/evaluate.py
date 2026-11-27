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

    def __init__(self, sim, device='cpu', chunk=1 << 21):
        self.sim = sim
        self.device = torch.device(device)
        self.chunk = int(chunk)
        self.orbit = Orbit.from_lattice(sim.lattice, self.device)
        self.stats = {}

    def _t(self, a, dtype=F64):
        return torch.as_tensor(np.asarray(a), dtype=dtype, device=self.device)

    # ------------------------------------------------------------------ layout (numpy, per point)
    def _layouts(self, s_arr, x_arr):
        sim = self.sim
        ip = sim.integration_params
        t = sim.beam.position
        nx = ip.xbins
        cols = {k: [] for k in ('pt', 'grp', 'lo', 'w', 'sp', 'ws', 'wc', 'R1', 'R2', 'ki', 'adapt')}
        patch, vxs = [], []
        g = 0
        for i, (s, x) in enumerate(zip(s_arr, x_arr)):
            lay = sim.wake_layout(s, x)
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
        return torch.stack([s, x, self._t(vxs)] + list(vec), dim=-1)      # (P, 9)

    def _eval(self, H, obs_pt, pt, xp, sp, taper):
        """gz, gx at nodes; pt: (n,) point index into obs_pt; taper: (R1, R2, ki) per node."""
        ip = self.sim.integration_params
        obs = obs_pt[pt].unbind(-1)
        return integrand(H, self.orbit, obs, xp, sp, float(self.sim.beam.position), taper,
                         ip.transverse_terms == 'W1')

    def _columns(self, H, obs_pt, C, idx, offs, h):
        """sum_k f(lo + offs_k * h) over the nodes of the columns idx, for W and F."""
        n_off = offs.numel()
        per = max(1, self.chunk // n_off)
        Sz = torch.empty(idx.numel(), dtype=F64, device=self.device)
        Sx = torch.empty_like(Sz)
        for a in range(0, idx.numel(), per):
            j = idx[a:a + per]
            xp = (C['lo'][j, None] + offs[None, :] * h[a:a + per, None]).reshape(-1)
            rep = lambda v: v[j, None].expand(-1, n_off).reshape(-1)
            gz, gx = self._eval(H, obs_pt, rep(C['pt']), xp, rep(C['sp']),
                                (rep(C['R1']), rep(C['R2']), rep(C['ki'])))
            Sz[a:a + per] = gz.view(-1, n_off).sum(1)
            Sx[a:a + per] = gx.view(-1, n_off).sum(1)
        return Sz, Sx

    def compute(self, s_arr, x_arr):
        sim = self.sim
        ip = sim.integration_params
        t0 = time.time()
        H = History.from_tracker(sim.DF_tracker, self.device)
        cat, n_grp, patch, vxs, nx = self._layouts(s_arr, x_arr)
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
            self._refine(H, obs_pt, C, cat, n_grp, Tz, Tx, h, nx)
        ws = self._t(cat['ws'])
        pt = C['pt']
        dE = torch.zeros(P, dtype=F64, device=self.device).index_add_(0, pt, ws * Tz) if ncol else torch.zeros(P, dtype=F64, device=self.device)
        xk = torch.zeros(P, dtype=F64, device=self.device).index_add_(0, pt, ws * Tx) if ncol else torch.zeros(P, dtype=F64, device=self.device)
        pz, px = self._patch(H, obs_pt, patch, P)
        dE = -sim.CSR_scaling * (dE + pz)
        xk = sim.CSR_scaling * (xk + px)
        out = dE.cpu().numpy(), xk.cpu().numpy()
        self.stats = dict(columns=ncol, points=P, t_layout=t_layout, t_eval=time.time() - t0 - t_layout)
        return out

    def _weighted_columns(self, H, obs_pt, C, idx, frac, trap, width):
        """sum_k trap_k f(lo + frac_k * width) per column (the base pass)."""
        n_off = frac.numel()
        per = max(1, self.chunk // n_off)
        Sz = torch.empty(idx.numel(), dtype=F64, device=self.device)
        Sx = torch.empty_like(Sz)
        for a in range(0, idx.numel(), per):
            j = idx[a:a + per]
            xp = (C['lo'][j, None] + frac[None, :] * width[j, None]).reshape(-1)
            rep = lambda v: v[j, None].expand(-1, n_off).reshape(-1)
            gz, gx = self._eval(H, obs_pt, rep(C['pt']), xp, rep(C['sp']),
                                (rep(C['R1']), rep(C['R2']), rep(C['ki'])))
            Sz[a:a + per] = (gz.view(-1, n_off) * trap).sum(1)
            Sx[a:a + per] = (gx.view(-1, n_off) * trap).sum(1)
        return Sz, Sx

    def _refine(self, H, obs_pt, C, cat, n_grp, Tz, Tx, h, nx):
        """_adapt_x_error for every refined group at once (Tz, Tx updated in place)."""
        ip = self.sim.integration_params
        active = np.flatnonzero(cat['adapt'])
        if active.size == 0:
            return
        grp = cat['grp']
        wc = cat['wc']
        h_c = h.clone()
        m = nx - 1
        for rnd in range(max(ip.x_adapt_rounds, 1)):
            if active.size == 0:
                break
            mm = m * 2 ** rnd
            offs = torch.arange(mm, dtype=F64, device=self.device) + 0.5
            idx = self._t(active, torch.long)
            Sz, Sx = self._columns(H, obs_pt, C, idx, offs, h_c[idx])
            Tz_new = 0.5 * Tz[idx] + 0.5 * h_c[idx] * Sz
            Tx_new = 0.5 * Tx[idx] + 0.5 * h_c[idx] * Sx
            ez = ((Tz_new - Tz[idx]).abs() / 3.0).cpu().numpy()
            ex = ((Tx_new - Tx[idx]).abs() / 3.0).cpu().numpy()
            Tz[idx] = Tz_new
            Tx[idx] = Tx_new
            h_c[idx] = h_c[idx] * 0.5
            g = grp[active]
            keep = np.zeros(active.size, dtype=bool)
            for e, T in ((ez, Tz), (ex, Tx)):
                budget = ip.x_adapt_tol * np.bincount(grp, wc * np.abs(T.cpu().numpy()), minlength=n_grp)
                keep |= _select(g, wc[active] * e, budget)
            active = active[keep]

    def _patch(self, H, obs_pt, patch, P):
        """The polar patch of every point that has one (_integrate_near_patch)."""
        z = torch.zeros(P, dtype=F64, device=self.device)
        zx = torch.zeros(P, dtype=F64, device=self.device)
        if not patch:
            return z, zx
        ip = self.sim.integration_params
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
        for a in range(0, pid.size, per):
            sl = slice(a, a + per)
            dr = R2[sl] / nr
            r = rr[None, :] * dr[:, None]                                       # (q, nr)
            xp = (x[sl, None, None] + r[:, :, None] * cphi[None, None, :]).reshape(-1)
            sp = (s[sl, None, None] + r[:, :, None] * sphi[None, None, :]).reshape(-1)
            q = r.shape[0]
            rep = lambda v: v[:, None].expand(-1, nr * nphi).reshape(-1)
            gz, gx = self._eval(H, obs_pt, rep(pid_t[sl]), xp, sp,
                                (rep(R1[sl]), rep(R2[sl]), torch.ones(q * nr * nphi, dtype=F64, device=self.device)))
            jac = (r * dr[:, None] * dphi)[:, :, None].expand(-1, -1, nphi).reshape(q, -1)
            z[pid_t[sl]] += (gz.view(q, -1) * jac).sum(1)
            zx[pid_t[sl]] += (gx.view(q, -1) * jac).sum(1)
        return z, zx
