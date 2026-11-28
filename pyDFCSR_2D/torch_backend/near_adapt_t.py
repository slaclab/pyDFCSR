"""
The near-adapt node construction (LayoutMixin._g_near_adapt_nodes) for all mesh points at once, for
the batched layout of the torch backend (CSR_computation.torch_bands).

Per point the algorithm is the per-point one: scan the column's retarded source at the fixed
near-region nodes, bisect every interval whose retarded z moves by more than near_adapt_scan_dz *
res_z inside the history's z support (or that holds a support edge), at most 4 rounds and
near_adapt_max_scan points, then equidistribute the monitor 1/h_fixed + |d(rho/rho_max)/ds'|/eps,
graded to near_adapt_max_ratio. Here every round handles all points together: one band solve (on
the device, through sim._band_solver) and one density lookup for all of them, and the interval
bookkeeping on flat arrays with a segment id per point. Only the final per-point steps (gradient,
grading, inverse CDF) loop over the points, which costs little. The result agrees with the
per-point code to round-off (the summation order of a few reductions differs); the numba path keeps
the per-point code and stays bit-identical.
"""
import math

import numpy as np
from numba import njit

from ..layout import _grade_log_monitor


@njit(cache=True)
def _final_nodes(sp, rho, bounds, s_obs, a_obs, b_obs, base_n, floor, g, use_floor, eps, lr, max_nodes):
    """
    The final steps of _g_near_adapt_nodes for every segment (one per point): the monitor
    1/h_fixed + |d(rho/rho_max)/ds'|/eps (np.gradient's formula), graded, its cumulative integral F
    and the inverse CDF at max(n, base size) equidistant values of F. Returns the nodes (flat), the
    count per segment (0: keep the base layout) and whether any count was capped.
    """
    n_seg = bounds.size - 1
    counts = np.zeros(n_seg, dtype=np.int64)
    want = np.zeros(n_seg, dtype=np.int64)
    capped = False
    Fs = np.zeros(sp.size)
    for j in range(n_seg):
        lo, hi = bounds[j], bounds[j + 1]
        n = hi - lo
        rmax = 0.0
        for i in range(lo, hi):
            if rho[i] > rmax:
                rmax = rho[i]
        if not rmax > 0.0 or n < 2:
            continue
        m = np.empty(n)
        for i in range(n):
            f_i = sp[lo + i]
            if use_floor:
                h = max(floor, g * abs(f_i - s_obs[j]))
            else:
                h = (b_obs[j] - a_obs[j]) / max(base_n[j] - 1, 1)
            if i == 0:
                gr = (rho[lo + 1] - rho[lo]) / rmax / (sp[lo + 1] - sp[lo])
            elif i == n - 1:
                gr = (rho[hi - 1] - rho[hi - 2]) / rmax / (sp[hi - 1] - sp[hi - 2])
            else:
                d1 = sp[lo + i] - sp[lo + i - 1]
                d2 = sp[lo + i + 1] - sp[lo + i]
                gr = (-(d2) / (d1 * (d1 + d2)) * rho[lo + i - 1] + (d2 - d1) / (d1 * d2) * rho[lo + i]
                      + d1 / (d2 * (d1 + d2)) * rho[lo + i + 1]) / rmax
            gr = abs(gr)
            if not math.isfinite(gr):
                gr = 0.0
            m[i] = 1.0 / h + gr / eps
        lm = _grade_log_monitor(np.log(m), sp[lo + 1:hi] - sp[lo:hi - 1], lr)
        F = 0.0
        Fs[lo] = 0.0
        for i in range(1, n):
            F += 0.5 * (math.exp(lm[i]) + math.exp(lm[i - 1])) * (sp[lo + i] - sp[lo + i - 1])
            Fs[lo + i] = F
        if not math.isfinite(F):
            continue
        k = int(math.ceil(F)) + 1
        if k > max_nodes:
            capped = True
            k = max_nodes
        want[j] = max(k, base_n[j])
    total = 0
    for j in range(n_seg):
        total += want[j]
    nodes = np.empty(total)
    pos = 0
    for j in range(n_seg):
        k = want[j]
        if k == 0:
            continue
        lo, hi = bounds[j], bounds[j + 1]
        Fj = Fs[lo:hi]
        q = np.linspace(0.0, Fj[-1], k)
        nodes[pos:pos + k] = np.interp(q, Fj, sp[lo:hi])
        counts[j] = k
        pos += k
    return nodes, counts, capped


def _scan(sim, solve, s_obs, x_obs, t, sp, n_iter):
    """_g_column_scan for flat columns (one observer per column): (z_ret, rho) at the band axis."""
    _, branches = solve(s_obs, x_obs, t, sp, n_iter=n_iter, scale_bands=True)
    zr = np.full(sp.shape, np.nan)
    rho = np.zeros(sp.shape)
    for br in branches:
        live = br['live']
        if not live.any():
            continue
        t_ret = t - br['r'][live]
        z_ret = sp[live] - t_ret
        r_b = sim._comoving_fields(br['xp'][live], z_ret, t_ret)[0]
        take = np.zeros(sp.shape, dtype=bool)
        take[live] = r_b >= rho[live]
        rho[take] = r_b[take[live]]
        zr[take] = z_ret[take[live]]
    return zr, rho


def near_adapt_batch(sim, reqs):
    """Near-region nodes for the requests [(s, x, t, a, b, base), ...]; returns a list of arrays."""
    ip = sim.integration_params
    tr = sim.DF_tracker
    out = [np.asarray(r[5], dtype=float) for r in reqs]
    res_z = sim._deposition_resolution()[0]
    if not res_z > 0:
        return out
    solve = sim._band_solver
    t = reqs[0][2]
    delta = ip.near_adapt_scan_dz * res_z
    n_hist = len(tr.time_log)
    zb, sz = np.asarray(tr.z_bar_arr)[:n_hist], np.asarray(tr.sigma_z_arr)[:n_hist]
    z_lo, z_hi = float(np.min(zb + tr.w_start * sz)), float(np.max(zb + tr.w_end * sz))
    act = [i for i, r in enumerate(reqs) if np.size(r[5]) >= 3]
    if not act:
        return out
    S = np.array([reqs[i][0] for i in act])
    X = np.array([reqs[i][1] for i in act])
    # flat arrays of all active points: segment id, s', and the scan
    seg = np.concatenate([np.full(np.size(reqs[i][5]), j) for j, i in enumerate(act)])
    sp = np.concatenate([np.asarray(reqs[i][5], dtype=float) for i in act])
    zr, rho = _scan(sim, solve, S[seg], X[seg], t, sp, ip.near_adapt_scan_iter)
    n_seg = len(act)
    done = np.zeros(n_seg, dtype=bool)
    for _ in range(4):
        size = np.bincount(seg, minlength=n_seg)
        done |= size >= ip.near_adapt_max_scan
        same = seg[1:] == seg[:-1]                          # interval j = (j, j+1) inside a segment
        ds = np.diff(sp)
        dz = np.abs(np.diff(zr))
        lo = np.fmin(zr[:-1], zr[1:])
        hi = np.fmax(zr[:-1], zr[1:])
        edge = (np.isnan(zr[:-1]) != np.isnan(zr[1:])) & (ds > delta)
        need = same & (((dz > delta) & (hi > z_lo) & (lo < z_hi)) | edge) & ~done[seg[:-1]]
        sg = seg[:-1][need]
        done |= np.bincount(sg, minlength=n_seg) == 0       # a segment with nothing to split stops
        if not need.any():
            break
        k = np.where(edge, 2, np.minimum(np.ceil(np.nan_to_num(dz, nan=0.0) / delta), 64)).astype(int)[need]
        room = np.maximum(ip.near_adapt_max_scan - size, 0)
        want = np.bincount(sg, weights=k - 1, minlength=n_seg)
        over = want > room
        if over.any():
            f = room / np.maximum(want, 1)
            k = np.where(over[sg], np.maximum(1, np.floor(k * f[sg])), k).astype(int)
        a0, a1 = sp[:-1][need], sp[1:][need]
        idx = np.repeat(np.arange(k.size), k - 1)
        j = np.arange(idx.size) - np.repeat(np.cumsum(k - 1) - (k - 1), k - 1) + 1
        new = a0[idx] + (a1[idx] - a0[idx]) * j / k[idx]
        if new.size == 0:
            break
        nseg = sg[idx]
        zm, rm = _scan(sim, solve, S[nseg], X[nseg], t, new, ip.near_adapt_scan_iter)
        # merge per segment, sorted, duplicates dropped (the old point kept), as np.unique does
        seg_all = np.concatenate([seg, nseg])
        sp_all = np.concatenate([sp, new])
        order = np.lexsort((sp_all, seg_all))
        seg_all, sp_all = seg_all[order], sp_all[order]
        keep = np.r_[True, (seg_all[1:] != seg_all[:-1]) | (sp_all[1:] != sp_all[:-1])]
        seg, sp = seg_all[keep], sp_all[keep]
        zr = np.concatenate([zr, zm])[order][keep]
        rho = np.concatenate([rho, rm])[order][keep]
    if (np.bincount(seg, minlength=n_seg) >= ip.near_adapt_max_scan).any():
        sim._note('adapt_scan', f'  [near_adapt] WARNING at s = {t:.4f} m: the column scan hit '
                                f'near_adapt_max_scan = {ip.near_adapt_max_scan} points before resolving '
                                f'{delta*1e6:.3g} um of retarded z per interval.', every_rank=True)
    # the final per-point steps of _g_near_adapt_nodes, in one numba pass over the segments
    bounds = np.r_[0, np.cumsum(np.bincount(seg, minlength=n_seg))].astype(np.int64)
    floor = ip.near_cell * sim.beam._sigma_x_transform
    g = ip.near_grade
    use_floor = bool(ip.near_cell and g and floor > 0.0)
    A = np.array([reqs[i][3] for i in act], dtype=float)
    B = np.array([reqs[i][4] for i in act], dtype=float)
    base_n = np.array([np.size(reqs[i][5]) for i in act], dtype=np.int64)
    nodes, counts, capped = _final_nodes(sp, rho, bounds, S.astype(float), A, B, base_n, float(floor), float(g),
                                         use_floor, float(ip.near_adapt_eps), float(np.log(ip.near_adapt_max_ratio)),
                                         int(ip.near_adapt_max_nodes))
    # nodes + [a, s, b], sorted and unique per segment, inside [a, b]
    has = counts > 0
    nseg_id = np.repeat(np.arange(n_seg), counts)
    extra_id = np.repeat(np.flatnonzero(has), 3)
    extra = np.stack([A, S, B], axis=1)[has].ravel()
    ids = np.concatenate([nseg_id, extra_id])
    vals = np.concatenate([nodes, extra])
    order = np.lexsort((vals, ids))
    ids, vals = ids[order], vals[order]
    keep = np.r_[True, (ids[1:] != ids[:-1]) | (vals[1:] != vals[:-1])] & (vals >= A[ids]) & (vals <= B[ids])
    ids, vals = ids[keep], vals[keep]
    cuts = np.searchsorted(ids, np.arange(n_seg + 1))
    for jj in np.flatnonzero(has):
        out[act[jj]] = vals[cuts[jj]:cuts[jj + 1]]
    if capped:
        sim._note('adapt_nodes', f'  [near_adapt] WARNING at s = {t:.4f} m: near-region nodes capped at '
                                 f'near_adapt_max_nodes = {ip.near_adapt_max_nodes}.', every_rank=True)
    return out
