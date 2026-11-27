"""
Where the wake integral puts its s' columns: the far / near regions and their extent, the extra
regions below the far edge (previous bend, straight section, causal drift region), the node
constructions in each, and the polar-patch radii (docs/DFCSR_2D_methods.md §5.2, §5.4).

Mixed into CSR.CSR2D; the methods use its state (beam, lattice, DF_tracker, integration_params).
"""
import math
from collections import namedtuple

import numpy as np
from numba import njit

# A band solve the layout needs: _retarded_xi_bands(s, x, t, sp, n_iter, scale_bands). The layout
# functions below are generators (prefix _g_) that yield these and receive (bands, branch records);
# LayoutMixin._drive answers them one by one (the plain methods, unchanged results), and
# _drive_many answers the pending requests of many observers in one solve (the batched layout).
BandRequest = namedtuple('BandRequest', 's x t sp n_iter scale_bands')


@njit(cache=True)
def _grade_log_monitor(lm, dg, lr):
    """Two forward/backward passes limiting |d ln m| per cell to lr (_g_near_adapt_nodes)."""
    for _ in range(2):
        for i in range(1, lm.size):
            lm[i] = max(lm[i], lm[i - 1] - lr * dg[i - 1] * math.exp(lm[i - 1]))
        for i in range(lm.size - 2, -1, -1):
            lm[i] = max(lm[i], lm[i + 1] - lr * dg[i] * math.exp(lm[i + 1]))
    return lm


def graded_u(u_near, u_far, g, k_min=0, close=False):
    """
    Distances u = u_near (1 + g)^k, k = 0..K, capped at u_far: geometric grading in the distance from
    the observer (equal steps in ln u), for sources whose contribution falls off smoothly with u.
    close: make sure the last node is exactly u_far.
    """
    k = max(int(np.ceil(np.log(u_far / u_near) / np.log1p(g))), k_min)
    u = np.minimum(u_near * (1.0 + g) ** np.arange(k + 1), u_far)
    if close and u[-1] < u_far:
        u = np.append(u, u_far)
    return u


class LayoutMixin:

    def _solve_request(self, req):
        bands = self._retarded_xi_bands(req.s, req.x, req.t, req.sp, n_iter=req.n_iter,
                                        scale_bands=req.scale_bands)
        return bands, self._branch_diag['branches']

    def _drive(self, gen):
        """Run one layout generator, answering each band request with its own solve."""
        try:
            req = next(gen)
            while True:
                req = gen.send(self._solve_request(req))
        except StopIteration as stop:
            return stop.value

    def _drive_many(self, gens):
        """
        Run many layout generators in lockstep: all pending requests with the same (n_iter,
        scale_bands, t) are concatenated into one solve (_retarded_xi_bands is elementwise in the
        columns, with the observer given per column), and the bands and branch records are split
        back. Every generator receives what its own solve would have returned.
        """
        out = [None] * len(gens)
        pending = {}
        for i, g in enumerate(gens):
            try:
                pending[i] = next(g)
            except StopIteration as stop:
                out[i] = stop.value
        while pending:
            groups = {}
            for i, req in pending.items():
                groups.setdefault((req.n_iter, req.scale_bands, req.t), []).append(i)
            nxt = {}
            for (n_iter, scale, t), ids in groups.items():
                reqs = [pending[i] for i in ids]
                sizes = [np.size(r.sp) for r in reqs]
                sp = np.concatenate([np.atleast_1d(np.asarray(r.sp, dtype=float)) for r in reqs])
                so = np.concatenate([np.full(n, float(r.s)) for r, n in zip(reqs, sizes)])
                xo = np.concatenate([np.full(n, float(r.x)) for r, n in zip(reqs, sizes)])
                bands = self._retarded_xi_bands(so, xo, t, sp, n_iter=n_iter, scale_bands=scale)
                diag = self._branch_diag['branches']
                a = 0
                for i, r, n in zip(ids, reqs, sizes):
                    b = a + n
                    cut = lambda v: v[a:b] if np.ndim(v) else v
                    res = ([(lo[a:b], hi[a:b]) for lo, hi in bands],
                           [{k: cut(v) for k, v in br.items()} for br in diag])
                    try:
                        nxt[i] = gens[i].send(res)
                    except StopIteration as stop:
                        out[i] = stop.value
                    a = b
            pending = nxt
        return out

    def _band_alive(self, s, x, t, sp):
        return self._drive(self._g_band_alive(s, x, t, sp))

    def _g_band_alive(self, s, x, t, sp):
        """Whether the retarded band has nonzero width, for each s' in the array."""
        sp = np.atleast_1d(np.asarray(sp, float))
        bands = self._disjoint_bands((yield BandRequest(s, x, t, sp, 12, True))[0])
        w = np.zeros(sp.size)
        for lo, hi in bands:
            w += np.maximum(hi - lo, 0.0)
        return w > 0.0

    def _causal_edge(self, s, x, t, sp_far, sp_near, n_scan=64):
        return self._drive(self._g_causal_edge(s, x, t, sp_far, sp_near, n_scan))

    def _g_causal_edge(self, s, x, t, sp_far, sp_near, n_scan=64):
        """
        Lowest s' in [sp_far, sp_near] whose retarded band still has nonzero width.

        Below this the retarded source has slipped outside the bunch and every column is
        identically zero. One vectorised scan; the last dead node below the first live one is
        returned, conservatively (truncating a live domain is a one-sided error). Not a
        root-find, a closed form or a refinement -- see docs/code_notes.md.
        Returns sp_far if the first sample is already alive, None if the whole interval is dead.
        """
        grid = np.linspace(sp_far, sp_near, n_scan)
        alive = yield from self._g_band_alive(s, x, t, grid)
        if not alive.any():
            return None
        if alive[0]:
            return sp_far
        return grid[max(int(np.argmax(alive)) - 1, 0)]

    def _layout_bounds(self, s, x):
        """
        The two s' regions, far (s1, s3) and near (s3, s4), for the observer at (s, x).

        Only the extent is decided here; which Eq 4.24 branches exist is a per-column property of
        the retarded tilt, left to _retarded_xi_bands. Everything is keyed on sin 2a and cos 2a,
        which are pole-free, never on tan 2a or |tau| (docs/DFCSR_2D_methods.md §5.2,
        docs/code_notes.md).
        """
        ip = self.integration_params
        b = self.beam
        sigma_z, sigma_x = b._sigma_z, b._sigma_x
        # the tilt of the density frame (core fit), not beams.slope (all particles)
        tau = self._frame_tilt()

        sin2a = 2.0 * tau / (1.0 + tau * tau)
        cos2a = (1.0 - tau * tau) / (1.0 + tau * tau)

        # Near-region length d: where the chirp branch leaves the beam, one continuous expression
        # saturated at branch_sin_min where the branches merge, floored at near_floor sigma_z so the
        # near region always holds the narrow band and the pole (docs/code_notes.md).
        d = ((10.0 * sigma_x + x - b._mean_x) * abs(cos2a)
             / max(abs(sin2a), ip.branch_sin_min))
        d = max(d, ip.near_floor * sigma_z)
        reach = ip.n_formation_length * self.formation_length
        if (ip.seam_ratio and ip.near_grade
                and not (ip.causal_clip and self._exit_transient())):
            # seam continuity: the near region's graded cell reaches 1/seam_ratio of the far region's
            # flat cell before the seam (bends only; past an exit the far layout is graded already)
            far_cell = ip.n_formation_length * self.formation_length / max(2 * ip.far_zbins, 3)
            d = max(d, far_cell / (ip.near_grade * ip.seam_ratio))
        # and never reach back further than the retained history supports
        d = min(max(d, 0.0), reach)

        s4 = s + 3.0 * sigma_z
        # the far reach may extend onto the synthetic straight pre-history s' in [-prehistory, 0)
        lo = -ip.prehistory
        s3 = max(lo, s - d)
        s1 = max(lo, s3 - reach)
        # drift_reach: past a bend the exit wake's source is the arc itself, so the far edge reaches
        # back to its entrance (the causal-edge scan trims it)
        s_bend = self._drift_reach_start()
        if s_bend is not None:
            if ip.drift_causal_reach:
                # the n_fl = 1 reach in a drift whatever n_fl is; earlier sources: _drift_causal_nodes
                s1 = max(lo, s3 - self.formation_length)
            s1 = min(s1, s_bend)
        return ((s1, s3), (s3, s4))

    def _region_node_counts(self, bounds):
        """
        Longitudinal node count for each s' region, near region last.

        A flat count per region is wrong once the regions differ in length by ~100x.
        The near region sets the requirement: its cell must be ~sigma_xi (measured to
        collapse across tilt amplification 66x and 167x), while its length grows with
        tilt, so its node count scales as sigma_x/sigma_xi. The far regions need far
        fewer nodes -- the far edge is invariant to machine precision there.

        near_cell = 0 restores the historical behaviour of `zbins` everywhere.
        """
        ip = self.integration_params
        if not ip.near_cell:
            self._last_region_nodes = [ip.zbins] * len(bounds)
            return self._last_region_nodes
        target = ip.near_cell * self.beam._sigma_x_transform
        counts = [ip.far_zbins] * len(bounds)
        a, b = bounds[-1]
        want = int(round(abs(b - a) / target))
        cap = 100 * ip.zbins
        # the cap can bind only without grading (with near_grade the count is not used)
        graded = bool(ip.near_grade)
        if want > cap and not graded:
            self._note('near_cap', f'WARNING: near-region nodes capped at {cap} (wanted {want}); near-region '
                                   f'cell is {want/cap:.1f}x coarser than near_cell requests. Raise zbins, '
                                   f'raise near_cell, or use graded nodes.', every_rank=True)
        counts[-1] = int(np.clip(want, 3, cap))
        self._last_region_nodes = counts
        return counts

    def _far_region_nodes(self, s, x, t, sp_far, sp_near, transient=None):
        return self._drive(self._g_far_region_nodes(s, x, t, sp_far, sp_near, transient))

    def _g_far_region_nodes(self, s, x, t, sp_far, sp_near, transient=None):
        """
        s' nodes for the merged far region, spanning sp_far (= s1, the causal reach
        s - n_formation_length*L_f) up to sp_near (= s3, where the near region starts).

        In a bend: 2*far_zbins uniform columns (not graded: at high tilt most of the far integral sits
        near the far end). Past a bend exit: the causal edge, a uniform fine window of
        far_window*L_f at far_cell*sigma_z above it, then graded towards the near region
        (docs/DFCSR_2D_methods.md §5.2, docs/code_notes.md).
        """
        ip = self.integration_params
        if sp_near <= sp_far:
            return np.array([sp_far, sp_near])

        if transient is None:
            transient = self._exit_transient()
        if not (ip.causal_clip and transient):
            n = max(2 * ip.far_zbins, 3)
            return self._record_far_nodes(np.linspace(sp_far, sp_near, n))

        lo = sp_far
        e = yield from self._g_causal_edge(s, x, t, sp_far, sp_near)
        if e is not None:
            lo = min(max(e, sp_far), sp_near)
            if e <= sp_far:
                # the reach, not causality, ended the region: counted
                self._count('far_clip_at_reach')

        parts = []
        top = lo
        if ip.far_window and ip.far_cell:
            top = min(lo + ip.far_window * self.formation_length, sp_near)
            cell = ip.far_cell * self.beam._sigma_z
            if ip.far_fine_max:
                # far_fine_max caps the fine window (short bunches asked for thousands of nodes)
                cell = max(cell, (top - lo) / ip.far_fine_max)
            n_fine = int(np.clip(round((top - lo) / cell) + 1, 3, 100 * ip.far_zbins))
            parts.append(np.linspace(lo, top, n_fine))
        if top < sp_near:
            g = ip.near_grade
            if g:
                u_hi, u_lo = s - top, max(s - sp_near, 1e-12)
                if u_hi > u_lo:
                    parts.append(s - graded_u(u_lo, u_hi, g))
            else:
                parts.append(np.linspace(top, sp_near, max(ip.far_zbins, 3)))
        if not parts:
            parts.append(np.linspace(lo, sp_near, max(ip.far_zbins, 3)))
        return self._record_far_nodes(np.unique(np.concatenate(parts)))

    def _record_far_nodes(self, sp):
        """Log the realised far node count, which is now derived rather than fixed."""
        self._last_region_nodes[0] = len(sp)
        return sp

    def _near_region_nodes(self, s, a, b, x=None, t=None):
        return self._drive(self._g_near_region_nodes(s, a, b, x, t))

    def _g_near_region_nodes(self, s, a, b, x=None, t=None):
        """
        s' nodes for the near region, graded so the cell grows with |s - s'|.

        The cell is  du = max(near_cell*sigma_xi, near_grade*u),  u = |s - s'|:
        an absolute floor set by the transverse support width close to the
        observation point, geometric growth outside. Built independently on each
        side, because the near region straddles s' = s.

        For u beyond the transverse offsets the column contribution falls as 1/u, so equal
        logarithmic intervals contribute equally: the node count grows as ln(u_max) with tilt
        instead of linearly.
        """
        ip = self.integration_params
        floor = ip.near_cell * self.beam._sigma_x_transform
        g = ip.near_grade
        if not ip.near_cell or not g or floor <= 0.0:
            base = np.linspace(a, b, self._region_node_counts(((a, b),))[-1])
            if ip.near_adapt and x is not None:
                return (yield from self._g_near_adapt_nodes(s, x, t, a, b, base))
            return base

        ustar = floor / g                      # where relative spacing takes over

        def side(umax):
            if umax <= 0.0:
                return np.zeros(1)
            u = np.arange(0.0, min(ustar, umax) + floor, floor)
            if umax > ustar:
                n = int(np.ceil(np.log(umax / ustar) / np.log1p(g)))
                u = np.concatenate((u, ustar * (1.0 + g) ** np.arange(1, n + 1)))
            return np.unique(np.minimum(u, umax))

        uL = side(s - a)
        uR = side(b - s)
        sp = np.concatenate((s - uL[::-1], s + uR))
        self._last_region_nodes[-1] = len(sp)
        sp = np.unique(sp)
        if ip.near_adapt and x is not None:
            return (yield from self._g_near_adapt_nodes(s, x, t, a, b, sp))
        return sp

    def _column_scan(self, s, x, t, sp):
        return self._drive(self._g_column_scan(s, x, t, sp))

    def _g_column_scan(self, s, x, t, sp):
        """
        The retarded source seen by the column at each s': (z_ret, rho) at the located band's
        axis point, the larger of the two Eq 4.24 branches. rho is 0 where no branch is live.
        """
        _, branches = yield BandRequest(s, x, t, sp, self.integration_params.near_adapt_scan_iter, True)
        zr = np.full(sp.shape, np.nan)
        rho = np.zeros(sp.shape)
        for br in branches:
            live = br['live']
            if not live.any():
                continue
            t_ret = t - br['r'][live]
            z_ret = sp[live] - t_ret
            r_b = self._comoving_fields(br['xp'][live], z_ret, t_ret)[0]
            take = np.zeros(sp.shape, dtype=bool)
            take[live] = r_b >= rho[live]
            rho[take] = r_b[take[live]]
            zr[take] = z_ret[take[live]]
        return zr, rho

    def _near_adapt_nodes(self, s, x, t, a, b, base):
        return self._drive(self._g_near_adapt_nodes(s, x, t, a, b, base))

    def _g_near_adapt_nodes(self, s, x, t, a, b, base):
        """
        Near-region s' nodes adapted to the retarded density along the column (Integration_params
        near_adapt). Starts from the fixed layout `base`, scans z_ret and rho along the column and
        bisects every interval whose z_ret still moves by more than near_adapt_scan_dz * res_z
        inside the history's z support, then equidistributes the monitor
            m(s') = 1 / h_fixed(s') + |d(rho / rho_max) / ds'| / near_adapt_eps,
        graded so neighbouring cells differ by at most near_adapt_max_ratio. The first term keeps
        every cell at least as fine as the fixed layout; the second puts ~2/eps nodes across each
        full-height rise and fall of the source, wherever it is and however many there are.
        Returns `base` unchanged when the column sees no charge.
        """
        ip = self.integration_params
        tr = self.DF_tracker
        if base.size < 3:
            return base
        res_z = self._deposition_resolution()[0]
        if not res_z > 0:
            return base
        delta = ip.near_adapt_scan_dz * res_z
        n_hist = len(tr.time_log)
        zb, sz = np.asarray(tr.z_bar_arr)[:n_hist], np.asarray(tr.sigma_z_arr)[:n_hist]
        z_lo, z_hi = float(np.min(zb + tr.w_start * sz)), float(np.max(zb + tr.w_end * sz))

        sp = np.asarray(base, dtype=float)
        zr, rho = yield from self._g_column_scan(s, x, t, sp)
        # Each round splits every interval that still needs it into ceil(|dz_ret| / delta) pieces
        # at once (z_ret is close to linear within an interval), so 2-3 rounds suffice; a round is
        # one band solve, ~4 ms of Python overhead whatever its size.
        for _ in range(4):
            ds = np.diff(sp)
            dz = np.abs(np.diff(zr))
            lo = np.fmin(zr[:-1], zr[1:])
            hi = np.fmax(zr[:-1], zr[1:])
            # a dead end (nan) next to a live one: the support edge lies inside; locate it to delta
            edge = (np.isnan(zr[:-1]) != np.isnan(zr[1:])) & (ds > delta)
            need = ((dz > delta) & (hi > z_lo) & (lo < z_hi)) | edge
            if not need.any() or sp.size >= ip.near_adapt_max_scan:
                break
            k = np.where(edge, 2, np.minimum(np.ceil(np.nan_to_num(dz, nan=0.0) / delta), 64)).astype(int)[need]
            room = max(ip.near_adapt_max_scan - sp.size, 0)
            if (k - 1).sum() > room:
                k = np.maximum(1, np.floor(k * room / max((k - 1).sum(), 1))).astype(int)
            a0, a1 = sp[:-1][need], sp[1:][need]
            new = np.concatenate([a0[i] + (a1[i] - a0[i]) * np.arange(1, k[i]) / k[i]
                                  for i in range(need.sum())]) if need.any() else np.zeros(0)
            if new.size == 0:
                break
            zm, rm = yield from self._g_column_scan(s, x, t, new)
            # unique: a split point can coincide with an existing one to round-off, and a zero
            # spacing would put inf/nan into the gradient monitor below
            sp, order = np.unique(np.concatenate([sp, new]), return_index=True)
            zr = np.concatenate([zr, zm])[order]
            rho = np.concatenate([rho, rm])[order]
        if sp.size >= ip.near_adapt_max_scan:
            self._note('adapt_scan', f'  [near_adapt] WARNING at s = {t:.4f} m: the column scan hit '
                                     f'near_adapt_max_scan = {ip.near_adapt_max_scan} points before resolving '
                                     f'{delta*1e6:.3g} um of retarded z per interval.', every_rank=True)
        rho_max = rho.max()
        if not rho_max > 0:
            return base

        floor = ip.near_cell * self.beam._sigma_x_transform
        g = ip.near_grade
        if ip.near_cell and g and floor > 0.0:
            h_fixed = np.maximum(floor, g * np.abs(sp - s))
        else:
            h_fixed = np.full(sp.shape, (b - a) / max(base.size - 1, 1))
        grad = np.abs(np.gradient(rho / rho_max, sp))
        m = 1.0 / h_fixed + np.where(np.isfinite(grad), grad, 0.0) / ip.near_adapt_eps
        # grading on the non-uniform scan grid: with h = 1/m, |d ln m| per cell <= ln(max_ratio)
        lr = np.log(ip.near_adapt_max_ratio)
        dg = np.diff(sp)
        lm = _grade_log_monitor(np.log(m), dg, lr)
        m = np.exp(lm)
        F = np.concatenate([[0.0], np.cumsum(0.5 * (m[1:] + m[:-1]) * dg)])
        if not np.isfinite(F[-1]):
            return base
        n = int(np.ceil(F[-1])) + 1
        if n > ip.near_adapt_max_nodes:
            self._note('adapt_nodes', f'  [near_adapt] WARNING at s = {t:.4f} m: {n} near-region nodes wanted, '
                                      f'capped at near_adapt_max_nodes = {ip.near_adapt_max_nodes}.', every_rank=True)
            n = ip.near_adapt_max_nodes
        nodes = np.interp(np.linspace(0.0, F[-1], max(n, base.size)), F, sp)
        nodes = np.unique(np.concatenate([nodes, [a, s, b]]))
        nodes = nodes[(nodes >= a) & (nodes <= b)]
        self._last_region_nodes[-1] = len(nodes)
        self._last_adapt = dict(scan=sp.size, nodes=len(nodes), base=base.size)
        return nodes

    def _near_patch_radii(self, s, s3, s4):
        """
        (R1, R2) of the polar near-field patch, or None if it is disabled.

        R2 is clipped so the disc stays strictly inside the s' region that owns it,
        which keeps the partition of unity exact: the Cartesian piece carries weight
        w, the disc carries 1 - w, and 1 - w vanishes for |r - r'| > R2.
        """
        n_sig = self.integration_params.near_patch
        if not n_sig:
            return None
        R2 = n_sig * self.beam._sigma_x_transform
        R2 = min(R2, 0.4 * (s4 - s), 0.4 * (s - s3))
        if R2 <= 0:
            return None
        return 0.5 * R2, R2

    def _extra_region_nodes(self, s, x, t, s1):
        return self._drive(self._g_extra_region_nodes(s, x, t, s1))

    def _g_extra_region_nodes(self, s, x, t, s1):
        """
        s' nodes of the regions below the regular far edge s1, in integration order:
          prev_bend_reach     inside a bend, the previous bend's exit tail, from its entrance
          reach_straight      inside a bend, the straight section before this bend's entrance
          drift_causal_reach  in the drift after a bend, the earlier sources from the causal edge
        """
        out = []
        pb = self._prev_bend_reach_start()
        if pb is not None and pb < s1:
            out.append(self._prev_region_nodes(s, pb, s1))
        lo_pre = self._straight_region_lo(s)
        if lo_pre is not None:
            lo_cur = min(s1, pb) if pb is not None else s1
            lo_cur = min(lo_cur, float(self.bend_start_rec))      # never the arc itself
            if lo_pre < lo_cur - 1e-12:
                out.append(self._straight_region_nodes(s, lo_pre, lo_cur))
        sp_drift = yield from self._g_drift_causal_nodes(s, x, t, s1)
        if sp_drift is not None:
            out.append(sp_drift)
        return out

    def _prev_region_nodes(self, s, pb, s1):
        """
        s' nodes of the prev_bend_reach region [pb, s1]: log-graded in u = s - s' at
        prev_bend_grade. Measured at the chicane's B4 entrance against a 4001-node
        reference, the region is a smooth 1/u-like tail of the drift history (53% within
        0.5 m of s1, 90% within 2 m), not a narrow band: grading 0.02 gives 113 nodes and
        <= 0.4% error, where the exit-transient layout used 544 nodes for 0.1%.
        """
        return np.unique(s - graded_u(max(s - s1, 1e-12), s - pb, self.integration_params.prev_bend_grade))

    def _straight_region_lo(self, s):
        """
        Lower s' bound of the straight-section extension for an observer at s, or None:
        s - _geo_reach(s), but never below the previous bend's exit (the section before the
        entrance is straight only from there; the previous bend's own tail is prev_bend_reach's
        business), never below the retained history unless that is the synthetic pre-history,
        and never below -prehistory.
        """
        g = self._geo_reach(s)
        if g is None:
            return None
        lo = s - g
        lo = max(lo, -self.integration_params.prehistory)
        pe = self.prev_bend_end
        if pe is not None:
            lo = max(lo, float(pe))
        tl = self.DF_tracker.time_log
        if len(tl) and float(tl[0]) > 1e-12:
            lo = max(lo, float(tl[0]))
        return lo

    def _straight_region_nodes(self, s, lo, hi):
        """
        s' nodes of the straight-section region [lo, hi): the column integral is a smooth ~1/u^2 there,
        so it is graded in u at 1 + near_grade (a few dozen nodes for 10 m).
        """
        g = self.integration_params.near_grade or 0.05
        return np.unique(s - graded_u(max(s - hi, 1e-9), s - lo, g, close=True))

    def _drift_causal_nodes(self, s, x, t, s1):
        return self._drive(self._g_drift_causal_nodes(s, x, t, s1))

    def _g_drift_causal_nodes(self, s, x, t, s1):
        """
        s' nodes of the causal drift region [edge, s1) below the regular drift layout, or None
        (Integration_params.drift_causal_reach): only in the drift after a bend, from the lowest s'
        whose retarded band still meets the bunch, bounded by s' = 0 and the retained history.
        """
        ip = self.integration_params
        if not (ip.drift_causal_reach and self._drift_reach_start() is not None):
            return None
        tl = self.DF_tracker.time_log
        h0 = max(0.0, float(tl[0]) if len(tl) else 0.0)
        if not h0 < s1 - 1e-9:
            return None
        lo = yield from self._g_causal_edge(s, x, t, h0, s1, n_scan=ip.drift_causal_scan)
        if lo is None or not lo < s1 - 1e-9:
            return None
        if lo <= h0 + 1e-12 and h0 > 0.0:
            # still alive at the oldest retained snapshot: the history, not causality, ends the region
            self._count('drift_causal_clipped')
        return np.unique(s - graded_u(max(s - s1, 1e-12), s - lo, ip.drift_causal_grade, k_min=2))

    def _geo_reach(self, s):
        """
        Straight-section reach for an observer inside a bend, measured back from s, or None.

        The straight before the bend's entrance is a source retarded by R phi^3/6 whose
        contribution decays only as 1/u, so it is integrated out to reach_straight * R phi before
        the entrance while that retardation is below reach_slip_max sigma_z
        (docs/DFCSR_2D_methods.md §5.2). Never in the drift after the bend (below).
        """
        ip = self.integration_params
        rs = ip.reach_straight
        sb = self.bend_start_rec
        if not rs or sb is None:
            return None
        R = abs(float(self.R_rec or 0.0))
        if R <= 0.0:
            return None
        a = float(s) - float(sb)
        if a <= 0.0:
            return None
        if not self.inbend:
            # in the drift the frozen straight history diverges as ln(reach) (talman_cancellation.md 7.7)
            return None
        phi = a / R
        slip = R * phi ** 3 / 6.0
        if slip > ip.reach_slip_max * self.beam._sigma_z:
            return None
        return a + rs * R * phi

    def _exit_transient(self):
        """
        Is the beam in a drift after a bend, i.e. computing an EXIT transient?

        This is the only regime where the causal edge is worth locating: it is the only
        one where it differs from the n_formation_length*L_f reach (u ~ 3 L_f past the
        exit against ~1 L_f inside the bend), and the only one where the contributing
        source is a narrow band far upstream rather than a 1/u tail.
        """
        return bool(self.afterbend and not self.inbend)

    def _prev_bend_reach_start(self):
        """Entrance of the previous bend while inside the next one, or None."""
        ip = self.integration_params
        if not (ip.drift_reach and ip.prev_bend_reach and self.inbend):
            return None
        pb = self.prev_bend_start
        return None if pb is None else max(0.0, float(pb))

    def _drift_reach_start(self):
        """Far edge for the exit transient (the last bend's entrance), or None."""
        if not (self.integration_params.drift_reach and self._exit_transient()):
            return None
        sb = self.bend_start_rec
        return None if sb is None else max(0.0, float(sb))

    def _bend_exit_rec(self):
        sb = self.bend_start_rec
        if sb is None or self.R_rec is None:
            return -np.inf
        return sb + abs(self.R_rec * self.phi_rec)
