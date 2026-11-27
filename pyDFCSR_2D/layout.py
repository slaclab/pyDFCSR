"""
Where the wake integral puts its s' columns: the far / near regions and their extent, the extra
regions below the far edge (previous bend, straight section, causal drift region), the node
constructions in each, and the polar-patch radii (docs/DFCSR_2D_methods.md §5.2, §5.4).

Mixed into CSR.CSR2D; the methods use its state (beam, lattice, DF_tracker, integration_params).
"""
import numpy as np

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
    def _band_alive(self, s, x, t, sp):
        """Whether the retarded band has nonzero width, for each s' in the array."""
        sp = np.atleast_1d(np.asarray(sp, float))
        bands = self._disjoint_bands(self._retarded_xi_bands(s, x, t, sp))
        w = np.zeros(sp.size)
        for lo, hi in bands:
            w += np.maximum(hi - lo, 0.0)
        return w > 0.0

    def _causal_edge(self, s, x, t, sp_far, sp_near, n_scan=64):
        """
        Lowest s' in [sp_far, sp_near] whose retarded band still has nonzero width.

        Below this the retarded source has slipped outside the bunch and every column is
        identically zero, so integrating there is pure cost. ONE vectorised scan, and
        the LAST DEAD node below the first alive one is returned -- deliberately
        conservative, since truncating a live domain is a one-sided error that cannot
        average out, while starting one cell early costs one cell.

        Three things rule out anything cleverer, all measured:

        - NOT a root-find. The alive mask is *not* monotone: at shear 20 it has a dead
          gap inside the live region at 2 of 40 mesh points, so bisection can land in
          the gap and return an edge far above the true one. "First alive from below"
          on a scan has no such failure mode.
        - NOT a closed form. The edge is where the retarded z leaves the deposition
          window, and the slippage that sets it is exact only on the ray the band
          actually sits on. Inverting it analytically on the x' = x ray is free
          (0.06 ms) and matches this to 4 far-cells at zero tilt, but at shear 20 the
          band sits at x' = +2.9..3.6 mm against an observer at x = -71 um, and the
          wrong chord put the edge 37 mm too high -- rel L2 1.0 instead of 1.3e-03.
        - NOT refined further. A 44-step scalar bisection located the edge to 1e-14 m
          and cost 75% of a whole wake point, for a number that is then discretised
          onto a far_cell*sigma_z = 673 um grid. Padding the target by 2 full zlim
          moves the answer by 0.3%.

        Returns sp_far if the first sample is already alive, or None if the whole
        interval is dead.
        """
        grid = np.linspace(sp_far, sp_near, n_scan)
        alive = self._band_alive(s, x, t, grid)
        if not alive.any():
            return None
        if alive[0]:
            return sp_far
        return grid[max(int(np.argmax(alive)) - 1, 0)]

    def _layout_bounds(self, s, x):
        """
        The two s' integration regions for the xi_bands path.

        Only the region EXTENT is decided here. Whether the two Eq 4.24 branches are
        resolvable is a per-column property of the retarded tilt a(s'), so it is left
        to _retarded_xi_bands; the extent cannot be, since it defines s3 itself.

        Two regions, not three. The old middle seam s2 = s3 - 200 sigma_z dated from the
        legacy decomposition, where it marked where the +-20 sigma_x WIDE transverse
        window stopped being needed. With per-column ribbons there is no wide window, so
        s2 no longer separates anything physical -- but because every region got the same
        NODE COUNT while their lengths differed ~45x, it was still acting as a crude
        two-level grading (cells 45.7 / 1.0 / 0.1 sigma_z). Merging the two far regions
        naively therefore LOST accuracy (3-10x at matched cost); the replacement is
        _far_region_nodes, which grades the merged region explicitly and is 28x better
        than the old split at a bend-exit transient for the same node count.

        Everything is keyed on sin 2a and cos 2a, never on tan 2a and never on |tau|:

          sin 2a = 2 tau/(1 + tau^2)      cos 2a = (1 - tau^2)/(1 + tau^2)

        Both are bounded and pole-free, whereas tan 2a = 2 tau/(1 - tau^2) has a pole at
        |tau| = 1. Thesis 4.4.2 notes that x2 = x - (s-s')tan(2a) degenerates to x2 = x
        at a = 0 AND at a = +-pi/2, so the chirp branch merges into the narrow one in
        BOTH limits -- and |sin 2a| is small in both, so a single test catches them.

        The legacy (xi_bands = False) decomposition in get_CSR_wake is deliberately left
        alone; it keys on |tan_theta| <= 1, which fires where the branches are maximally
        SEPARATED and which sent the near region swinging
        d = 99.9 mm -> 807.7 mm -> (branch flip) s3 = s - 50.9 um -> 99.9 mm
        within ~1 mm of path length across a longitudinal waist.
        """
        ip = self.integration_params
        b = self.beam
        sigma_z, sigma_x = b._sigma_z, b._sigma_x
        # Use the SAME tilt the density frame uses (core fit) rather than beams.slope
        # (all particles). They disagree most exactly where the correlation vanishes.
        tau = self._frame_tilt()

        sin2a = 2.0 * tau / (1.0 + tau * tau)
        cos2a = (1.0 - tau * tau) / (1.0 + tau * tau)

        # Near-region length as ONE continuous expression, with no branch on it at all.
        # An earlier version switched the extent formula at branch_sin_min and
        # reintroduced a discontinuity (125 mm -> 0.347 mm across the threshold).
        #
        #   d = (10 sigma_x + x - xmean) |cos 2a| / max(|sin 2a|, branch_sin_min)
        #
        # This is the legacy d = (10 sigma_x + x - xmean)/|tan 2a| wherever the branches
        # are resolvable, and it saturates at branch_sin_min as sin 2a -> 0, where the
        # branches merge and d stops meaning anything.
        #
        # It is then floored at near_floor sigma_z, which is NOT cosmetic. As |tau| -> 1,
        # cos 2a -> 0 and the chirp band becomes vertical, so it does exit the beam at
        # once -- but d is the length of the NEAR region, which must also contain the
        # NARROW band and the neighbourhood of the 1/|r-r'| pole. Sizing it from the
        # chirp exit alone starves both. Measured at shear 50, 0.80 m into the dipole
        # (tau = -0.999, cos 2a = 8.8e-4): the upstream reach collapsed to 15 um =
        # 0.009 sigma_z against sigma_z = 1744 um, so 1-2 of the 88 graded nodes landed
        # on the causal side and the pole sat 15 um from the region edge, leaving
        # region 2's uniform 1.74 mm cells to carry it. The wake went visibly blocky.
        # The floor is the s3 = s - 20 sigma_z that the deleted |tan theta| <= 1 branch
        # used to provide; removing that branch in 6o dropped it silently, and 6f's
        # coarser s grid (tau = -1.225 then -0.814) stepped over the region where it
        # matters -- the defect only bites within ~1% of |tau| = 1.
        #
        # The (x - xmean) term is kept deliberately: the chirp branch starts at the
        # observation point, so one starting near the beam edge exits sooner than one
        # starting at the centre, and d legitimately varies ~1.9x across a wake mesh.
        # Dropping it makes the wake look ~4x smoother at fixed resolution, but the two
        # variants converge to the same answer (rel L2 0.045 -> 0.0017 as zbins/xbins
        # refine), so that smoothness is per-column node-layout jitter being removed
        # from the roughness METRIC, not accuracy being gained. Optimising the metric
        # rather than the answer is not worth losing the correct exit condition.
        d = ((10.0 * sigma_x + x - b._mean_x) * abs(cos2a)
             / max(abs(sin2a), ip.branch_sin_min))
        d = max(d, ip.near_floor * sigma_z)
        reach = ip.n_formation_length * self.formation_length
        if (ip.seam_ratio and ip.near_grade
                and not (ip.causal_clip and self._exit_transient())):
            # Seam continuity: keep grading the near region until its cell near_grade*u has
            # reached 1/seam_ratio of the far region's flat cell, so the far columns never
            # inherit structure finer than they can carry (Integration_params.seam_ratio).
            # Only where the far layout IS flat (in a bend); past a bend exit the far region
            # is itself graded at near_grade from the causal edge, so the seam is continuous
            # already and extending the near region there only moved the 0.4% node-phase
            # jitter of the golden drift kicks.
            far_cell = ip.n_formation_length * self.formation_length / max(2 * ip.far_zbins, 3)
            d = max(d, far_cell / (ip.near_grade * ip.seam_ratio))
        # and never reach back further than the retained history supports
        d = min(max(d, 0.0), reach)

        s4 = s + 3.0 * sigma_z
        # prehistory: the lattice start is not a wall. Sources on the synthetic straight section
        # s' in [-prehistory, 0) are the first snapshot translated along the initial tangent
        # (Integration_params.prehistory); 0 keeps the old clamp at s' = 0.
        lo = -ip.prehistory
        s3 = max(lo, s - d)
        # The far edge is the causal reach measured from s3, and it is clamped at 0
        # because s' < 0 is before the lattice exists -- interpolate1D returns 0 there
        # silently, so nodes below 0 are not merely wasted but invisible. The old
        # s1 = s2 - n_fl*L_f with s2 = s3 - 200 sigma_z reached back 200 sigma_z FURTHER
        # than the retained history supports, and inverted (s1 > s2) whenever s2 went
        # negative while s1 clamped.
        s1 = max(lo, s3 - reach)
        # In a drift after a bend the source of the exit wake is the bend's own arc,
        # which recedes behind the observer; a fixed n_fl*L_f reach loses it ~1.7 R*phi
        # past the exit and the wake then changes sign (progress.md, 2026-11-03).
        # drift_reach extends the far edge back to the bend's entrance; the causal-edge
        # scan in _far_region_nodes then trims it to where the source still contributes.
        s_bend = self._drift_reach_start()
        if s_bend is not None:
            if ip.drift_causal_reach:
                # the regular exit-transient layout always has the n_formation_length = 1 reach in a drift,
                # so its fine window stays on the last bend's arc whatever n_fl is (at n_fl = 6 it slid
                # into D31 and B1RE's exit wake came out 65% wrong); earlier sources go to the causal
                # drift region (Integration_params.drift_causal_reach)
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
        # Warn once per run when the cap binds. It binds where |tan 2 alpha| collapses
        # -- near a bend entrance the chirp band can stretch to d = 10 sigma_x /
        # |tan 2 alpha| ~ 150 mm while the cell stays at sigma_xi, asking for ~25000
        # columns. Silently clipping that returns an under-resolved wake that looks
        # perfectly plausible, which is exactly how it went unnoticed once already.
        # ... but only when this count is what the near region will actually use.
        # With near_grade set, _near_region_nodes discards `want` entirely and builds a
        # graded grid whose finest cell IS near_cell*sigma_xi, so the cap never binds
        # and warning about it is not merely noise: it reports a resolution loss that
        # is not happening, and advises "use graded nodes" when they are already on.
        graded = bool(ip.near_grade)
        if want > cap and not graded:
            self._note('near_cap', f'WARNING: near-region nodes capped at {cap} (wanted {want}); near-region '
                                   f'cell is {want/cap:.1f}x coarser than near_cell requests. Raise zbins, '
                                   f'raise near_cell, or use graded nodes.', every_rank=True)
        counts[-1] = int(np.clip(want, 3, cap))
        self._last_region_nodes = counts
        return counts

    def _far_region_nodes(self, s, x, t, sp_far, sp_near, transient=None):
        """
        s' nodes for the merged far region, spanning sp_far (= s1, the causal reach
        s - n_formation_length*L_f) up to sp_near (= s3, where the near region starts).

        TWO constructions, because the far integrand has two quite different shapes:

        IN A BEND (or before any bend) -- UNIFORM at 2*far_zbins nodes. This is the
        cost-neutral replacement for the old (s1,s2)+(s2,s3) pair, which gave each
        sub-region far_zbins nodes. It is deliberately NOT log-graded: grading assumes
        the contribution decays as 1/u away from the observer, and in a bend at high
        tilt it does not -- measured at shear 20, 0.30 m in, 10-99% of the far integral
        sits at u/L_f = 0.12-0.56, i.e. near the FAR end, where distance grading is
        coarsest. Pure log grading there gave rel L2 1.79 against 5.3e-02 for the old
        layout, and refining near_grade 0.05 -> 0.005 only reached 1.2e-02 at 456 nodes.

        PAST A BEND EXIT -- causal clip, then a uniform fine window, then log grading.
        Here the source is the arc still inside the dipole, so the contribution is a
        narrow band far upstream (u/L_f ~ 2.7-3.1, 10-99% of it inside ~0.3 L_f) and
        everything below the causal edge is identically zero. Log grading alone put a
        single 30 mm cell across the entire contribution and was 400x worse than the old
        layout; with the window it is 28x better at the same node count. See
        _exit_transient, _causal_edge and Integration_params.
        """
        ip = self.integration_params
        if sp_near <= sp_far:
            return np.array([sp_far, sp_near])

        if transient is None:
            transient = self._exit_transient()
        if not (ip.causal_clip and transient):
            # Not an exit transient: uniform, and no band solve. The causal edge would
            # be redundant here anyway -- measured in a bend it sits at u ~ 1.0 L_f
            # (0.994 at shear 0, 0.783 at shear 20), which is what n_formation_length
            # already provides.
            n = max(2 * ip.far_zbins, 3)
            return self._record_far_nodes(np.linspace(sp_far, sp_near, n))

        lo = sp_far
        e = self._causal_edge(s, x, t, sp_far, sp_near)
        if e is not None:
            lo = min(max(e, sp_far), sp_near)
            if e <= sp_far:
                # The edge should be the tighter limit; if the reach binds instead, the
                # integral is truncated by the history budget rather than by causality.
                # Honest, but a different limitation -- count it rather than hide it.
                self._count('far_clip_at_reach')

        parts = []
        top = lo
        if ip.far_window and ip.far_cell:
            top = min(lo + ip.far_window * self.formation_length, sp_near)
            cell = ip.far_cell * self.beam._sigma_z
            if ip.far_fine_max:
                # cap the fine-window node count: far_cell*sigma_z is set by the bunch,
                # but the band it resolves scales with L_f, so a short bunch (sigma_z
                # 19 um in D3) put 3300 nodes in a window that ~300 resolve to 3e-4
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
        """
        s' nodes for the near region, graded so the cell grows with |s - s'|.

        The cell is  du = max(near_cell*sigma_xi, near_grade*u),  u = |s - s'|:
        an absolute floor set by the transverse support width close to the
        observation point, geometric growth outside. Built independently on each
        side, because the near region straddles s' = s.

        Rationale: for u larger than the transverse offsets, |r - r'| ~ u, so the
        per-column contribution falls as 1/u and equal contributions come from equal
        logarithmic intervals. Uniform nodes are therefore under-resolved where the
        accuracy is set and over-resolved where the cost is, forcing
        N ~ u_max/sigma_xi -- which is why the count had to scale with tilt. Graded,
        N ~ 1/near_grade + ln(u_max/u*)/ln(1+near_grade), growing only
        logarithmically with tilt.

        The outer integral uses np.trapz(..., x=sp), which handles the non-uniform
        spacing correctly.
        """
        ip = self.integration_params
        floor = ip.near_cell * self.beam._sigma_x_transform
        g = ip.near_grade
        if not ip.near_cell or not g or floor <= 0.0:
            base = np.linspace(a, b, self._region_node_counts(((a, b),))[-1])
            if ip.near_adapt and x is not None:
                return self._near_adapt_nodes(s, x, t, a, b, base)
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
            return self._near_adapt_nodes(s, x, t, a, b, sp)
        return sp

    def _column_scan(self, s, x, t, sp):
        """
        The retarded source seen by the column at each s': (z_ret, rho) at the located band's
        axis point, the larger of the two Eq 4.24 branches. rho is 0 where no branch is live.
        """
        self._retarded_xi_bands(s, x, t, sp, n_iter=self.integration_params.near_adapt_scan_iter)
        diag = getattr(self, '_branch_diag', None)
        zr = np.full(sp.shape, np.nan)
        rho = np.zeros(sp.shape)
        if diag is None or diag['sp'] is not sp:
            return zr, rho
        for br in diag['branches']:
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
        zr, rho = self._column_scan(s, x, t, sp)
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
            zm, rm = self._column_scan(s, x, t, new)
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
        lm = np.log(m)
        for _ in range(2):
            for i in range(1, lm.size):
                lm[i] = max(lm[i], lm[i - 1] - lr * dg[i - 1] * np.exp(lm[i - 1]))
            for i in range(lm.size - 2, -1, -1):
                lm[i] = max(lm[i], lm[i + 1] - lr * dg[i] * np.exp(lm[i + 1]))
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
        sp_drift = self._drift_causal_nodes(s, x, t, s1)
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
        s' nodes for the straight-section extension [lo, hi) below the regular layout
        (reach_straight). Its sources are far from the observer and their retarded z is nearly
        constant, so the column integral is a smooth ~1/u^2 in u = s - s': geometric grading in u
        with ratio 1 + near_grade, ~ln(u_lo/u_hi)/near_grade nodes (41 for 10 m behind a 0.5 m
        bend at 5%), where the fixed far cell would need thousands.
        """
        g = self.integration_params.near_grade or 0.05
        return np.unique(s - graded_u(max(s - hi, 1e-9), s - lo, g, close=True))

    def _drift_causal_nodes(self, s, x, t, s1):
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
        lo = self._causal_edge(s, x, t, h0, s1, n_scan=ip.drift_causal_scan)
        if lo is None or not lo < s1 - 1e-9:
            return None
        if lo <= h0 + 1e-12 and h0 > 0.0:
            # still alive at the oldest retained snapshot: the history, not causality, ends the region
            self._count('drift_causal_clipped')
        return np.unique(s - graded_u(max(s - s1, 1e-12), s - lo, ip.drift_causal_grade, k_min=2))

    def _geo_reach(self, s):
        """
        Entrance-transient reach for an observer at s, measured back from s, or None.

        The straight section the beam came along before the last bend entrance is a source
        retarded by ~ R phi^3/6 (phi = arc angle traversed) whose contribution decays only as
        1/u with the distance u before the entrance, so it is needed out to
        reach_straight * R phi, i.e. reach = (s - entrance) + reach_straight * R phi. It is
        applied while that retardation is within reach_slip_max sigma_z of the bunch
        (Integration_params.reach_straight): inside the bend slip = a^3/(6R^2) with a the arc
        length past the entrance; in the drift x past the exit the straight sources slip a
        further x phi^2/2 and fall out of the bunch within a fraction of R phi.
        Inside the bend only (see below). The straight section is integrated as its own graded
        region below the regular layout (_straight_region_nodes); the retention floor and check use
        the same reach.
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
            # NOT in the drift after the bend: there the frozen straight history is seen from a
            # different straight line, each column contributes ~ phi^2 lambda'/u and the sum grows
            # as ln(reach) -- measured on Indranil's gain beam 0.19 m past B1LE, -0.18 MeV/m per
            # doubling of the reach out to 170 m (55x the 1D wake at 40 R phi), with or without the
            # velocity terms. Its value is set by the real length of the upstream straight, not by
            # local physics, so the regular layout (n_formation_length * L_f) is kept there. Inside
            # the bend the same contribution converges as 1/U (talman_cancellation.md 7.7).
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
