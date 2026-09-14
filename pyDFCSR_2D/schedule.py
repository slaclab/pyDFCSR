"""
The step schedule: where the beam is tracked to, where density snapshots are taken, and where
CSR wakes are computed and applied.

Why this exists
---------------
One uniform `step_size` used to drive three grids whose costs differ by orders of magnitude
(all measured in progress Step 10 and §6bb):

    tracking   ~0.001 s per step
    snapshot   ~0.08 s per step, plus 0.7-1.6 MB of memory each
    CSR kick   ~5-7 s per evaluation (a full wake mesh)

So kicks are ~5000x more expensive than snapshots: **kicks decide run time, snapshots decide
memory**, and they need different criteria. They used to be tied together by an integer `nsep`.

The physics forces the issue. §6x measured that a 2.39 mm longitudinal waist needs
`step_size <= 0.0006 m` (the wake is 79% wrong at 0.05 m), while §6bb measured that applying
that step size *uniformly* through a chicane costs 22-32 GB of history against 0.15 GB for local
refinement. No single uniform step can serve both.

What this module guarantees
--------------------------
* The **total** step count is known before tracking starts. `CSR2D.init_statistics` preallocates
  arrays of length `lattice.total_steps` and `update_statistics(step)` indexes them directly, so
  the schedule must be precomputed, never adapted mid-run.
* Kicks are a **subset** of snapshots, so a kick always has a density history at its own position.
* In `manual` and `auto`, every element boundary is a node. Each step then lies wholly inside one
  element, which is what lets the run loop stop splitting steps across boundaries -- and fixes
  the fact that a boundary previously got no snapshot at all (the beam was tracked to the
  boundary outside the step loop, then past it, and only then was a snapshot taken).

Bit-identity
------------
`legacy` reproduces the old node set and kick cadence exactly, INCLUDING not placing nodes on
element boundaries. That is deliberate: boundary nodes add snapshots and change the sequence of
bmad-x calls, so `manual` and `auto` cannot be bit-identical to previously recorded results and
are not expected to be. Only `legacy` carries that guarantee.
"""
import math

import numpy as np


class StepSchedule:
    """
    An immutable step plan.

    Arrays are all length N = number of NODES. Node 0 is the lattice entrance at s = 0 and is not
    a step, so `dl[0] = 0` and there are N-1 actual steps. Index i > 0 means "the step that ends
    at s_nodes[i]".

    s_nodes  (N,)  monotone increasing, s_nodes[0] = 0, s_nodes[-1] = lattice_length
    ele_of   (N,)  index of the element the step ENDS in
    dl       (N,)  step lengths, dl[i] = s_nodes[i] - s_nodes[i-1], dl[0] = 0
    is_snap  (N,)  take a density snapshot here
    is_kick  (N,)  compute and apply a CSR wake here; must imply is_snap
    kick_lo  (N,)  start of the arc this kick integrates over  (valid where is_kick)
    kick_hi  (N,)  end of that arc                              (valid where is_kick)
    """

    def __init__(self, s_nodes, ele_of, is_snap, is_kick, kick_lo, kick_hi,
                 mode, lattice_length, spe, is_step, nsep=None, step_size=None,
                 dl=None, kick_interval='trailing'):
        self.s_nodes = np.asarray(s_nodes, dtype=np.float64)
        self.ele_of = np.asarray(ele_of, dtype=np.int64)
        self.is_snap = np.asarray(is_snap, dtype=bool)
        self.is_kick = np.asarray(is_kick, dtype=bool)
        self.kick_lo = np.asarray(kick_lo, dtype=np.float64)
        self.kick_hi = np.asarray(kick_hi, dtype=np.float64)
        self.mode = mode
        self.lattice_length = float(lattice_length)
        self._spe = np.asarray(spe, dtype=int)
        # Nodes the run loop actually advances to. `legacy` can carry a trailing node BEYOND the
        # lattice end, because np.arange(0, L + h/2, h) overshoots whenever L is not a multiple
        # of h (L = 2.5, h = 0.07 gives a final node at 2.52). The old code silently dropped it
        # from steps_per_element, so the loop never reached it -- this is the `run(stop_time=T)`
        # overshoot recorded in Step 7. It is preserved here bug-for-bug, because
        # `_positions_record` sizes the preallocated statistics arrays and is written to output.
        # `manual` and `auto` land the last node exactly on lattice_length instead.
        self.is_step = np.asarray(is_step, dtype=bool)
        self.nsep = nsep
        self.step_size = step_size
        self.kick_interval = kick_interval

        if dl is None:
            self.dl = np.zeros_like(self.s_nodes)
            self.dl[1:] = np.diff(self.s_nodes)
        else:
            # Supplied explicitly, because np.diff(np.arange(0, L+h/2, h)) is NOT exactly h:
            # measured 35 of 37 steps differing by ~1e-16 for L=1.85, h=0.05. The run loop feeds
            # this length straight into bmad-x tracking, so a last-bit difference changes the
            # trajectory and destroys the bit-identity guarantee that `legacy` exists to provide.
            self.dl = np.asarray(dl, dtype=np.float64).copy()
            assert self.dl.size == self.s_nodes.size
            self.dl[0] = 0.0
            drift = np.abs(np.cumsum(self.dl) - self.s_nodes).max()
            assert drift < 1e-9 * max(self.lattice_length, 1.0), (
                f'explicit dl is inconsistent with s_nodes by {drift:.3e} m')
        self._validate()

    def _validate(self):
        n = self.s_nodes.size
        for name in ('ele_of', 'is_snap', 'is_kick', 'kick_lo', 'kick_hi'):
            assert getattr(self, name).size == n, f'{name} length {getattr(self, name).size} != {n}'
        assert n >= 2, f'need at least 2 nodes, got {n}'
        assert self.s_nodes[0] == 0.0, f's_nodes[0] must be 0, got {self.s_nodes[0]}'
        assert self._spe.sum() == int(self.is_step.sum()), (
            f'steps_per_element sums to {self._spe.sum()} but is_step marks '
            f'{int(self.is_step.sum())} steps')
        assert np.all(self.dl[1:] > 0.0), (
            'step lengths must be strictly positive; a zero-length step would give two snapshots '
            'at the same time and divide by zero in the non-uniform interpolation weight')
        # kicks must be a subset of snapshots, or a kick has no history at its own position
        bad = int((self.is_kick & ~self.is_snap).sum())
        assert bad == 0, f'{bad} kick nodes are not snapshot nodes'
        assert np.all(np.diff(self.ele_of) >= 0), 'ele_of must be non-decreasing'
        if self.kick_interval == 'midpoint' and self.is_kick.any():
            # The single assertion that catches almost every quadrature bug here: the kick
            # intervals must tile the lattice exactly, with no gap and no overlap.
            got = self.total_arc_kicked()
            assert abs(got - self.lattice_length) < 1e-9 * max(self.lattice_length, 1.0), (
                f'midpoint kick intervals sum to {got:.12g} m but the lattice is '
                f'{self.lattice_length:.12g} m -- they must tile it exactly')

    # ---- counts -------------------------------------------------------------------------
    @property
    def n_nodes(self):
        return self.s_nodes.size

    @property
    def n_steps(self):
        """Steps the loop actually executes -- excludes any trailing overshoot node."""
        return int(self.is_step.sum())

    @property
    def n_snap(self):
        return int(self.is_snap.sum())

    @property
    def n_kick(self):
        return int(self.is_kick.sum())

    @property
    def kick_indices(self):
        return np.flatnonzero(self.is_kick)

    # ---- compatibility aliases, so existing consumers keep working ----------------------
    @property
    def positions_record(self):
        """What lattice._positions_record used to be."""
        return self.s_nodes

    def steps_per_element(self, n_element=None):
        """
        Steps attributed to each element.

        Stored rather than derived. For `legacy` it is computed with the historical algorithm
        verbatim, including the `- 1` on the first element (s = 0 is a node but not a step) and
        the exclusion of any trailing overshoot node. Deriving it from `ele_of` instead gave the
        right answer in 39 of 48 randomised lattices and was off by one in the last element for
        the other 9 -- always the overshoot case.
        """
        if n_element is None or n_element == self._spe.size:
            return self._spe
        out = np.zeros(n_element, dtype=int)
        k = min(n_element, self._spe.size)
        out[:k] = self._spe[:k]
        return out

    def total_arc_kicked(self):
        """Sum of the kick intervals. Must equal the arc the kicks are responsible for."""
        k = self.is_kick
        return float(np.sum(self.kick_hi[k] - self.kick_lo[k]))

    def summary(self, max_lines=12):
        lines = [f'  [schedule] mode={self.mode}  nodes={self.n_nodes}  steps={self.n_steps}  '
                 f'snapshots={self.n_snap}  kicks={self.n_kick}']
        d = self.dl[1:]
        lines.append(f'  step size: min {d.min():.6g} m, max {d.max():.6g} m, '
                     f'ratio {d.max()/d.min():.1f}')
        return '\n'.join(lines[:max_lines])


def midpoint_intervals(s_nodes, kick_idx, boundaries, s_end):
    """
    Centre each kick's integration interval on its own sample point.

    The kick is a quadrature of int(W ds). Sampling W at s_i and applying it over the arc
    BEHIND s_i is a trailing rectangle rule, error O(h * dW/ds). Centring the interval on s_i
    makes it a midpoint rule, O(h^2), for free -- the wake is still evaluated at s_i using
    history up to s_i, only the weight moves.

        kick_lo[i] = (s_prev + s_i)/2        kick_hi[i] = (s_i + s_next)/2

    This needs the NEXT kick position, which is why it is only possible with a precomputed
    schedule and was deferred out of §6dd.

    Intervals are clipped at element boundaries, because W genuinely jumps at a dipole edge and
    an interval spanning one would smear the transient across it. Clipping only moves the split
    point between two neighbouring kicks, so the total arc is preserved exactly.
    """
    s_nodes = np.asarray(s_nodes, dtype=np.float64)
    ki = np.asarray(kick_idx, dtype=np.int64)
    n = s_nodes.size
    lo = np.zeros(n)
    hi = np.zeros(n)
    if ki.size == 0:
        return lo, hi

    sk = s_nodes[ki]
    mid = 0.5 * (sk[:-1] + sk[1:])
    lo[ki[0]] = 0.0
    hi[ki[-1]] = float(s_end)
    for j in range(ki.size - 1):
        hi[ki[j]] = mid[j]
        lo[ki[j + 1]] = mid[j]

    # clip at element boundaries: move any split point that crosses one onto the boundary
    b = np.asarray([x for x in boundaries if 0.0 < x < s_end], dtype=np.float64)
    for j in range(ki.size - 1):
        a0, a1 = sk[j], sk[j + 1]
        crossed = b[(b > a0) & (b < a1)]
        if crossed.size:
            # the boundary nearest the current split point wins
            cut = float(crossed[np.argmin(np.abs(crossed - hi[ki[j]]))])
            hi[ki[j]] = cut
            lo[ki[j + 1]] = cut
    return lo, hi


def build_legacy(distance, nsep, lattice_length, step_size, n_element,
                 kick_interval='trailing'):
    """
    Reproduce the historical node set and kick cadence EXACTLY.

    Two details that must not be 'cleaned up', because recorded baselines depend on them:

    1. The node set is `np.arange(0, lattice_length + step_size/2, step_size)`. Element
       boundaries are NOT nodes, so a step can straddle one; the run loop still has to split
       such a step. Placing nodes on boundaries would change the snapshot set.
    2. The kick test was `step % nsep[ele] == 0` where `step` restarts at 0 in every element.
       That is why kicks land on the first step of every element regardless of how far the
       previous kick was -- and why the kick interval had to be measured as elapsed arc rather
       than `nsep * step_size` (§6dd, which was a 14.3% error at nsep = 3).
    """
    s_nodes = np.arange(0, lattice_length + step_size / 2, step_size)
    n = s_nodes.size

    # steps_per_element, verbatim from the historical get_steps()
    spe = np.zeros((n_element,), dtype=int)
    prev_ind = 0
    for count, d in enumerate(np.asarray(distance, dtype=float)):
        ind = int(np.searchsorted(s_nodes, d, side='right'))
        spe[count] = ind - prev_ind - (1 if count == 0 else 0)
        prev_ind = ind

    # expand spe into a per-node element index; nodes past the sum are the overshoot and are
    # never stepped to
    n_stepped = int(spe.sum())
    ele_of = np.zeros(n, dtype=np.int64)
    ele_of[1:n_stepped + 1] = np.repeat(np.arange(n_element), spe)
    if n_stepped + 1 < n:
        ele_of[n_stepped + 1:] = n_element - 1
    is_step = np.zeros(n, dtype=bool)
    is_step[1:n_stepped + 1] = True

    is_snap = is_step.copy()    # s = 0 is handled by CSR2D.initialization, not by the loop

    # reproduce `step % nsep == 0` with `step` restarting per element
    is_kick = np.zeros(n, dtype=bool)
    step_in_ele = 0
    prev_ele = ele_of[1] if n > 1 else 0
    for i in range(1, n_stepped + 1):
        e = ele_of[i]
        if e != prev_ele:
            step_in_ele = 0
            prev_ele = e
        if step_in_ele % int(nsep[e]) == 0:
            is_kick[i] = True
        step_in_ele += 1

    if kick_interval == 'midpoint':
        kick_lo, kick_hi = midpoint_intervals(s_nodes, np.flatnonzero(is_kick),
                                              np.asarray(distance, dtype=float), lattice_length)
    else:
        # trailing intervals: each kick owns the arc since the previous kick (the §6dd form)
        kick_lo = np.zeros(n)
        kick_hi = np.zeros(n)
        last = 0.0
        for i in np.flatnonzero(is_kick):
            kick_lo[i] = last
            kick_hi[i] = s_nodes[i]
            last = s_nodes[i]

    # exactly step_size, never diff(s_nodes) -- see the note in StepSchedule.__init__
    dl = np.full(n, float(step_size))
    dl[0] = 0.0

    return StepSchedule(s_nodes, ele_of, is_snap, is_kick, kick_lo, kick_hi,
                        mode='legacy', lattice_length=lattice_length,
                        spe=spe, is_step=is_step, nsep=nsep, step_size=step_size, dl=dl,
                        kick_interval=kick_interval)


# schedule-only per-element keys. They must be stripped before the element dict reaches
# get_bmadx_element, which forwards whatever is left as **kwargs to SBend/Quadrupole/Sextupole.
SCHEDULE_ELEMENT_KEYS = ('steps', 'kick_every')


def build_manual(lattice_config, distance, lattice_length, n_element,
                 default_steps=None, default_kick_every=1, step_size=None,
                 kick_interval='trailing', nsep=None):
    """
    User-specified step count per element.

    Per element in the lattice YAML:

        element_2:
          type: dipole
          L: 1.0
          angle: 1.0
          nsep: 1
          steps: 200          # this element gets exactly 200 equal steps
          kick_every: 10      # a CSR kick on every 10th step

    Unlike `legacy`, **element boundaries are exact nodes**. Each element's steps are
    `L_e / n_e` and the last node of each element is set to the boundary exactly, so:

    * the boundary gets a density snapshot, which it never did before -- the beam used to be
      tracked to the boundary outside the step loop and past it inside, with the snapshot taken
      only afterwards, so the position where the CSR transient turns on was absent from the
      history entirely;
    * no step straddles a boundary, so `DL_1` is 0 and the run loop's split path never fires;
    * the last node lands on `lattice_length` exactly, rather than overshooting it the way
      `np.arange(0, L + h/2, h)` does.

    The last step of every element is always a kick node, regardless of `kick_every`. Midpoint
    intervals are clipped at boundaries (`W` jumps at a dipole edge), and that clipping is only
    meaningful if the boundary is itself a kick.
    """
    ele_keys = [k for k in lattice_config if k != 'step_size']
    distance = np.asarray(distance, dtype=np.float64)
    starts = np.concatenate([[0.0], distance[:-1]])

    s_list = [0.0]
    ele_list = [0]
    kick_list = [False]
    spe = np.zeros(n_element, dtype=int)

    for e, key in enumerate(ele_keys):
        cfg = lattice_config[key]
        L_e = float(cfg['L'])
        n_e = int(cfg.get('steps', default_steps if default_steps is not None
                          else max(int(round(L_e / step_size)) if step_size else 1, 1)))
        if n_e < 1:
            raise ValueError(f"{key}: steps must be >= 1, got {n_e}")
        ke = max(int(cfg.get('kick_every', default_kick_every)), 1)
        h = L_e / n_e
        spe[e] = n_e
        for j in range(1, n_e + 1):
            # last node of the element is the boundary EXACTLY, not starts[e] + n_e*h
            s_list.append(float(distance[e]) if j == n_e else starts[e] + j * h)
            ele_list.append(e)
            kick_list.append(((j - 1) % ke == 0) or (j == n_e))

    s_nodes = np.asarray(s_list, dtype=np.float64)
    ele_of = np.asarray(ele_list, dtype=np.int64)
    is_kick = np.asarray(kick_list, dtype=bool)
    is_step = np.ones(s_nodes.size, dtype=bool)
    is_step[0] = False
    is_snap = is_step.copy()

    if kick_interval == 'midpoint':
        kick_lo, kick_hi = midpoint_intervals(s_nodes, np.flatnonzero(is_kick),
                                              distance, lattice_length)
    else:
        kick_lo = np.zeros(s_nodes.size)
        kick_hi = np.zeros(s_nodes.size)
        last = 0.0
        for i in np.flatnonzero(is_kick):
            kick_lo[i] = last
            kick_hi[i] = s_nodes[i]
            last = s_nodes[i]

    # dl is derived from the nodes here, unlike `legacy`. The nodes are the truth in this mode and
    # there is no bit-identity guarantee to protect, so accumulated rounding is not a hazard.
    return StepSchedule(s_nodes, ele_of, is_snap, is_kick, kick_lo, kick_hi,
                        mode='manual', lattice_length=lattice_length,
                        spe=spe, is_step=is_step, nsep=nsep, step_size=step_size,
                        kick_interval=kick_interval)


# ---------------------------------------------------------------------------------------------
# `auto`: refine where the physics demands it, coarsen where it does not.
#
# PROVISIONAL CONSTANTS. m_steps inherits §6x's calibration (min_steps = 2, measured against wake
# convergence at a waist). eps_tr and kappa are first estimates and have NOT been calibrated
# against anything -- doing so is a measured campaign in its own right, like §6x. They are exposed
# in the config for exactly that reason. Do not treat them as validated.
# ---------------------------------------------------------------------------------------------
AUTO_DEFAULTS = dict(
    m_steps=2.0,        # steps across the longitudinal scale L_z; from §6x
    edge_steps=20.0,    # steps across a formation length at a bend edge; PROVISIONAL
    kappa=8.0,          # kick spacing as a multiple of the snapshot spacing; PROVISIONAL
    h_min=None,         # hard floor; defaults to lattice_length / 2e6
    h_max=None,         # hard ceiling; defaults to step_size
    dyadic=True,        # snap step sizes to h_max / 2^j
    r_floor=1e-3,       # relevance below this counts as zero: do not refine at all
)


def _dyadic_snap(h, h_max, tol=0.01):
    """
    Snap DOWN to h_max / 2^j, with a tolerance so a near-rung value stays on its rung.

    Quantizing to a dyadic ladder is what makes the realised schedule piecewise uniform in long
    stretches, which keeps the history's segment structure simple, gives integer step counts per
    element for free, and makes a single-rung schedule reduce to the uniform case exactly.

    The tolerance is not cosmetic. Without it, an h a hair below a rung snaps a full factor of 2
    finer: measured, a relevance of exp(-2.8/0.18) ~ 1e-7 in a far drift was enough to put h_eff
    just under h_max and cost a uniform 2x over-refinement across the whole drift, for nothing.
    """
    h = np.asarray(h, dtype=np.float64)
    ratio = np.maximum(h_max / np.maximum(h, 1e-300), 1.0)
    j = np.ceil(np.log2(ratio) - np.log2(1.0 + tol))
    j = np.maximum(j, 0.0)
    return h_max / np.power(2.0, j)


def auto_step_profile(s, sz, rho, L_f, distance, lattice_length, opts,
                      bend_edges=None, bend_R=None, bend_phi=None, bend_is_exit=None):
    """
    The local required step h_eff(s), and the pieces that made it, for reporting.

    Four drivers, but only three terms -- "the beam varies fast" and "the wake varies fast" are
    the SAME curve, because the 1D steady-state wake scales as W ~ Q/(R^(2/3) sigma_z^(4/3)), so
    d ln W/ds = -(4/3) d ln sigma_z/ds.

    1. Longitudinal scale.

           L_z = sz / sqrt( (dsz/ds)^2 + sz*|d2sz/ds2| )

       The second term is not decoration. A plain sz/|dsz/ds| criterion DIVIDES BY ZERO at a
       waist minimum -- exactly where the finest steps are needed, since dsz/ds vanishes there.
       L_z instead reduces to sz/|sz'| on the slopes and to the waist half-width sqrt(sz/sz'') at
       the minimum, so it matches waist_width to O(1) and inherits §6x's m_steps calibration.

    2. Bend transient: eps_tr * max(L_f, |s - nearest edge|). Growing linearly away from the edge
       gives geometric refinement -- a handful of steps near the edge rather than a uniformly fine
       window across the whole formation length.

    3. Relevance, which is what PERMITS coarsening: r = (W/W_max) * exp(-d_after_bend/(n_fl*L_f)).
       Far down a drift after a bend, the transient has decayed and there is nothing to resolve.

    Combined RECIPROCALLY rather than by min(), so h_eff is smooth. A min() has kinks, and the
    equidistribution integrator below would chase them.
    """
    s = np.asarray(s, float)
    sz = np.asarray(sz, float)
    h_max = opts['h_max']
    h_min = opts['h_min']

    d1 = np.gradient(sz, s)
    d2 = np.gradient(d1, s)
    L_z = sz / np.sqrt(d1 ** 2 + sz * np.abs(d2) + 1e-300)
    h_1 = np.maximum(L_z / opts['m_steps'], h_min)

    # Distance to the nearest BEND edge -- not to any element boundary. A drift-drift junction
    # has no transient and must not attract refinement.
    if bend_edges is None or len(bend_edges) == 0:
        d_edge = np.full_like(s, np.inf)
        L_f_edge = np.maximum(L_f, h_min)
        upstream_of_entrance = np.zeros_like(s, dtype=bool)
    else:
        be = np.asarray(bend_edges, float)
        dmat = np.abs(s[:, None] - be[None, :])
        near = np.argmin(dmat, axis=1)
        d_edge = dmat[np.arange(s.size), near]
        # The transient scale belongs to the BEND, not to the upstream drift. L_f as passed in
        # follows CSR2D's convention, where in the pre-first-bend drift it is the ACCUMULATED
        # DRIFT LENGTH -- 0.35 m for a 0.35 m drift. Using that as the decay scale made
        # exp(-d_edge/L_f) ~ 0.5 across the entire drift, so the whole drift was refined uniformly
        # and there was no differential at the edge at all (measured). Recompute it from the
        # nearest bend's radius instead: L_f = (24 R^2 * 5 sigma_z)^(1/3).
        Rn = np.asarray(bend_R, float)[near]
        phin = np.asarray(bend_phi, float)[near]
        is_exit = np.asarray(bend_is_exit, bool)[near]

        # The transient scale is DIRECTION-DEPENDENT. Stupakov & Emma, EPAC 2002
        # (SLAC-PUB-9242) give different physics at the two faces:
        #
        #   ENTRANCE (their Case A / Sec 3.1, Fig 4): steady state is reached after the
        #   "overtaking length" L_0 = (24 sigma_z R^2)^(1/3). That is the scale over which the
        #   entrance transient dies away INSIDE the magnet, and it is what this code already
        #   used -- correctly.
        #
        #   EXIT (their Case C, Eq. 10): W ~ (1/(phi_m + 2x)) with x the downstream distance in
        #   units of R, so the amplitude HALVES at x = phi_m/2, i.e. a physical R*phi_m/2. That
        #   is gamma-independent, always positive and pole-free.
        #
        # Using the steady-state form at the exit too was wrong. For a strong bend the two
        # nearly coincide (0.39 m vs 0.50 m at R=1, phi=1), which is exactly why the strong-bend
        # test lattice hid it; for a weak bend it is 5-8x too large, so the exit step came out
        # 5-8x too coarse in precisely the chicane regime (their Fig. 3 shows the wake still
        # changing shape there).
        #
        # Note the code's own unused out-of-bend branch,
        # 3 R^2 phi^4 / (4 (R phi^3 - 6 sigma_z)), descends from their Eq. 13 but has a pole at
        # R phi^3 = 6 sigma_z and goes NEGATIVE below it -- true for a weak bend with a long
        # bunch, which is presumably why it was disabled.
        L_entrance = (24.0 * Rn ** 2 * 5.0 * np.maximum(sz, 1e-30)) ** (1.0 / 3.0)
        L_exit = 0.5 * Rn * np.abs(phin)
        L_f_edge = np.maximum(np.where(is_exit, L_exit, L_entrance), h_min)

        # Do NOT refine the straight section UPSTREAM of an entrance. Measured directly: refining
        # the pre-bend drift from 5 to 93 snapshots changed the wake 5 steps inside the entrance
        # by < 1e-6 relative -- successive differences all zero to 6 decimals.
        #
        # The drift is not irrelevant to the integral; the opposite. It carries 64.6% of the
        # sampled integrand points, reaching back to s' = 0.587 for an observation at 1.025, and
        # the integrand there is large. What it does not have is fast VARIATION: in a drift sigma_z
        # is constant and the tilt evolves linearly, so linear interpolation of the frame is
        # already exact and extra snapshots buy nothing. Contributing a lot is not the same as
        # needing fine sampling.
        #
        # L_z encodes this correctly on its own -- sigma_z' and sigma_z'' both vanish in a drift, so
        # L_z -> infinity and no refinement is asked for. The edge term was overriding that.
        upstream_of_entrance = (~is_exit) & (s < np.asarray(bend_edges, float)[near])

    # Steps across the transient. This is a DIVISOR, not a fraction: the CSR wake turns on over a
    # formation length, so the requirement at the edge is L_f/edge_steps. The first version used
    # `eps_tr * max(L_f, d_edge)` with eps_tr = 0.25, which at L_f ~ 0.4 m gives h_2 = 0.1 m --
    # COARSER than h_max, so it could only ever relax the step and never refine it. Measured: the
    # dipole exit got no refinement at all.
    #
    # max(L_f, d_edge) then relaxes the requirement linearly once further from the edge than a
    # formation length, which is geometric refinement: a handful of steps near the edge rather than
    # a uniformly fine window across the whole L_f.
    h_2 = np.where(np.isfinite(d_edge),
                   np.maximum(np.maximum(L_f_edge, d_edge) / opts['edge_steps'], h_min),
                   h_max)

    # relevance: wake magnitude proxy, damped by distance since the last bend
    inbend = np.abs(rho) > 0.0
    R = np.where(inbend, 1.0 / np.maximum(np.abs(rho), 1e-30), np.inf)
    W = np.where(np.isfinite(R), 1.0 / (np.maximum(R, 1e-30) ** (2.0 / 3.0)
                                        * np.maximum(sz, 1e-30) ** (4.0 / 3.0)), 0.0)
    # carry the wake forward through drifts, decaying over a formation length
    d_since = np.zeros_like(s)
    last_bend_s = -np.inf
    W_carry = np.zeros_like(s)
    w_last = 0.0
    for i in range(s.size):
        if inbend[i]:
            last_bend_s = s[i]
            w_last = W[i]
        d_since[i] = 0.0 if not np.isfinite(last_bend_s) else s[i] - last_bend_s
        W_carry[i] = w_last
    Wmax = W_carry.max() if W_carry.max() > 0 else 1.0
    decay = np.exp(-d_since / np.maximum(L_f, 1e-12))
    r_wake = np.clip(W_carry / Wmax, 0.0, 1.0) * decay

    # Proximity to a bend edge is its OWN relevance, on both sides. The first version used only
    # r_wake, which is zero in the drift BEFORE any bend (no wake has been generated yet), and
    # since r multiplies the whole refinement term that switched off h_2 as well -- so the
    # approach to the first dipole entrance stayed at h_max right up to the boundary. Measured:
    # no upstream refinement at all. Relevance must mean "something is about to happen here" as
    # well as "something just happened here".
    r_edge = np.where(np.isfinite(d_edge),
                      np.exp(-d_edge / L_f_edge), 0.0)
    r_edge = np.where(upstream_of_entrance, 0.0, r_edge)
    r = np.clip(np.maximum(r_wake, r_edge), 0.0, 1.0)
    # A negligible relevance must mean NO refinement, not a little. Left unfloored, an r of 1e-7
    # still pushes h_eff below h_max and the dyadic snap then charges a full factor of 2.
    r = np.where(r < opts['r_floor'], 0.0, r)

    inv_req = 1.0 / h_max + 1.0 / h_1 + 1.0 / h_2
    inv_eff = r * inv_req + (1.0 - r) / h_max
    h_eff = np.clip(1.0 / inv_eff, h_min, h_max)
    if opts['dyadic']:
        h_eff = _dyadic_snap(h_eff, h_max)
    h_eff = np.where(r <= 0.0, h_max, h_eff)     # irrelevant regions get the ceiling exactly
    return h_eff, dict(L_z=L_z, h_1=h_1, h_2=h_2, r=r, d_edge=d_edge,
                       L_f_edge=L_f_edge, r_edge=r_edge, r_wake=r_wake)


def build_auto(lattice_config, distance, lattice_length, n_element, s_scan, sz_scan,
               rho_scan, L_f_scan, step_size=None, kick_interval='trailing', nsep=None,
               **overrides):
    """
    Build a schedule by equidistributing 1/h_eff, so element boundaries land exactly.

    phi(s) = integral of 1/h_eff. Element e then gets
        n_e = max(ceil(phi(b_e) - phi(a_e)), ceil(L_e/h_max), 1)
    steps, placed at equal increments of phi via inverse interpolation. Boundaries are exact by
    construction and every count is an integer, which is what `init_statistics` needs.
    """
    opts = dict(AUTO_DEFAULTS)
    opts.update({k: v for k, v in overrides.items() if v is not None})
    if opts['h_max'] is None:
        opts['h_max'] = float(step_size) if step_size else lattice_length / 100.0
    if opts['h_min'] is None:
        opts['h_min'] = lattice_length / 2.0e6

    # bend edges only: entrance and exit s of every element that actually bends
    ele_keys = [k for k in lattice_config if k != 'step_size']
    lens = np.array([float(lattice_config[k]['L']) for k in ele_keys])
    ends = np.cumsum(lens)
    starts = np.concatenate([[0.0], ends[:-1]])
    bend_edges, bend_R, bend_phi, bend_is_exit = [], [], [], []
    for e, k in enumerate(ele_keys):
        cfgk = lattice_config[k]
        if cfgk.get('type') == 'dipole' and float(cfgk.get('angle', 0.0)) != 0.0:
            ang = float(cfgk['angle'])
            Re = abs(float(lens[e]) / ang)
            bend_edges += [float(starts[e]), float(ends[e])]
            bend_R += [Re, Re]
            bend_phi += [ang, ang]
            bend_is_exit += [False, True]

    h_eff, parts = auto_step_profile(s_scan, sz_scan, rho_scan, L_f_scan,
                                     distance, lattice_length, opts,
                                     bend_edges=bend_edges, bend_R=bend_R,
                                     bend_phi=bend_phi, bend_is_exit=bend_is_exit)

    phi = np.concatenate([[0.0], np.cumsum(0.5 * (1.0 / h_eff[1:] + 1.0 / h_eff[:-1])
                                          * np.diff(s_scan))])
    distance = np.asarray(distance, float)
    starts = np.concatenate([[0.0], distance[:-1]])

    s_list = [0.0]
    ele_list = [0]
    spe = np.zeros(n_element, dtype=int)
    for e in range(n_element):
        a, b = float(starts[e]), float(distance[e])
        pa, pb = np.interp([a, b], s_scan, phi)
        L_e = b - a
        n_e = int(max(math.ceil(pb - pa), math.ceil(L_e / opts['h_max']), 1))
        targets = pa + (pb - pa) * np.arange(1, n_e + 1) / n_e
        nodes = np.interp(targets, phi, s_scan)
        nodes[-1] = b                      # boundary exact
        # guard against a non-monotone node from interpolation flatness
        prev = s_list[-1]
        for k in range(n_e):
            nk = max(float(nodes[k]), prev + opts['h_min'] * 1e-6)
            if k == n_e - 1:
                nk = b
            s_list.append(nk)
            ele_list.append(e)
            prev = nk
        spe[e] = n_e

    s_nodes = np.asarray(s_list, float)
    ele_of = np.asarray(ele_list, np.int64)
    is_step = np.ones(s_nodes.size, bool)
    is_step[0] = False
    is_snap = is_step.copy()

    # kicks: coarser than snapshots by kappa, snapped to snapshot nodes, forced at every boundary
    is_kick = np.zeros(s_nodes.size, bool)
    kick_h = opts['kappa'] * np.interp(s_nodes, s_scan, h_eff)
    nxt = s_nodes[0]
    for i in range(1, s_nodes.size):
        if s_nodes[i] >= nxt or np.any(np.abs(distance - s_nodes[i]) < 1e-12):
            is_kick[i] = True
            nxt = s_nodes[i] + kick_h[i]

    if kick_interval == 'midpoint':
        kick_lo, kick_hi = midpoint_intervals(s_nodes, np.flatnonzero(is_kick),
                                              distance, lattice_length)
    else:
        kick_lo = np.zeros(s_nodes.size)
        kick_hi = np.zeros(s_nodes.size)
        last = 0.0
        for i in np.flatnonzero(is_kick):
            kick_lo[i] = last
            kick_hi[i] = s_nodes[i]
            last = s_nodes[i]

    sch = StepSchedule(s_nodes, ele_of, is_snap, is_kick, kick_lo, kick_hi,
                       mode='auto', lattice_length=lattice_length,
                       spe=spe, is_step=is_step, nsep=nsep, step_size=step_size,
                       kick_interval=kick_interval)
    sch.auto_opts = opts
    sch.auto_parts = parts
    sch.auto_h_eff = h_eff
    sch.auto_s_scan = np.asarray(s_scan, float)
    return sch
