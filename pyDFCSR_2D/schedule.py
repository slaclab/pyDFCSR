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
                 mode, lattice_length, spe, is_step, nsep=None, step_size=None):
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

        self.dl = np.zeros_like(self.s_nodes)
        self.dl[1:] = np.diff(self.s_nodes)
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


def build_legacy(distance, nsep, lattice_length, step_size, n_element):
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

    # trailing intervals: each kick owns the arc since the previous kick (the §6dd form)
    kick_lo = np.zeros(n)
    kick_hi = np.zeros(n)
    last = 0.0
    for i in np.flatnonzero(is_kick):
        kick_lo[i] = last
        kick_hi[i] = s_nodes[i]
        last = s_nodes[i]

    return StepSchedule(s_nodes, ele_of, is_snap, is_kick, kick_lo, kick_hi,
                        mode='legacy', lattice_length=lattice_length,
                        spe=spe, is_step=is_step, nsep=nsep, step_size=step_size)
