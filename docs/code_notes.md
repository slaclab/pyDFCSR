# Code notes

*Design notes and history that used to sit as long comments in the code, moved here in the 2026-11-27 cleanup. The code keeps a short statement of what each piece does and points here. Parameter histories are in `parameter_notes.md`.*

## `CSR2D.run`

**Not resumable.** `run` restarts its element loop at the first element and `step_count` at 1, so a second call re-tracks an already-advanced beam from the lattice entrance: asked to stop at s = 0.750 after a first call to 0.450, it landed at 0.778 with a density history that was not one forward pass. It refuses instead of returning a plausible-looking wrong answer; construct a fresh CSR2D per stop_time.

**Steps come from the schedule.** Every per-step length and decision comes from the precomputed schedule. `step_count` starts at 1 and increments once per step, so it is the schedule's node index: node i is the end of step i, node 0 the s = 0 entrance handled by `initialization()`. Under mode 'legacy' the schedule reproduces the old uniform node set, kick cadence and steps_per_element exactly. The kick cadence used to be `step % nsep[ele_count] == 0` with `step` restarting per element; the schedule lets 'auto' and 'manual' choose kicks independently of the snapshot spacing, since a kick costs ~5000x a snapshot.

**L_f is refreshed every step**, not only on element entry: sigma_z evolves within an element, and L_f truncates the history and sets the integration reach (57 -> 1362 um inside one bend at shear 50; pinning L_f to the entrance value left the wake 12.9% wrong at shear 20).

**drift_cutoff.** Stop computing wakes once the beam is far enough past a bend that the exit transient has died. The scale is R*phi of the bend just left, not a multiple of the formation length (which is what the old commented form got wrong, and why it was never enabled). Stupakov & Emma Eq. 10 gives W ~ 1/(phi + 2 d/R), halving at d = R*phi/2, and the retarded slippage saturates at the same place. Measured on the 1 rad test dipole, on-axis peak relative to the exit face: 0.338 at d = 0.1 R*phi, 0.173 at 0.2, 0.054 at 0.5; the real decay is faster than Eq. 10 (which assumes phi << 1), so R*phi is a conservative cutoff for a strong bend. Skipped steps and kicks are counted (a skipped kick must be visible: silently returning no wake cost a set of wrong figures in §6f). abs() on both R and phi: R_rec = L/angle carries the sign of the bend, so a negative-angle dipole gave a negative threshold and every step in its drift was blocked (the chicane's B2 and B3 lost every kick past the exit face). Steps, not kicks: the test is per step, and reporting it as "kicks skipped" overstated it 5x.

**The kick length.** The kick is a quadrature of int W ds, so its length must be the arc actually covered. DL*nsep was wrong at the first kick of the run (applied after a single step but weighted by nsep steps, +200% at nsep = 3) and at element boundaries, where `step` restarts and the boundary step is split in two (+50%); on a 4-element test lattice with nsep = 3 it over-weighted the integrated kick by 14.3%. With `kick_interval: midpoint` the kick is centred on the sample point, a midpoint rule (O(h^2)) instead of a trailing rectangle (O(h)); it needs the next kick position, hence a precomputed schedule (§6dd). The trailing length is taken from beam.position, not from the schedule: position accumulates by `position += step_size` and differs from the node i*h in the last bits for 31 of 38 nodes, so substituting would break bit-identity.

**stop_time.** `>=` within a tolerance, not `>`: with step_control's force_nodes a node can land exactly on stop_time, and a strict `>` walked a full step past it. The node grid is fixed, so a stop_time between nodes lands on the first node past it; that used to be silent and overshot by up to a full step (at step_size 0.1, run(stop_time=0.60) landed at 0.700 with a 27% different beam slope, which confounded a convergence study in §6i), so it is now printed.

**Element edges.** The exit edge used to be applied only by the split step that crosses into the next element, and that piece is skipped when it is shorter than 1e-6 m; when a node lies on the boundary (always in modes 'manual' and 'auto') E2 was silently dropped. `_ends_element` uses the same 1e-6, so exactly one of the two paths applies it.

## layout.py

### _causal_edge: why a plain scan

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

### _layout_bounds: two regions, keyed on sin 2a

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

### _layout_bounds: which tilt

Use the SAME tilt the density frame uses (core fit) rather than beams.slope
(all particles). They disagree most exactly where the correlation vanishes.

### _layout_bounds: the near-region length

Near-region length as ONE continuous expression, with no branch on it at all.
An earlier version switched the extent formula at branch_sin_min and
reintroduced a discontinuity (125 mm -> 0.347 mm across the threshold).

  d = (10 sigma_x + x - xmean) |cos 2a| / max(|sin 2a|, branch_sin_min)

This is the legacy d = (10 sigma_x + x - xmean)/|tan 2a| wherever the branches
are resolvable, and it saturates at branch_sin_min as sin 2a -> 0, where the
branches merge and d stops meaning anything.

It is then floored at near_floor sigma_z, which is NOT cosmetic. As |tau| -> 1,
cos 2a -> 0 and the chirp band becomes vertical, so it does exit the beam at
once -- but d is the length of the NEAR region, which must also contain the
NARROW band and the neighbourhood of the 1/|r-r'| pole. Sizing it from the
chirp exit alone starves both. Measured at shear 50, 0.80 m into the dipole
(tau = -0.999, cos 2a = 8.8e-4): the upstream reach collapsed to 15 um =
0.009 sigma_z against sigma_z = 1744 um, so 1-2 of the 88 graded nodes landed
on the causal side and the pole sat 15 um from the region edge, leaving
region 2's uniform 1.74 mm cells to carry it. The wake went visibly blocky.
The floor is the s3 = s - 20 sigma_z that the deleted |tan theta| <= 1 branch
used to provide; removing that branch in 6o dropped it silently, and 6f's
coarser s grid (tau = -1.225 then -0.814) stepped over the region where it
matters -- the defect only bites within ~1% of |tau| = 1.

The (x - xmean) term is kept deliberately: the chirp branch starts at the
observation point, so one starting near the beam edge exits sooner than one
starting at the centre, and d legitimately varies ~1.9x across a wake mesh.
Dropping it makes the wake look ~4x smoother at fixed resolution, but the two
variants converge to the same answer (rel L2 0.045 -> 0.0017 as zbins/xbins
refine), so that smoothness is per-column node-layout jitter being removed
from the roughness METRIC, not accuracy being gained. Optimising the metric
rather than the answer is not worth losing the correct exit condition.

### _layout_bounds: seam_ratio

Seam continuity: keep grading the near region until its cell near_grade*u has
reached 1/seam_ratio of the far region's flat cell, so the far columns never
inherit structure finer than they can carry (Integration_params.seam_ratio).
Only where the far layout IS flat (in a bend); past a bend exit the far region
is itself graded at near_grade from the causal edge, so the seam is continuous
already and extending the near region there only moved the 0.4% node-phase
jitter of the golden drift kicks.

### _layout_bounds: prehistory

prehistory: the lattice start is not a wall. Sources on the synthetic straight section
s' in [-prehistory, 0) are the first snapshot translated along the initial tangent
(Integration_params.prehistory); 0 keeps the old clamp at s' = 0.

### _layout_bounds: the far edge

The far edge is the causal reach measured from s3, and it is clamped at 0
because s' < 0 is before the lattice exists -- interpolate1D returns 0 there
silently, so nodes below 0 are not merely wasted but invisible. The old
s1 = s2 - n_fl*L_f with s2 = s3 - 200 sigma_z reached back 200 sigma_z FURTHER
than the retained history supports, and inverted (s1 > s2) whenever s2 went
negative while s1 clamped.

### _layout_bounds: drift_reach

In a drift after a bend the source of the exit wake is the bend's own arc,
which recedes behind the observer; a fixed n_fl*L_f reach loses it ~1.7 R*phi
past the exit and the wake then changes sign (progress.md, 2026-11-03).
drift_reach extends the far edge back to the bend's entrance; the causal-edge
scan in _far_region_nodes then trims it to where the source still contributes.

### _layout_bounds: fixed drift reach

the regular exit-transient layout always has the n_formation_length = 1 reach in a drift,
so its fine window stays on the last bend's arc whatever n_fl is (at n_fl = 6 it slid
into D31 and B1RE's exit wake came out 65% wrong); earlier sources go to the causal
drift region (Integration_params.drift_causal_reach)

### _region_node_counts: when the cap warning applies

Warn once per run when the cap binds. It binds where |tan 2 alpha| collapses
-- near a bend entrance the chirp band can stretch to d = 10 sigma_x /
|tan 2 alpha| ~ 150 mm while the cell stays at sigma_xi, asking for ~25000
columns. Silently clipping that returns an under-resolved wake that looks
perfectly plausible, which is exactly how it went unnoticed once already.
... but only when this count is what the near region will actually use.
With near_grade set, _near_region_nodes discards `want` entirely and builds a
graded grid whose finest cell IS near_cell*sigma_xi, so the cap never binds
and warning about it is not merely noise: it reports a resolution loss that
is not happening, and advises "use graded nodes" when they are already on.

### _far_region_nodes: two constructions

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

### _far_region_nodes: no causal edge in a bend

Not an exit transient: uniform, and no band solve. The causal edge would
be redundant here anyway -- measured in a bend it sits at u ~ 1.0 L_f
(0.994 at shear 0, 0.783 at shear 20), which is what n_formation_length
already provides.

### _far_region_nodes: far_clip_at_reach

The edge should be the tighter limit; if the reach binds instead, the
integral is truncated by the history budget rather than by causality.
Honest, but a different limitation -- count it rather than hide it.

### _far_region_nodes: far_fine_max

cap the fine-window node count: far_cell*sigma_z is set by the bunch,
but the band it resolves scales with L_f, so a short bunch (sigma_z
19 um in D3) put 3300 nodes in a window that ~300 resolve to 3e-4

### _near_region_nodes: why graded

Rationale: for u larger than the transverse offsets, |r - r'| ~ u, so the
per-column contribution falls as 1/u and equal contributions come from equal
logarithmic intervals. Uniform nodes are therefore under-resolved where the
accuracy is set and over-resolved where the cost is, forcing
N ~ u_max/sigma_xi -- which is why the count had to scale with tilt. Graded,
N ~ 1/near_grade + ln(u_max/u*)/ln(1+near_grade), growing only
logarithmically with tilt.

The outer integral uses np.trapz(..., x=sp), which handles the non-uniform
spacing correctly.

## layout.py (2)

### _geo_reach

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

### _geo_reach: not in the drift

NOT in the drift after the bend: there the frozen straight history is seen from a
different straight line, each column contributes ~ phi^2 lambda'/u and the sum grows
as ln(reach) -- measured on Indranil's gain beam 0.19 m past B1LE, -0.18 MeV/m per
doubling of the reach out to 170 m (55x the 1D wake at 40 R phi), with or without the
velocity terms. Its value is set by the real length of the upstream straight, not by
local physics, so the regular layout (n_formation_length * L_f) is kept there. Inside
the bend the same contribution converges as 1/U (talman_cancellation.md 7.7).

### _straight_region_nodes

s' nodes for the straight-section extension [lo, hi) below the regular layout
(reach_straight). Its sources are far from the observer and their retarded z is nearly
constant, so the column integral is a smooth ~1/u^2 in u = s - s': geometric grading in u
with ratio 1 + near_grade, ~ln(u_lo/u_hi)/near_grade nodes (41 for 10 m behind a 0.5 m
bend at 5%), where the fixed far cell would need thousands.

## bands.py

### _retarded_xi_bands: both roots, always

BOTH roots are always taken. The branch degeneracy of thesis 4.4.2 is a
property of the tilt at the RETARDED time a(s'), which varies column to
column -- the retarded tilt sweeps +19.4 -> -0.22 -> -19.8 across one near
region -- so it cannot be decided from a(s) at the observation point. Doing
so discarded 28-31% of the wake wherever it fired.
The per-column decision needs no new logic: on a degenerate column the
second root either fails the _eq424 guards (rad < 0, l <= 0) and is parked
dead, or lands on top of the first and is collapsed to zero width by
_disjoint_bands.

### _retarded_xi_bands: band_jacobian_min

The band is sized in lab x', but the beam is sized in u at the
retarded time, and z_ret moves with x' through |r - r'|:
    du/dx' = (1 - tau * d|r-r'|/dx') / sigma_xi
(the frame's own change across one band is second order). Where the
line of sight is oblique to a tilted beam -- columns in a bend seen
from the drift after it -- |J| falls to ~0.25, so +-10 sigma_xi of x'
covers only +-2.5 sigma_xi of the beam (18% of the D3 far integral
missed). Scale the band by 1/|J| so it covers the same +-xlim*margin
in u everywhere, and so the same nodes per beam width.

### _retarded_xi_bands: parking a dead branch

A dead branch carries no density whatever band it is given, so it gets
zero width and _integrate_xi_region drops its contribution outright.
Its position still has to be finite and sane: on a dead column the
fixed point had no root to converge to and can wander (45 m has been
observed), which contributes nothing but produces absurd node
coordinates in debug output and risks landing on the 1/|r-r'| pole.
Park it a few band widths off the observation point instead: finite,
in a zero-density region, and never exactly at r = 0.

## ibp.py

### _ibp_integrand: when a column keeps the shipped form

Column-wise choice of form. The identity holds column by column, so a column can keep
the shipped form without any boundary term. Where J gets small on a node that carries
charge, the chi partition inside the column becomes a steep ramp in x' (J changes at
A (1 - R_x^2)/R near the observer, A ~ the beam tilt) whose chi and 1 - chi pieces
cancel each other, so the by-parts column is far less converged than the shipped one.
Measured on the chicane, frozen at B3 (tau = -30) and D3 (tau = -49): with the
partition alone the by-parts wake was off by up to 30x at 3200 nodes. So a column uses
the by-parts form only if |J| > 2 J0 at every node with charge; chi then vanishes there
except on empty nodes, where it only guards the 1/J.

J must also keep ONE SIGN over the whole column. Along a column R_x is monotonic and A
varies slowly, so J crosses zero at most once, but at high tilt it can do so between
two nodes: chicane D3 (tau = -49), a column 40 um from the observer has J from -20 to
+20, its 200 nodes miss the crossing (smallest sampled |J| 3.0), and the 1/J pole
between them made that one column 100x too large. A sign test sees it at any node count.

## deposit_smooth.py

### DF_tracker_comoving: ring buffer (why not a rebuilt stack)

---- ring buffer for the five large field arrays -------------------------
The deque only changes by one append and a few poplefts per step, but
build_interpolant used to rebuild the whole stack with
np.array([e[k] for e in DF_log]) -- O(N) copying for an O(1) change, every
step. Since N is bounded by the history window n_fl*L_f/step_size, total
cost went as (1/step_size)^2, and allocating a fresh 164 MB set of arrays
every step drove peak RSS to 8.75 GB at 200^2, which is what blocked the
fine stepping 6x showed is needed at a waist.

Storage is preallocated and written one slice at a time. The live region is
kept CONTIGUOUS (compacted when the cursor reaches the end) rather than
wrapping modularly, so the interpolant is a zero-copy VIEW and
interpolate3D_comoving_fields needs no change at all.

### DF_tracker_comoving: the frame options

Which normalized frame the fixed (u, w) grid is laid in, per snapshot:
'rms'       w = (z - mean)/std over all particles (the original frame)
'quantile'  the grid spans [z_q, z_(1-q)] widened by frame_margin*span on each side,
            q = frame_quantile; particles are weighted by T(z), 1 on [z_q, z_(1-q)] and
            a C2 smootherstep to 0 across the margin, so the far tails are dropped as
            a source smoothly instead of being clipped. A short or non-Gaussian bunch
            (BC20: full span 208 um against +-5 rms = 515 um) gets bins fitted to
            where its charge is, and quantiles are continuous in the particle
            coordinates, so the frame does not jitter or jump between horns.
'peak'      the quantile window, narrowed to the highest current peak: full weight over
            centre +- peak_halfwidth * FWHM, a taper over peak_taper * FWHM, never wider
            than the quantile window. A long bunch gets exactly the quantile frame; once a
            spike forms, the low-current pedestal behind it is dropped as a source (its
            CSR is small; CSR2D reports its 1D share per kick and warns above
            CSR_computation.pedestal_warn, which also catches a second peak outside the
            window). Centre and FWHM from peakfind.peak_center_fwhm, rate-limited between
            snapshots (|d ln H| <= peak_rate, |d centre| <= peak_shift * H), because the
            history lookup blends neighbouring snapshots' frames.
'auto'      decided once, before the first deposit, by CSR2D._resolve_auto_frame
            (choose_auto_frame): 'peak' unless the charge the peak window would drop
            carries more than auto_pedestal_max of the 1D energy loss or of the 1D field
            on the window at the exit of the first bend (a second horn, a strong
            pedestal), then 'quantile'. The choice is printed and kept for the run; the
            per-kick pedestal diagnostic keeps watching it.

### DF_tracker_comoving: transverse edge taper

Transverse edge taper (progress_cancellation.md §4c-d). The deposit used to end in a
hard cut at u = +-xlim: particles beyond were dropped and the interpolant is zero past
the grid, so wherever the beam reaches the edge rho has a step there that the stored
d rho/du does not contain. The longitudinal integrand is G * d rho/dx, whose +- lobes
cancel by a factor C = 60-1000 (C ~ 188 sigma_z^(2/3)/sigma_xi um), and an error in
d rho/dx with a nonzero integral across the beam is amplified by ~C: clipping 0.3% of
the charge moved the wake by 3%, against 0.19% when the cut is consistent.
So each particle's deposit weight is multiplied by T(u): 1 for |u| <= e - u_taper,
a C2 smootherstep to 0 at |u| = e, with e = xlim - 3 smoothing_sigma delta_u so the
smoothed field is down to ~1e-2 of the ramp's density at the grid edge. The charge
removed is then removed consistently (the stored derivatives are those of the tapered
field), and counts as clipped in the clip_warn diagnostic. A z cut does not need this:
T(z) rho has d/dx = T(z) d rho/dx, still cancelling; frames quantile/peak taper z
for the physical step in the current. u_taper = 0: the hard edge, as before.

### DF_tracker_comoving: frame_blend options

Which parametrisation of the frame is interpolated BETWEEN snapshots.
Deposition is identical either way; only the blend differs.

'coeff'  blends tau = tan(alpha) linearly. tan is the wrong chart across a
         longitudinal waist: the beam's orientation goes +87 deg -> 90 deg
         (vertical) -> -87 deg, and no continuous path in tau can pass
         through 90 deg -- it must pass through 0, i.e. HORIZONTAL. Measured
         at a shear-20 waist, the blended frame then spends 2.75% of the near
         region at |alpha| < 45 deg, orientations the beam never occupies,
         and |tan 2a| = 2 tau/(1 - tau^2) reaches 285 against endpoint values
         of 0.10 because tau crosses the poles at +-1 twice.
'moment' blends (var_z, cov, var_x) linearly and derives
         tau = cov/var_z, sigma_z = sqrt(var_z),
         sigma_xi = sqrt(var_x - cov^2/var_z).
         Same measurement: 0.03% and |tan 2a| <= 1.234. But it pays for that
         with a width defect: a convex mix of two thin ellipses at different
         tilts is FATTER than either, by a(1-a)(dtau)^2 var_z, so mid-interval
         sigma_xi comes out 8x-10x too large at ordinary high tilt and the
         wake stops converging under step refinement. Kept because it is the
         honest full-covariance option and the defect is measured, not
         assumed.
'orient' takes the orientation from the moment ratio and the widths
         log-linearly, which is where each chart is well conditioned. Only
         tau needs the moment chart; sigma_xi and sigma_z do not.

Default stays 'coeff' so every recorded baseline reproduces byte-identically.

### DF_tracker_comoving: vx_model control variate

Control variate for the velocity field.
'none'   deposit px itself: v_x = smooth(rho px) / (smooth(rho) + eps).
'affine' fit px ~ b x + g(z) (g cubic in w) over the core, deposit only the
         residual, and add b x + g(z) back analytically (b to vx_u exactly).
Why: smoothing a ratio is biased by ~ sigma_k^2 grad(ln rho) . grad(px). On a
strongly tilted beam px varies fastest ALONG the beam (x ~ tau z, so even
px = b x has d px/dw = b tau sigma_z), and when the beam is also curved in
(u, w) -- the straight tilt fit leaves a banana -- d^2 ln rho / du dw changes
along the bunch, so the bias in d vx/du acquires a z-trend. Measured in the
chicane D3 (tau = -49): the code's d vx/dx ran -0.03 .. -0.42 /m tail to head
against a true -0.26 .. -0.30, and that trend drove the spurious x-structure of
the far-drift wake. With 'affine' the error is <= 0.02 /m in every slice.
Default 'affine' since 2026-11-04 (progress.md). It also removes the 6% low
bias of the FODO W2 test, which was the smoothing shrinking d vx/dx, so
the W2 ratio there moved 0.1327..0.1356 -> 0.1415..0.1418. Results recorded
before that date were computed with 'none'; set it explicitly to reproduce.

### DF_tracker_comoving: ring shrink test

Shrink is tested on EVERY push, not only when the cursor reaches the end.
The history grows deep during the pre-first-bend drift, where
formation_length is an accumulated drift length and nothing is truncated,
then collapses once a bend sets L_f from sigma_z. Checking only at
compaction meant the cursor needed cap - n further pushes to notice, which
is more than a short run has: the buffer sat at 1.64 GB for 251 live slices
needing 0.40 GB. After shrinking, cap = 2n, so the condition is immediately
false again and this cannot thrash.

## interp3D.py

### interp3D.py: co-moving interpolation

Co-moving (Lagrangian) interpolation of the density history.

A lab-frame blend (the removed bspline_fft path) evaluates each snapshot
in its OWN tilted frame and then averages the two results. That is a
superposition of two shear-displaced copies of the beam, so for a beam whose
tilt rate changes by dp' between snapshots it returns two peaks where there is
one, separated by G*sigma_xi with G = |dp'|*sigma_z/sigma_xi. It is O(1) wrong
once G >~ 1, which a chicane reaches easily.

Here the FRAME is interpolated in time instead of the field. Every snapshot
stores its shape on the same normalized grid u = (xi - xi_bar)/sigma_xi,
w = (z - z_bar)/sigma_z, so index (i, j) refers to the same material point of
the beam at every snapshot. Blending at fixed (u, w) is then exact for any
affine evolution of the beam -- shear plus scaling -- which is what linear
optics does between history steps.

Note the normalized grid is shared in INDEX space only. Each snapshot's
physical cell is still delta_xi_k = 2*xlim*sigma_xi_k/nbins, so no transverse
resolution is given up.
---------------------------------------------------------------------------

### interp3D.py: 'orient' widths

'orient': widths log-linear, as in 'coeff'.

Blending the full covariance INFLATES sigma_xi badly, because a
convex mix of two thin ellipses at different tilts is fatter than
either -- by a(1-a)(dtau)^2 var_z. Measured at shear 20 in the body
of the dipole, mid-interval sigma_xi comes out 8x-10x too large
(20.3 um -> 196.9 um at tau: -9.92 -> -6.58), and the wake then
fails to converge under step refinement. The widths do not need the
moment chart -- only the orientation does -- so take just that.

### interp3D.py: shared basis weights

--- evaluate both snapshots at the SAME (u, w), then blend ---

This used to be ten bspline_eval_single calls (5 fields x snapshots k, k+1). All
ten share the same (u_cell, w_cell), hence the same 4x4 stencil and the same eight
basis weights -- but cubic_bspline_w(w_cell - j) sat INSIDE the dj loop, so every
call recomputed the four w-weights sixteen times, and LLVM cannot common that up
across ten separate calls over ten different arrays. It came to ~200 weight
evaluations per query where only 8 distinct values exist. Profiling put this
kernel at 67% of wake-mesh time, so this was the whole hot spot.

Weights are now computed once and all ten accumulators filled in one pass.

BIT-IDENTICAL, deliberately: each accumulator receives the same terms in the same
order as before, and the multiply keeps the original left-to-right association
(d * wi) * ww -- the algebraically equal d * (wi * ww) would change the last bits.
cubic_bspline_w is pure, so hoisting it cannot change a value.

## wake.py

### _integrate_xi_region: x' refinement and the by-parts form (history)

the x' refinement (near_adapt_x / far_adapt_x) and the by-parts form now coexist: a
refined column -- one where the retarded density jumps between x' nodes, i.e. a horn
crossed obliquely -- keeps the shipped form on its own refined nodes (the by-parts
form's gain is where the density is smooth across the band), every other column
takes the by-parts integrand on the uniform nodes. Until 2026-11-23 ibp switched
the refinement off, and on the loss beam that brought the aliased cells back
(max |dW| 34% of the horn's W at 200 nodes, ibp_nodes.py).

Since the error-based refinement (2026-11-26) every live column of a refined region is refined, so `ibp` has no effect there; `_ibp_consistency_check` switches the refinement off.
