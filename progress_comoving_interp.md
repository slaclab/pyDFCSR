# Fix noisy CSR wakes at large x–z tilt

*(Originally titled "co-moving (Lagrangian) density-history interpolation" — that turned out to be
the fix for a different defect. The noise fix is the integration-quadrature alignment in §3.0.)*

**Started:** 2026-09-08 · **Last updated:** 2026-09-09

### What is actually implemented right now

| | status |
|---|---|
| Steps 1–3 — diagnosis | ✅ done (tests only, no production code touched) |
| **Step 4 — quadrature band fix** (`_retarded_xi_band`, per-column nodes in `get_CSR_wake`) | ✅ **implemented** |
| **Step 4 addendum — polar patch** for the `r → 0` singularity | ✅ **implemented** |
| Step 4c — fix the confounded acceptance test | ✅ done |
| Step 4e — diagnose the remaining `sp3` failure | ✅ done |
| Step 4f — verify thesis Eq 4.24 | ✅ done |
| **Step 5 — co-moving interpolant** (`DF_tracker_comoving`, `interpolate3D_comoving_fields`) | ✅ **implemented, `method: bspline_comoving`** |
| Step 5c/5d — Eq 4.24 branches vs the measured integrand, at one tilt then swept over 6 shears | ✅ done (141 predictions inside support, 0 missed) |
| **Step 6 — two-branch band location from Eq 4.24** (`_retarded_xi_bands`, `_eq424`, `_disjoint_bands`) | ✅ **implemented**; design in §6a/6b, result in §6c |
| Step 7 — remaining secondary fixes (`lattice.py` key order, `bilinear_single` OOB on the *old* paths, a pre-existing test failure) | ❌ not started |
| Step 8 — validate and document | ❌ not started |

**Three methods are now selectable** via `particle_deposition: method:` — `legacy`, `bspline_fft`,
`bspline_comoving` — so all three stay runnable side by side for benchmarking. The first two are
byte-unchanged by this work apart from `get_CSR_wake`'s node placement, which both B-spline paths share.

**Where it stands.** `bspline_comoving` eliminates ghosting exactly (G-independent to six digits) and,
once the band locator was made co-moving too, cuts wake roughness 34× and converges at 100² instead of
400². Step 6 then closed the correctness gap: the single-band locator was missing a physical chirp
branch worth **rel L2 1.04 of the whole wake**, and with both Eq 4.24 branches located the transverse
domain is invariant to 0.00000 over its last doubling and the result agrees with the old ±20σ_x bands
to **1e-5** at low tilt, where those old bands are trustworthy. Cost 1.5×. The two older methods
(`legacy`, `bspline_fft`) are untouched and still selectable.

The two-branch wake is ~25× *smaller* than the single-branch one and of opposite sign — the branches
nearly cancel. That was confirmed independently against a brute-force integral over the full ±20σ_x
rectangle, which converges to the two-branch value and excludes the single-branch one by 1–2 orders.

**Is the wake smooth?** Yes (§6d). Absolute roughness is 16× lower than single-branch, converges at
second order, and what remains is genuine curvature: the second difference flips sign 11 times in 37
(noise would flip ~half), and a degree-8 polynomial fits to 1.3% of peak. Relative roughness at this
tilt has gone 3.0 (old bands) → 0.72 → 0.18 → **0.020**.

**A production bug found late and fixed (§6g diagnosis, §6h fix).** `formation_length` was computed once
per element on entry, so it never saw in-bend bunch lengthening — 181.7 mm instead of 395.5 mm at shear
20, 3× low at shear 50 — which truncated the retained history to 3 snapshots and left the wake **12.9%
wrong at the shipped default**. Now refreshed every step: `L_f` matches the analytic value exactly, the
history deepens to 6–8 snapshots, and the default `n_formation_length = 1.5` is converged to 0.00000.
Every Step 6 conclusion survives; the absolute magnitudes quoted in §6c–§6f were measured pre-fix and are
~13% low.

**Known open items.** (i) The problem is now a **cancellation** one, so accuracy must be judged on the
total, never on one branch. (ii) `CSR_integration: zbins` should default to **400**, not 200 — at 200 the
quadrature noise is still 4× above the curvature floor. (iii) Axis D, the near-patch radius invariance,
degraded from 0.011 to 0.085 and is not yet attributed; `near_patch = 5.0` may no longer be right.

### Where the diagnosis landed

The noise is **not** ghosting (§2.1), which the original plan assumed — that was disproved in Steps 1–2.
It is a **sampling failure of the outer `(x′, s′)` quadrature** (§2.0, §2.0b): the integration bands are
sized from `σ_x` while the deposition grid is sized from `σ_ξ`, leaving as few as **0.38 quadrature
nodes inside the beam** with 99.3–100% of nodes returning a hard zero. §4.4.2 of the thesis in fact
prescribes the *slice* rms width (= σ_ξ), so the Step 4 fix is **restoring the thesis's own
prescription**, not inventing a scheme.

Step 4 fixed that (cell 13.2 σ_ξ → 0.21 σ_ξ, noise down 3–5×, 1.5× faster, exact to 0.035% at low
tilt) but was **not sufficient**: it located only one of up to four nonzero `x′` intervals per column,
missing a chirp branch worth up to **67%** of the contribution in region `sp3` (Step 4e). Step 5
(co-moving interpolant) collapsed the four candidate branches to the two physical ones by removing
ghosting, and Step 6 then located both — which turned out to change the high-tilt wake by rel L2 1.04,
i.e. the missing branch was as large as the answer itself.

**One-paragraph summary of the diagnosis.** The noise is not an interpolation error. The ξ-frame
deposition shrank the density support by the tilt amplification factor `σ_x/σ_ξ` (65.9× in the test
case) while `get_CSR_wake` continued to size its `(x′, s′)` integration bands from `σ_x`. The result
is a transverse quadrature cell of 13.2 σ_ξ and **0.76 integration nodes inside the beam**, with
99.98–100% of nodes returning a hard zero. The undersampling the new deposition was built to fix was
relocated from the deposition grid to the outer quadrature. Ghosting (lab-frame time blending of a
shearing beam) is a separate, genuine O(1) accuracy defect that affects the old and new schemes
*equally* and is therefore not the regression.

This is a focused work log for one investigation. The broader project history lives in
`progress.md`; a summary entry will be appended there when this work concludes.

---

## 1. Problem statement

`pyDFCSR_2D` reconstructs the CSR wake by integrating over the *history* of the beam density,
evaluated at retarded times and positions. The history is a stack of 2D snapshots queried by a
3D interpolant.

- **Old scheme** (`deposit.py`, `interpolate3D`): CIC deposition on a uniform **lab-frame**
  `(x, z)` grid spanning ±`xlim`·σ_x. Smooth wakes, but at high tilt σ_x ≫ σ_ξ, so the transverse
  cell is ≈ σ_x/10 and the beam core is badly undersampled.
- **New scheme** (`deposit_smooth.py`, `interpolate3D_transformed`): remove the x–z tilt with a
  per-snapshot polynomial ξ = x − p_k(z), deposit with a cubic B-spline on a ±`xlim`·σ_ξ grid,
  smooth and differentiate by FFT. Density and derivatives are genuinely well resolved.

The new scheme produces **noisy, physically wrong wakes whenever the beam has a large x–z tilt** —
exactly the regime it was built for. `progress.md` records ~27 investigation steps that correctly
localized the *symptoms* (far-retarded-time queries, sensitivity to integration resolution,
`poly_degree=0` stable / `poly_degree=3` divergent) but treated them with parameter band-aids
(`xlim` 5→10, `smoothing_sigma` 3→10, integration grid 100²→300²/500²), each trading one regime
for another.

**Goal:** remove the defect so the well-resolved density produces smooth wakes with no
regime-dependent parameter tuning, and keep the result autodiff-friendly for development goal #2.

---

## 2. Root cause

### 2.0 PRIMARY CAUSE (confirmed in Step 3): the CSR integration bands are sized by σ_x, but the deposition grid is sized by σ_ξ

This section supersedes the original diagnosis. Measured at s = 0.70 m in a 1 rad dipole with a
`shear z:x = 20` beam:

> **Correction (2026-09-09).** The derivation originally written here was wrong, and the corrected
> version is in §2.0b below. It compared the integration band width in `x′` against the deposition
> grid's width in **ξ**, which is not a valid comparison: the interpolant applies `ξ′ = x′ − p(z′)`
> before the lookup, so the grid's support in lab `(x′, z′)` space is a *wide slanted ribbon*, not a
> narrow one. A query at large `x′` can and does land inside the grid. The conclusion survives, but
> for a different reason, and the numbers were understated. **The load-bearing evidence was never the
> ratio below — it was the directly counted hard-zero fraction, which requires no such argument.**

| quantity | value |
|---|---|
| σ_x (lab frame, tilt included) | 826.6 µm |
| σ_ξ (tilt removed) | 12.5 µm |
| **tilt amplification σ_x/σ_ξ** | **65.9 ×** |
| integration band half-width (`20·σ_x`, region 1) | **16 531 µm** |
| integration cell along x′ at 100 bins | **13.2 σ_ξ** |
| **fraction of integration nodes returning a hard 0.0** | **99.32 – 100 %** ← the real evidence |

The transverse integral is being evaluated with **less than one quadrature node inside the beam.**
Effectively 100% of the `(x′, s′)` integration mesh falls outside the deposition grid and returns
`bilinear_single`'s hard `0.0`; the entire wake comes from the handful of nodes that happen to land
on the density ribbon. Which nodes those are changes discontinuously as the observation point moves
across the wake mesh. **That is the noise.**

### 2.0b The correct derivation: the light cone pins z_ret, so the support is thin *in the integration plane*

The right question is not how wide the deposition grid is in ξ, but: **at fixed `s′`, over what range
of `x′` is the integrand nonzero?** That is what a rectangular `(x′, s′)` mesh actually samples.
Measured directly by scanning `x′` across the full `±20σ_x` band at one wake point:

```
   s_prime  (s-s_prime)   nonzero x-extent  /sigma_xi  #intervals   z_ret span
    in sp1     0.243390              63.4u        5.1           1        8.01u
    in sp1     0.134363             110.6u        8.8           1        7.58u
    in sp2     0.055562             120.5u        9.6           1        3.38u
   near s3     0.004118             128.1u       10.2           1        0.61u
    in sp3     0.000200             592.3u       47.2           1      376.06u
```

**Why it is thin despite the tilt removal.** At fixed `s′`, `z_ret = s′ − t + |r − r′|`, and
`|r − r′|` is dominated by the arc distance `|s − s′|`, varying only weakly with `x′`. The last column
shows `z_ret` moving by just **0.6–8 µm** across the whole nonzero region. Tilt removal would let a
distant `x′` map into the grid, but only if `z_ret` followed the tilt line: `Δx′ = p′·Δz_ret`. Reaching
`Δx′ = 16 000 µm` at `p′ ≈ 1.6` needs `Δz_ret ≈ 19σ_z` — far outside the beam. So the grid's support is
a wide slanted ribbon in `(x′, z′)`, but the map `(x′, s′) → z_ret` is nearly `x′`-independent and the
ribbon is thin, and contiguous, **in the integration plane.**

Corrected node counts, using the measured extents rather than the ξ-grid width:

| region | band width | nonzero extent | fraction | nodes | **nodes in support** |
|---|---|---|---|---|---|
| R1 (`x0 ± 20σ_x`, over sp1) | 33 062 µm | 63.4 µm | 0.19 % | 200 | **0.38** |
| R2 (`x0 ± 5σ_x`, over sp2) | 8 266 µm | 120.5 µm | 1.5 % | 100 | **1.5** |
| R3+R4 (over sp3) | 10 745 µm | 592 µm | 5.5 % | 200 | **11** |

Step 3 reported 0.76 nodes for R1 from the ξ-grid proxy; the correct value is **0.38**. The
undersampling was twice as bad as first stated.

**And this yields a lead on the one remaining failure (Step 4c axis A).** In `sp3`, close to `s′ = s`,
`|r − r′| → 0` so `∂z_ret/∂x′` blows up: the `z_ret` span jumps from ~1 µm to **376 µm** and the
nonzero extent from 128 µm to 592 µm. The light cone stops pinning `z_ret`. That means
`_retarded_xi_band`'s central assumption — locate the ribbon from the *column-centre* `z_ret` because
`z_ret` barely varies across a column — **holds in sp1 and sp2 and breaks down precisely in sp3**,
which is the only region where axis A fails. This is better supported than the near/far-cancellation
guess in Step 4c, and it is falsifiable: it predicts the failure should disappear if the band location
is iterated per node rather than per column in `sp3` (option (c) of §3.0, decision 1).

`get_CSR_wake` (`CSR.py:460–608`) sizes every integration band as a multiple of
`self.beam._sigma_x` — `x1_w = x0 ± 20·sigma_x`, `x3_l = x0 ± 5·sigma_x`, and so on. That was
correct for the legacy scheme, whose deposition grid is *also* ±`xlim`·σ_x: the density support and
the integration band were the same size, so ~50 nodes landed inside the beam. The new scheme shrank
the density support by the tilt amplification factor (66× here) and **left the integration bands
untouched.** The undersampling that the ξ-frame deposition was built to *fix* was simply relocated
from the deposition grid to the outer quadrature.

This is not an interpolation error. It is a **sampling failure of the outer 2D integral.**

**It explains every band-aid in `progress.md`, all of which turn the same single knob — the number
of quadrature nodes inside the density support:**

| band-aid | why it helped |
|---|---|
| integration grid 100² → 300²/500² (Step 24) | more nodes ⇒ more of them land in the ribbon |
| `xlim` 5 → 10 | widens the deposition grid ⇒ wider support |
| `smoothing_sigma` 3 → 10 | broadens ρ in ξ ⇒ wider support |
| `poly_degree` 0 stable, 3 divergent | deg 0 removes no tilt, so σ_ξ ≈ σ_x and support matches the band (legacy-like); deg 3 removes the most tilt, giving the thinnest ribbon and the worst sampling |

None of them addresses the mismatch, which is why each traded one regime for another.

**Consequence for the plan:** the co-moving interpolant of §3 does **not** fix this. It fixes §2.1,
which is a real but separate defect. The fix for the noise is to build the integration bands from the
*same* tilt-removed transform the deposition uses — see §3.0.

### 2.1 SECOND, INDEPENDENT DEFECT: the time interpolation is done in the lab frame, which ghosts a shearing beam

`interp3D.py:275–294` evaluates **each snapshot at the lab query point** and then blends:

```python
xi_k  = xval[i] - eval_poly(poly_coeffs[k],   zval[i])   # snapshot k's frame
xi_k1 = xval[i] - eval_poly(poly_coeffs[k+1], zval[i])   # snapshot k+1's frame
val_k  = bilinear_single(data[k],   xi_idx_k,  z_idx_k,  ...)
val_k1 = bilinear_single(data[k+1], xi_idx_k1, z_idx_k1, ...)
result[i] = (1-alpha)*val_k + alpha*val_k1
```

Each term is the correct lab density *of its own snapshot*. Their linear combination is **not** the
density at the intermediate time — it is a **superposition of two shear-displaced copies of the
beam**, the classic ghost/double-image of linear frame blending. For a beam whose x–z shear p′
changes by Δp′ between snapshots, the two images are separated in x by Δp′·z. Define the
**ghosting parameter**

```
G  =  |Δp′| · σ_z / σ_ξ        (per history step)
```

- `G ≪ 1` → images overlap, blend is fine.
- `G ≳ 1` → the interpolant returns two peaks where there is one, and **zero density where the beam
  actually is**.

`progress.md` Step 20 recorded the signature without naming the mechanism: a query point maps to
"ξ≈42 µm (core, high density) at one step but ξ≈152 µm (3σ tail, near zero) at the adjacent step,"
with the slope swinging +5.5 → −10 between steps. With σ_z ≈ σ_ξ ≈ 50 µm that is G ≈ 15.

### 2.2 The test that "validated" the interpolator could not detect this

`test/test_interp_tilted_gaussians.py` built its "exact" reference as

```python
rho_exact = (1 - alpha) * analytical_density(x, z, slopes[k])   \
          +      alpha  * analytical_density(x, z, slopes[k+1])
```

i.e. **the reference was itself the lab-frame blend** — the approximation under test. The test
therefore verified that the interpolator reproduces the very thing that is wrong, which is why it
reported the new method as "15× more accurate than legacy" at slope +5.5 → −10 and the defect
survived. Line 174 even computed the physically correct `slope_blended` and never used it.

### 2.3 Secondary defects

| # | Defect | Location | Effect |
|---|--------|----------|--------|
| a | Chain rule for ∂ρ/∂z baked in per snapshot with p′_k, then two *lab-frame* values blended. A co-moving blend must apply the chain rule with the **blended** p′_α. `get_poly_deriv_blended()` was written for exactly this and is never called. | `deposit_smooth.py:244–246`; `CSR.py:697–698`; `interp3D.py:193` | inconsistent derivatives; worst where p′ varies fastest |
| b | Out-of-bounds returns a **hard 0.0**, not a clamp or taper. The ±5σ_ξ grid is a narrow slanted band in lab space, so far-retarded queries fall outside it. | `interp3D.py:159–160` | step discontinuity in the integrand → quadrature noise (Step 26/27's finding) |
| c | **Grid registration bug.** `histogram_bspline_2d` deposits at bin *centers* (`-0.5` at `deposit_smooth.py:46–47`, spacing `(end-start)/nbins`), but `xi_grids = np.linspace(start, end, nbins)` gives *nodes* with spacing `(end-start)/(nbins-1)`. That array feeds the trapz normalization, the spectral `dx`/`dz`, `p'(z_grids)`, and `min_xi_arr`/`delta_xi_arr`. Legacy `histogram_cic_2d` has no `-0.5`, so this half-bin shift is new. | `deposit_smooth.py:46–47, 197–200, 216, 245, 311–314` | ½-cell misregistration + 1/(n−1) derivative scale error, systematic |
| d | `poly_degree=3` default: the cubic is fit globally then **extrapolated to ±5σ_z** where there are no particles, so p(z), p′(z) blow up in the z tails. Also inconsistent with the deg-1 slope used by the wake mesh (`CSR.py:374`) and integration bands (`CSR.py:469–472`). | `deposit_smooth.py:158, 181` | garbage frame map and chain-rule term at grid z-edges; explains "deg 0 stable, deg 3 divergent" |
| e | `dsum = trapz(trapz(density))`, `density /= dsum` renormalizes clipped charge back to 1. The clipped fraction differs snapshot to snapshot. | `deposit_smooth.py:216–218` | per-snapshot normalization jitter → temporal noise |
| f | `epsilon = max(density_smooth)/velocity_threhold` and the mask `(ρ/(ρ+ε))³` are snapshot-dependent envelopes on `vx`, `∂vx/∂x`. | `deposit_smooth.py:230–239` | temporal inconsistency in the W2/div(v) term |
| g | **C⁰ interpolant ⇒ quadrature jitter.** `bilinear_single` is continuous but its first derivative jumps across every cell boundary, so the integrand has kinks on a lattice of lines in the `(x', s')` plane. Trapz is 2nd-order only for smooth integrands; across a kink it is 1st-order and the error depends on *where* the kink falls relative to the quadrature nodes. As the observation point steps across the wake mesh, the kink lattice shifts against the fixed nodes, so quadrature error jitters point-to-point. Worse for derivatives: `interpolate3D_transformed_with_derivs` finite-differences the bilinear interpolant, giving piecewise-constant (discontinuous) ∂ρ/∂ξ. | `interp3D.py:154–175, 348–354` | point-to-point wake noise; explains why Step 24's 100²→300² integration grid helped so much, at 9× cost |

---

## 3. Planned fix

Two independent fixes are needed, in this order:

- **§3.0 — align the CSR integration quadrature with the deposition support.** Fixes the *noise*
  (§2.0). This is the primary fix and must come first.
- **§3.1–3.4 — the co-moving interpolant, added as `method: 'bspline_comoving'`.** Fixes the *O(1)
  accuracy error* from ghosting (§2.1) and the secondary defects (§2.3). Does not fix the noise.

### 3.0 PRIMARY FIX: resolve the density support with the `(x′, s′)` quadrature

**The requirement.** `get_CSR_wake` must place enough quadrature nodes inside the retarded density
support to integrate it. Today it places `0.76`. The target is the ~50 that legacy achieves, i.e.
a transverse integration cell of order `σ_ξ/5`, *not* `13.2 σ_ξ`.

**Why simply shrinking the bands does not work.** The obvious move — replace `20·sigma_x` with
`20·sigma_xi` — is wrong. The bands are not sized to cover the *current* beam; they cover the
**union of the beam's support over the whole retained history**, because a column at `s′` is
evaluated at retarded time `t_ret(x′, s′)` and that time sweeps back over the formation length. Over
that window the centroid moves and the tilt `p′` changes, so in lab `x′` the union of supports is
genuinely wide — which is exactly why `20·sigma_x` "worked" and why it is generous rather than
arbitrary. Shrinking it to `σ_ξ` would clip real contributions and trade noise for bias.

The support is **thin but slanted and curved** in the `(x′, s′)` plane: at each `s′` the density
occupies `|x′ − p(z_ret) − ξ̄| ≲ xlim·σ_ξ`, a ribbon of width `~σ_ξ` whose centreline moves with
`s′`. A **rectangular mesh in lab `x′` cannot resolve a thin slanted ribbon without resolving the
whole bounding box** — that is the entire problem in one sentence, and it is why every band-aid in
`progress.md` amounted to refining the bounding box.

**The fix: integrate along the ribbon, not across the bounding box.** Change the transverse
integration variable from lab `x′` to the tilt-removed `ξ′`, so the mesh follows the support:

```
for each s' column:
    t_ret, z_ret  <- light-cone solve at the column centre
    k, alpha      <- history index and time fraction for t_ret
    p             <- poly_coeffs_interp[k]            # already stored
    xi_lo = min_xi_arr[k]                             # already stored
    xi_hi = xi_lo + (n_xi - 1) * delta_xi_arr[k]      # already stored
    xi_nodes = linspace(xi_lo, xi_hi, nx)             # exactly the deposition support
    x'_nodes = xi_nodes + eval_poly(p, z_ret)         # nodes now land ON the ribbon
    inner trapz over x'_nodes for this column
outer trapz over s'
```

The mesh becomes **per-column** rather than a single rectangle, so the inner `np.trapz` needs its own
`x=` array per column instead of one shared `xp_w`/`xp_n`. That is the main structural change to
`get_CSR_wake`. Node count stays the same; the nodes simply stop being wasted on empty space. The
existing 3-region / 4-region `s′` decomposition and the chirp-band logic are untouched — only the
transverse extents and node placement change.

**This fix is fully standalone — it does not need the co-moving tracker.** `DF_tracker_smooth`
already stores everything required, per snapshot: `poly_coeffs_interp`, `min_xi_arr`,
`delta_xi_arr`, `min_z_arr`, `delta_z_arr` (`deposit_smooth.py:311–317`), plus `min_x` and `delta_x`
for the time axis. Reading the ξ-extent straight off `min_xi_arr` / `delta_xi_arr` is also *better*
than recomputing `xlim·σ_ξ`: it guarantees the quadrature nodes bracket the deposition grid exactly by
construction, so the two can never drift apart again — which is the failure mode this whole
investigation is about.

**Design decisions still open (resolve before implementing):**

1. **`z_ret` depends on `x′`, so the mapping is implicit.** Options: (a) one Newton/fixed-point pass
   per column using the column-centre `z_ret`, accepting a slightly off-centre ribbon; (b) widen the
   ξ′ range by a safety factor (e.g. 2×) so a one-pass estimate still brackets the support;
   (c) iterate to convergence per column. Start with (b) — cheapest and robust — and measure the
   clipped fraction to confirm it is negligible.
2. **Two snapshots straddle each `t_ret`.** The interpolant blends snapshots `k` and `k+1`, whose
   ξ-extents differ. The node range must cover the **union** of the two, or the blend will be clipped
   on one side. Cheap and safe: `xi_lo = min(over k, k+1)`, `xi_hi = max(over k, k+1)`.
3. **Jacobian.** Substituting `x′ = ξ′ + p(z_ret)` at fixed `s′` has unit Jacobian
   (`∂x′/∂ξ′ = 1`) only if `p(z_ret)` is treated as constant within the column. If the per-column
   `z_ret(x′)` variation is retained, the Jacobian is `1 + p′·∂z_ret/∂ξ′` and must be included.
   Options (a)/(b) above make this exactly 1 by construction; option (c) does not.
4. **The `s′` band extents may also need revisiting.** `s2 = s − 500·σ_z`, `s3 = s − 20·σ_z` etc. use
   `σ_z`, which is *not* tilt-amplified, so the longitudinal sampling is far less pathological than
   the transverse. Step 3 did not measure longitudinal nodes-in-support separately — worth checking
   once the transverse fix is in, but not expected to be the issue.

**Diagnostic to keep permanently.** Log "integration nodes inside the density support" per region,
as `test_integrand_anatomy.py` now does. It is a single number that would have caught this
immediately, and it should gate any future change to either the bands or the deposition grid.

### 3.1 The co-moving interpolant

Per snapshot *k*, store the *dimensionless shape* on a **fixed normalized grid** shared by all
snapshots, `u ∈ [−xlim, xlim]`, `w ∈ [−zlim, zlim]`:

```
u = (ξ - ξ̄_k)/σ_ξk ,   ξ = x - p_k(z)          w = (z - z̄_k)/σ_zk
ρ̂_k(u, w)  with  ∫∫ ρ̂_k du dw = 1
plus  ∂ρ̂_k/∂u,  ∂ρ̂_k/∂w   (normalized frame — NOT the pre-baked lab-frame ∂ρ/∂z)
frame params: p_k (deg 1), ξ̄_k, σ_ξk, z̄_k, σ_zk
```

Query at `(x, z, t)` with `α` the time fraction between snapshots *k*, *k+1*:

```
p_α  = (1-α)p_k + αp_{k+1}          ξ̄_α, z̄_α: linear in α
σ_ξα = exp((1-α)ln σ_ξk + α ln σ_ξk₊₁)      (log-linear: positive and smooth)
u = (x - p_α(z) - ξ̄_α)/σ_ξα        w = (z - z̄_α)/σ_zα
ρ̂ = (1-α) ρ̂_k(u,w) + α ρ̂_{k+1}(u,w)
ρ         = ρ̂ / (σ_ξα σ_zα)                          # unit-Jacobian shear ⇒ J = σ_ξα σ_zα
∂ρ/∂x     = ρ̂_u / (σ_ξα² σ_zα)
∂ρ/∂z|_x  = [ ρ̂_w/σ_zα - p′_α(z)·ρ̂_u/σ_ξα ] / (σ_ξα σ_zα)
```

Properties:
- **Exact for any affine beam evolution** (shear + scaling) — i.e. exact for linear optics, the
  dominant evolution between history steps. Ghosting is eliminated by construction, not reduced.
- All snapshots share one normalized grid ⇒ no per-timestep grid metadata, and out-of-bounds is
  consistent across snapshots (addresses defect b's inconsistency).
- The chain rule uses the blended p′_α (fixes defect a).
- `G ≳ 1` no longer matters, so no `xlim` / `smoothing_sigma` / integration-grid retuning per regime.

**The shared grid is *normalized*, not physical — no resolution is given up.** The grid shared
across snapshots is the *index* space `(u, w)`. Snapshot *k*'s **physical** cell size is still set
by that snapshot's own beam size:

```
Δξ_k = 2·xlim·σ_ξk / nx      ← identical to what 'bspline_fft' does today
Δz_k = 2·zlim·σ_zk / nz
```

The physical grid still breathes and shears with the beam, and transverse resolution stays
`σ_x/σ_ξ` better than legacy at high tilt. Normalization changes only the *labelling*: index
`(i, j)` now refers to the same material point of the beam at every snapshot, which is exactly what
makes the time blend ghost-free. Side benefit: this permanently retires the "re-interpolate all
history when σ changes by 2×" path (`deposit.py:339–391`, `DF_interp`).

*Caveat:* normalizing by σ makes the blend exact for affine evolution. If the beam *shape* in
normalized coordinates changes fast (strong nonlinear compression forming a spike), the blend is
still only first-order in Δt — but that is far weaker than the current `G ≪ 1`, which fails
outright in a chicane.

### 3.2 The evaluation kernel (fixes defect g)

Replace `bilinear_single` on this path with evaluation against the **same cubic B-spline kernel used
for deposition**: 4×4 stencil, C² continuous, with `∂ρ̂/∂u`, `∂ρ̂/∂w` obtained **analytically by
differentiating the kernel** rather than by finite differences.

- Integrand becomes smooth ⇒ trapz recovers full 2nd-order convergence ⇒ we should be able to return
  to ~100² integration bins instead of 300²–500². That pays for the 4× stencil cost, so this is
  likely net-faster.
- The point-to-point quadrature jitter disappears rather than being averaged down.
- Analytic derivatives with no finite differences and no branching is what makes this path
  autodiff-friendly.

Note: evaluating with B-spline weights on values that were themselves B-spline-*deposited* applies
the kernel twice, i.e. extra smoothing. Compensate by reducing `smoothing_sigma` — the FFT Gaussian
already dominates the effective kernel width, so this is a one-parameter recalibration. If exact
interpolation of grid values ever matters, the proper alternative is a tridiagonal B-spline
prefilter; recorded as an option, not done now.

### 3.3 Registration convention (defect c), applied consistently

```python
delta = (end - start) / nbins                     # matches histogram_bspline_2d
grid  = start + (np.arange(nbins) + 0.5) * delta  # bin centers
# use this `delta` for spectral dx/dz and for the stored delta_* metadata
```

### 3.4 Normalization (defect e)

Replace the trapz renormalization with the exact, grid-independent
`density /= npart * delta_u * delta_w`, and record the clipped-charge fraction
(`1 - sum(density)*delta_u*delta_w`) per step; warn above a tolerance instead of silently
renormalizing it away.

### 3.5 Files to change

| File | Change |
|------|--------|
| `pyDFCSR_2D/deposit_smooth.py` | New `DF_tracker_comoving`. Reuse `cubic_bspline`, `histogram_bspline_2d`, `smooth_and_differentiate`. Deposit in normalized `(u, w)`; store ρ̂, ρ̂_u, ρ̂_w + frame params. Fix (c) registration, (e) normalization, (f) vx envelope, (d) default `poly_degree=1`. |
| `pyDFCSR_2D/interp3D.py` | New `interpolate3D_comoving(...)` and `interpolate3D_comoving_with_derivs(...)`. Reuse `eval_poly`, `eval_poly_deriv`. New `bspline_eval_single` (C² 4×4 kernel, clamp-to-edge). Leave `interpolate3D`, `interpolate3D_transformed` untouched. |
| `pyDFCSR_2D/CSR.py` | **Primary (§3.0):** `get_CSR_wake` (`:460–608`) — per-column transverse nodes placed in ξ′ from the history frame at `t_ret`, replacing the σ_x-scaled rectangular `xp_w`/`xp_n`/`xp1..4`; per-column `x=` arrays in the inner `np.trapz`. **Secondary:** `method == 'bspline_comoving'` branch at `:58–65`; third branch in `get_CSR_integrand` (`:616`, `:671`); wake mesh (`:374`) to read the same deg-1 transform as the deposition. |
| `pyDFCSR_2D/test/test_interp_tilted_gaussians.py` | Fix the reference. ✅ done in Step 1 |
| `pyDFCSR_2D/lattice.py` | `:18–39` look `step_size` up by name instead of assuming it is the first key (bug found in Step 2). |

---

## 4. Step log

### Step 1 — Prove the diagnosis (2026-09-08) ✅

**What was done.** Built a new acceptance test `pyDFCSR_2D/test/test_ghosting.py` designed so that
the *only* error source is the time-interpolation scheme:

- Snapshots are built by **sampling the analytic density** — no deposition, no filtering — so
  deposition noise and smoothing are excluded.
- Slopes are `[0, ds, 2·ds]`, so p(t) is linear in t. Any scheme that interpolates the *frame* in
  time is therefore **exact** here, and the true intermediate-time density is exactly a Gaussian
  sheared by `ds/2`.
- Query points are laid out **in the true beam frame at the query time**
  (`x = ξ + p_α(z)`), so every point sits where the beam actually is. No masking heuristics.
- Two stacks are built, mirroring the two production paths: per-snapshot ξ-frame grids as
  `DF_tracker_smooth` stores them, and one shared lab grid as legacy `DF_tracker` does.
- Sweep `G` from 0.05 to 20 at σ_ξ = σ_z = 50 µm, 200 bins, `xlim` = 5.

Separately, patched `test_interp_tilted_gaussians.py` to compare against the true intermediate-time
density `analytical_density(x, z, slope_blended)`, keeping the old blend reference as a reported
contrast.

**Result — `test_ghosting.py`:** relative L2 error vs the true density.

```
      G |   new rho    new dx    new dz |   leg rho    leg dx    leg dz | dx_leg/sxi
------------------------------------------------------------------------------------
   0.05 |    0.0005    0.0009    0.0005 |    0.0005    0.0010    0.0006 |       0.06
   0.10 |    0.0011    0.0024    0.0012 |    0.0012    0.0025    0.0012 |       0.06
   0.20 |    0.0038    0.0085    0.0054 |    0.0039    0.0088    0.0054 |       0.07
   0.50 |    0.0228    0.0502    0.0357 |    0.0231    0.0508    0.0355 |       0.10
   1.00 |    0.0831    0.1772    0.1425 |    0.0837    0.1782    0.1418 |       0.15
   2.00 |    0.2524    0.4904    0.4713 |    0.2527    0.4899    0.4675 |       0.25
   5.00 |    0.6185    0.9814    1.0778 |    0.6129    0.9669    1.0485 |       0.55
  10.00 |    0.8109    1.0932    1.1997 |    0.7961    1.0594    1.1305 |       1.06
  20.00 |    0.9080    1.0724    1.1435 |    0.8844    1.0381    1.0313 |       2.06
```

![Error vs G](pyDFCSR/pyDFCSR_2D/test/benchmark_results/ghosting/error_vs_G.png)

![Ghost cut at G=10](pyDFCSR/pyDFCSR_2D/test/benchmark_results/ghosting/ghost_cut_G10.png)

![Ghost cut at G=2](pyDFCSR/pyDFCSR_2D/test/benchmark_results/ghosting/ghost_cut_G2.png)

**Findings.**

1. **Ghosting is confirmed and quantitatively O(1) for `G ≳ 1`.** Error grows ≈ G² at small G,
   crosses 8% at G = 1, 25% at G = 2, and saturates near 100% by G ≈ 5–10. Derivatives are worse
   than the density, as expected, exceeding 100% relative error by G ≈ 5.
2. **The 1D cut at G = 10 is unambiguous.** Truth is a single peak at x = 500 µm. Both interpolants
   return **two half-amplitude peaks at 0 and 1000 µm and essentially zero density where the beam
   actually is.** This is the double image, seen directly.
3. **The old test's reference was wrong by exactly the amount the interpolator was wrong.**
   `blend-vs-truth` matches `new_rho` to 3–4 significant figures at every G (e.g. 0.8110 vs 0.8109
   at G = 10). The test was structurally incapable of detecting this defect.
4. **Patched `test_interp_tilted_gaussians.py` now fails loudly** where it used to pass. At its
   high-tilt case (slope +5.5 → −10, G = 15.5), against the true density: ρ error 0.858,
   ∂ρ/∂x 1.079, ∂ρ/∂z 1.689 for the new method — and 0.857 / 1.078 / 1.676 for legacy. The old
   reference was itself 0.858 wrong.

**Important refinement to the diagnosis — this is what sent Steps 2 and 3 looking elsewhere.**

The new and legacy error curves lie **on top of each other** (differences in the 3rd significant
figure). Ghosting therefore affects both schemes equally and is **not, by itself, the regression
that made the new wakes noisy while legacy stayed smooth.** The original framing ("legacy's coarse
cells blur the two ghost images together") is not supported: at G = 10 the ghosts are separated by
20σ_ξ while legacy's cell is only ≈1σ_ξ, nowhere near enough to merge them.

What this changes and what it does not:

- **Ghosting is a real, long-standing O(1) accuracy defect in both schemes**, worth fixing on its
  own merits. The co-moving interpolant remains the right fix. This is now proven, not conjectured.
- **The new-vs-legacy noise difference must be explained by something else.** The leading hypothesis
  is that the *L2 magnitude* of the ghosting error is the same in both schemes but its *spatial
  frequency content* is not: with legacy, the ghost peaks are broad and low, so the wake integral
  over the retarded band varies smoothly with the observation point; with the new scheme the peaks
  are narrow and tall (0.05σ_ξ cells), so as the band sweeps it alternately hits and misses sharp
  spikes, injecting high-frequency structure into the integrand. Quadrature noise depends on
  frequency content, not on L2 norm. The secondary defects (b hard-zero OOB, c registration,
  d poly_degree=3 blowup, g C⁰ quadrature jitter) plausibly dominate the *noise* even though
  ghosting dominates the *error*.
- **This hypothesis is untested at the time of writing.** Steps 2 and 3 exist to test it, and they
  should now be treated as load-bearing rather than optional.

> **Added after Step 3: the frequency-content hypothesis above was WRONG too.** Step 2 shrank the
> ghost separation 4.8× with no effect on the noise, and Step 3 found the actual mechanism — the
> outer `(x′, s′)` quadrature has 0.76 nodes inside the density support. The paragraph is left in
> place because the reasoning it records ("quadrature noise depends on frequency content, not on L2
> norm") turned out to point at the right *subsystem* — the quadrature — for the wrong reason.
> See §2.0 and Step 3.

One caveat on the test's fairness to legacy: it gives legacy only its coarser grid, not the
Savitzky–Golay filter the production path applies. Real legacy smoothing is therefore wider than
modelled here. That strengthens the frequency-content hypothesis without changing the ghosting
conclusion.

**Pass criterion status.** Criterion 6 has two halves. The half that proves the ghosting diagnosis
("current `bspline_fft` error demonstrably O(1) for G ≳ 1") is **met**. The half that validates the
fix ("co-moving interpolant error at round-off for pure affine shear at all G") is **pending** —
`test_ghosting.py` already imports `interpolate3D_comoving` behind a `HAVE_COMOVING` guard and will
add that column automatically once Step 5 lands.

**Files touched.**
- `pyDFCSR_2D/test/test_ghosting.py` (new)
- `pyDFCSR_2D/test/test_interp_tilted_gaussians.py` (reference corrected; contrast reporting added)

**Reproduce.**
```bash
conda activate pydfcsr
cd .../pyDFCSR_claude/pyDFCSR
python pyDFCSR_2D/test/test_ghosting.py
python pyDFCSR_2D/test/test_interp_tilted_gaussians.py
```

---

### Step 2 — Does the noise track G? (2026-09-08) ✅ — **No. Ghosting ruled out as the noise mechanism.**

**What was done.** New test `pyDFCSR_2D/test/test_G_scaling.py`. The existing `test_tilt_sweep.py`
varies the tilt, which changes `G`, the transverse resolution *and* the integration-band geometry all
at once — it cannot separate them. This test varies **only the lattice `step_size`**, so at a fixed
physical position the beam has a fixed tilt `p′` and a fixed σ, but

```
G = |p′(t_{k+1}) − p′(t_k)| · σ_z/σ_ξ   ∝   step_size
```

`apply_CSR = 0` so every run sees identical beam evolution and the wakes are compared at identical
physical positions (`s = 0.600 m`). Noise metric: relative L2 norm of the second difference of the
wake along z, which isolates point-to-point jitter from the wake's shape and amplitude.

**Result.** `shear z:x = 20`, wake mesh 5×40, integration 100², deposition 200².

```
method          step        G   |slope|   s_xi/um   rough(dE)   rough(xk)   s_cmp
---------------------------------------------------------------------------------
bspline_fft    0.100   30.042     2.043      9.58     2.26924     2.10021   0.600
bspline_fft    0.050   13.246     2.043      9.58     2.44714     2.72745   0.600
bspline_fft    0.025    6.258     2.043      9.58     2.19550     2.53075   0.600
legacy         0.100   30.042     2.043      9.58     0.54196     0.05689   0.600
legacy         0.050   13.246     2.043      9.58     0.67776     0.05724   0.600
legacy         0.025    6.258     2.043      9.58     0.36648     0.05666   0.600

Scaling of roughness with G (ghosting predicts ~G^2, i.e. 4x per halving):
  bspline_fft:
    G 30.042 -> 13.246 (2.27x) : roughness 2.26924 -> 2.44714 (0.93x)  => effective power -0.09
    G 13.246 ->  6.258 (2.12x) : roughness 2.44714 -> 2.19550 (1.11x)  => effective power  0.14
```

![Roughness vs G](pyDFCSR/pyDFCSR_2D/test/benchmark_results/G_scaling/roughness_vs_G.png)

**Findings.**

1. **Reducing G by 4.8× changed the noise by nothing** (effective power −0.09 and +0.14, i.e. flat).
   Ghosting predicts `~G²`, which over this range would have been a **23× reduction.** The noise is
   not ghosting.
2. **At identical G, `bspline_fft` is 5× rougher than legacy on dE/dct and ~40× rougher on
   `x_kick`** (2.27 vs 0.54, and 2.10 vs 0.057). Step 1 proved both schemes have *the same* ghosting
   error, so this gap cannot be ghosting either. Something present in the new scheme and absent from
   legacy is responsible.
3. The Step 1 "frequency-content" hypothesis is therefore also unsupported in the form stated: it
   predicted noise scaling with the ghost-peak sharpness, but the ghost separation shrank 4.8× with
   no effect at all.

**Caveat, recorded honestly.** All three step sizes sit in the *saturated* ghosting regime
(`G ≥ 6.3`), where Step 1 measured error already flat at ~80–100%. So this test never probed the
`G ≲ 1` region where the `G²` law lives, and strictly speaking it shows "no sensitivity within the
saturated regime". Finding 2 is the load-bearing one and does not depend on the regime: at *identical*
G with *identical* beams, the two schemes differ by 5–40× in noise.

**Files touched.** `pyDFCSR_2D/test/test_G_scaling.py` (new).

**Bug found while writing this test.** `yaml.dump` sorts keys by default, which put `step_size` after
`element_1..3`. `lattice.py:18–39` does `Nelement = len(lattice_config) - 1` and iterates
`list(lattice_config.keys())[1:]`, i.e. it assumes the **first key is `step_size`** and every
remaining key is an element. Wrong order gives
`TypeError: 'float' object is not subscriptable` at `lattice.py:26`. Generated lattice YAMLs must use
`sort_keys=False`. Worth a real fix in `lattice.py` (look the key up by name) — noted, not done.

**Reproduce.**
```bash
python pyDFCSR_2D/test/test_G_scaling.py    # ~15 min, 6 runs
```

---

### Step 3 — Integrand anatomy at one noisy point (2026-09-08) ✅ — **ROOT CAUSE FOUND**

**What was done.** New test `pyDFCSR_2D/test/test_integrand_anatomy.py`. At one high-tilt wake-mesh
point it:

- installs a spy on `interpolate3D_transformed` to capture the exact `(x′, z_ret, t_ret)` query
  points the wake integral actually uses;
- reproduces the index arithmetic of `bilinear_single` to classify every query point as **inside**,
  **out-of-bounds (hard 0.0)**, or **silently extrapolated** (the `int()` truncation-toward-zero hole:
  indices in `(−1, 0)` pass the `x0 < 0` guard and get extrapolated with negative weights);
- measures the integration-band width against the deposition-grid width, and counts **how many
  quadrature nodes land inside the density support**;
- scans the integration grid over 50², 100², 200², 400² to separate quadrature error from density
  error.

**Result.** `s_obs = 0.70 m`, `shear z:x = 20`, `|slope| = 1.603`, `poly_degree = 3`.

```
--- (0) integration-band width vs deposition-grid width ---
  sigma_x  (lab)         =       826.57 um
  sigma_xi (tilt removed)=        12.54 um
  tilt amplification     =         65.9 x
  deposition grid half-width in xi =      62.67 um (= 5.0 sigma_xi)
  deposition cell in xi            =      0.630 um

  region                   x-half-width   cells across  cell/sigma_xi  pts in density
  R1 far (20 sigma_x)          16531.4u            200           13.2            0.76
  R3 near (5 sigma_x)           4132.9u            100            6.6            1.52
  R near (10 sigma_x)           8265.7u            100           13.2            0.76

--- (i)/(ii) out-of-bounds and silent-extrapolation fractions ---
region                  n_pts    OOB %  extrap %  |integrand|max
----------------------------------------------------------------
R1 (far, wide x)        20000    99.98      0.01      3.2629e+12
R2 (chirp band)         10000   100.00      0.01      3.9865e+11
R3 (near, band a)       10000   100.00      0.02      1.3619e+10
R4 (near, band b)       10000   100.00      0.06      5.0711e+11

--- (iv) quadrature convergence of the wake along z ---
  grid   50^2 : roughness(dE) = 1.95429   roughness(x_kick) = 1.79718
  grid  100^2 : roughness(dE) = 2.33626   roughness(x_kick) = 2.20589
  grid  200^2 : roughness(dE) = 1.84767   roughness(x_kick) = 1.22583
  grid  400^2 : roughness(dE) = 1.47434   roughness(x_kick) = 0.56928

  Convergence toward the 400^2 answer:
    grid   50^2 : rel. diff dE = 38.46244   x_kick = 19.50372
    grid  100^2 : rel. diff dE = 15.18054   x_kick =  8.07953
    grid  200^2 : rel. diff dE =  7.41733   x_kick =  2.79230
```

![Integrand map](pyDFCSR/pyDFCSR_2D/test/benchmark_results/integrand_anatomy/integrand_map.png)

![Quadrature convergence](pyDFCSR/pyDFCSR_2D/test/benchmark_results/integrand_anatomy/quadrature_convergence.png)

**Findings.**

1. **The transverse integral is evaluated with less than one quadrature node inside the beam.**
   `0.76` nodes for the two widest regions, `1.52` for the narrowest. **99.98–100% of all query
   points return `bilinear_single`'s hard `0.0`.** The whole wake comes from the one or two nodes
   that happen to land on the density ribbon.
2. **The integrand map shows this directly:** instead of a smooth ribbon it is a handful of isolated
   nonzero pixels, and the 1D `x′` cuts are delta-like spikes rather than resolved profiles. The
   `|integrand|max` values (up to 3.3e12) are the amplitude of those isolated spikes.
3. **Which nodes land on the ribbon changes discontinuously as the observation point moves** across
   the wake mesh. That is the noise — it is a **sampling failure of the outer 2D quadrature**, not an
   interpolation error.
4. **Confirmed by the grid scan.** The wake is nowhere near converged: 100² differs from 400² by
   **15× on dE and 8× on `x_kick`** — relative differences of 1518% and 808%, not percent-level. The
   convergence plot shows ±150 MeV/m oscillations at 50²/100² collapsing to a small smooth curve at
   400². Refining the grid monotonically improves things because it is the only knob that increases
   nodes-in-support, which is precisely why Step 24 of `progress.md` found 300²–500² "fixed" it.
5. **Root cause, stated precisely:** `get_CSR_wake` sizes every integration band as a multiple of
   `self.beam._sigma_x` (lab frame, tilt included), while the new deposition grid is sized by `σ_ξ`
   (tilt removed). At 65.9× tilt amplification the band is **264× wider than the density support**
   and the transverse cell is **13.2 σ_ξ**. The ξ-frame deposition did not remove the undersampling —
   it **relocated it** from the deposition grid to the outer quadrature.
6. **Legacy is smooth for a mundane reason:** its deposition grid is *also* `±xlim·σ_x`, so the
   density support and the integration band are the same size and ~50 nodes land inside. It was never
   better-behaved; it was merely self-consistent.
7. **Silent extrapolation is real but negligible** here (0.01–0.06%). Worth fixing as hygiene
   (defect b), not a driver.

**This supersedes the Step 1 diagnosis as the explanation for the noise.** Ghosting (§2.1) remains a
proven O(1) *accuracy* defect in both schemes and still needs the co-moving interpolant — but it is
not what makes the new wakes noisy. See §2.0 and §3.0.

**Files touched.** `pyDFCSR_2D/test/test_integrand_anatomy.py` (new).

**Reproduce.**
```bash
python pyDFCSR_2D/test/test_integrand_anatomy.py    # ~10 min
```

---

### Step 4 — Align the integration quadrature with the deposition support (2026-09-08) 🟡 **implemented; correct but not sufficient**

**What was done.** Implemented §3.0 in `CSR.py`, gated on `CSR_integration: xi_bands` (default `True`,
active only for the `bspline_fft` deposition).

- `_retarded_xi_band(s, x, t, sp)` — locates the retarded density support in lab x′ per s′ column, by
  a 3-pass fixed point on `t_ret ↔ x'`. Reads the ξ extent straight off `min_xi_arr` /
  `delta_xi_arr` / `poly_coeffs_interp`, so the quadrature domain is the deposition grid *by
  construction* and the two cannot drift apart again. Snapshots whose z grid does not reach `z_ret`
  are excluded from the union, so they cannot widen the band without contributing.
- `_integrate_xi_region(...)` — per-column uniform x′ nodes; inner `np.trapz` at unit spacing scaled
  by each column's `dx` (exact for a uniform grid).
- `get_CSR_wake` gains an early branch: **3 regions** (ribbon over sp1, sp2, sp3), keeping the
  existing `s1..s4` decomposition untouched. The chirp case's old regions 3 and 4 tiled x′ over the
  *same* s′ range, so once both become the same ribbon they must be merged or the ribbon is counted
  twice. Debug mode returns a dict in this path.

**Result 1 — sampling and noise, at 65.9× tilt amplification (`s = 0.70 m`, shear 20).**

```
mode           grid    n_query   hard-zero %
old (sigma_x)    100     250000         99.32
xi_bands         100     150000         77.71

  region        width med    cell/s_xi med    cell/s_xi max
  sp1             2258.9u             1.82             1.82
  sp2              257.3u             0.21             0.72
  sp3              255.8u             0.21             1.64
  (old bands: cell = 13.2 sigma_xi at 100 nodes)

mode             grid   rough(dE)   rough(xk)
old (sigma_x)     100     2.33626     2.20589
old (sigma_x)     400     1.47434     0.56928
xi_bands          100     0.73034     0.44533
xi_bands          400     0.44963     0.14489
```

Median transverse cell **13.2 σ_ξ → 0.21 σ_ξ, a 63× improvement.** Noise down 3.2× on dE and 5× on
`x_kick` at equal grid. `xi_bands` at 100² is smoother than the old bands at 800². It is also
**1.5× faster** (3 regions instead of 4): 36 ms vs 55 ms per wake point at 400².

**Result 2 — the fix is exact where it can be checked (`test_xi_bands_equiv.py`).**

At low tilt the old bands are well sampled, so both modes must agree. They do:

```
shear   amplification   old vs new @800^2 (dE)   old vs new (x_kick)
    0            1.1 x                 0.00035              0.00235
    2            1.3 x                 0.00036              0.00348
   20           65.9 x                 1.05311              0.37384
```

**0.035% agreement at low tilt.** The band construction discards nothing. And at high tilt the old
bands walk monotonically toward the new answer as they are refined (gap 35.6 → 16.9 → 2.74 → 1.05 for
grids 100 → 800), while `xi_bands` self-converges cleanly (0.275 → 0.104 → 0.033). Both point the same
way: the old bands were the undersampled ones.

![xi_bands A/B](pyDFCSR/pyDFCSR_2D/test/benchmark_results/xi_bands/xi_bands_ab.png)

![Old vs new across tilt](pyDFCSR/pyDFCSR_2D/test/benchmark_results/xi_bands/xi_bands_equiv.png)

**Result 3 — but the high-tilt answer is NOT domain-converged. A second defect was masked.**

Widening the band (`xi_band_margin`) should not change the answer, since the added nodes sit where the
density is zero. It does:

```
margin   rough(dE)   dE vs margin=8
   1.0     0.18398         20.18116
   2.0     0.44963          5.15088
   3.0     1.90172          0.91904
   8.0     2.09759          0.00000
```

Per-region breakdown at one wake point, and the cause:

```
margin  region     dE contrib   min|r-r'|      |integrand|max
   2.0     sp1   -1.27812e-01   107047 um          2.968e+12
   2.0     sp2    4.68929e-02     4018 um          2.356e+12
   2.0     sp3    1.47058e-01     1.72 um          6.009e+13
   2.0   TOTAL    6.61387e-02

sp3, excising a neighbourhood of r = 0:
   excise r <   10 um :  -32.5% change, from  0.02% of the nodes
   excise r <  100 um :  -68.5% change, from  1.21% of the nodes
```

**Two things, both newly exposed rather than newly created:**

1. **An unhandled integrable singularity at `r = |r − r′| → 0`.** Region sp3 spans
   `s3 → s4 = s + 5σ_z`, so it straddles `s′ = s`, and the observation point sits inside the beam, so
   `x′ = x` is inside the band. There `W1 ∝ part1/r³` with `part1 = (r−r′)·(n−n′) ~ r²/R`, i.e. the
   integrand diverges like `1/r`. That is integrable in 2D, but a plain uniform trapezoid mesh
   converges slowly and erratically across it. **32% of sp3 now comes from 0.02% of the nodes.** The
   old σ_x-wide bands never got closer than ~80 µm to the singularity, so they never sampled the
   blow-up — the coarseness that undersampled the density also hid this. Resolving the density exposed
   it. The original chirp-branch split at `x ± 0.1σ_x` was presumably managing this, and the 3-region
   merge removed that structure.
2. **The total is a near-cancellation.** `sp1 = −0.128` against `sp2 + sp3 = +0.194`, total `+0.066`.
   The parts are ~3× the sum, so any residual error in a part is amplified ~3× in the answer. This is
   the "cancellation regime" `progress.md` suspected in chicane B2, now measured directly.

**Honest status.** The quadrature-alignment fix is **necessary, correct, cheap and a large
improvement**, but **not sufficient**. It converts a density-sampling failure into a
singularity-sampling failure. The remaining error is now concentrated in a tiny, identifiable region
of the integration plane rather than smeared over the whole thing, which is a much better place to be
— but the high-tilt wake should not yet be trusted quantitatively.

**Files touched.**
- `pyDFCSR_2D/params.py` — `xi_bands`, `xi_band_margin` in `Integration_params`
- `pyDFCSR_2D/CSR.py` — `_lattice_at`, `_retarded_xi_band`, `_eval_poly_rows`,
  `_integrate_xi_region`, and the new branch in `get_CSR_wake`
- `pyDFCSR_2D/test/test_xi_bands.py` (new), `pyDFCSR_2D/test/test_xi_bands_equiv.py` (new)

**Note.** The older one-off diagnostics (`test_region1_nonzero.py`, `test_integrand_diagnostic.py`,
`test_integrand_step17.py`, `test_integrand_anatomy.py`) unpack `get_CSR_wake(debug=True)`
positionally and need `CSR_integration: {xi_bands: False}` to keep working.

**Reproduce.**
```bash
python pyDFCSR_2D/test/test_xi_bands.py         # sampling, noise, convergence
python pyDFCSR_2D/test/test_xi_bands_equiv.py   # the falsification test
```

### Step 4c — Fix the acceptance test, then re-read the result (2026-09-09) ✅ **test was confounded; polar patch is correct; one real failure remains**

**Why the earlier margin sweep was invalid.** Step 4's "not domain-converged" verdict came from
sweeping `xi_band_margin` at a **fixed node count**. Widening the band at fixed `xbins` also coarsens
the cell, so that sweep varied the domain *and* the resolution together. The rising roughness
(0.18 → 2.10) is the signature of coarsening, not of a domain error. The conclusion was not supported
by the measurement.

**The fixed test** (`test_xi_bands_converge.py`) varies one thing at a time, and separates
**invariances** (the answer must not move at all) from **convergences** (it must settle):

| axis | kind | result | worst dE deviation |
|---|---|---|---|
| A. transverse domain, **cell size held fixed** (`xbins ∝ margin`) | invariance | ❌ **fails** | **1.017** |
| B. transverse resolution, fixed domain | convergence | ✅ 0.236 → 0.095 → 0.018 → 0 | 0.236 |
| C. longitudinal resolution | convergence | ✅ 0.134 → 0.011 → 0.005 → 0 | 0.134 |
| D. **near-patch radius R2** | invariance | ✅ ~1% across R2 = 2 → 12 σ_ξ | 0.011 |
| E. near-patch resolution | convergence | ✅ converged at nr = 50 | 0.00001 |
| F. patch on vs off, both resolved | — | 4.2% dE, 8.8% x_kick, **resolution-independent** | — |

**What this establishes.**

1. **The polar patch is correct.** Axis D is the sharp test: since `w + (1−w) = 1` identically, the
   total cannot depend on where the Cartesian/polar crossover sits. It doesn't, to ~1%. The partition
   of unity is implemented right.
2. **The polar substitution works as predicted.** Axis E converges at **nr = 50** — deviation 1e-5.
   The `r dr dφ` Jacobian cancels the `1/r` exactly, leaving an integrand so smooth that 50 radial
   nodes suffice. Confirmed independently by measuring `integrand × r → −1.036e8`, constant over four
   decades in r and independent of φ.
3. **The patch is doing real, necessary work.** Axis F: turning it off shifts dE by 4.2% and `x_kick`
   by 8.8%, and that shift is **independent of resolution** (0.0416 / 0.0421 / 0.0422 at
   `xbins` 200 / 400 / 800). A Cartesian mesh never removes it by refinement, because the near-field
   contribution accumulates equally per decade of r. So Step 4's reading — "the patch didn't work" —
   was also an artifact of the confounded test.
4. **Both resolution axes converge cleanly**, at roughly the expected order.

**The one real failure, and it is well localized.** Per-region breakdown vs domain width at fixed
cell size:

```
margin  xbins           sp1           sp2           sp3         patch         TOTAL
   1.0    100  -7.19459e-02   4.62530e-02   1.39460e-01   0.00000e+00   1.13767e-01
   2.0    200  -5.52064e-02   4.61205e-02   1.46993e-01   0.00000e+00   1.37907e-01
   4.0    400  -5.43031e-02   4.60174e-02   1.51052e-01   0.00000e+00   1.42766e-01
   8.0    800  -5.64981e-02   4.59668e-02   1.58648e-01   0.00000e+00   1.48117e-01
```

- `sp2` is invariant to **0.6%** ✅
- the patch is invariant to **machine precision** (it does not depend on the band at all) ✅
- `sp1` settles for margin ≥ 2 ✅
- **`sp3` grows without settling: +0.0075, +0.0041, +0.0076 per doubling of the domain** ❌

Roughly *constant per doubling* — i.e. **logarithmic in the transverse domain width.** Edge
diagnostics show 6–8 of 400 `sp3` columns have their peak |integrand| *at the band edge*, and that
persists at margin 8, so this is not the density ribbon being clipped (`sp1`/`sp2` show zero edge
leakage at every margin).

**Most likely cause, not yet confirmed.** This is exactly the structure of Stupakov,
PRAB 25 014401 §IV Eqs (35)–(37): the near-region integral **is** genuinely log-divergent in the
size of the near region, and the `ln Δs` terms **cancel** against the far-region (line-charge)
contribution — *"As expected, the auxiliary variable Δs disappeared from the final result."* The
3-region merge in Step 4 collapsed the original chirp branch's 4 regions into 3 and may have broken
that cancellation. If so the fix is to restore a consistent near/far split rather than to widen the
band.

**Caveat on scale.** The per-region numbers above are one mid-z wake point, where the total moves
~13% over margin 1 → 8. The relative L2 over the whole 40-point z-scan is 1.0, so other z points move
far more. The mid-z point is not representative and the region breakdown should be repeated at a
worst-case point before acting.

**Files touched.**
- `pyDFCSR_2D/params.py` — `near_patch`, `near_patch_nr`, `near_patch_nphi`
- `pyDFCSR_2D/beams.py` — enabled the already-present-but-commented `_sigma_x_transform` cache
- `pyDFCSR_2D/CSR.py` — `taper` kwarg on `get_CSR_integrand`, `_near_patch_radii`,
  `_integrate_near_patch`
- `pyDFCSR_2D/test/test_xi_bands_converge.py` (new)

![Convergence, one axis at a time](pyDFCSR/pyDFCSR_2D/test/benchmark_results/xi_bands/xi_bands_convergence.png)

**Reproduce.**
```bash
python pyDFCSR_2D/test/test_xi_bands_converge.py    # ~4 min
```

### Step 4d — (superseded by Step 4e) Restore a consistent near/far split in the s′ decomposition ❌ **hypothesis abandoned**

The `ln Δs` near/far cancellation of Stupakov §IV was the guess for the `sp3` log-divergence. It was
wrong. Step 4e found the actual cause: `_retarded_xi_band` locates only **one** of the up-to-four
nonzero `x′` intervals per column. Kept here for the record only.

### Step 4e — The real cause of the sp3 failure: multiple localization branches (2026-09-09) ✅ **diagnosed**

**What the code misses.** At fixed `s′` the integrand is nonzero on up to **four** disjoint `x′`
intervals, not one. Measured by scanning `x′` finely at fixed `s′` and counting disjoint nonzero runs:

```
  s-s_prime  #intervals   intervals [um]
    0.00020           1   [   -604.0,    -11.6]
    0.00050           3   [  -1221.7,   -933.1] , [   -756.1,   -588.6] , [   -141.6,     -6.2]
    0.00100           3   [  -2243.9,  -1957.7] , [  -1401.0,  -1240.1] , [   -134.3,     -3.7]
    0.00200           3   [  -4138.8,  -3999.4] , [  -2691.7,  -2533.3] , [   -131.3,     -2.3]
    0.00402           1   [   -129.7,     -1.5]
    0.00600           1   [   -128.9,     -1.1]
```

**Where the count comes from: 2 branches × 2 snapshots.**

1. *Two branches per snapshot.* For one snapshot, "query point lies on that snapshot's beam axis"
   (a line, Eq 4.20) combined with "query point lies on the light cone" (a circle, Eq 4.19) is a
   line–circle intersection ⇒ a **quadratic** ⇒ two roots. These are the thesis's narrow band
   (`x₁ ≈ x`, weakly dependent on τ) and chirp band (`x₂ = x − (s−s′)·tan 2α`, entirely set by τ).
2. *Two snapshots.* `interpolate3D_transformed` evaluates snapshot `k` in **its own** frame `p_k` and
   snapshot `k+1` in **its own** frame `p_{k+1}`, then blends the two results. The support is the
   **union** of two independent supports, so each snapshot contributes its own pair of roots.
3. *Why 3 and not 4.* The narrow roots sit at ≈ `x` for both snapshots, so they overlap and merge into
   one interval. The chirp roots depend strongly on τ, and the two snapshots have different τ, so they
   stay separate. `1 + 2 = 3`.

Cross-check: recovering `tan 2α` from each chirp interval's centre via `(x − x₂)/(s − s′)` at
`ds = 1 mm` gives `2.043`, matching the code's `tan_alpha = 2.0413` computed from the current beam
slope, and `1.256` for the other, implying an effective `τ ≈ −2.07` for the other snapshot — a Δτ ≈ 0.47
over one 0.1 m step, consistent with the measured `G ≈ 30`.

**Weights: only one chirp branch matters, but it matters a lot.**

```
  s-s' = 0.5 mm   alpha = 0.9950
      x' in [-1221.7, -933.1]   contribution = 66.62%   <- chirp, dominant snapshot
      x' in [ -756.1, -588.6]   contribution =  0.39%   <- chirp, ghost image
      x' in [ -141.6,   -6.2]   contribution = 32.98%   <- narrow (merged)
  s-s' = 1.0 mm   alpha = 0.9900
      x' in [-2243.9,-1957.7]   contribution = 38.24%
      x' in [-1401.0,-1240.1]   contribution =  1.41%
      x' in [ -134.3,   -3.7]   contribution = 60.35%
  s-s' = 2.0 mm   alpha = 0.9800
      x' in [-4138.8,-3999.4]   contribution =  0.00%
      x' in [-2691.7,-2533.3]   contribution =  0.62%
      x' in [ -131.3,   -2.3]   contribution = 99.38%
```

The two chirp branches carry the blend weights `α` and `1−α`. Here `α ≈ 0.99` throughout, because the
chirp branch only lives within `d ≈ 4 mm` of `s` while the history step is `dt = 100 mm`: `t_ret` never
leaves the immediate neighbourhood of a snapshot while the chirp branch exists. **That is a consequence
of `d ≪ dt`, not a general truth** — at fine `step_size` both would carry comparable weight, so an
implementation must handle both with a weight threshold rather than assume one dominates.

**Conclusion.** `_retarded_xi_band` initialises its fixed point at `xp = x`, so it converges to the
root nearest `x` — the narrow branch — and **never finds the chirp branch, which carries up to 67% of
the column's contribution.** The chirp branch exists only for `ds ≲ 3 mm`, i.e. only inside `sp3`,
which is exactly and only where axis A of `test_xi_bands_converge.py` fails. The old code's
`[x − 10σ_x, x + 3σ_x]` near-region bands did cover it, crudely; the Step 4 merge of R3+R4 threw it
away. This is a regression introduced in Step 4, not a pre-existing defect.

**Relation to ghosting (§2.1).** The two chirp branches *are* the ghosting, seen in the integration
plane instead of in the density. Physically there is only **one** chirp branch, at the true slope
`τ(t_ret)`. The lab-frame blend places weight at `τ_k` and at `τ_{k+1}` and nothing in between, so
*neither* visible branch is in the right place. The co-moving interpolant of §3.1 would collapse them
into one correctly-placed branch. **Note this is not yet implemented** — there is no `comoving` code in
the tree; `CSR.py` still calls `interpolate3D_transformed` throughout. Doing §3.1 first would make the
band-location problem strictly easier, since there would be one unambiguous `τ(t_ret)` to feed Eq 4.24
rather than two competing ones.

### Step 4f — Verification of thesis Eq 4.24: the printed formula has a typo (2026-09-09) ✅

Eq 4.24 of `reference/Jingyi_Tang_PhD_Thesis` §4.4.2 gives the localization branch positions in closed
form, and is the natural replacement for the numerical fixed point in `_retarded_xi_band`. Before
building on it, it was checked. **As printed it is wrong: the `tan²α₀` prefactor on the square root
should be `tan α₀`.**

#### The derivation

Two constraints, both from §4.4.2. With `q ≡ n(s)x + r₀(s) − r₀(s′)`, `T ≡ t − s′`, `τ ≡ tan α₀`:

*Light cone* (Eq 4.19), using `|n(s′)| = 1`:

```
l = t − t_ret ,    l² = |q − n(s′)x′|²  =  q² − 2x′(n′·q) + x′²
```

*Beam axis* (Eq 4.20) — the query point lies on the beam's central axis at the retarded time. Since
`z′ = s′ − t_ret`:

```
x′ = τ·(s′ − t_ret)      ⇒     t_ret = s′ − x′/τ     ⇒     l = t − t_ret = T + x′/τ
```

Equating the two expressions for `l²` and collecting powers of `x′`:

```
A = 1 − 1/τ²            B = −2T/τ − 2(n′·q)            C = q² − T²
```

whence

```
x′± = τ [ T + (n′·q)τ  ±  √rad ] / (τ² − 1) ,    rad = (τ² − 1)(T² − q²) + ((n′·q)τ + T)²
```

**Both terms carry exactly one power of τ.** The thesis's first term,
`(n′·q tan²α₀ + (t−s′)tan α₀)/(tan²α₀−1)`, is precisely `τ(τ(n′·q) + T)/(τ²−1)` — correct. But its
square-root prefactor is `tan²α₀/(tan²α₀−1)` where it must be `tan α₀/(tan²α₀−1)`; the printed equation
is internally inconsistent with its own first term.

Telling detail: **the thesis radicand matches this derivation exactly**, term for term. So the thesis
followed this same path and the `τ²` is a transcription slip in the final line, not a different
formulation. Note also that both versions are dimensionally consistent, so dimensional analysis cannot
catch it — the failure is a broken cancellation. The narrow branch `x₁ ≈ x` requires the two terms to
nearly cancel, which only happens when the powers of τ match.

#### Symbolic check — how it works

A root of a polynomial, substituted back into that polynomial, must give identically zero. So: build
the quadratic `quad = l²_geometric − l²_axis` symbolically in sympy, substitute each candidate closed
form for `x′`, and simplify. Zero ⇒ the candidate is a root; anything else ⇒ it is not. This tests the
formula without needing any numbers, geometry, or assumptions about the lattice.

```python
xp, T, nq, q2, tau = sp.symbols("x' T n_dot_q q2 tau", real=True)
quad = sp.expand((q2 - 2*xp*nq + xp**2) - (T + xp/tau)**2)     # = 0 defines x'
thesis = (nq*tau**2 + T*tau)/(tau**2-1) + tau**2/(tau**2-1)*sp.sqrt(...)   # as printed
fixed  = (nq*tau**2 + T*tau)/(tau**2-1) + tau   /(tau**2-1)*sp.sqrt(...)   # single power
sp.simplify(quad.subs(xp, thesis))   # -> T**2*tau**2 + 2*T*n_dot_q*tau + ... (NOT zero)
sp.simplify(quad.subs(xp, fixed))    # -> 0
```

Result: the printed form leaves a residual equal to the radicand itself; the single-power form leaves
exactly `0`. sympy's own `solve()` independently returns
`tau*(T + n_dot_q*tau ± sqrt(...))/(tau**2 - 1)` — one power of τ, confirming it a third way.

#### Numerical check — how it works

The symbolic check verifies the algebra but shares its starting point, so it cannot catch a
misunderstanding of what Eq 4.19/4.20 mean. The numerical check is independent: it never uses the
closed form at all. Instead it solves the **original two-equation system directly** by root-finding, in
real lattice geometry taken from a running simulation, and asks which closed form reproduces the
answer.

Define the residual of the combined system as a function of `x′` alone at fixed `s′`:

```python
def resid(xp_, sp_):
    q, nsp = geom(sp_)                       # q and n(s') from lattice.coords, lattice.n_vec
    return np.linalg.norm(q - nsp*xp_) - ((t - sp_) + xp_/tau)     # Eq 4.19 minus Eq 4.20
```

`resid = 0` is exactly the pair of constraints, with no algebra applied. Then scan `x′` coarsely to
find sign changes, bracket each, and polish with `scipy.optimize.brentq` to `xtol = 1e-14`. Compare the
roots against both closed forms. Geometry (`q`, `n(s′)`, `τ`, `x`, `s`, `t`) is read from an actual
`CSR2D` run at `s = 0.700040 m`, `τ = −1.6034`, so the test exercises a real curved trajectory rather
than a contrived case.

```
  s-s'    | numeric root 1  numeric root 2 |  corrected -   corrected + |   thesis -      thesis +
  0.00100 |     -2107.1052       -63.4733  |     -63.4733    -2107.1047 | -2723.7158     553.1378
  0.00402 |     -8292.6937       -63.1619  |     -63.1619    -8292.6916 |-10775.7323    2419.8788
  0.01000 |    -20641.5055       -62.4958  |     -62.4958   -20641.5032 |-26850.6687    6146.6697
  0.05000 |   -106230.0553       -50.6381  |     -50.6381  -106230.0416 |-138266.8391  31986.1594
  0.13000 |         81.2371            nan |      81.2371  -293188.5051 |-381674.8126   88567.5446
  0.24000 |        729.5001            nan |     729.5001  -584564.1546 |-761160.8740  177326.2194
```

All µm. The corrected form matches to 4–6 significant figures at every `s′`; the printed form matches
nowhere, missing the narrow branch by 40× (−2724 vs −63.5 µm). The `nan` entries are the second root
falling outside the ±0.2 m scan window, not a formula failure.

#### Corrected and generalized for the code

`get_CSR_wake` needs the axis intercept too, since `beam.slope = polyfit(z, x, 1)` has
`slope[1] ≠ 0`. Redoing the derivation with `x′ = τ·z′ + b` gives `l = T + (x′−b)/τ = (T − b/τ) + x′/τ`,
so the intercept enters **only** as a shift of `T` — verified symbolically (sympy returns the identical
expression under `T → Tb`):

```
Tb  = (t − s′) − b/τ                     τ = tan α₀ = slope[0],  b = slope[1]
rad = (τ² − 1)(Tb² − q²) + ((n′·q)τ + Tb)²
x′± = τ [ Tb + (n′·q)τ ± √rad ] / (τ² − 1)
```

Guards required when implementing:

| case | condition | handling |
|---|---|---|
| degenerate | `τ² = 1`, i.e. `A = (τ−1)(τ+1)/τ² = 0` | quadratic becomes linear, `x′ = −C/B`. This is the thesis's `α = ±π/4` case and the origin of the code's `|tanθ| ≤ 1` branch threshold |
| no intersection | `rad < 0` | that branch is absent at this `s′` |
| spurious root | `l ≤ 0` | squaring admits solutions with `t_ret > t`; filter them |
| curved axis | `poly_degree > 1` | Eq 4.24 assumes a **straight** axis. At the default `poly_degree = 3` the axis is a cubic, so the closed form is approximate and must be refined. A second, independent argument for the `poly_degree = 1` default already listed as defect (d) |

#### Everything else in §4.4.2 checks out

An isolated typo, not a flaw in the analysis. Independently confirmed against the code and the
measurements:

- Eq 4.22's `tan 2α` **is** the code's `tan_alpha = -2*tan_theta/(1 - tan_theta**2)` ✓
- the code's `d` is where the chirp branch crosses `10σ_x`: numerically at `ds = 4.018 mm` the chirp
  root is at **−8292.7 µm** against `10σ_x = 8265.7 µm` ✓
- the two-branch structure, and the claim that the chirp branch's `s′` extent is short while the narrow
  branch runs for metres, both hold — see Step 4e ✓
- §4.4.2 prescribes the envelope as `x′(s′) ± 6σ_x` with **"σ_x(s) is slice rms transverse size"** —
  i.e. `beams.py:214 sigma_x_transform`, the tilt-removed σ_ξ. The code uses `beam._sigma_x`, the
  *projected* size. **That is the §2.0 defect, restated as a code-vs-thesis discrepancy**: the thesis
  prescribes `±6σ_slice` = ±75 µm here, the code implements `±20σ_projected` = ±16 531 µm, a 220×
  superset. A superset is safe, which is why it went unnoticed with lab-frame deposition; the ξ-frame
  deposition made it fatal. **The Step 4 fix is therefore restoring the thesis's own prescription, not
  inventing a new scheme.**

Worth recording as an erratum — Eq 4.24 is the formula anyone reimplementing DFCSR would copy.

#### Quantified verdict

```
symbolic : single-power form is a root = True,  as-printed is a root = False
numerical: worst relative mismatch, corrected form : 2.552e-07
           worst relative mismatch, printed form   : 1.089e+03
```

The corrected form reproduces the root-found branches to **2.6e-07** relative; the printed form is off
by a factor of **1089**.

**Files touched.** `pyDFCSR_2D/test/test_eq424_localization.py` (new) — self-contained, runs both
checks and writes `benchmark_results/eq424/eq424_log.txt`. Its module docstring carries the full
derivation, so the reasoning travels with the code. Exposes `eq424_corrected(...)` and
`eq424_general(...)` (the intercept form) for reuse by `CSR.py` when the band locator is rewritten.

**Reproduce.**
```bash
python pyDFCSR_2D/test/test_eq424_localization.py     # ~1 min; symbolic check needs no simulation
```

### Step 4b — (superseded by 4c) Handle the r → 0 singularity in the near region ✅ **done in Step 4**

Discovered in Step 4. The near region's integral is dominated by a neighbourhood of `r = 0` that a
uniform trapezoid mesh cannot resolve. Options, roughly in order of effort:

1. **Restore a singularity-aware split.** Keep the ribbon bands but subdivide the near region into an
   inner disc/annulus around `(x, s)` and an outer remainder, as the original `x ± 0.1σ_x` boundary
   did. Cheapest, least principled.
2. **Polar quadrature around the singularity.** Substitute `x′ − x = r cos φ`, `s′ − s = r sin φ`
   locally; the Jacobian `r` cancels the `1/r`, giving a smooth integrand. Principled and standard,
   but needs a matching/blending rule with the surrounding Cartesian regions.
3. **Analytic subtraction.** Split `f = (f − f_sing) + f_sing` with `f_sing` the leading `ρ(x,z)/(R r)`
   term integrated in closed form over the near region. Most accurate, most algebra, and needs the
   thesis derivation to get the leading coefficient right.

**Acceptance:** the wake insensitive to `xi_band_margin` (today 5.2 between margin 2 and 8) and to
excising a small disc around `r = 0` (today 32% for a 10 µm disc).

### Step 5 — Implement the co-moving interpolant (2026-09-09) ✅ **implemented and verified; ghosting eliminated**

**What was built.**

| file | addition |
|---|---|
| `interp3D.py` | `cubic_bspline_w`, `bspline_eval_single` (C², 4×4, `math.floor` not `int()`), `interpolate3D_comoving_fields`, `interpolate3D_comoving` |
| `deposit_smooth.py` | `DF_tracker_comoving` |
| `CSR.py` | `method == 'bspline_comoving'`, `use_comoving`, `_comoving_fields`, `_grid_extents`, co-moving branch in `_retarded_xi_band` and `get_CSR_integrand` |
| `test/test_ghosting.py` | `build_comoving_stack` + the CMV columns — the acceptance test |

Design points worth recording:

- **All five fields come from one pass.** `get_CSR_integrand` previously made five separate
  `interpolate3D_transformed` calls, each redoing the frame construction and stencil indices.
  `interpolate3D_comoving_fields` returns `(ρ, ∂ρ/∂x, ∂ρ/∂z|ₓ, vx, ∂vx/∂x)` together.
- **The chain rule is applied at query time with the blended `p′_α`** — fixing defect (a). The old path
  baked `p′_k` in per snapshot and then blended lab-frame values, which is inconsistent.
- **`eval_poly` is linear in its coefficients**, so `p_α(z) = (1−α)p_k(z) + αp_{k+1}(z)`; no blended
  coefficient arrays needed.
- Secondary fixes folded in: registration at bin centres with `delta = (end−start)/nbins` (c);
  `density /= npart·Δu·Δw` with a reported clipped-charge fraction (e); snapshot-independent velocity
  floor referenced to `1/2π`, the peak of a normalized Gaussian, instead of this snapshot's own peak (f);
  `poly_degree = 1` default with the fit restricted to `|z| < 3σ_z` (d).
- The shared normalized grid **retires the adaptive re-gridding path** entirely — `legacy` still prints
  `start reinterpolation ... xbins = 510 zbins = 1490` mid-run; the co-moving tracker never re-grids.

**Result 1 — ghosting is eliminated, exactly (`test_ghosting.py`).**

```
      G |   new rho    new dx    new dz |   leg rho    leg dx    leg dz |   CMV rho    CMV dx    CMV dz
   0.05 |    0.0005    0.0009    0.0005 |    0.0005    0.0010    0.0006 |  0.000589  0.001019  0.001019
   1.00 |    0.0831    0.1772    0.1425 |    0.0837    0.1782    0.1418 |  0.000589  0.001019  0.001019
   5.00 |    0.6185    0.9814    1.0778 |    0.6129    0.9669    1.0485 |  0.000589  0.001019  0.001019
  20.00 |    0.9080    1.0724    1.1435 |    0.8844    1.0381    1.0313 |  0.000589  0.001019  0.001019
```

![Error vs G, all three methods](pyDFCSR/pyDFCSR_2D/test/benchmark_results/ghosting/error_vs_G.png)

*(this is the Step 1 figure regenerated; the green `bspline_comoving` curve is new)*

The CMV columns are **flat to six digits across three decades of G**. Both lab-frame blends degrade to
O(1). At `G = 20` the density error improves **1500×** and the derivatives **1000×**. The residual
5.9e-04 is pure B-spline grid error, not ghosting — it is identical at every G because the stored
normalized shape is identical at every snapshot, which is the whole point. Pass criterion 6 is met
(strictly: flat at the *grid* error rather than at round-off, and exactly G-independent).

Physical cell size preserved: `δξ = 0.42 / 0.48 / 0.63 µm` across the three snapshots, matching
`bspline_fft`'s 0.63 µm. **No transverse resolution was traded for the shared index space.** Clipped
charge ≈ 0 (−2e−16).

**Result 2 — the band locator had to be made co-moving too, and that mattered enormously.**

First attempt left `_retarded_xi_band` taking the union of the two snapshots' frames. Wake noise then
barely moved (0.72 → 0.63), because the *quadrature domain* was still sized by the ghost separation
even though the *interpolant* no longer ghosted. After making the locator mirror the interpolant's
blending exactly (linear in poly and means, log-linear in σ):

```
                     band width med / max [um]        rough(dE)          self-conv
                     sp1          sp2          sp3     @100²    @400²    100 vs 400
bspline_fft      786/2233     251/256     256/2085   0.71956  0.44653      0.2544
bspline_comoving  169/190     216/248     250/251    0.02101  0.01501      0.0057
```

- max band width **2085 → 251 µm**, and max ≈ median now: the ghost-union outliers are gone
- roughness **34× lower** at 100², 30× at 400²
- **self-convergence 0.6% at 100², versus 25% for `bspline_fft`** — a 45× improvement

Cost: 8.9 ms/pt versus 3.9 for `bspline_fft` at 100² (the 4×4 stencil is 16 taps versus 4, partly
offset by one pass instead of five). But since 100² now suffices where `bspline_fft` needed 400²,
the like-for-like comparison is **8.9 ms at 0.6% convergence versus 36.7 ms at 25%** — roughly 4×
faster *and* far better converged.

**Result 3 — but the wake is still not correct: it is smooth and incomplete.**

The domain-invariance test (axis A) still fails, and the reason is explicit:

```
 margin   xbins   rough(dE)   dE vs m=1
    1.0     100     0.01631     0.00000
    2.0     200     0.01546     0.56203
    8.0     800     0.02170     0.99184

ds = 0.5 mm   located band = [-189.0, +61.4] um,  2 nonzero intervals:
      [-1214.5,  -926.7]  67.18% of the column   inside band: False   <-- MISSED
      [ -135.5,    -5.7]  32.82% of the column   inside band: True
```

The co-moving fix did its job — **3 intervals → 2**, the ghost image is gone, exactly as predicted in
Step 4e. But the surviving chirp branch is *physical*, carries up to **67%** of the column, and lies
outside the located band, because `_retarded_xi_band`'s fixed point initialises at `xp = x` and
converges to the narrow branch only. So the wake is smooth, self-converged, and **quantitatively
wrong** — the exact "smooth but wrong" trap the falsification test in Step 4c was built to catch.

**Net effect.** This step removed a proven O(1) accuracy defect and, via the band locator, most of the
residual noise. It did **not** make the wake correct. The remaining work is the two-branch band
location from Eq 4.24 (Step 6), which the co-moving interpolant makes materially simpler: one
unambiguous `τ(t_ret)` to feed the formula, and 2 candidate branches instead of 4.

**Reproduce.**
```bash
python pyDFCSR_2D/test/test_ghosting.py           # the acceptance test, CMV columns flat vs G
python pyDFCSR_2D/test/test_xi_bands_converge.py  # axis A still fails; see Step 6
```

### Step 5c — Localization branches: corrected Eq 4.24 against the measured integrand (2026-09-09) ✅

Step 4f verified Eq 4.24 against a root-find of **its own defining system** — that was about the
*algebra*. This checks the *physics*: do the predicted branches sit where the integrand actually is?
Same comparison as thesis Fig. 4.3(b–c), but with the corrected formula and against the code.

The branch positions are implicit — `τ` and the intercept come from the frame at `t_ret`, and `t_ret`
depends on `x′` — so each branch is found by the fixed point
`x′ → t_ret → blended frame → Eq 4.24 → x′`. For the co-moving path the axis is
`x′ = p_α(z′) + ξ̄_α` with `p` degree 1, so `τ = p_α[0]` and the effective intercept is
`p_α[1] + ξ̄_α`.

**Result — co-moving: exactly two branches, both predicted correctly.**

```
   s-s_prime  #found |  Eq4.24 narrow     measured   rel err |  Eq4.24 chirp     measured   rel err
     0.00020       2 |          -63.8        -90.2    inside |        -470.8       -444.2    inside
     0.00050       2 |          -63.8        -70.6    inside |       -1077.5      -1070.6    inside
     0.00100       2 |          -63.8        -67.1    inside |       -2077.7      -2074.3    inside
     0.00150       2 |          -63.8        -66.0    inside |       -3065.6      -3063.4    inside
     0.00200       2 |          -63.9        -65.5    inside |       -4041.5      -4030.5    inside
     0.00300       1 |          -63.9        -65.0    inside |       -5958.7           --    absent
     0.00402       1 |          -64.0        -64.8    inside |       -7865.9           --    absent
     0.01000       1 |          -64.4        -64.7    inside |      -18315.7           --    absent
```

Every prediction lands **inside** the measured interval. The chirp branch tracks to ~0.3%
(−1077.5 predicted vs −1070.6 measured; −4041.5 vs −4030.5). Beyond `ds ≈ 2–3 mm` it is reported
`absent`: the formula still returns a position, correctly, but `z′` has left the beam so there is no
density there — consistent with the code's `d = 4018 µm` being *defined* as where the chirp branch
exits `10σ_x`.

**Result — interval count is the direct signature of ghosting.**

```
bspline_fft      : [1, 3, 3, 3, 3]   <- each snapshot contributes its own pair of roots
bspline_comoving : [2, 2, 2, 2, 2]   <- one blended frame, so one pair
```

Eq 4.24 predicts **two**. The co-moving path produces exactly two; the lab-frame blend produces three.
And Eq 4.24 **cannot describe `bspline_fft` at all** — the radicand goes negative (`nan`) for
`ds < 3 mm`, and where it does return a value it lies on neither ridge. That is not a failure of the
formula: it is a statement that the lab-frame blend **has no well-defined localization geometry**,
because it superposes two beam axes. A scheme with no single axis has no Eq 4.24.

![Branches, co-moving](pyDFCSR/pyDFCSR_2D/test/benchmark_results/localization/branches_bspline_comoving.png)

![Branches, lab-frame blend](pyDFCSR/pyDFCSR_2D/test/benchmark_results/localization/branches_bspline_fft.png)

The figures are the clearest artefact this investigation has produced. In the co-moving panels the
integrand sits on **one** horizontal ridge (narrow branch) and **one** diagonal ridge (chirp branch),
with the red and white-dashed Eq 4.24 curves lying exactly on top of them. In the `bspline_fft` panels
there are **two** diagonal ridges — the ghost double image, visible directly in the integration plane
rather than inferred — and the predicted curves match neither.

**What this unlocks.** Eq 4.24 is now validated as a *predictor of where to put quadrature nodes*, on
the co-moving path, to sub-percent accuracy. Step 6 can build on it directly, and
`test_localization_branches.py:Geometry.branch()` already implements the fixed point it needs.

**Files touched.** `pyDFCSR_2D/test/test_localization_branches.py` (new) — writes
`benchmark_results/localization/`.

**Reproduce.**
```bash
python pyDFCSR_2D/test/test_localization_branches.py    # ~3 min, both methods
```

### Step 5d — Does Step 5c hold at other beam tilts? Shear sweep (2026-09-09) ✅ — **yes, and it caught a bug in my own code**

Step 5c checked one shear (`z:x = 20`, `dx/dz = −1.60`, amplification 65.9×) at one observation point.
That is thin evidence: the branch geometry depends on the local tilt `τ`, on the observation point
`(x, s)`, and on **which side of the `|tan θ| = 1` degeneracy** the beam sits — the point where Eq
4.24's `(τ²−1)` denominator vanishes and the code switches branch treatment. So this sweeps all three:

| axis | values |
|---|---|
| `shear z:x` | 0, 1, 2, 5, 20, 50 |
| observation point | 3 positions on the wake mesh in z (fractions 0.2 / 0.5 / 0.8) |
| `s − s′` | 0.2, 0.5, 1, 2, 4, 10 mm |

Shears 1 and 2 are deliberately chosen to straddle `|τ| = 1`, so the linear-fallback guard is exercised.

**Classification uses no distance tolerance.** A prediction is `absent` if the integrand evaluated
*at the predicted point* is `< 1e-6` of that column's peak — that branch carries no density. Otherwise
it must land inside a measured nonzero interval or it is a `MISS`. (The first version of this test used
an absolute `20σ_ξ` window, which at low tilt is 1.2 mm — far too loose. See below.)

```
  shear     dx/dz     amp     branch  inside  absent   MISS  worst err  max #int
      0   -0.3715     1.1  non-chirp      18      18      0     0.0000         1
      1   -0.2587     1.0  non-chirp      18      18      0     0.0000         1
      2   -2.0695     1.3      chirp      20      16      0     0.0000         1
      5   -2.1971     8.9      chirp      25      11      0     0.0000         2
     20   -1.6034    65.9      chirp      28       8      0     0.0000         2
     50   -1.5154   166.8      chirp      32       4      0     0.0000         2

TOTAL: 141 predictions landed inside measured support, 0 missed.
Max intervals seen at any (shear, obs point): 2 (Eq 4.24 predicts at most 2)
VERDICT: consistent across all tilts tested
```

Two structural trends, both as the thesis predicts:

- **`max #int` is 1 for amplification ≤ 1.3× and 2 for ≥ 8.9×.** At low tilt the chirp root
  degenerates toward `x₂ = x` and merges with the narrow branch, so there is only one resolvable
  ridge. Two distinct branches appear only once the beam is genuinely chirped.
- **`absent` falls monotonically (18, 18, 16, 11, 8, 4) as amplification rises.** The more the beam is
  sheared, the further the chirp branch stays inside the density in `z′`, so fewer predictions fall
  where there is nothing.

Note also that `dx/dz` is *not* monotonic in the input shear (−0.37, −0.26, −2.07, −2.20, −1.60, −1.51):
the dipole generates its own tilt, which adds to or partly cancels the imposed one. The amplification
`σ_x/σ_ξ` is the meaningful ordering, and it is monotonic.

**The sweep found a real bug — in `eq424()`, not in Eq 4.24.** The first run reported
**INCONSISTENT, 17 misses**. Rather than accept the verdict I measured `|integrand|` at each predicted
position: exactly `0.000e+00`, with `z_ret` sitting 2.0–19.2 σ_z outside the beam. Two causes, both mine:

1. **The `l > 0` spurious-root guard was missing.** Step 4f *documented* it as required and I did not
   implement it. The derivation squares `l = Tb + x′/τ`, which admits solutions with `l < 0` —
   `t_ret > t`, a source point in the **future**. For shear 1 at `ds = 1 mm`, `l = 1000 − 2175 =
   −1175 µm`. The fix, now in `eq424()`:

   ```python
   if Tb + xp / tau <= 0:
       return np.nan     # l must be a positive distance
   ```

   This guard is **required by Step 6** — without it the band locator would place quadrature nodes at
   confidently-computed positions where the integrand is identically zero. It was only caught at *low*
   tilt, where the spurious root lands a few hundred µm from the real one instead of far away.

2. **The classifier's absolute tolerance was too loose.** Replaced with the direct integrand
   evaluation described above, which has no length scale in it at all.

Both fixed; re-run gives 141/0. **This sweep should be kept as a regression test for Step 6**, since it
is what exercises the `l ≤ 0` and `|τ| ≈ 1` guards.

![Branch prediction vs measured, swept over shear](pyDFCSR/pyDFCSR_2D/test/benchmark_results/localization/branch_shear_sweep.png)

Each panel is predicted (Eq 4.24) vs measured branch centre for one shear, with the `y = x` line;
green circles are `inside`, red × would be misses. There are no red markers in any panel.

**Files touched.** `pyDFCSR_2D/test/test_localization_shear_sweep.py` (new);
`test_localization_branches.py` — `write_inputs`/`run` parametrized by `shear`, and the `l > 0` guard
added to `eq424()`.

**Reproduce.**
```bash
python pyDFCSR_2D/test/test_localization_shear_sweep.py   # ~15 min, 6 shears
# writes benchmark_results/localization/{branch_shear_sweep.png,shear_sweep_log.txt}
```

### Step 6 — Two-branch band location from the corrected Eq 4.24 (2026-09-10) ✅ **implemented; the missing branch was worth the whole wake**

Per Step 4f's verified formula and Step 5's Result 3. For each `s′` column compute **both** roots of

```
Tb = (t − s′) − b/τ ,   rad = (τ²−1)(Tb² − q²) + ((n′·q)τ + Tb)²
x′± = τ [ Tb + (n′·q)τ ± √rad ] / (τ² − 1)
```

with `τ`, `b` from the *blended* co-moving frame at `t_ret`, then refine each with the per-column fixed
point and give each its own sub-band, merging where they overlap. Guards: `τ²=1` → linear;
`rad < 0` → branch absent; `l ≤ 0` → spurious root. The polar patch stays as is for `r < R2`.

All three guards are already implemented and exercised in `test_localization_branches.py:eq424()` —
reuse it rather than rewriting. The `l ≤ 0` guard is **not optional**: Step 5d shows that omitting it
puts nodes where the integrand is identically zero, and `test_localization_shear_sweep.py` is the
regression test that catches it.

Step 5c removed the guesswork: the formula predicts both branch positions to ~0.3% on the co-moving
path, and `test_localization_branches.py:Geometry.branch()` and `eq424()` already implement the fixed
point and the guards. The remaining work is to vectorize them over `sp`, derive each sub-band's *width*
from the deposition grid (as `_retarded_xi_band` already does for the narrow branch), and integrate the
sub-bands separately. Note the "skip negligible branches by blend weight" idea from Step 4e is no
longer needed on this path — there is only one frame, so both branches are physical.

**Acceptance:** axis A of `test_xi_bands_converge.py` reads ~0 at the worst z point, and the located
sub-bands cover every measured nonzero interval.

#### 6a. Geometry of the branch crossing, and how it meets the polar patch

Worked out before writing any code, because it settles two design choices and exposes one gap.

**The two branches intersect exactly at the singular point.** At `s′ = s` the chirp branch is
`x₂ = x − (s−s′)·tan 2α = x`, and the narrow branch is `x₁ ≈ x`. Both pass through `(x′ = x, s′ = s)`,
which is precisely where `|r − r′| = 0` and the kernel diverges. In Eq 4.24 that is `rad → 0`: the two
roots `±√rad` collide, and `l = Tb + x′/τ → 0`. So the branch crossing **is** the disc centre.

```
        x'
         |      Cartesian ribbon bands (nodes ride the density)
         |   ====== branch A (narrow, x' ~ x) ==================
         |                       .-''-.
    x ---+----------------------(  ( x )  )   <- polar disc, R2 = 5 sigma_xi
         |                    ,'  `-'  /`.       branches cross AT the centre
         |                 ,-'    `-..-'   `-.
         |              ,-'   branch B (chirp)  `-.
         +---------------------|---------------------- s'
                              s'=s
```

**Consequence 1 — the branches are not labelled.** The implementation takes `sign = ±1`, not "narrow"
and "chirp". Which sign is which flips when `τ²` crosses 1, and Step 5d showed the two roots *merge*
into one resolvable ridge for amplification ≤ 1.3×. Treating the two roots symmetrically makes the
labelling irrelevant, which removes a whole class of branch-assignment bug.

**Consequence 2 — the taper must be applied to *both* branch meshes.** The partition of unity
`w(r) + (1−w(r)) = 1` is pointwise in the `(x′, s′)` plane: every Cartesian piece carries `w`, the disc
carries `1 − w`. If only one branch mesh were tapered the disc would over- or under-count by the
untapered branch's contribution inside `R2`.

**Consequence 3 — the merge rule is interval *subtraction*, not union.** The first design replaced two
overlapping bands with their union. That is exact but it *coarsens*: `nx` nodes then span up to 2× the
width, doubling the cell size precisely in the near-singular region where the integrand is largest —
the opposite of what Step 4 was for. Instead: both bands have identical width
`W = 2·xlim·σ_ξ·margin`, so if `lo_B > lo_A` they overlap one-sidedly and the uncovered part of B is
`[hi_A, hi_B]`. Clip B to that. Each mesh keeps `nx` nodes over a width ≤ `W`, so the cell is never
coarser than the single-branch case, identical bands collapse B to zero width automatically, and it
stays fully vectorized.

#### 6b. How the polar patch works (recorded because §4's Step 4 addendum never spelled it out)

`_near_patch_radii` (CSR.py:623):

```python
R2 = near_patch * self.beam._sigma_x_transform      # default near_patch = 5.0
R2 = min(R2, 0.4 * (s4 - s), 0.4 * (s - s3))
R1 = 0.5 * R2
```

- **The radius is 5 σ_ξ, not 5 σ_x.** In the main test case σ_x ≈ 827 µm at 65.9× amplification, so
  σ_ξ ≈ 12.5 µm and **R2 ≈ 63 µm**, against an `sp3` region spanning 25 σ_z. The clip to
  `0.4·(s4−s)` and `0.4·(s−s3)` keeps the disc strictly inside the one `s′` sub-region that owns it.
- **Only `sp3` gets a disc** (`tapers = [None, None, (*radii, False)]`, CSR.py:772) because it is the
  only region straddling `s′ = s`.
- **Why polar.** The integrand is `~1/r`, which is structural rather than merely measured: the
  longitudinal one is literally `num1/r_minus_rp + num2/r_minus_rp` (CSR.py:1066), and for the
  transverse `W1 = part1/r³` with `part1 = (r−r′)·(n−n′) ~ r·(r/R_bend)`, giving `1/(r·R_bend)`. The
  polar element `r dr dφ` (CSR.py:668) cancels that `1/r` **exactly**, leaving a bounded smooth
  integrand — the precondition for trapz to be 2nd order. On a uniform Cartesian mesh the integral
  still converges, but at `O(h)` with an error depending on where nodes fall relative to `r = 0`;
  that is the point-to-point wake jitter. Radial nodes at cell midpoints
  (`(arange(nr)+0.5)*dr`) mean `r = 0` is never evaluated.
- **It does not need to work anywhere else.** Since `w + (1−w) = 1` identically, the split is
  algebraically exact for *any* `R1`, `R2` — which is exactly what Step 4c axis D verified (total
  invariant to ~1% as `R2` swept 2 → 12 σ_ξ).

**Caveat i — the mesh radius and the taper radius are different quantities.** The mesh is a disc of
Euclidean radius `R2` in the `(x′, s′)` *parameter* plane; `w` is a function of `r_minus_rp`, the
physical lab chord (CSR.py:924–926). In a bend the chord is slightly shorter than the arc, so
`{|r−r′| ≤ R2}` is marginally *larger* than the parameter disc, leaving a sliver where `1−w > 0` that
the polar mesh never covers. Relative size `(R2/R_bend)²/24` ≈ **1e-10** at `R2 ≈ 63 µm` and
metre-scale bend radius. Negligible — but it is an approximation, recorded so it is not rediscovered.

**Caveat ii — the disc ignores the localization entirely.** Full `2π` in `φ`, no ribbon. Affordable
only because it is tiny and sits inside the beam.

**The gap this exposes, to be instrumented before it is trusted.** As `r → 0`, `rad → 0` from above,
so roundoff can push it slightly negative and the `rad < 0` guard would declare *both* branches absent.
Inside `R2` that is harmless (the disc covers it and `w → 0` kills the Cartesian piece). But branch
separation only exceeds the band width at

```
s − s′  ≈  2·xlim·σ_ξ·margin / |tan 2α|  ≈  20 σ_ξ     for |tan 2α| ~ 1
```

while the disc stops at `5 σ_ξ`. So there is an **annulus `5 σ_ξ < r < 20 σ_ξ`** where `w` is already 1
and the branches are still merged — handled by neither the disc nor by clean branch separation, only by
the interval-subtraction merge. A spurious `rad < 0` there would silently drop real support. Step 6 must
therefore log the count of `rad < 0` columns as a function of `r` rather than assume the guard is safe.

#### 6c. Result — the controlled A/B

Axis A on the co-moving path went 1.017 → 0.057, but that number conflates two changes, because Step 5
(co-moving interpolant) and Step 6 (two-branch locator) landed together. `test_two_branch_bands.py`
does the attribution properly: same beam, same interpolant, same everything, monkeypatching
**only** `_retarded_xi_bands` back to single-band behaviour.

```
                margin   xbins   rough(dE)   dE vs m=1   xk vs m=1
  single-branch    1.0     100     0.01514     0.00000     0.00000
  single-branch    2.0     200     0.01546     0.56203     0.25143
  single-branch    4.0     400     0.01809     0.86111     0.43354
  single-branch    8.0     800     0.03822     0.99183     0.54372

  two-branch       1.0     100     0.08615     0.00000     0.00000
  two-branch       2.0     200     0.08134     0.05487     0.10988
  two-branch       4.0     400     0.08119     0.05657     0.11076
  two-branch       8.0     800     0.08119     0.05657     0.11076
```

The single-branch column is still climbing on the last doubling (**+0.131**); the two-branch column
changes by **exactly 0.00000** from margin 4 to 8. That is the acceptance criterion in the only form
that means anything: *the domain has stopped mattering.* A wider transverse domain no longer finds
support the locator missed.

**Honest reading of the residual 0.057.** The `m = 1` reference point uses `xbins = 100`, and axis B
independently shows `xbins = 100` is itself **0.090** away from converged. So the `m = 1` offset is
within the resolution error of the reference, not evidence of a domain error — and the family
`m ∈ {2, 4, 8}` agrees with itself to five decimals. There is also a mechanism: after interval
subtraction the second band's width is the *branch separation*, which is margin-independent, so
`xbins ∝ margin` over-resolves it rather than holding its cell size fixed. Axis A's "fixed cell size"
premise therefore holds for band 1 but not for the clipped band 2. Worth remembering before reading too
much into that column.

**How much was actually missing.** At the base setting, switching the locator changes the wake by

```
  dE     single -> two branch : rel L2 change 1.04128
  x_kick single -> two branch : rel L2 change 0.52854
```

The missing chirp branch was of order the **entire** wake, not a correction to it. That is consistent
with Step 4e's per-column measurement (up to 67% of one column) and it retires the "smooth,
self-converged, and quantitatively wrong" caveat that has been at the top of this document since
Step 5.

**Falsification at low tilt — the test that matters most.** `test_xi_bands_equiv.py`, now on the
co-moving path, compares the new bands against the *old* ±20σ_x rectangle where the old quadrature is
demonstrably well sampled:

| shear | amplification | old vs new dE @ 800² | old self-conv | new self-conv |
|---|---|---|---|---|
| 0 | 1.1× | **0.00001** | 0.00056 → 0 | 0.00012 → 0 |
| 2 | 1.3× | 0.00125 | 0.00089 → 0 | 0.00083 → 0 |
| 20 | 65.9× | 1.46766 | 207 → 79 → 8.4 → 0 | 0.137 → 0.023 → 0.005 → 0 |

At amplification 1.1× the two schemes agree to **1e-5**. The single-band version agreed to 3.5e-4, so
adding the second branch made the low-tilt agreement **35× better** — exactly what should happen if the
missing branch is real: at low tilt the two branches nearly merge, so the missing piece is small but
nonzero, and it is now captured. At shear 20 the old bands never converge at all (self-convergence
207 → 0 only because the 800² reference is itself garbage; roughness 1.97 → 3.01, non-monotonic) while
the new bands converge cleanly, so the 1.47 gap is the old bands' error.

**The wake gets much SMALLER, and that needed independent confirmation.** The two-branch wake is not a
corrected version of the single-branch one — it is ~25× smaller and of the *opposite* sign at mid-z
(single-branch dips to −2.48, two-branch peaks at +0.10). The second branch very nearly cancels the
first. That is expected in this regime (it is the same near/far cancellation as Stupakov PRAB 25 014401
§IV, where the `ln Δs` terms cancel between regions) but "my new code makes the answer 25× smaller" is
exactly the claim that must not be taken on trust. So it was checked against a brute-force integral over
the **full ±20σ_x rectangle** (`xi_bands = False`), same co-moving interpolant, refining the transverse
node count for three single wake points:

```
z-idx 10:  bands single -0.05931   bands TWO +0.00516
           brute force xbins=  200 : -1.25506      800 : -0.05890     3200 : +0.01796
z-idx 20:  bands single -0.72398   bands TWO +0.06689
           brute force xbins=  200 : -8.72138      800 : +0.19078     3200 : +0.07394
z-idx 30:  bands single -2.47562   bands TWO +0.09903
           brute force xbins=  200 : +4.31923      800 : +0.01617     3200 : +0.04828
```

The brute force converges toward the **two-branch** value in sign and magnitude (z-idx 20:
+0.0739 vs +0.0669) and **excludes the single-branch value by one to two orders**: nothing in the
refinement sequence goes anywhere near −2.48. Note the brute force is still visibly noisy at
`xbins = 3200` — 3200 nodes across 33 mm is a 10 µm cell, only 0.8 σ_ξ — so this confirms sign and
magnitude to a few tens of percent, not more. That is enough to settle the question it was asked.

**Consequence worth carrying forward: this is now a cancellation problem.** The total is a small
difference of two large branch contributions, so the *relative* accuracy needed on each branch is
roughly (cancellation factor) × (accuracy wanted on the total). At 25× cancellation, 1% on each branch
buys only ~25% on the total. Any future convergence claim has to be made on the total, never on a
single branch, and the `zbins`/`near_patch` settings matter more than they did before.

**The `rad < 0` guard: instrumented, not assumed.** §6b flagged that roundoff could push `rad` slightly
negative near the branch crossing and silently drop real support in the annulus where the taper has
already reached 1. Measured over ~16 000 columns of the `sp3` region across the whole z scan:

```
    r / sigma_xi   columns    rad<0     frac                   zone
      1 -      2        40        0   0.0000      inside polar disc
      2 -      5       120        0   0.0000      inside polar disc
      5 -     10       120        0   0.0000    ANNULUS (taper = 1)
     10 -     20       280        0   0.0000    ANNULUS (taper = 1)
     20 -     50       760        0   0.0000     branches separated
     50 -    inf     14680        0   0.0000     branches separated
```

**Zero occurrences anywhere**, including all 400 annulus columns. The guard is safe here. Recorded as a
measurement rather than an argument, so if it ever does fire the diagnostic already exists.

The same diagnostic surfaced a second worry — columns near the pole where **both** branches are marked
dead by the z-support test, hence contributing nothing while the taper is already 1. Checked directly by
scanning `x′` over ±14σ_x at those columns:

```
  r/s_xi  live0  live1   peak|g_z| over +-14 s_x   peak inside band
   14.07   True   True                1.7473e+12         1.7473e+12
    9.30  False  False                0.0000e+00         0.0000e+00
   14.46  False  False                0.0000e+00         0.0000e+00
   19.63  False  False                0.0000e+00         0.0000e+00
```

The dead columns are **genuinely empty** — the integrand is exactly zero across the full scan, not merely
outside the band — and the live column's peak sits entirely inside its band. So the `live` guard is
dropping nothing. Not a hole.

**Cross-check of the vectorized formula.** `CSR2D._eq424` is a fresh vectorized reimplementation, not a
call into the validated scalar `test_localization_branches.eq424`, so the two were compared directly on
200 000 random cases deliberately spanning `τ ≈ 0`, `τ ≈ ±1` and `|τ| > 1`:

```
sign -1: nan-agree   55519  num-agree  144481  EXISTENCE MISMATCH 0   max rel diff 8.07e-12
sign +1: nan-agree  121094  num-agree   78906  EXISTENCE MISMATCH 0   max rel diff 2.41e-13
```

Zero disagreements about whether a branch *exists* — which is the failure mode that matters, since an
existence error places nodes in the wrong place rather than slightly misplacing them.

**Cost: 1.5×, not the 2× predicted.** 20.4 → 30.7 ms per wake point. Cheaper than feared because many
columns' second branch is dead or fully covered, and a region with no live second branch anywhere is
skipped outright.

#### 6d. Is the wake actually smooth? (correcting a misreading of my own metric)

I first reported "roughness rose 0.0155 → 0.0813, so the two-branch wake is rougher." **That was
wrong**, and wrong in an avoidable way: the roughness metric used throughout this document is
`‖Δ²w‖ / ‖w‖`, and the two-branch wake is ~25× smaller. Dividing by a 25× smaller norm inflates the
number by itself. The correct comparison is the **unnormalized** second difference:

```
locator         zbins      ||w||   rough ABS   rough REL
single-branch     100     8.9342     0.18561     0.02077
single-branch     200     8.9342     0.13816     0.01546
single-branch     400     8.9342     0.13410     0.01501
single-branch     800     8.9342     0.13410     0.01501

two-branch        100     0.4017     0.11258     0.28028
two-branch        200     0.3998     0.03252     0.08134
two-branch        400     0.3998     0.00853     0.02133
two-branch        800     0.3999     0.00819     0.02048
```

In absolute terms the two-branch wake is **16× smoother** (0.0082 vs 0.1341), not rougher.

**Second correction: the saturation is not noise.** `Δ²w ≈ h²w''` where `h` is the *CSR mesh* spacing,
which `zbins` does not change. So the floor each column settles on is the wake's genuine curvature, and
only the *excess* above it is quadrature noise. Both locators reach their floor; relative to their own
magnitude the floors are comparable (0.0150 vs 0.0205).

**Direct evidence that what remains is shape, not noise**, at `zbins = 800`:

- the second difference flips sign **11 times out of 37** — pure noise flips about half the time,
  smooth curvature only where `w''` does;
- the residual about a smooth polynomial fit is **1.28% of peak** at degree 8 and **0.52%** at degree 10.
  The wake is a smooth low-order curve to ~1%.

So: **yes, smooth.** For scale, relative roughness at this tilt was 3.0 with the old bands, 0.72
pre-Step-4, 0.18–0.21 post-Step-4, and is **0.020** now.

**The actionable finding is a default, not a defect.** Absolute roughness falls 0.113 → 0.033 → 0.0085
→ 0.0082, i.e. ~4× per doubling (second order, as trapz should be) and then hits the curvature floor at
`zbins ≈ 400`. The base setting `zbins = 200` sits **4× above the floor**. `CSR_integration: zbins`
should default to 400 on this path; the earlier "roughness is `zbins`-limited" note was right about the
cause and wrong to file it under "got worse".

#### 6e. Open item, figures, and files touched

**One thing did get worse, recorded rather than buried.**

- **Axis D (near-patch radius invariance) degraded from 0.011 to 0.085** across `R2 = 2 → 12 σ_ξ`, and
   roughness falls monotonically with `R2` (0.244 → 0.023). Plausible cause: two branch bands put more
   Cartesian area near the `1/r` pole than one did, so the small-`R2` end is less resolved than before —
   consistent with axis F, where turning the patch off now costs 0.220 in dE (was 0.042) at every
   resolution. **This was not A/B'd, so it is not attributed.** It is the first thing to look at next,
   and it suggests `near_patch = 5.0` may no longer be the right default.

![Two-branch band location](pyDFCSR/pyDFCSR_2D/test/benchmark_results/xi_bands/two_branch_bands.png)

![Convergence axes, co-moving](pyDFCSR/pyDFCSR_2D/test/benchmark_results/xi_bands/xi_bands_convergence_bspline_comoving.png)

![Falsification vs the old bands, co-moving](pyDFCSR/pyDFCSR_2D/test/benchmark_results/xi_bands/xi_bands_equiv_bspline_comoving.png)

**Files touched.**
- `pyDFCSR_2D/CSR.py` — new `_comoving_frame_at`, `_eq424`, `_retarded_xi_bands`, `_disjoint_bands`;
  `_integrate_xi_region` now loops over branch bands. `_retarded_xi_band` is kept and is still what
  `legacy` and `bspline_fft` use, so those two paths are unchanged.
- `pyDFCSR_2D/test/test_two_branch_bands.py` (new) — the controlled A/B and the `rad < 0` diagnostic.
- `pyDFCSR_2D/test/test_xi_bands_converge.py`, `test_xi_bands_equiv.py` — parametrized by deposition
  method (default `bspline_comoving`), outputs suffixed by method so both remain comparable.

**A follow-up fix, and a trap in reading `_branch_diag`.** Plotting the nodes (§6f) showed some at
±10 **metres**. On a *dead* column the fixed point has no root to converge to and can wander, and the
band was being centred there. It contributes exactly zero either way — re-running the A/B after the fix
gives byte-identical numbers (1.04128, 0.52854) — but the coordinates are absurd, waste evaluations, and
could in principle land on the `1/|r−r′|` pole. Dead branches are now parked at `x + 4·half`: finite, in
a zero-density region, never exactly at `r = 0`.

The same investigation nearly produced a false alarm. `_disjoint_bands` **reorders the bands per
column** (`a_first = loA <= loB` swaps them wherever the second branch starts first), so
`_branch_diag['branches'][j]` does *not* correspond to the returned `bands[j]`. My diagnostic paired
them anyway and reported "dead columns have nonzero width", which looked like a correctness bug and was
not — `_integrate_xi_region` tests `width > 0` on the returned bands, never the `live` mask, so it was
right all along. Now noted in the `_disjoint_bands` docstring so the next reader does not repeat it.

**Reproduce.**
```bash
python pyDFCSR_2D/test/test_two_branch_bands.py                        # the A/B, ~6 min
python pyDFCSR_2D/test/test_xi_bands_converge.py bspline_comoving      # ~5 min
python pyDFCSR_2D/test/test_xi_bands_equiv.py   bspline_comoving       # ~20 min
```

The brute-force cross-check and the dead-column scan were run inline rather than committed as tests; the
numbers are above, and both are a few lines on top of `test_two_branch_bands.py`'s fixture
(`xi_bands = False` with `xbins` swept, and `get_CSR_integrand` over `±14σ_x` at a fixed `s′`).

#### 6f. What the integrand and the wakes actually look like (2026-09-10)

Everything above is scalar metrics. Those decide whether something is converged, but they
hide what the answer looks like, so here are the pictures — swept over shear
`z:x = 0, 2, 20, 50` (amplification 1.1x -> 166.8x).

**READ THIS FIRST: the first version of this section was wrong, and the way it was wrong is
the most useful thing in it.** To make the diagnostic fast I set `compute_CSR: 0`, assuming it
only skipped the wake evaluation. It does not. `CSR.py:311` is

```python
if debug or self.CSR_params.compute_CSR:
    self.DF_tracker.get_DF(...); self.DF_tracker.append_DF()
    self.DF_tracker.append_interpolant(...); self.DF_tracker.build_interpolant()
```

so `compute_CSR: 0` gates the **entire density-history build**. The tracker was left with
**1 snapshot instead of 3**, `get_CSR_integrand` returned near-zeros, and every figure and
number still looked plausible. It failed *silently* — no exception, no empty array, just a
thin sliver of fake support.

It was caught only because the figure disagreed with Step 5c. Reproducing Step 5c's exact
config and observation point gave its published numbers to 0.1 um, while an identical
observation point with identical beam statistics (`sigma_x = 826.57`, `sigma_xi = 12.54`,
`sigma_z = 515.44`, `slope = -1.6034`, `x_obs = -63.74 um`) gave **zero** intervals. Same
beam, same point, different integrand -> it had to be the history.

```
                 snapshots   intervals at ds = 1 mm
debug=False              1   0
debug=True               3   2, at -2074.3 um and -67.1 um   <- matches Step 5c exactly
```

The fix is `run(stop_time=..., debug=True)`, which builds the history while `compute_CSR: 0`
still skips the wake (gated separately at `:334`). Runtime stays ~10 s. **Lesson worth
keeping: `compute_CSR: 0` is not a performance knob for anything that later calls
`get_CSR_integrand`.** All numbers below are post-fix.

**Plotting the integrand also needed a change of coordinates.** A plain linear `(x', s')`
colormap cannot work at high tilt: the chirp ridge is about `sigma_xi` = 12 um wide but sweeps
across ~10 mm of `x'` over the `s'` range at shear 50, so it is thinner than one pixel on any
linear axis. Two views are therefore produced.

**View 1 — lab `(x', s')`, where the thesis 4.4.2 picture appears literally.** Support is
measured, not modelled: each `s'` column is scanned over a fixed, *prediction-independent*
window `x - 14 sigma_x` to `x + 6 sigma_x` (anchoring the window to the predicted centres, as
a first version did, makes the figure structurally unable to reveal support the prediction
missed).

![Integrand support in lab coordinates](pyDFCSR/pyDFCSR_2D/test/benchmark_results/wake_maps/integrand_lab_geometry.png)

This is the clearest confirmation of the localization geometry produced so far. A narrow band
sits flat at `x' = x` parallel to the `s'` axis, and a chirp band leaves the observation point
diagonally — and **its measured slope is `tan 2alpha`**, independently:

| shear | measured diagonal slope | `tan 2alpha` | chirp reach in `s'` |
|---|---|---|---|
| 0 | — (branch absent, 0% live) | -0.86 | — |
| 2 | 300 um / 0.24 mm = **1.25** | +1.26 | -0.24 mm |
| 20 | 4400 um / 2.15 mm = **2.05** | +2.04 | -2.15 mm |
| 50 | 10500 um / 4.8 mm = **2.19** | +2.34 | -4.8 mm |

At shear 0 the chirp branch does not exist (`|tan theta| < 1`, the root degenerates into the
narrow one), and at shear 20 and 50 the chirp band **dominates** the support while the narrow
band is a thin line at `x' = x`. Both are as thesis 4.4.2 predicts.

**View 2 — each branch in its own band coordinate** `v = (x' - centre_k(s'))/half_k(s')`, so
`v = 0` is the branch and `|v| <= 1` is the band the quadrature covers. This straightens each
ridge so the colormap is resolvable, and answers "is the support inside the band?".

![Integrand in band-relative coordinates](pyDFCSR/pyDFCSR_2D/test/benchmark_results/wake_maps/integrand_band_relative.png)

**Caution on reading View 2:** it *co-moves with each branch*, so a correctly tracked chirp
band is drawn **flat**, and its characteristic diagonal is subtracted away by construction. A
diagonal streak in this view is a tracking *residual*, not a strong chirp band — the opposite
of how it reads at a glance. View 1 is the one for geometry; View 2 is only for band fit.

**Coverage.** Fraction of `|integrand|` inside each individual band, and inside the **union**
of the located bands, over a 40-half-width window:

```
 shear  region  live %  inside |v|<=1  |  UNION coverage
     0     sp3   88.3%         1.0000  |         1.00000
     2     sp3   89.3%         1.0000  |         0.98018
    20     sp3   72.3%         0.9397  |         0.99751
    50     sp3   68.3%         0.9728  |         0.99994
   all     sp1  50-100%        1.0000  |         1.00000
   all     sp2   100.0%        1.0000  |         1.00000
```

**Union coverage >= 0.98018 across all 12 (shear, region) combinations.** That is an
independent confirmation of axis A from the opposite direction: axis A *infers* coverage from
the wake's insensitivity to a wider domain, this *measures* it on the integrand. Where an
individual band is off-centre, the other band covers the remainder.

**The wakes.**

![Wakes on the tilt-removed mesh](pyDFCSR/pyDFCSR_2D/test/benchmark_results/wake_maps/wake_transformed_vs_shear.png)

![Wakes in the physical x-z plane](pyDFCSR/pyDFCSR_2D/test/benchmark_results/wake_maps/wake_xz_vs_shear.png)

- Shear 0 and 2 give the textbook structure: `dE/dct` negative through the core turning
  positive at the head, essentially a function of `z`; the transverse kick is a smooth centred
  blob.
- The magnitude collapses with tilt — `dE/dct` range +-6.09 MeV/m at shear 2 versus +-0.12 at
  shear 20 and +-0.07 at shear 50. This is the 6c branch cancellation as a global feature
  rather than a single-point number.
- **Residual noise at shear 50 is confined to the transverse mesh edges**
  (`|x - p(z)| >~ 25 um`), where density is low; the core is smooth.
- In physical `(x, z)` the mesh is a sheared parallelogram collapsing to a thin diagonal
  ribbon at high shear, so wake detail is unreadable there. The tilt-removed view is the one
  to judge smoothness on.

These wake maps were produced with `compute_CSR: 1` and are unaffected by the history bug.

**A second, unrelated performance trap.** `test_wake_maps.py` keeps `compute_CSR: 1` with a
21x51 wake mesh, so `csr.run()` computes the full 2D wake at every lattice step — about 7500
wake points — before the script does its own work. That is why it costs 8-9 min per shear
rather than the ~90 s its own work justifies. Correct for wake maps, pure waste for anything
that only needs the history.

**Files.** `pyDFCSR_2D/test/test_wake_maps.py` (new, wake maps),
`pyDFCSR_2D/test/test_integrand_plane.py` (new, lab geometry + band-relative + union
coverage).

```bash
python pyDFCSR_2D/test/test_integrand_plane.py   # ~10 s  (needs debug=True)
python pyDFCSR_2D/test/test_wake_maps.py         # ~35 min
```

#### 6g. Two production bugs found by widening one plot axis (2026-09-10) ⚠️ **affects absolute accuracy**

Extending the branch-0 panel of §6f out to a formation length meant printing `L_f`, which turned out to
be **identical for all four shears** — and that unravelled two real bugs in the production code, neither
of which any existing test would catch.

**Bug 1 — `formation_length` never sees in-bend bunch lengthening.** `get_formation_length` is called at
`CSR.py:270` once per *element, on entry*, so it uses σ_z at the dipole entrance and never updates as the
bunch lengthens inside the bend. `CSR.py:317` has exactly the per-step fix commented out:
`#self.get_formation_length(R=R, sigma_z=self.beam.sigma_z)`.

| shear | σ_z at s = 0.6 m | `csr.formation_length` | `(24R²·5σ_z)^(1/3)` |
|---|---|---|---|
| 0 | 57.5 µm | 181.7 mm | 190.4 mm |
| 20 | 515.4 µm | **181.7 mm** | 395.5 mm |
| 50 | 1361.9 µm | **181.7 mm** | 546.7 mm |

181.7 mm is the value for σ_z = 50 µm, the *initial* bunch length. At shear 50 it is **3× too small**.
`L_f` sets both `s1 = max(0, s2 − n_fl·L_f)` (how far back the integration reaches) and the history
truncation in `append_interpolant`, so it is not a cosmetic quantity.

**Bug 2 — the default `n_formation_length = 1.5` discards history that matters, by 12.9%.** Integration
extent held fixed at 1.5 `L_f`, only the retained history depth varied:

```
   n_fl (history)  snapshots   rel L2 vs deepest
              1.5          3             0.12875
              3.0          6             0.00000
              6.0          8             0.00000
             12.0          8             0.00000
             24.0          8             0.00000
```

**The default retains 3 snapshots and is 12.9% wrong.** Converged from `n_fl ≥ 3.0`.

**Longitudinal integration *extent* is not the problem — history *depth* is.** These two are easy to
conflate because the same `n_formation_length` controls both. Varying only the integration reach, with the
history built deep enough that truncation never binds:

```
history depth       integrate to 1.5 L_f    3.0 L_f    6.0 L_f   12.0 L_f
n_fl = 1.5  (3 snaps)          0.00000    0.02398    0.02398    0.02398
n_fl = 12.0 (8 snaps)          0.00000    0.00214    0.00214    0.00214
```

With an adequate history the extent effect is **0.2%**, i.e. `1.5 L_f` of reach is fine. My first attempt
at this measurement read 2.4% and I nearly logged it as an extent error — wrong, because I varied
`n_formation_length` *after* the run, so the history had already been truncated at 1.5 `L_f` and the extra
domain returned exact zeros. **Extending the integration domain past the retained history is silently
free and silently useless.**

**The two bugs compound.** `L_f` is ~2× small at shear 20 *and* `n_fl = 1.5` is ~2× small, so the retained
history is roughly 4× shorter than it should be. Fixing Bug 1 alone would raise the effective reach at
shear 20 from 273 mm to 593 mm, which is past the convergence point measured above — so **fixing the
formation length makes the existing default correct**, which is the cheapest of the available fixes.

**Does this invalidate Step 6?** No. Every comparison in §6c–§6f used identical history settings in both
arms, so the controlled A/B (rel L2 1.04 for the missing branch), the low-tilt falsification (1e-5), and
the union coverage all stand. But the **absolute** wake magnitudes quoted there are ~13% from converged,
and should be restated once Bug 1 is fixed.

**Still-untested axis.** `n_formation_length` sets how far *back* history is kept; the lattice
`step_size` sets how *finely*. Only 8 snapshots exist over 0.7 m here, so "converged at 6 snapshots"
really means "uses all available history" — it cannot distinguish convergence from saturation. The
history *step size* axis, which the original plan's Step 6 verification explicitly asked for, has still
never been varied.

**Reproduce.** Measured inline on `test_two_branch_bands.py`'s fixture: rewrite the config's
`n_formation_length`, `run(..., debug=True)`, then sweep `csr.integration_params.n_formation_length`
before calling `z_scan`. Worth promoting to a committed test alongside axes A–F.

#### 6h. Bug 1 fixed: refresh the formation length every step (2026-09-10) ✅

New `CSR2D._refresh_formation_length(R)` factors out the existing branching (in-bend → `R`; after-bend
drift → `R_rec`; pre-first-bend drift → returns `False`, since there `formation_length` is an accumulated
drift length and must *not* be recomputed from σ_z). It is now called both on element entry, where the
old code called `get_formation_length` directly, and **once per step** after tracking and before
`append_interpolant` — which is where `CSR.py:338` had the fix commented out all along. Note the
commented line passed `sigma_z=self.beam.sigma_z` without the `5×` factor the entry call uses; the
5× is kept, so this changes *when* `L_f` is evaluated, not its definition.

**`L_f` now matches the analytic value exactly, and the history deepens as a result:**

```
   shear    sigma_z    L_f before   L_f now   expected   snapshots before -> after
       0      57.5um      181.7mm   190.4mm    190.4mm            3 -> 3
      20     515.4um      181.7mm   395.5mm    395.5mm            3 -> 6
      50    1361.9um      181.7mm   546.7mm    546.7mm            3 -> 8
```

**And it does exactly what §6g predicted — the default is now converged.** History-depth sweep,
integration extent held at 1.5 `L_f`:

```
   n_fl (history)  snapshots  L_f (mm)   rel L2 vs deepest
              1.5          6     395.5             0.00000     <- was 0.12875
              3.0          8     395.5             0.00000
              6.0          8     395.5             0.00000
             24.0          8     395.5             0.00000
```

The 12.9% history-truncation error is gone at the shipped default, without changing
`n_formation_length`. That was the cheapest of the available fixes and it is the one that was already
identified in a stale comment.

**Regression check — every Step 6 conclusion survives**, with absolute numbers shifted as expected since
the wake is now computed from a deeper history:

| | before fix | after fix |
|---|---|---|
| axis A, single-branch, worst | 0.99183 | 0.97179 (still climbing) |
| axis A, two-branch, last doubling | 0.00000 | **0.00000** (0.06212 → 0.06212) |
| missing-branch effect, dE | 1.04128 | **1.06466** |
| `rad < 0` occurrences | 0 | **0** |
| unit tests | 13 pass, 1 pre-existing fail | unchanged |

**The low-tilt falsification is unchanged, which is the important one.** Re-run post-fix:

| shear | amplification | old vs new dE @ 800² | before fix |
|---|---|---|---|
| 0 | 1.1× | **0.00001** | 0.00001 |
| 2 | 1.3× | **0.00125** | 0.00125 |
| 20 | 65.9× | 29.00313 | 1.46766 |

Shear 0 and 2 — where the old ±20σ_x bands are well sampled and therefore trustworthy — agree to
**1e-5** exactly as before. The deeper history did not disturb the regime where the answer is known.

The shear-20 number went from 1.47 to 29.0, and the self-convergence columns show why it is the *old*
bands degrading, not the new ones:

```
 grid   rough old   rough new  old self-conv  new self-conv   old vs new
  100     2.67649     0.28973       23.23772        0.21296    623.94312
  200     2.30836     0.08777        6.52830        0.02540    186.53100
  400     3.51705     0.01938        3.51956        0.00560     99.52106
  800     0.37804     0.01818        0.00000        0.00000     29.00313
```

The new bands self-converge monotonically (0.213 → 0.025 → 0.0056 → 0) with roughness falling
0.29 → 0.018. The old rectangle changes by **352%** on its final refinement and its roughness is
non-monotonic (2.68, 2.31, 3.52, 0.38) — it is not converged at any grid tested, so `old vs new` is not
a statement about the new bands' accuracy. A deeper history carries more retarded-time structure, so the
under-resolved old quadrature does *worse* than before, which is the expected direction.

Bug 2 (`n_formation_length = 1.5` being too small) needs no separate fix — it was only a problem because
`L_f` was too small.

**Not yet done.** (i) The history *step size* axis is still untested. (ii) The absolute magnitudes quoted
in §6c–§6f were measured pre-fix and are ~13% low. (iii) **The brute-force cross-check in §6c — the one
that independently confirmed the two-branch magnitude and sign against a full ±20σ_x integral at 3200
transverse bins — was run pre-fix and should be repeated.** It is the only independent confirmation of
the absolute answer at high tilt, and the numbers above show the old-band reference is fragile, so it
needs redoing with the deeper history before the high-tilt magnitude is claimed as verified.

**Files touched.** `pyDFCSR_2D/CSR.py` — new `_refresh_formation_length`; element-entry calls routed
through it; per-step call added before `append_interpolant`.

#### 6i. Closing the three open items from §6h (2026-09-10)

**(1) Brute-force cross-check, redone post-fix — weaker than before, and it says why.**

```
z-idx 20:  bands single -0.64954   bands TWO +0.06911
    xbins  cell/sigma_xi         dE
     1600          1.648       +0.07957
     3200          0.824       +0.03277
     6400          0.412       +0.10569
    12800          0.206       +0.09118
```

At `xbins = 12800` the brute-force cell (0.206 σ_ξ) matches the bands' own, and it gives **+0.091 against
the two-branch +0.069** — same sign, factor 1.3 — while the single-branch answer is **−0.650**, wrong
sign and 7× too large. So the discrimination is decisive but the agreement is only ~30%, versus the
tighter pre-fix match.

The residual is explained, not mysterious: with `xi_bands = False` the code takes the legacy path, which
has **no polar patch**, so the `1/r` singularity is integrated on a plain Cartesian mesh that converges
only at `O(h)`. Axis F already measured that omission as a 22% shift in dE. A ~30% residual against a
reference carrying a known ~22% near-field error is consistent. **Conclusion: the brute force is a sanity
check on sign and order of magnitude, and cannot be a precision reference by construction.** The
deeper history makes it worse because there is more retarded-time structure for an `O(h)` scheme to miss.

**(2) History step size — I was wrong in §6g, and the error was a confound in my own harness.**

`run(stop_time=0.60)` stops after the first step that *reaches* 0.60, so **the final position depends on
`step_size`**:

```
   step (m)  stops at   sigma_z   sigma_x     slope
     0.1000  0.700000   515.4um   826.6um   -1.6034     <- the default OVERSHOOTS by 0.1 m
     0.0500  0.650000   473.4um   853.7um   -1.8032
     0.0250  0.600000   430.1um   878.8um   -2.0431
     0.0125  0.612500   441.1um   872.7um   -1.9786
     0.0080  0.600000   430.1um   878.8um   -2.0431
     0.0010  0.600000   430.1um   878.8um   -2.0431
```

The slope differs by **27%** between the 0.1 and 0.025 runs. My "28% history-step-size error" was
therefore mostly *comparing the wake at s = 0.70 against s = 0.60* — a different point in the lattice —
not a discretization error. Restricting to step sizes that all stop at exactly `s = 0.600000`:

```
   step (m)   stops at  snaps  dt/sig_z  dE vs finest
     0.0250   0.600000     23      58.1       0.00111
     0.0080   0.600000     70      18.6       0.00021
     0.0040   0.600000    140       9.3       0.00007
     0.0020   0.600000    276       4.7       0.00002
     0.0010   0.600000    551       2.3       0.00000
```

**The history step size is converged** — 0.11% at 23 snapshots, falling ~3–5× per 2–3× refinement, i.e.
roughly second order. The axis the original plan asked for is clean. The confounded version read
0.280 / 0.147 / 0.125 / 0.067 and would have sent me chasing a nonexistent bug.

Two things worth keeping from this:

- **`run(stop_time=T)` can overshoot by nearly a full step and does so silently.** At the default
  `step_size = 0.1` a request for 0.60 lands at 0.70 — 17% past. Any study that varies `step_size`,
  or that compares against a published number, must assert on `beam.position` afterwards.
- The alternative explanation was tested and rejected: varying `n_particle` 50 k → 800 k moves the wake
  by only **0.6–1.2%**, an order below the step-size signal, so this was never shot noise. (Note it does
  not scale as `1/√N` either — the beam is a Hammersley quasi-random sequence, so its "noise" is
  deterministic.)

**(3) Restated magnitudes.** §6f's numbers are **unchanged post-fix, bit for bit** — union coverage
1.00000 / 0.98018 / 0.99751 / 0.99994 and every `tan 2α`. That is expected rather than suspicious: union
coverage measures the support *footprint* and `tan 2α` the beam slope, and neither depends on how finely
the history is sampled in time. §6d's smoothness numbers shift slightly and every conclusion holds:

| §6d quantity | pre-fix | post-fix |
|---|---|---|
| abs roughness, single-branch (converged) | 0.13410 | 0.09079 |
| abs roughness, two-branch (converged) | 0.00819 | **0.00667** |
| ratio (two-branch smoother by) | 16× | **13.6×** |
| ‖w‖ single / two (cancellation factor) | 22× | 14.9× |
| 2nd-difference sign flips | 11 / 37 | **11 / 37** |
| degree-8 polynomial residual | 1.28% | 1.32% |
| `zbins` at which roughness reaches the floor | 400 | 400 |

§6c's headline numbers were already restated in §6h (missing-branch effect 1.041 → **1.065**, axis A
two-branch last doubling still **0.00000**).

**The wake maps, regenerated post-fix.** The high-tilt magnitudes drop ~9–11%, as predicted, and the
low-tilt cases are untouched — consistent with §6h, where the history only deepened at high tilt (3 → 6
and 3 → 8 snapshots, versus 3 → 3 at shear 0):

| shear | `dE/dct` pre-fix (MeV/m) | post-fix | change |
|---|---|---|---|
| 0 | [−2.4898, +0.8674] | [−2.4898, +0.8674] | none |
| 2 | [−6.0949, +2.2235] | [−6.0949, +2.2234] | none |
| 20 | [−0.0360, +0.1165] | **[−0.0304, +0.1032]** | −11% |
| 50 | [−0.0232, +0.0736] | **[−0.0204, +0.0667]** | −9% |

The qualitative structure is unchanged, so §6f's description still stands. A side benefit of the
dead-column parking fix (§6e) also shows up here: the auto-scaled integrand window at shear 20 shrank
from 45 967 µm to **8 091 µm**, because it is no longer being stretched by branch centres that had
wandered to tens of metres.

*A process note, since it nearly caused a false claim:* I read this log file while the regenerating job
was still mid-write, saw values bit-identical to pre-fix, and was about to report the wake as invariant.
It was stale. Bit-identical output after a change that should have moved it is a reason to check file
timestamps, not to conclude invariance — the script overwrites in place, so a stale read is
indistinguishable from a real result by content alone.

**Net effect of all three:** the Step 6 result is unchanged and better supported; one claimed bug (history
step size) turned out to be my own harness confound and is withdrawn; one new trap (`stop_time` overshoot)
is recorded.

### Step 7 — Remaining secondary fixes ⬜

Most of §2.3 was folded into `DF_tracker_comoving` in Step 5 — (c) registration, (e) normalization plus
the clipped-charge diagnostic, (f) snapshot-independent vx floor, (d) `poly_degree=1` with the fit
restricted to |z| < 3σ_z, and (g) the C² evaluation kernel with `math.floor` closing the `int()`
truncation hole. **On the co-moving path only.** Still outstanding:

- (b) `bilinear_single`'s hard-zero OOB and its `int()` truncation hole remain on the **`bspline_fft`
  and legacy paths**, which still use it. Fix or leave, but do not assume Step 5 touched them.
- `lattice.py:18–39` assumes `step_size` is the first YAML key (found in Step 2); look it up by name.
- `test_deposit_smooth.py::test_zero_field_gives_zero_derivative` **fails, and did so before this
  work.** It asserts a constant field has zero spectral derivative, but `smooth_and_differentiate`
  zero-pads before the FFT, so the array edge is a genuine step:
  `max|dfdx|` = 1.8e+01 untrimmed, 2.6e-02 trimming 5 cells, 8.8e-03 trimming 31. Boundary-only, and
  harmless for real densities which vanish at ±5σ — but the *test* asserts an invariant the padding
  breaks by design and should trim the boundary. The other 13 tests in that file pass.
- The older one-off diagnostics (`test_region1_nonzero.py`, `test_integrand_diagnostic.py`,
  `test_integrand_step17.py`, `test_integrand_anatomy.py`) unpack `get_CSR_wake(debug=True)`
  positionally and need `CSR_integration: {xi_bands: False}` since Step 4 returns a dict on that path.
- ~~`formation_length` computed only on element entry~~ — **fixed in §6h.**
- ~~The **history step size** axis has never been varied~~ — **done in §6i: converged**, 0.11% at 23
  snapshots, ~2nd order.
- **`run(stop_time=T)` overshoots by up to a full step, silently.** At the default `step_size = 0.1`,
  `run(stop_time=0.60)` lands at `s = 0.700000` — 17% past the request, with a 27% different beam slope.
  This confounded a whole convergence study in §6i before it was caught. Either stop at the requested
  `s` by splitting the final step, or warn. Until then, any script comparing runs at different
  `step_size` must assert on `beam.position`.
- **`compute_CSR: 0` silently produces a beam with no density history** (`CSR.py:311` gates
  `get_DF`/`append_DF`/`append_interpolant`/`build_interpolant` on `debug or compute_CSR`). Anything that
  calls `get_CSR_integrand` afterwards gets near-zeros with no error raised — see §6f, where it cost a
  full set of wrong figures. The name suggests a pure output switch and it is not. Either rename it,
  split history-building from wake-evaluation, or raise if `get_CSR_integrand` is called with fewer than
  two snapshots. The last is cheapest and would have caught this immediately.

### Step 8 — Validate and document ⬜

- Re-run Step 3 → nodes-in-support and the 100²-vs-400² gap are the primary acceptance metrics.
- Re-run Step 1 → co-moving must be exact for affine shear. ✅ **already met in Step 5**
- Axis A of `test_xi_bands_converge.py` on the co-moving path must read ~0 — currently 0.99, the
  outstanding correctness gap (Step 6).
- `test/test_tilt_sweep.py`, `test/benchmark_chirp_highres_sweep.py`: the error-vs-tilt curve should
  flatten instead of blowing up past |slope| ≈ 2–5.
- **Three-way convergence study** at one high-tilt case, varying independently: history step size,
  deposition bins, CSR integration bins. A converged answer must be insensitive to all three. Then
  compare the joint-limit result against legacy *at legacy's own resolution* — this shows whether
  legacy was the under-resolved one, and settles whether the remaining chicane-B2 disagreement is
  physics (cancellation regime) or numerics.
- Save all plots under `pyDFCSR_2D/test/benchmark_results/` and append a summary step to
  `progress.md`.

---

## 5. Verification suite

```bash
conda activate pydfcsr
cd .../pyDFCSR_claude/pyDFCSR

# 0a. Thesis Eq 4.24 verification (symbolic + numerical); no simulation for check 1
python pyDFCSR_2D/test/test_eq424_localization.py

# 0b. Eq 4.24 branches vs the measured integrand support, both B-spline methods
python pyDFCSR_2D/test/test_localization_branches.py

# 0c. Same, swept over beam shear 0..50 — regression test for the l>0 and |tau|=1 guards
python pyDFCSR_2D/test/test_localization_shear_sweep.py

# 0d. Step 6 acceptance: two-branch vs single-branch bands, same interpolant,
#     plus the rad<0 guard diagnostic binned by |r-r'|
python pyDFCSR_2D/test/test_two_branch_bands.py

# 1a. PRIMARY acceptance test: nodes-in-support + quadrature convergence
python pyDFCSR_2D/test/test_integrand_anatomy.py

# 1a2. Convergence with the axes properly separated (supersedes the margin sweep)
python pyDFCSR_2D/test/test_xi_bands_converge.py

# 1b. Interpolation accuracy vs ghosting parameter G (accuracy, not noise)
python pyDFCSR_2D/test/test_ghosting.py

# 1c. Noise-vs-G control: confirms noise does NOT track G
python pyDFCSR_2D/test/test_G_scaling.py

# 2. Unit tests for deposition/interpolation still pass
pytest pyDFCSR_2D/test/test_deposit_smooth.py -v
pytest pyDFCSR_2D/test/test_import.py -v

# 3. High-tilt end-to-end, CSR off (isolates deposition+integration from CSR dynamics)
python pyDFCSR_2D/test/benchmark_detailed.py   # single_dipole_tilted_config.yaml

# 4. Tilt sweep: error-vs-tilt curve should flatten
python pyDFCSR_2D/test/test_tilt_sweep.py

# 5. Full chicane, all three methods
mpirun -np 4 python pyDFCSR_2D/test/run_chicane_mpi_new.py

ruff check .
```

**Pass criteria**

The primary criteria are 1 and 2 — they target the noise. The rest target accuracy and hygiene.

1. **Nodes inside the density support ≥ ~20 per integration region** at 65× tilt amplification, and
   the hard-zero fraction well below 100%. — *today: 0.76 nodes, 99.98–100% hard zeros (Step 3).*
2. **~100² integration bins reproduce the 400² answer to a few percent.** — *today: 1518% off on
   dE/dct and 808% on `x_kick` (Step 3).* This is the direct test that the quadrature is sampling the
   integrand rather than aliasing it, and it should also let us return to 100² from the 300²–500²
   band-aid, recovering 9–25× in cost.
3. Wake-vs-tilt disagreement no longer grows with |slope| — no `xlim` / `smoothing_sigma` /
   integration-grid retuning per regime.
4. Wakes visually smooth at high tilt with the density still sharply resolved. Physical deposition
   cell size must stay at today's `bspline_fft` value (`Δξ = 2·xlim·σ_ξk/nx`) — confirm no resolution
   was traded away for the shared index space.
5. Converged under independent refinement of history step, deposition bins, and integration bins.
6. Co-moving interpolant error at round-off for pure affine shear at all G; current `bspline_fft`
   error demonstrably O(1) for G ≳ 1. — *second half met in Step 1; first half pending.* Accuracy,
   not noise.
