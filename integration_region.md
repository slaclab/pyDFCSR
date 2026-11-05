# How the CSR integration region is determined

Reference for the 2D CSR wake integral in `pyDFCSR_2D/CSR.py`, and how the beam tilt `τ` enters it.

- **Code version:** branch `csr1d_mode`, commit `98175a2` (2026-11-02). The integration code is unchanged
  since `integrand_cancellation` @ `136a48a`.
- **Line numbers:** they refer to that commit. Update this file when the integration code changes.

---

## 0. Which code path runs

`get_CSR_wake(s, x)` (CSR.py:1522) computes the wake at one observation point `(x, s)`. It has two
paths:

| Path | Runs when | Status |
|---|---|---|
| **xi_bands** (CSR.py:1609–1643) | `use_smooth_deposit and integration_params.xi_bands` | **Live.** `xi_bands = True` by default (params.py:21). `use_smooth_deposit` is True for `bspline_comoving` and `bspline_fft`. |
| legacy (CSR.py:1532–1590, 1629–1690) | `xi_bands = False`, or the `legacy` deposition | Dead in every current config. Kept byte-identical for `test_xi_bands_equiv.py`. |

**The legacy path:**
- it uses fixed transverse windows (±20σx far, ±10σx near), plus four chirp sub-regions when
  `|tan θ| > 1`;
- it splits s′ at `s2 = s3 − 200σz` and `s3 = s − 20σz`;
- the xi_bands path discards all of these.

Everything below describes the **live** path.

---

## 1. Structure of the integral

The integrand is integrated over `(x′, s′)`, transverse position and arc length of the source, at
the retarded time `t_ret = t − |r − r′|`. The live path does it in two steps:

1. **Longitudinal (s′):** split the range behind the observation point into a **far** and a **near**
   region, each with its own node layout. A **polar patch** around the singular point
   `(x′, s′) = (x, s)` is added on top.
2. **Transverse (x′):** on each s′ column, integrate only over one or two narrow **bands** where the
   retarded beam actually is. They are found per column from the light-cone geometry.

```
  total = far region  +  near region (tapered away from the pole)  +  polar patch (taper complement)
          _far_region_nodes   _near_region_nodes                    _integrate_near_patch
                      \             /
            each integrated by _integrate_xi_region: per-column x' bands from _retarded_xi_bands
```

Inner and outer integrals both use the trapezoid rule; `np.trapz(x=sp)` handles non-uniform s′ nodes.

---

## 2. Longitudinal extent: `_layout_bounds` (CSR.py:1276)

`τ` here is the **density frame's** tilt (`_frame_tilt`, CSR.py:1368): the core-fit slope `poly_coeffs[0]`
of the comoving tracker. It is not `beam.slope`, which fits all particles; the two differ most where
the x–z correlation vanishes. Define `α = arctan τ`, the beam's angle in x–z.

```
sin 2α = 2τ/(1 + τ²)          cos 2α = (1 − τ²)/(1 + τ²)          # bounded, no pole

d  = (10 σx + x − x̄) · |cos 2α| / max(|sin 2α|, branch_sin_min)
d  = max(d, near_floor · σz)
d  = min(d, n_formation_length · L_f)

s4 = s + 3 σz
s3 = max(0, s − d)
s1 = max(0, s3 − n_formation_length · L_f)

far  region : (s1, s3)
near region : (s3, s4)
```

| Term | Meaning |
|---|---|
| `10σx + x − x̄` | The chirp branch `x′ = x − (s − s′)·tan 2α` starts at the observer and leaves the beam (10σx from the centre) after `d = (10σx + x − x̄)/\|tan 2α\|`. Written with sin/cos so there is no pole at \|τ\| = 1. The `x − x̄` term is kept on purpose: d varies ~1.9× across the wake mesh. |
| `branch_sin_min = 0.05` | Saturates d where the two branches merge (\|sin 2α\| → 0). This happens at **both** \|τ\| ≪ 1 and \|τ\| ≫ 1. Threshold set by geometry: branches unresolvable once `\|tan 2α\|·L < 2·margin·xlim·σξ`. |
| `near_floor = 20` | Lower bound on d in σz. At \|τ\| → 1, cos 2α → 0 and d would collapse; the near region must still contain the narrow band and the 1/\|r − r′\| pole. Measured motivation: shear 50, τ = −0.999, reach had fallen to 15 µm against σz = 1744 µm. |
| `n_formation_length · L_f` | d and the far reach are capped by the retained history. `L_f = (24 R² · 5σz)^(1/3)`, refreshed every step from the current σz (`_refresh_formation_length`). In a drift, R is the last bend's. |
| `s + 3σz` | Upstream end of the near region: sources ahead of the observer. |
| `s′ ≥ 0` clamp | Before the lattice start, `interpolate1D` returns 0 silently. |

**Near-region length vs τ** (σx = 1 mm, as in the chicane's D3 drift):

| \|τ\| | 0.1 | 0.5 | 1 | 2 | 10 | 20 | 40 | 80 | 100 |
|---|---|---|---|---|---|---|---|---|---|
| 2α [deg] | 11 | 53 | 90 | 127 | 169 | 174 | 177 | 179 | 179 |
| geometric `10σx/\|tan 2α\|` [mm] | 50 | 7.5 | 0 | 7.5 | 50 | 100 | 200 | 400 | 500 |
| **d used** [mm] | 50 | 7.5 | 0.4 (floor) | 7.5 | 50 | 100 | 200 | **200 (sat.)** | **200 (sat.)** |

For |τ| ≳ 40 the near region stops growing at about 200σx. **Beyond that, the part of the chirp
branch that is still inside the beam lies in the far region** (see §8).

---

## 3. Far-region s′ nodes: `_far_region_nodes` (CSR.py:1191)

Two layouts, chosen by `_exit_transient()` (CSR.py:1249) = `afterbend and not inbend`.

**(a) Inside a bend, or before any bend:** uniform, `2·far_zbins` = 400 nodes over `(s1, s3)`.
- **Why not graded:** grading assumes a 1/u decay. At high tilt in a bend that fails: at shear 20,
  10–99% of the far integral sits at u/L_f = 0.12–0.56, at the far end.
- **Why no causal clip:** in a bend the causal edge sits at u ≈ 1.0 L_f, which `n_formation_length`
  already provides.

**(b) In a drift after a bend (exit transient), with `causal_clip = True`:**
1. **Causal edge** (`_causal_edge`, CSR.py:1153): a 64-point scan over `(s1, s3)`, using
   `_band_alive` = any band with nonzero width (§5). The far region starts at the last dead node
   below the first live one. A scan, not a root-find, because the alive mask is not monotone. If the
   edge falls at or below `s1`, the history reach is the binding limit, counted in
   `_far_clip_at_reach`.
2. **Fine window:** uniform cells of `far_cell · σz` (10σz) over `far_window · L_f` (1.0 L_f) above the
   edge, between 3 and `100·far_zbins` nodes. Rationale: past an exit the source is the arc still
   inside the dipole, a strip about 0.3 L_f wide at u/L_f ≈ 2.7–3.1, just above the causal edge.
3. **Above the window:** log-graded toward `s3` with ratio `1 + near_grade`. If `near_grade = 0`,
   `far_zbins` uniform nodes instead.

---

## 4. Near-region s′ nodes and the polar patch

**Near nodes** (`_near_region_nodes`, CSR.py:1426). Graded on each side of `s′ = s`, with `u = |s − s′|`:

```
du = max(near_cell · σξ, near_grade · u)          # near_cell = 0.5, near_grade = 0.05
```

- **Fine end:** uniform cells of `near_cell·σξ` out to `u* = floor/near_grade`.
- **Beyond u\*:** geometric growth.
- **Why:** once `|r − r′| ≈ u`, the per-column contribution falls as 1/u, so equal contributions
  come from equal log intervals.
- **Node count:** grows only as `ln(u_max/u*)` with tilt.
- **σξ:** `beam._sigma_x_transform` = rms of x after removing the linear x–z fit.
- **Defaults:** `near_cell = 0` or `near_grade = 0` give `np.linspace` with `_region_node_counts`
  nodes, capped at `100·zbins`, with a one-time warning if the cap binds.

**Polar patch** (`_near_patch_radii`, CSR.py:1471; `_integrate_near_patch`, CSR.py:1488):
- **Radii:** `R2 = min(near_patch · σξ, 0.4(s4 − s), 0.4(s − s3))` and `R1 = R2/2`, with
  `near_patch = 5`. None if `near_patch = 0` or `R2 ≤ 0`.
- **Mesh:** polar, centred on `(x′ = x, s′ = s)`, `near_patch_nr = 100` radial midpoints by
  `near_patch_nphi = 180` angles. The area element `r dr dφ` cancels the `1/|r − r′|` singularity.
- **Taper:** a C² smootherstep `w(|r − r′|)` rising 0 → 1 across `[R1, R2]` (CSR.py:1965). The near
  region's Cartesian mesh carries `w`, the patch carries `1 − w`, so the pieces sum to the untapered
  integral exactly. The far region gets no taper.

---

## 5. Transverse extent: the x′ bands, `_retarded_xi_bands` (CSR.py:983)

Applies only with the comoving tracker and `poly_degree = 1`, since Eq 4.24 assumes a straight beam
axis. Otherwise it falls back to the single-band locator `_retarded_xi_band` (CSR.py:742), which iterates
3 times from `x′ = x` and covers the deposition grid's span at the retarded time.

**For each s′ column, both roots (sign = ±1) of the corrected thesis Eq 4.24** (`_eq424`, CSR.py:921).
The beam axis at the retarded time is `x′ = τ_ret·z′ + b`:

```
q = r0(s) − r0(s′) + x n(s),   nq = n(s′)·q,   q2 = |q|²,   T = t − s′,   Tb = T − b/τ_ret
rad = (τ_ret² − 1)(Tb² − q2) + (nq τ_ret + Tb)²
x′  = τ_ret [ Tb + nq τ_ret ± sqrt(rad) ] / (τ_ret² − 1)
```

1. **Fixed point:** start at `x′ = x` and do 12 iterations. Each iteration recomputes `t_ret` from the
   current `x′`, takes the blended frame at `t_ret` (§6), and re-solves Eq 4.24.
2. **Guards in `_eq424`** (each was needed, per the code comments):
   - **\|τ\| < 1e-9:** the axis is horizontal; single root `x′ = b`.
   - **\|\|τ\| − 1\| < 1e-9:** `τ² − 1 → 0`; the quadratic becomes linear, `x′ = −(q2 − T²)/(−2T/τ − 2nq)`.
   - **`rad < 0`:** the light cone misses the beam *axis* at this s′. With `tangent_bands: true`
     (the default since 2026-11-04) the band goes to the column's closest approach to the axis. With
     `false`, the old behaviour, the branch is treated as absent and the column dropped, which is
     wrong near the observer (§5a).
   - **`l = Tb + x′/τ ≤ 0`:** spurious root from squaring (source in the future); rejected.
3. **Band:** centred on the retarded beam axis at the converged root. The half-width is
   `0.5·(u_end − u_start)·σξ_ret·xi_band_margin = xlim·σξ_ret·margin = 5 × 2 × σξ_ret = 10σξ_ret`.
   `xlim` is the deposition grid half-width (5) and `xi_band_margin = 2`.
4. **Dead branch:** any guard failed, or the retarded z falls outside the deposition grid
   (`w = (z_ret − z̄)/σz_ret ∉ [−zlim, zlim)`). It gets zero width, parked at `x + 4·half` so it can
   never hit the pole.
5. **Overlap** (`_disjoint_bands`, CSR.py:1072): when the two bands overlap, subtract rather than
   union, so the width and node density never change. One band becomes `[hi1, hi2]`, and a band fully
   inside the other collapses to zero width. **The returned bands are reordered by position per
   column**, so band j is not branch j.

**Integration** (`_integrate_xi_region`, CSR.py:1104):
- each live band gets `xbins` uniform nodes;
- the inner trapezoid is scaled by that column's spacing `width/(xbins − 1)`;
- zero-width columns are selected out, not multiplied by zero, so a node on the pole cannot produce
  NaN.

**Why both roots are always taken:** whether a column has one or two branches depends on the tilt at
that column's **retarded** time. The retarded tilt can sweep +19.4 → −0.22 → −19.8 across one near region.
Deciding from the observation-point tilt discarded 28–31% of the wake (Step 6).

---

## 5a. Dropped columns: the light cone misses the beam centre line but still passes through the beam (`tangent_bands`)

Found 2026-11-03 while diagnosing the chicane's D3 wake. Option `CSR_integration.tangent_bands`,
default `True` since 2026-11-04. Figure: [tangent_bands_explained.png](pyDFCSR_2D/test/benchmark_results/d3_diagnostic/tangent_bands_explained.png)
(script `test/plot_tangent_bands.py`).

![tangent bands](pyDFCSR_2D/test/benchmark_results/d3_diagnostic/tangent_bands_explained.png)

### What the band locator assumes

The source integral is done column by column. For a fixed s′, the code sweeps x′. It needs to know
where on that column the source point lies inside the beam *at its retarded time*, because the
integrand is zero everywhere else. It puts a ±10σξ band there and integrates over the band only
(§5).

To find "where", Eq 4.24 replaces the beam by its **centre line** ξ = 0, a zero-width line, and solves
for where the retarded light cone crosses that line. One root gives one band, two roots give two. **If
there is no root (`rad < 0`), the column is marked "branch absent" and skipped entirely.** The ±10σξ
band width does not help, because no band is ever created.

### What a column actually sees

Take a drift, an observer at offset ξ_obs from the beam axis, and a column a distance D = s − s′
behind it. Parametrise the column by y = x′ − x. Moving along the column changes |r − r′|, which
changes the retarded time, which changes the bunch coordinate z′ at which the source is sampled. On
a tilted beam the axis moves by τ·Δz′ in x for a shift Δz′. For |y| ≪ D the retarded offset of the
source point from the axis is

```
ξ_ret(y) = ξ_obs + y − τ·y²/(2D)
```

This is a parabola in y (panel a). At y = 0 (x′ = x) it gives ξ_ret = ξ_obs: the column passes
through the observer's own position in the beam, which is always inside the beam.

The centre-line crossings are the roots of ξ_ret = 0. They exist only when 1 + 2τ·ξ_obs/D ≥ 0. So
**when τ·ξ_obs < 0 and the column is close, D < 2|τ·ξ_obs|**, the parabola's extremum stays on the
observer's side and never reaches ξ = 0. Eq 4.24 has no root and the column is dropped. But the
extremum is at ξ_ret = ξ_obs + D/(2τ), between the axis and the observer, i.e. **in the beam core**.
The column crosses dense charge; it just never touches the zero-width line exactly.

### A concrete case: D3, s = 12.30 m

Here τ = −6.9 and σξ = 34.7 µm. Take the row 6 observer at ξ_obs = +0.37σξ = 12.8 µm and a column at
D = 100 µm:

| position on the column y [µm] | −30 | −14.5 (extremum) | 0 (observer) | +30 |
|---|---|---|---|---|
| ξ_ret [µm] | 13.9 | **5.6** | 12.8 | 73.9 |
| ξ_ret / σξ | 0.40 | **0.16** | 0.37 | 2.1 |

The column comes within 0.16σξ of the centre line, about the densest place in the beam, but never
reaches 0, so the default code drops it. The dropped range is predicted to be D < 2|τ|·ξ_obs (panel b):

| observer | predicted dropped range | measured (band diagnostics) |
|---|---|---|
| row 6, ξ_obs = +0.37σξ | D < 177 µm | dead at 139 µm, alive at 243 µm |
| row 7, ξ_obs = +1.1σξ | D < 527 µm | dead at 442 µm, alive at 593 µm |
| row 5, ξ_obs = −0.36σξ | none | every column alive |

The measured cut-offs fall between the near-region grid's columns, consistent with the prediction.

### Why the wake jumped across the axis

- **Only one side is hit.** It happens only for τ·ξ_obs < 0. On the other side the parabola always
  crosses the axis twice and the column always gets its bands. Rows 5 and 6 straddle ξ = 0, so the
  error switches on between them (panel c).
- **The closest columns are the ones lost.** At small D, the self-field (v_x) part of the integrand,
  which goes as 1/|r − r′|, is largest. In a drift the v_x-free part of the near integrand is
  identically zero (parallel motion, τ(s) = τ(s′)), which is why switching v_x off hid the defect.
- **The lost range grows with |τ·ξ_obs|.** It is worst for outer rows and at high tilt: in B3 at
  τ = −91 it reaches tens of mm. In a bend τ(s) ≠ τ(s′), so the near integrand is nonzero even
  without v_x, and the defect shows there too.

### The fix (`tangent_bands: true`)

A column with no root gets its band at the **closest approach to the axis**, the extremum of ξ_ret
over x′, and the usual ±10σξ band then covers the beam. The extremum is where d|r − r′|/dx′ = 1/τ,
which gives exactly (not to parabola accuracy)

```
x′ = nq + sign(τ)·sqrt((q2 − nq²)/(τ² − 1)),      |τ| > 1
```

(`_eq424(..., tangent=True)`). The vertex of the squared quadratic agrees with this only to leading
order in 1/τ; `test_tangent_bands.py` checks that the returned point is a true extremum. If the light
cone really does miss the beam, the column is still discarded by the existing z_ret-off-grid check.
Columns that already had a root are untouched, so the result is byte-identical to the default
wherever no column was dropped.

### Verification against a band-free brute force (`d3_wake_diagnostic.py coltruth`)

Each column is also integrated over a fine x′ grid with 40,001 nodes spanning the whole lab-frame
ribbon (±6σx + 6σξ), with no bands, the same integrand and the same taper. Near region only, in MeV/m:

| point | default | `tangent_bands` | brute force |
|---|---|---|---|
| B3 7.40 m, row 11 | −2.369 | −0.511 | −0.511 |
| B3 7.60 m, row 11 | −5.607 | +0.806 | +0.805 |
| B3 7.60 m, row 6 | −5.155 | −5.475 | −5.475 |
| D1 3.02 m, row 11 | −0.043 | +0.051 | +0.051 |
| D3 8.57 m, row 9 | +0.186 | −0.135 | −0.135 |
| D3 12.35 m, row 7 | +0.830 | −0.185 | −0.175 |

With the fix, the D3 v_x contribution matches an independent Coulomb-work estimate
2K∫ρ(v_x − px′)/(x − x′)dx′ computed from the particles: correlation 0.99, scale 0.98, at 10.06 and
12.30 m. The x-spread of the D3 map falls from 0.224 to 0.035 MeV/m at 12.30 m.

**Status:** **default on since 2026-11-04**, after the chicane, single-dipole and FODO W2 validation
(`progress.md`, "Validation of `tangent_bands` and `vx_model: affine`"). Results recorded before
that date were computed with it off; set `tangent_bands: false` to reproduce them.

---

## 6. Where τ enters, in full

1. **Near-region length** — `sin 2α`, `cos 2α` from the observation-time frame tilt, with
   `branch_sin_min` and `near_floor` (§2).
2. **Band position and existence** — Eq 4.24 with the **retarded** tilt `τ_ret`, solved per column with
   the guards above (§5).
3. **Frame at the retarded time** (`_comoving_frame_at`, CSR.py:870, mirroring the interpolant
   exactly). `frame_blend` (default `'orient'`) sets how the frame is blended between the two
   snapshots `k, k+1` around `t_ret`:
   - **`'coeff'`:** `τ` blended linearly. Wrong across a longitudinal waist, where the orientation goes
     +87° → 90° → −87° and a linear path in τ passes through horizontal (§6q).
   - **`'moment'`:** `var_z`, `cov`, `var_x` blended linearly, with `τ = cov/var_z`. Correct orientation, but
     σξ comes out 8–10× too wide mid-interval at high tilt.
   - **`'orient'` (default):** `τ = cov/var_z` from the blended moments, with σξ and σz blended
     log-linearly.
4. **Step schedule** (`schedule.py:516`, `step_control: auto`) — limits how far the band centre can move
   per step:

   ```
   h_τ = tau_frac · σξ / (|dτ/ds| · σz),    tau_frac = 2
   1/h_eff includes 1/h_τ
   ```

   Measured insensitive over 32× (§11l) and 8× in D1 (§11r).

---

## 7. Defaults (`Integration_params`, params.py:15; values in `configure_params`, params.py:20)

| Parameter | Default | Used in |
|---|---|---|
| `xi_bands` | True | selects the live path |
| `xbins` | 200 | x′ nodes per band (chicane runs: 200) |
| `zbins` | 200 | legacy path; cap `100·zbins` on uniform near nodes |
| `n_formation_length` | 4 (chicane runs: 1) | far reach, and the cap on d |
| `far_zbins` | 200 | far nodes (`2×` in a bend) |
| `near_cell` | 0.5 σξ | finest near cell |
| `near_grade` | 0.05 | geometric growth of near and far cells |
| `branch_sin_min` | 0.05 | d saturation at merged branches |
| `near_floor` | 20 σz | minimum near reach |
| `xi_band_margin` | 2.0 | band half-width multiplier |
| `near_patch` | 5.0 σξ | polar patch radius R2 |
| `near_patch_nr`, `near_patch_nphi` | 100, 180 | polar mesh |
| `causal_clip` | True | far start at the causal edge, exit transient only |
| `far_window` | 1.0 L_f | fine far window |
| `far_cell` | 10 σz | fine far cell |
| `far_fine_max` | 500 | cap on fine-window nodes; raises `far_cell` only where it is excessive (since 2026-11-04) |
| `tangent_bands` | True | band at the closest approach where Eq 4.24 has no root (§5a; since 2026-11-04) |
| `drift_reach` | True | in a drift after a bend, far edge back to that bend's entrance (since 2026-11-04) |
| `drift_snap_thin` | 0.01 | drop drift snapshots that interpolation reproduces (since 2026-11-04) |
| `drift_cutoff` | 0 (was 1.0 \|R·φ\| before 2026-11-04) | no wake beyond this distance past an exit; 0 = drifts integrated in full |

Deposition (`particle_deposition`): `xlim = zlim = 5` sets the grid extent and the band half-width;
`frame_blend = 'orient'`; `poly_degree = 1` is required for `_retarded_xi_bands`; `vx_model = 'affine'`
(since 2026-11-04).

---

## 8. Known limits (as of 2026-11-03)

- **|τ| ≫ 1: branches parallel, not coincident.** As `tan 2α → 0` the two Eq 4.24 branches become
  parallel. The chirp branch stays inside the beam over a distance that grows like `σx/|tan 2α|`, but
  d saturates at about 200σx (|τ| ≳ 40), so the rest of it falls into the far region. The far layout
  was designed for a narrow strip far upstream (bend exit) or a uniform spread (in a bend), not for a
  live branch running along the whole drift. §6m recorded that no affordable 1D grid converged in this
  regime: 32,191 uniform vs 780 graded nodes disagreed by 22%.
- **Pointwise cancellation (Step 12).**
  - **Size:** the integrand is linear in `∂ρ/∂x`, whose ± lobes cancel. The cancellation factor
    `Σ|parts|/|Σparts|` is 60 in a bend, 640 past an exit, and 1113 at shear 20, almost entirely
    between adjacent nodes.
  - **Consequence:** noise in `∂ρ/∂x` reaches the wake with ~100× the gain of noise in `ρ`, and
    correlated noise is 5.7× worse than white.
  - **Proposed fix:** a control variate, computing the fitted-Gaussian part of `∂ρ/∂x` analytically.
- **Drift wakes: diagnosed 2026-11-03** (`test/d3_wake_diagnostic.py`; results in
  `test/benchmark_results/d3_diagnostic/`, written up in `progress.md`). The chicane's D3 far-drift
  failure is **two separate defects**, neither in the quadrature:
  1. **The reach does not get back to the bend.** In a drift, `L_f` stays the in-bend formation length
     of the last bend, `(24 R²·5σz)^(1/3)` (`_refresh_formation_length`), so the integral reaches only
     `d + n_formation_length·L_f` behind the observer. Both the s′ range (`_layout_bounds`) and the
     retained history are cut there. But the post-bend source is the bend itself, which recedes. With
     the chicane's `n_formation_length = 1`, B3 leaves the reach about 1.7 R·φ past its exit, and the
     wake then has the wrong sign (energy gain). Covering the bend (n_fl 6 at s = 8.57 and 10.06 m)
     flips the x-averaged core from +0.03 to −0.26 and −0.08 MeV/m (1D: −0.35, −0.15). n_fl 6 and 8 are
     identical to every digit.
  2. **The transverse-velocity terms produce a growing x-structure.** With `ignore_vx`, the D3 wake
     becomes x-independent: the x-spread falls 0.043 → 0.010, 0.063 → 0.0003 and 0.224 → 0.0000 MeV/m at
     s = 8.57, 10.06, 12.30 m. The core then sits 15–25% below 1D at every position. The v_x term grows
     with the beam's convergence, ⟨xx′⟩/⟨x²⟩ = −0.24 → −2.6 /m toward B4, which is 10–50× what the FODO
     W2 test validated (±0.05 /m).
     **Resolved 2026-11-03: mostly physics plus a band-locator defect, not deposition noise.**
     - **The physics:** the v_x terms carry the self-field work on a converging beam, e·v_x·E_x. A
       direct Coulomb estimate reproduces them (corr 0.99, scale 0.98).
     - **The defect, which made the x-structure:** `_eq424` returned nan wherever the retarded light
       cone misses the beam *axis* (rad < 0), so `_retarded_xi_bands` dropped that column. For an
       observer with τ·ξ_obs < 0, the near columns with s − s′ ≲ 2|τ ξ_obs| miss the axis but still
       pass through the beam, and all of them were lost.
     - **The fix, opt-in (§5a):** `CSR_integration.tangent_bands: true` puts the band at the column's
       closest approach, x′ = nq + sign(τ)√((q² − nq²)/(τ² − 1)). The near region then matches a
       brute-force x′ sum to < 0.1% at five test points and to 6% at the sixth.
     - **Also in B3:** outer-row wakes change by up to 6.4 MeV/m.
     - **Separate, small effect:** a ∂v_x/∂x smoothing bias on banana-shaped beams. It has an opt-in
       fix, `particle_deposition.vx_model: affine`, which changes the D3 maps by only a few percent.
       It is a deposition change, not an integration-region change: it fits
       px ≈ c₀ + b_u·u + cubic(w) on the core, deposits only the residual, and adds the fit back
       analytically (∂v_x/∂u += b_u exactly). Full explanation and numbers in `progress.md`,
       "How the v_x fix works".
  - **The integration itself is converged there:** at s = 10.058 m, `near_grade`, `near_cell`, `xbins`,
    `xi_band_margin`, `branch_sin_min`, `near_patch`, `far_cell` and `far_window` move the map by
    ≤ 18% and the x-average by ≤ 4%, and none changes the sign or the x-pattern.
  - **That region still matters:** 1D shows it carries 37% of the energy loss and 58% of the emittance
    growth (`progress.md`, top TODO). `drift_cutoff = 1.0` remains the safe default until both defects
    are fixed.
  - **Tail roughness in D1/D3 (§11r, §11v):**
    - **Not caused by:** a noise floor; the bunch core; `tau_frac`; snapshot density.
    - **Possibly** a roughness-metric artefact (it correlates with amplitude).
  - **s = 8.317 m:** a separate defect, located in the bunch core.
- **Dropped columns where Eq 4.24 has no root:** fixed by `tangent_bands`, **default on since
  2026-11-04** (full explanation in §5a). The defect reached into bends: 2D chicane results recorded
  before that date, including the chirp-scan tables, are superseded.
- **The drift reach (the other half of the D3 failure) — fixed, ON by default since 2026-11-04 (`drift_reach`, `drift_snap_thin` 0.01, `far_fine_max` 500), and `drift_cutoff` now defaults to 0:**
  - **`drift_reach: true`:** in a drift after a bend, `_layout_bounds` puts s1 at the last bend's entrance, `_far_region_nodes`'s causal-edge scan trims it, and `_retention_floor` keeps the history back to it until the next bend.
  - **`drift_snap_thin`:** drops drift snapshots that linear interpolation reproduces (`DF_tracker_comoving.thin_previous`).
  - **`far_fine_max`:** caps the far fine-window node count, which was the actual cost, ~3,300 nodes per D3 kick. Cap 500 holds the default's accuracy everywhere tested.
  - **Results:** with these, the chicane runs without `drift_cutoff` and D3 is a clean energy loss with the 1D z-profile (`progress.md`, "The D3 reach fix").
  - **Still open:** (i) on entering the NEXT bend the reach drops the previous bend's tail (first B4 kick); (ii) a drift only keeps its last bend, so older bends (B2 seen from D3) are excluded.
- **Never tested in the drift:** deposition resolution (the deposition sweep was started and stopped
  when the reach finding came up).

---

## 9. Source references

- **Code:**
  - `pyDFCSR_2D/CSR.py`: `get_CSR_wake` 1522, `_layout_bounds` 1276, `_frame_tilt` 1368,
    `_far_region_nodes` 1191, `_causal_edge` 1153, `_near_region_nodes` 1426, `_near_patch_radii` 1471,
    `_integrate_near_patch` 1488, `_retarded_xi_bands` 983, `_eq424` 921, `_disjoint_bands` 1072,
    `_integrate_xi_region` 1104, `_comoving_frame_at` 870, taper 1981.
  - `pyDFCSR_2D/params.py`: `Integration_params` 15, with the measurement-backed comments for each
    parameter.
  - `pyDFCSR_2D/schedule.py`: `h_τ` 516.
- **Design history** (`progress_comoving_interp.md`):
  - §2.0–3.0: band sizing by σξ;
  - Step 4e/4f/5c/5d: Eq 4.24, both branches;
  - Step 6: per-column branches;
  - §6m: graded near region and the |τ| ≫ 1 degeneracy;
  - §6n/§6q/§6r/§6u: frame blend;
  - §11s: live vs legacy path;
  - §11u: the far-region rebuild;
  - §11r/§11v: drift roughness;
  - Step 12: cancellation.
- **Physics:** thesis §4.4.2 (integrand localization, Eq 4.24); Stupakov PRAB 25, 014401 Sec. IV (the
  near-field split).
