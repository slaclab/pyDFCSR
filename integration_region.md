# How the CSR integration region is determined

Reference for the 2D CSR wake integral in `pyDFCSR_2D/CSR.py`, and how the beam tilt `τ` enters it.

- **Code version:** branch `csr1d_mode`, commit `98175a2` (2026-11-02). The integration code is unchanged
  since `integrand_cancellation` @ `136a48a`.
- **Line numbers:** they refer to that commit. Update this file when the integration code changes.

---

## 0. Which code path runs

`get_CSR_wake(s, x)` (CSR.py:1506) computes the wake at one observation point `(x, s)`. It has two
paths:

| Path | Runs when | Status |
|---|---|---|
| **xi_bands** (CSR.py:1593–1627) | `use_smooth_deposit and integration_params.xi_bands` | **Live.** `xi_bands = True` by default (params.py:21). `use_smooth_deposit` is True for `bspline_comoving` and `bspline_fft`. |
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

## 2. Longitudinal extent: `_layout_bounds` (CSR.py:1260)

`τ` here is the **density frame's** tilt (`_frame_tilt`, CSR.py:1352): the core-fit slope `poly_coeffs[0]`
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

## 3. Far-region s′ nodes: `_far_region_nodes` (CSR.py:1175)

Two layouts, chosen by `_exit_transient()` (CSR.py:1249) = `afterbend and not inbend`.

**(a) Inside a bend, or before any bend:** uniform, `2·far_zbins` = 400 nodes over `(s1, s3)`.
- **Why not graded:** grading assumes a 1/u decay. At high tilt in a bend that fails: at shear 20,
  10–99% of the far integral sits at u/L_f = 0.12–0.56, at the far end.
- **Why no causal clip:** in a bend the causal edge sits at u ≈ 1.0 L_f, which `n_formation_length`
  already provides.

**(b) In a drift after a bend (exit transient), with `causal_clip = True`:**
1. **Causal edge** (`_causal_edge`, CSR.py:1137): a 64-point scan over `(s1, s3)`, using
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

**Near nodes** (`_near_region_nodes`, CSR.py:1410). Graded on each side of `s′ = s`, with `u = |s − s′|`:

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

**Polar patch** (`_near_patch_radii`, CSR.py:1455; `_integrate_near_patch`, CSR.py:1472):
- **Radii:** `R2 = min(near_patch · σξ, 0.4(s4 − s), 0.4(s − s3))` and `R1 = R2/2`, with
  `near_patch = 5`. None if `near_patch = 0` or `R2 ≤ 0`.
- **Mesh:** polar, centred on `(x′ = x, s′ = s)`, `near_patch_nr = 100` radial midpoints by
  `near_patch_nphi = 180` angles. The area element `r dr dφ` cancels the `1/|r − r′|` singularity.
- **Taper:** a C² smootherstep `w(|r − r′|)` rising 0 → 1 across `[R1, R2]` (CSR.py:1965). The near
  region's Cartesian mesh carries `w`, the patch carries `1 − w`, so the pieces sum to the untapered
  integral exactly. The far region gets no taper.

---

## 5. Transverse extent: the x′ bands, `_retarded_xi_bands` (CSR.py:968)

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
   - **`rad < 0`:** the light cone misses the beam axis at this s′; the branch is absent.
   - **`l = Tb + x′/τ ≤ 0`:** spurious root from squaring (source in the future); rejected.
3. **Band:** centred on the retarded beam axis at the converged root. The half-width is
   `0.5·(u_end − u_start)·σξ_ret·xi_band_margin = xlim·σξ_ret·margin = 5 × 2 × σξ_ret = 10σξ_ret`.
   `xlim` is the deposition grid half-width (5) and `xi_band_margin = 2`.
4. **Dead branch:** any guard failed, or the retarded z falls outside the deposition grid
   (`w = (z_ret − z̄)/σz_ret ∉ [−zlim, zlim)`). It gets zero width, parked at `x + 4·half` so it can
   never hit the pole.
5. **Overlap** (`_disjoint_bands`, CSR.py:1056): when the two bands overlap, subtract rather than
   union, so the width and node density never change. One band becomes `[hi1, hi2]`, and a band fully
   inside the other collapses to zero width. **The returned bands are reordered by position per
   column**, so band j is not branch j.

**Integration** (`_integrate_xi_region`, CSR.py:1088):
- each live band gets `xbins` uniform nodes;
- the inner trapezoid is scaled by that column's spacing `width/(xbins − 1)`;
- zero-width columns are selected out, not multiplied by zero, so a node on the pole cannot produce
  NaN.

**Why both roots are always taken:** whether a column has one or two branches depends on the tilt at
that column's **retarded** time. The retarded tilt can sweep +19.4 → −0.22 → −19.8 across one near region.
Deciding from the observation-point tilt discarded 28–31% of the wake (Step 6).

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
| `drift_cutoff` | 1.0 \|R·φ\| | no wake beyond this distance past an exit (not part of the region, but it decides whether the wake is computed at all) |

Deposition (`particle_deposition`): `xlim = zlim = 5` sets the grid extent and the band half-width;
`frame_blend = 'orient'`; `poly_degree = 1` is required for `_retarded_xi_bands`.

---

## 8. Known limits (as of 2026-11-02)

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
- **Drift wakes.**
  - **Failure past ~1.9 R·φ:** the 2D wake in the chicane's D3 turns bipolar and grows toward B4,
    while 1D decays. This is the known failure mode `drift_cutoff = 1.0` exists for.
  - **That region still matters:** 1D shows it carries 37% of the energy loss and 58% of the emittance
    growth (`progress.md`, top TODO).
  - **Tail roughness in D1/D3 (§11r, §11v):**
    - **Not caused by:** a noise floor; the bunch core; `tau_frac`; snapshot density.
    - **Possibly** a roughness-metric artefact (it correlates with amplitude).
  - **s = 8.317 m:** a separate defect, located in the bunch core.
- **Never tested in the drift:** deposition resolution, `CSR_integration` bins, and the band geometry
  at extreme tilt.

---

## 9. Source references

- **Code:**
  - `pyDFCSR_2D/CSR.py`: `get_CSR_wake` 1506, `_layout_bounds` 1260, `_frame_tilt` 1352,
    `_far_region_nodes` 1175, `_causal_edge` 1137, `_near_region_nodes` 1410, `_near_patch_radii` 1455,
    `_integrate_near_patch` 1472, `_retarded_xi_bands` 968, `_eq424` 921, `_disjoint_bands` 1056,
    `_integrate_xi_region` 1088, `_comoving_frame_at` 870, taper 1965.
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
