# pyDFCSR 2D: how the code works

*Methods document for the 2D density-function CSR code in `pyDFCSR_2D/`, written against the code on branch `near_adapt` (commit ea4f41f, 2026-11-23) and the development logs `progress.md`, `progress_facet_bc20.md`, `progress_cancellation.md`, `progress_comoving_interp.md` and `integration_region.md`. Every default quoted below was read from `params.py` or `deposit_smooth.py`; every benchmark number carries the section of the log it comes from. Where something could not be verified it is marked as such.*

**Contents**

1. Overview and physical model
2. The simulation loop
3. Density estimation
4. The wake mesh
5. The integration
6. Kick application and the step/kick schedule
7. The 1D model used for the pre-pass and the diagnostics
8. Parallelisation
9. Benchmarks and run times
10. Known limitations and open items
11. Parameter reference

---

## 1. Overview and physical model

### 1.1 What the code computes

pyDFCSR tracks a bunch of macro-particles through a lattice of dipoles, drifts, quadrupoles and sextupoles and, at chosen positions along the way, computes the coherent synchrotron radiation (CSR) field that the bunch exerts on itself and applies it as an energy kick and a transverse momentum kick. The field is computed not from particle-to-particle Liénard–Wiechert sums but from the bunch's *density function* ρ(x, z, t) and its velocity field v_x(x, z, t), stored over the recent history, so the cost of a field evaluation does not grow with the number of particles. The method is the one of Stupakov and Tang (PRAB 24, 020701, 2021) for the longitudinal force and of Stupakov (PRAB 25, 014401, 2022) for the transverse force, implemented in the 2D (x, z) plane of the bend. Chapter 4 of the thesis in `reference/` describes the original algorithm; this document describes the code as it is now, which differs from the thesis in most of its numerics.

### 1.2 Coordinates

The beam is described in the Frenet–Serret frame of the reference orbit: s is the arc length, x the offset along the normal n(s), z = s − ct the longitudinal position inside the bunch (head at +z). The lab-frame position of a source point is r′ = r₀(s′) + x′ n(s′); r₀, n and the tangent τ are tabulated once from the lattice (`lattice.get_referece_traj`, 2000 samples per element, padded past the end with a straight line and, since b15f71a, continued straight before the start). "2D" means the integrals over the source run over (x′, s′) only: the vertical extent of the bunch is not represented, which is appropriate when the bunch is much wider in the bending plane than vertically, the situation in a bunch compressor.

### 1.3 The longitudinal force

With the retarded potentials the rate of energy change of a particle at r moving with β is W = v·E = W₁ + W₂ + W₃ (thesis Eq. 4.3–4.4),

$$W_1 = -c\int \frac{d^2 r'}{|r'-r|}\,[\beta - (\beta\cdot\beta')\beta']\cdot\nabla' \rho',\qquad
W_2 = c\int \frac{d^2 r'}{|r'-r|}\,(\beta\cdot\beta')\,\rho'\,\nabla'\!\cdot\beta',\qquad
W_3 = -\int \frac{d^2 r'}{|r'-r|}\,\rho'\,\beta\cdot\partial_{t_{ret}}\beta',$$

where primed quantities are taken at the source point r′ and at the retarded time t_ret = t − |r′ − r|/c. The code evaluates W₁ and W₂ (`get_CSR_integrand`): with β = v_s τ + v_x n, v_s = 1, and the source gradient written in the curvilinear frame,

```
∇'ρ' = ∂ρ/∂x'  n(s')  +  ∂ρ/∂z' / (1 + κ x')  τ(s')          # κ = curvature at s'
W1 integrand = (1 + κ x') [ (β − (β·β')β') · ∇'ρ' ] / |r − r'|
W2 integrand = −(1 + κ x') (β·β') ρ' (∂v_x/∂x') / |r − r'|
```

The factor (1 + κx′) is the area element of the curvilinear coordinates. W₃, the term in ∂β′/∂t, is dropped (`vs_t = 0`): in a drift the velocities are constant and in a bend the velocity change over the retarded interval is of order the bend angle, which the thesis argues is negligible for relativistic beams. The sum has a removable 1/|r − r′| singularity at the observer, which is what makes the 2D integral finite and what the polar patch of §5.5 handles. The result is scaled to MeV/m by `CSR_scaling = 8.98755e3 × Q[C]` for a density normalised to unit integral over (x, z) in m² (`CSR.py`, `self.CSR_scaling`).

### 1.4 The transverse force

With β = τ(s) for every particle (thesis Eq. 4.7), the normal component of the Lorentz force is (thesis Eq. 4.8)

$$F_\perp = \frac{1}{c}\int \frac{d^2 r'\,\partial_{t_{ret}}\rho'}{|r'-r|^2}\,[(r-r')\cdot(n-n') - |r'-r|\,(n\cdot\tau')] + \int \frac{d^2 r'\,\rho'}{|r'-r|^3}\,(r-r')\cdot(n-n'),$$

and ∂ρ′/∂t_ret is replaced through the continuity equation by −∇′·(ρ′β′) = −β′·∇′ρ′ − ρ′∇′·β′. In the code the three pieces are W1, W2, W3 of the *transverse* integrand (`CSR_integrand_x = W1 + W2 + W3`, with W1 the ρ′/|r−r′|³ term, W2 the ∂ₜρ (r−r′)·(n−n′)/|r−r′|² term and W3 the −∂ₜρ (n·τ′)/|r−r′| term). `transverse_terms: 'W1'` keeps only the first piece; it is a diagnostic, not a model: W2 + W3 are 10–20% of the force inside a bend and cancel W1 in the drift after it (progress_facet_bc20.md §9k).

### 1.5 What is neglected

* W₃ of the longitudinal force and ∂ₜβ′ of the transverse force (relativistic approximation β′ = τ(s′) + v_x n).
* The formulas are the γ → ∞ limit: there is no space-charge 1/γ² term. A consequence worth knowing: in a straight line the W₁ integrand vanishes identically (β = β′ makes the bracket zero), so the pure-drift longitudinal field is carried only by the velocity terms, whose numerical floor is measured in §10.
* The vertical plane: ρ(x, z) is the projection over y.
* Shielding by the vacuum chamber.

---

## 2. The simulation loop

The run is driven by a precomputed *step schedule* (§6): a set of nodes along s at which the beam is tracked to, a subset at which the density is deposited into the history ("snapshots"), and a subset of those at which the wake is computed and applied ("kicks"). The loop in `CSR2D.run` is, per node:

```
initialisation:  deposit the initial beam, push it into the history
for each node i (step i ends at s_i, inside element e):
    track the particles from s_{i-1} to s_i with bmad-x      (sextupoles: Bmad's kick-drift-kick)
    refresh the formation length L_f from the current sigma_z (R of the current or last bend)
    if is_snap[i]:
        deposit rho, d rho/du, d rho/dw, v_x, d v_x/du on the comoving grid   (section 3)
        push the snapshot; drop snapshots older than the retention floor
        optionally thin the previous snapshot (drift_snap_thin)
        rebuild the time table of the history
    if is_kick[i]:
        build the wake mesh (x_k, z_k) around the beam                          (section 4)
        for every mesh point (serial, or split across MPI ranks):
            W, F = get_CSR_wake(s_i + z_k, x_k)                                  (section 5)
        gather W and F on every rank
        run the per-kick diagnostics (quantile frame, ibp check)
        interpolate W, F from the mesh onto each particle; apply the energy kick
            dE = W * L_kick and the momentum kick d px = F * L_kick / E0,
            L_kick = the midpoint interval of this kick                         (section 6)
    record statistics
write statistics.h5 and wakes.h5
```

This differs from thesis Algorithm 1 in three ways that matter for cost and correctness: the three grids (tracking, snapshots, kicks) have separate spacings instead of one `step_size`; the history is never re-gridded (the deposition grid is fixed in a normalised comoving frame, §3.5); and the kick is applied over a midpoint interval centred on the kick position rather than over the step behind it.

---

## 3. Density estimation

### 3.1 The comoving frame

A compressed bunch in a chicane is a thin ribbon tilted in (x, z) by the dispersion: dx/dz = τ can be 3–40 while σ_x/σ_z is of order 1–100. A rectangular (x, z) grid aligned with the lab axes would waste almost all its cells on empty space and smear the ribbon across them (progress.md Steps 7–12, 17–21, where the tilted-Gaussian tests showed the wake error growing with |τ|). The deposition therefore works in a normalised, tilt-removed frame (`DF_tracker_comoving.get_DF`):

```
fit the tilt on the core (|z - mean z| < fit_zlim sigma_z, fit_zlim = 3):  x = p(z),  poly_degree = 1
xi   = x - p(z)                              # tilt-removed transverse coordinate
u    = (xi - xi_bar) / sigma_xi              # normalised across the ribbon
w    = (z  -  z_bar) / sigma_z               # normalised along it
optional curvature shear (curvature = True): u <- u - g(w), g = c2 (w^2 - 1) + c3 w^3 fitted on |w| < curv_zlim = 4
```

The grid is fixed in (u, w): `xbins` × `zbins` cells over [−xlim, xlim] × [−zlim, zlim] (defaults 128 × 128, ±5 × ±5, nodes at bin centres). Only the frame parameters (p, ξ̄, σ_ξ, z̄, σ_z, c₂, c₃) change from snapshot to snapshot, and they are stored with it. A lab-frame query (x, z, t) is mapped into the frame of the two snapshots that bracket t and read off the fixed grid (§3.6).

### 3.2 Deposition and the velocity field

Particles are deposited with cubic B-spline weights (`histogram_bspline_2d`: each particle spreads over 4 × 4 cells), normalised by the total particle count and the cell area, so ∫ρ du dw = 1 (or the windowed fraction, §3.3). The transverse velocity field is deposited as the weighted sum Σ p_x and divided by the density:

```
vx_model 'affine':  fit px ≈ c0 + b u + g1 w + g2 w^2 + g3 w^3 on the core, subtract it before depositing,
                    deposit the residual, add the fit back analytically (held flat beyond |w| = fit_zlim)
vx      = smooth(rho v_res) / (max(rho_s, 0) + eps) + fit,      eps = (1/2pi) / velocity_threhold
dvx/du  = spectral derivative of vx, times the mask (rho_+/(rho_+ + eps))^3
```

The affine model exists because the fitted part is noise-free and exact for the linear x–p_x correlations a lattice produces, while the deposited residual carries the shot noise; before it, the whole velocity field came from the ratio of two noisy deposits (progress.md, "Validation of `tangent_bands` and `vx_model: affine`"). The floor ε keeps the ratio finite where the density is small and the mask tapers ∂v_x/∂x to zero there. `velocity_threhold` defaults to 5; every BC20 and chicane configuration uses 1000. Both the floor and the mask clamp the density at zero since 8a66afa (§3.4 explains why).

### 3.3 Frames: rms, quantile, peak, auto

How the frame's z window is chosen decides what the grid resolves (`frame`):

| frame | window | when |
|---|---|---|
| `rms` (default) | w = (z − ⟨z⟩)/σ_z over all particles, grid ±zlim σ_z | Gaussian-like bunches |
| `quantile` | [z_q, z_{1−q}] of the charge (q = `frame_quantile` = 0.005), widened by `frame_margin` = 0.1 of its span on each side; particles weighted by a taper T(z), 1 inside, C² smootherstep to 0 across the margin | compressed bunches with spikes or horns |
| `peak` | the quantile window narrowed to the highest current peak: full weight over centre ± `peak_halfwidth` (3) × FWHM, taper over `peak_taper` (2) × FWHM; centre and FWHM from `peakfind.peak_center_fwhm`, rate-limited between snapshots (`peak_rate` 0.2 in ln H, `peak_shift` 0.25 H) | a single spike on a pedestal |
| `auto` | decided once from the initial beam (§3.3.1) | when in doubt |

Why: the FACET-II BC20 bunch ends as a 4 µm FWHM spike holding 40% of the charge on a 200 µm pedestal, with σ_z = 51 µm. An rms grid of 300 bins over ±5σ_z has 1.7 µm cells and, after a 3-bin smoothing, ~5 µm resolution; it saw half of the CSR (−3.95 MeV against −7.6 MeV converged 1D, progress_facet_bc20.md §7). The quantile window puts the same 300 bins over the 208 µm the charge occupies; the peak window puts them over ±3 FWHM of the spike (§9a, §9h). Quantiles are continuous in the particle coordinates, so the window never jumps; the taper makes the dropped tails disappear smoothly rather than as a step, because a density step at a grid edge is amplified by the cancellation factor of §5.7 (progress_cancellation.md §4c–e; the same reason the deposit is tapered in u over `u_taper` = 0.5 σ_ξ before the ±xlim edge).

The peak frame drops the pedestal as a *source*. That is legitimate inside a bend, where the pedestal's own wake is weak, but not in a drift after a bend, where the pedestal's radiation from the bend still arrives (§9h); and it cannot represent two horns. CSR2D therefore evaluates at every kick the 1D wake of the dropped charge (`_quantile_diagnostics`, §7) and warns when it exceeds `pedestal_warn` = 0.05 of the 1D loss or field.

**3.3.1 `frame: auto`** (427ec3e). Before the first deposit, CSR2D previews the peak window on the initial beam and asks the 1D model what the charge it would drop contributes to the wake at the exit of the first bend (`choose_auto_frame`, `csr1d.dropped_wake_fraction`). Above `auto_pedestal_max` = 0.05 of the loss or of the field on the window the frame is `quantile`, else `peak`. The decision is printed and stored in `statistics.h5`. On Indranil's beams: gain → peak (window holds 100%), loss → quantile (the window holds 24% of the charge, the dropped part carries 58% of the 1D loss); Sergei's BC20 beam → peak (progress_facet_bc20.md §11d). The frame cannot change during a run: the grid shape is fixed and a window jump would put a step into the time-interpolated history. The per-kick diagnostic keeps watching.

### 3.4 Smoothing and derivatives

The deposited fields are smoothed by a Gaussian applied in Fourier space with zero padding (`smooth_and_differentiate`), and the derivatives ∂ρ/∂u, ∂ρ/∂w are the spectral derivatives of the *same* smoothed field, so ρ and ∇ρ are consistent to machine precision. The widths are `smoothing_sigma` (3 bins) in u and `smoothing_sigma_z` (default the same) in w.

Two caps act on the z width in the quantile and peak frames:

* `smoothing_res_fwhm` (e.g. 0.1): the effective z resolution, cell × √(σ_s² + 2/3) (the B-spline deposit and its evaluation each add 1/3 cell² of variance), is kept below this fraction of the highest peak's FWHM. The smoothing is in bins and the bins follow the window, so a width chosen for a long bunch (6.5 bins = 14 µm at the BC20 entrance) would be 0.6–1 µm on the spike, comparable to its rms (§9j).
* `smoothing_min_bins` = 1.0: the cap never goes below one bin. A Gaussian narrower than about a bin is not band-limited (at 0.5 bins its response at the Nyquist frequency is 0.29), its real-space kernel rings, and the smoothed density of a sharp edge goes negative by ~1e-4 of the peak. On Indranil's loss beam, whose horns fall 100× in 0.5 µm, that reached the velocity floor ε = 1.6e-4, the ratio ρv/(ρ + ε) and the mask (ρ/(ρ+ε))³ passed through a zero denominator in a few cells of a few snapshots, ∂v_x/∂u reached 2000 instead of 2e-4, and the W₂ term gave 7654 MeV/m at the second kick and 1e20 MeV/m in the exit drift (progress_facet_bc20.md §11b). The clamp max(ρ, 0) in the floor and the mask, and the 1-bin floor, removed it; the second kick is then 12.8 MeV/m with 100 u bins and 13.1 with 320.

`smoothing_adaptive` (quantile frame) widens the z smoothing cell by cell where the pilot density is sparse so that a kernel holds at least `smoothing_min_particles` = 100 particles, blended between widths σ₀·2ᵏ up to `smoothing_sigma_z_max` = 16 bins (`smooth_levels`; §9e). It is off by default and was not used in the reference runs.

### 3.5 The history

Snapshots are kept in a preallocated ring (5 fields × capacity × xbins × zbins, 1.2 MB each at 100 × 300) with the per-snapshot frame scalars in deques. Three rules decide what is kept:

* **Formation length.** L_f = (24 R² σ_z)^{1/3} inside a bend, refreshed every step from the current σ_z (`_refresh_formation_length`; pinning it to the entrance value left the wake 12.9% wrong at shear 20, progress_comoving_interp.md §6h). Snapshots older than `n_formation_length` (1) × L_f behind the beam are dropped.
* **Retention floor.** Dropping on the instantaneous L_f is safe only while σ_z decreases; at a longitudinal waist L_f collapses and recovers, and the snapshots the later steps need are gone. `build_retention_plan` precomputes, from the linear optics, how far back every later kick will reach and keeps the suffix minimum; `drift_reach` and `prev_bend_reach` add the start of the current and previous bend (`_retention_floor`). `_check_retention` compares what was asked against what survived at every step.
* **Drift thinning** (`drift_snap_thin` = 0.01): in a drift after a bend, the second-newest snapshot is dropped if linear interpolation in t between its neighbours reproduces its density, v_x and frame within 1%.

### 3.6 Interpolation at the retarded point

`interpolate3D_comoving_fields` (numba) returns all five fields the integrand needs at an array of lab points (x′, z′, t_ret) in one pass. For each point it finds the two snapshots k, k+1 bracketing t_ret (`lookup_hybrid`: an O(1) division for uniform snapshot times, a bucket table otherwise, measured at +1.8% of the mesh time), blends the *frame* between them, maps (x′, z′) into (u, w) of the blended frame, and reads the fixed grid trilinearly (in u, w and t). Derivatives are transformed back to lab x and z with the blended p′ (chain rule). The frame blend is `frame_blend = 'orient'`: orientation τ = cov/var_z from linearly blended second moments, widths log-linear. The alternatives are kept for reference: blending τ itself (`coeff`) is wrong across a longitudinal waist, where the orientation passes through 90° and a linear path in τ passes through 0°; blending the full covariance (`moment`) gets the orientation right but fattens the mid-interval σ_ξ by a(1−a)(Δτ)² var_z, 8–10× at high tilt (`configure_params` comments; progress_comoving_interp.md §6n–q). Blending the frame rather than the field is what removed the "ghosting" of a tilted ribbon between two snapshots (progress.md Step 20–21, progress_comoving_interp.md Step 5).

---

## 4. The wake mesh

The wake is computed on an `xbins` × `zbins` tensor mesh of observer points around the beam (defaults 20 × 30; the BC20 runs use 12 × 40 or 12 × 80) in the tilt-removed coordinate ξ = x − polyval(slope, z) (the slope here is the fit over *all* particles, `beams.slope`), then mapped to lab x. Two modes (`mesh_mode`):

* `uniform`: ξ over ±xlim σ_ξ, z over ±zlim σ_z (defaults 5, 5).
* `charge`: each axis equidistributes the monitor M = `mesh_uniform_frac`/L + (1 − `mesh_uniform_frac`) λ̃, λ̃ the particle histogram smoothed over `mesh_smooth` (3) deposition resolution lengths, graded so neighbouring spacings differ by at most `mesh_max_ratio` (1.5), over the particles' span widened by `mesh_pad` (2%) (`peakfind.charge_nodes`, `equidistribute`, `grade_monitor`). No peak is detected: every horn gets nodes in proportion to its charge, and the smoothing puts nodes just outside a spike where its field still varies.
* `mesh_x_quantile` (e.g. 0.002): the ξ axis spans the [q, 1−q] quantiles instead of every particle, and particles beyond it receive the edge row's kick (`Beam.apply_wakes(clamp_x=True)`) instead of none. On heavy-tailed beams 0.002 still puts rows at ±6σ_ξ where there are no particles; 0.01 is the better choice (§10).

With `curvature` on, the rows follow the curved ridge σ_ξ g(w) (`_mesh_ridge`), otherwise the outer rows land outside the sheared deposition grid. In the quantile and peak frames the observer's own v_x in the integrand is T·v_x(source field) + (1 − T)·v_x(beam-wide cubic fit) so that a mesh point outside the window is not given v_x = 0 (`_observer_vx`, `_fit_beam_vx`).

---

## 5. The integration

`get_CSR_wake(s, x)` returns (W, F) at one observer. The integrand is a function of the source position (x′, s′), with the retarded time fixed by |r − r′|. It is localised in two ways: in s′ by splitting the range behind the observer into regions with different node layouts, and in x′ by integrating, on each s′ column, only over narrow bands where the retarded beam actually is. Both integrals use the trapezoid rule; the outer one on non-uniform s′ nodes (`np.trapz(x=sp)`), the inner one on uniform or refined x′ nodes per column. The legacy fixed-window path (`xi_bands: false`) is kept for a regression test and is not described here.

### 5.1 Pseudocode

```
get_CSR_wake(s, x):
    t = beam position
    (s1, s3), (s3, s4) = layout_bounds(s, x)              # far region, near region (5.2)
    sp_far  = far_region_nodes(s1, s3)                    # uniform in a bend; causal clip + fine window + log grading after an exit
    sp_near = near_region_nodes(s3, s4)                   # graded about s' = s; near_adapt: scanned and equidistributed (5.4)
    (R1, R2) = near_patch_radii(s, s3, s4)                # polar patch about the observer (5.5)
    W = F = 0
    for region, nodes, taper in [(far, sp_far, none), (near, sp_near, rise 0->1 across [R1, R2])]:
        bands = disjoint( retarded_xi_bands(s, x, t, nodes) )          # one or two (lo, hi) per column (5.3)
        for band in bands:
            xp[:, j] = lo_j + frac * (hi_j - lo_j)                     # xbins uniform x' nodes per column
            gz, gx = integrand(s, x, t, xp, sp, taper)                 # fields at (x', z' = s' - t_ret, t_ret)
            if adapt_x (near always; far if far_adapt_x):
                refine the columns where rho jumps between neighbouring x' nodes (5.4)
            W += trapz_s'( trapz_x'(gz) ), F likewise
    if a previous bend's exit tail still arrives (prev_bend_reach): add its region (causal-edge layout)
    if the patch exists: W, F += polar integral over the disc, weighted 1 - taper
    return CSR_scaling * (W, F)
```

### 5.2 Longitudinal layout (`_layout_bounds`)

With τ the frame tilt (the core fit of the deposition, not the all-particle slope), sin2α = 2τ/(1+τ²), cos2α = (1−τ²)/(1+τ²):

```
d  = (10 sigma_x + x - mean x) |cos 2a| / max(|sin 2a|, branch_sin_min)    # branch_sin_min = 0.05
d  = max(d, near_floor sigma_z)                                             # near_floor = 20
d  = max(d, far_cell / (near_grade seam_ratio))   in a bend; far_cell = n_fl L_f / (2 far_zbins), seam_ratio = 2
d  = min(d, n_formation_length L_f)
reach = n_formation_length L_f
s4 = s + 3 sigma_z;   s3 = max(-prehistory, s - d);   s1 = max(-prehistory, s3 - reach)   # prehistory = 0 by default
in the drift after a bend (drift_reach): s1 = min(s1, start of that bend)
```

The near region (s3, s4) is where the second ("chirp") branch of the integrand, which leaves the beam after d = 10σ_x/|tan 2α|, still lies inside the beam, and where the 1/|r−r′| pole sits; the far region (s1, s3) is the rest of the causal reach. `near_floor` keeps the near region from collapsing at |τ| → 1, and the saturation at |sin 2α| < 0.05 keeps d finite where the two branches merge (|τ| ≪ 1 and ≫ 1). `integration_region.md` §2 tabulates d against τ. The `seam_ratio` floor (2026-11-24) makes the s′ cell continuous across the far/near seam: the near region's graded cell is `near_grade`·u, the far region's is flat, and without the floor an observer on the near-x side of a tilted beam (x − x̄ < 0, |cos 2α/sin 2α| ≈ 5 at τ ≈ 10) had d = 3 mm, so 0.15 mm cells handed over to 5 mm cells while the column integral still varied on the 1–10 mm scale of the observer's transverse distance from the charge. On Indranil's loss beam at B2RE the far-column count then moved W by 18% at the outer rows and 10% in the horn row's core; with the floor, 400 and 800 far columns agree to 0.5% of max|W| (progress_facet_bc20.md §11k). The reach itself is converged at `n_formation_length` = 1: doubling and tripling it at constant far cell moves the mean wake by 0.01–0.03%; at fixed `far_zbins` it coarsens the far cell and that, not the reach, is what changes the result (−1.35% at 10 mm cells).

Far-region nodes (`_far_region_nodes`): in a bend, 2 × `far_zbins` = 400 uniform nodes (grading assumes a 1/u decay that a tilted beam in a bend does not have: at shear 20, 10–99% of the far integral sat at the far end). After a bend exit the source is the arc still inside the dipole, a strip far upstream and everything below it identically zero, so the layout is: a 64-point causal-edge scan for the lowest s′ whose band has nonzero width (`_causal_edge`), a uniform fine window of `far_cell` (10) σ_z cells over `far_window` (1) L_f above it (3 to `far_fine_max` = 500 nodes), then log grading toward s3 with ratio 1 + `near_grade`. `prev_bend_reach` adds, inside a bend, the previous bend's exit tail with the same layout (1D: −0.05 MeV/m at the chicane's B4 entrance). `reach_straight` (2026-11-24) adds, inside a bend and in the drift after it, the straight section before that bend's entrance down to reach_straight·Rφ before it, as a separate region graded in u = s − s′ (`_straight_region_nodes`), while its retardation is within reach_slip_max σ_z (5), never below the previous bend's exit (which `prev_bend_reach` covers, so it acts on the first bend and the drift after it): its sources contribute at nearly one retarded z with a 1/u tail that the flat reach truncated (−40% of ⟨W⟩ at the chicane's B1 exit; `talman_cancellation.md` §7.5).

### 5.3 Transverse bands (`_retarded_xi_bands`)

On each column the retarded beam axis x′ = τ_ret z′ + b meets the light cone where (thesis Eq. 4.24 in the form corrected in progress_comoving_interp.md Step 4f)

```
q   = r0(s) - r0(s') + x n(s);   nq = n(s') . q;   q2 = |q|^2;   T = t - s';   Tb = T - b/tau
rad = (tau^2 - 1)(Tb^2 - q2) + (nq tau + Tb)^2
x'  = tau [ Tb + nq tau +- sqrt(rad) ] / (tau^2 - 1)
```

Both roots are always taken, each refined by 12 fixed-point passes (the retarded frame depends on x′ through t_ret), each giving a band centred on the retarded axis with half-width xlim σ_ξ,ret × `xi_band_margin` (2) = 10 σ_ξ,ret, shifted by the curvature ridge. Deciding from the observation-time tilt which branch exists discarded 28–31% of the wake, because the retarded tilt sweeps across one near region (progress_comoving_interp.md §6). Guards: |τ| ≈ 0 (one root), |τ| ≈ 1 (the quadratic degenerates to a linear equation), rad < 0 (the cone misses the axis), l ≤ 0 (a root from squaring, a source in the future). Three later additions:

* `tangent_bands` (default on): a column whose cone misses the zero-width axis but still crosses the beam gets its band at the closest approach, x′ = nq + sign(τ)√((q2 − nq²)/(τ² − 1)). Without it, columns close to an off-axis observer with τ ξ_obs < 0 were dropped entirely, and the D3 wake of the chicane jumped across the axis (integration_region.md §5a, verified against a 40 001-node brute force to 3 digits).
* `xi_band_margin_degenerate` (3): the margin is multiplied by this, ramped in with a smoothstep between |sin 2α| = 0.8 and 0.95, where the two branches merge and the default margin under-covered the source by up to 40% (progress_facet_bc20.md §9m).
* `band_jacobian_min` (0.2): the far region's oblique columns (|J| = |1 − τ ∂r/∂x′| down to 0.25) have their band widened by 1/max(|J|, 0.2), since the band is sized in lab x′ but the beam in u.

Overlapping bands are made disjoint by subtraction, never union, so the x′ cell size never doubles where the integrand peaks (`_disjoint_bands`). A dead branch gets zero width and is selected out, so a node landing on the pole cannot produce NaN.

### 5.4 Node placement along and across the columns

**Near region, fixed layout.** Graded about s′ = s on each side: cell du = max(`near_cell` σ_ξ, `near_grade` u) with `near_cell` = 0.25, `near_grade` = 0.05, i.e. uniform 0.25 σ_ξ cells out to u* = 5 σ_ξ and geometric growth beyond, because once |r − r′| ≈ u the column contribution falls as 1/u (`_near_region_nodes`; count grows as ln(u_max/u*) with tilt instead of linearly).

**Why that fails on a spike, and `near_adapt`.** Behind the observer the retarded coordinate moves slowly, z_ret ≈ z − u³/24R², so even a spike is a slow source along the column. Ahead of it, z_ret ≈ z + 2u: a 3 µm spike is crossed in 1.5 µm of s′, and a 41 µm cell steps over it. That is where the B1RE transverse force on the BC20 pedestal was speckle (§9k). With `near_adapt`, each column is first scanned (band solve with `near_adapt_scan_iter` = 4 passes and a density lookup at the band axis), every scan interval is subdivided until z_ret moves by at most `near_adapt_scan_dz` (0.25) × res_z inside the history's z support (at most `near_adapt_max_scan` = 4000 points), and the nodes equidistribute

```
m(s') = 1 / h_fixed(s')  +  |d(rho/rho_max)/ds'| / near_adapt_eps        # near_adapt_eps = 0.02
```

graded to `near_adapt_max_ratio` (1.5), at most `near_adapt_max_nodes` (2000). The first term keeps every cell at least as fine as the fixed layout, the second puts ~2/ε nodes across every rise and fall of the source, whatever its shape and however many there are. On the BC20 B1RE pedestal it reproduces a 64× brute-force refinement (~5000 nodes) with 146 nodes (§9l).

**Across the column, `near_adapt_x` and `far_adapt_x`.** z_ret depends on x′ too (dz_ret/dx′ ≈ Δx/|r − r′|, of order 1 for a source ahead), so a spike is crossed within the band over a few µm of x′ against a uniform cell of band width / `xbins` (200), 10–16 µm on BC20. After the uniform evaluation, every column whose retarded density jumps by more than ε ρ_max between neighbouring x′ nodes is re-evaluated on x′ nodes equidistributing 1/h_uniform + |d(ρ/ρ_max)/dx′|/ε, up to `near_adapt_x_rounds` (4) times and `near_adapt_x_max` (4) × xbins nodes; all flagged columns get the same count so it is one more rectangular evaluation (`_adapt_x_columns`). It was first applied to the near region only (§9l); `far_adapt_x` (ea4f41f, default True, inert unless `near_adapt` and `near_adapt_x` are on) applies it to the far region as well, after Indranil's loss beam showed isolated far-region cells of F up to 10× the horn's force and of W off by 34% of the horn's W where a horn is seen through the oblique branch of the band from far behind (§11f, §9 of this document).

### 5.5 The polar near patch

The integrand diverges as 1/|r − r′| at the observer. Along rays out of that point, integrand × |r − r′| is constant over four decades, so a polar mesh centred on (x′ = x, s′ = s) with area element r dr dφ has a bounded, smooth integrand; a Cartesian mesh accumulates equal contributions per decade of r and never converges (`_integrate_near_patch`). The disc has radius R2 = min(`near_patch` (5) σ_ξ, 0.4 (s4 − s), 0.4 (s − s3)), `near_patch_nr` = 100 radial midpoints and `near_patch_nphi` = 180 angles (18 000 nodes). A C² smootherstep w(|r − r′|) rising from 0 at R1 = R2/2 to 1 at R2 multiplies the near region's Cartesian integrand, and 1 − w the patch's, a partition of unity that sums to the untapered integral exactly.

### 5.6 Cost anatomy of a wake point

On the Gaussian dipole, cutting the x′ nodes from 200 to 16 saves only ~55% of a wake point's 16 ms; the rest is the band location (12 fixed-point passes per column), the s′ layout with its causal-edge scan, and the 18 000-node patch (progress_cancellation.md §7b). With the adaptive options the picture changes: on Indranil's loss beam the refinement itself dominates (§9.4 below).

### 5.7 The ±∂ρ/∂x cancellation

The W₁ integrand's leading term is G·∂ρ/∂x′ with G nearly constant across the thin ribbon, so the + and − lobes on the two sides of the axis cancel on each column to a small residual: that residual, reached through the retarded-time chain rule, is the physical 1D-like ∂ρ/∂z wake. The cancellation factor C = Σ|node contributions|/|total| is 60 at the centre of the test dipole, 200–250 for a 12.5 µm thin beam, ~1100 at shear 20, and collapses onto C ≈ 188 σ_z^{2/3}/σ_ξ (µm, R = 1 m) to ±20% in all 13 measured cases, σ_ξ being the slice width (progress_cancellation.md §0, §3). Its consequences, all measured there:

* it does **not** amplify shot noise (the stored ∂ρ/∂x is the consistent derivative of the same smoothed ρ: the seed scatter is identical with and without the cancellation);
* it needs ~96–128 x′ nodes per band for 1e-3–1e-4 accuracy (the default 200 is converged at ~5e-7);
* any error in ∂ρ/∂x that does not integrate to zero across the beam is amplified by ~C: a hard deposit edge clipping 0.3% of the charge moved the wake by 3%, which is why the deposit is tapered in u (§3.3).

`ibp: true` integrates the main term by parts along each column, which removes the cancellation (C → 1–3), is immune to a ρ/∂ρ inconsistency and gives the same beam as the shipped form to 1e-5 on the dipole and 2.4e-4 on the chicane (progress_cancellation.md §7). **It is a robustness option, not a speed or convergence one.** Measured on the BC20 loss beam against an 8× x′ reference (progress_facet_bc20.md §11i): with the x′ refinement running alongside it (since 3cd9528; before that `ibp` switched the refinement off and the horns aliased), by parts reproduces the shipped result exactly wherever the shipped result is converged, is better only where the shipped form is already poor (25 x′ nodes: 41% against 237% of the horn's W), and costs 40% more per wake point for its geometry evaluations. The x′ node count is set by resolving the horns, not by the cancellation, so by parts cannot lower it. Where it earns its cost: a deposition whose derivatives are not the exact derivatives of its values (sub-bin smoothing, a future density estimator), and as a check. `ibp_check: 3` compares both forms at three mesh points per kick, at about one wake point of cost, and warns above `ibp_check_tol` = 1e-3; it is part of the BC20 reference recipe since 2026-11-24 and recommended for any new kind of beam or deposition setting. `ibp` itself stays off by default.

The cheap gain on this front is the uniform x′ count: 100 nodes instead of 200 cost 1.35% instead of 0.97% of the horn's W at the worst mesh cell (0.19% rms) and save 15% of the wake time on the loss beam (§11i). The default stays 200 because the dipole and chicane validations were done there.

---

## 6. Kick application and the step/kick schedule

### 6.1 Applying the kick

The mesh values are interpolated linearly onto each particle at (ξ_particle − ridge, z) (`RegularGridInterpolator`, zero outside, or the edge row with `clamp_x`), and

```
dpz = W [MeV/m] * L_kick * 1e6 / E0 [eV],      dpx = F [MeV/m] * L_kick * 1e6 / E0
```

with E0 the reference energy. L_kick is the kick's midpoint interval: kick_lo = (s_prev + s_i)/2, kick_hi = (s_i + s_next)/2, clipped at element boundaries because W jumps at a dipole edge (`schedule.midpoint_intervals`). Sampling W at s_i and applying it over the arc behind s_i would be a trailing rectangle rule, O(h dW/ds); the midpoint rule is O(h²) for free, since the wake is still evaluated with the history up to s_i.

### 6.2 Steps and snapshots (`step_control.mode: auto`)

Tracking costs ~1 ms per step, a snapshot ~0.08–0.9 s (20k vs 1M particles) plus ~1 MB, a kick 5–60 s; so kicks decide the run time and snapshots the memory, and they need different criteria (schedule.py module docstring). The auto schedule (`build_auto`) first runs a linear-optics scan of σ_z, σ_ξ, τ and σ_x along the lattice and builds an effective step h_eff(s) from: `m_steps` = 2 steps across the longitudinal scale, `m_steps_xi` = 2 across the transverse scale, `tau_frac` = 2 (band-centre drift per step as a fraction of the band half-width), `edge_steps` = 8 across the transient scale at each bend edge, capped by h_max (= `step_size`) and snapped to h_max/2ʲ (`dyadic`). Nodes are placed by equidistributing 1/h_eff within each element, so element boundaries are nodes exactly (a step never straddles a boundary). Every node is a snapshot. The calibrations behind the constants are in progress.md §11l–o (`AUTO_DEFAULTS` comments).

### 6.3 Where to kick

* `kick_placement: kappa` (default): one kick every `kappa` = 4 snapshots. Calibrated on the final beam: 0.04% in σ_E and 0.22% in emittance against a kick at every snapshot, for 0.23× the cost (AUTO_DEFAULTS). It wastes kicks in quiet drifts: on BC20, 333 of 440 kicks before Q2ER produced ~5% of the energy loss (progress_facet_bc20.md §9d).
* `kick_placement: csr1d`: a 1D pre-pass (§7) on the same beam, lattice and snapshot schedule, with a kick at every snapshot, gives r(s) = rms of the 1D wake over the bunch. Kicks are placed at the integers of ∫n ds with

```
n(s) = r/(tol R) + (|r''| / (24 tol R))^(1/3) + 1/kick_h_max,        R = int r ds
```

the first term capping each kick's share of the total CSR at `kick_tol`, the second bounding the midpoint rule's local error, the third capping the spacing where there is no CSR (`kicks_from_profile`). Dipole edges are always kicks. Two terms cover what the 1D profile cannot see: `kick_kappa_refine` k adds a kick per k steps wherever the auto profile refines the steps (h_eff below 0.75 of its median: a waist, a bend edge), and `kick_h_exit` caps the spacing at that value for R·φ past each bend exit, the exit wake's decay length. `kick_prepass_particles` (20 000) runs the pre-pass on a subsample (1M particles: ~6 min; 20k: ~13–30 s). The BC20 reference uses tol 0.005, h_max 0.5 m, kappa_refine 4, h_exit 0.1 m: 269 kicks, reproducing a 297-kick hand-made dense schedule to within 0.1 MeV total and 1% in emittance (§9m–n).

---

## 7. The 1D model used for the pre-pass and the diagnostics

`csr1d.py` implements the line-charge CSR wake of P. Emma's `csr_calc` on top of the same tracking, schedule and output as CSR2D (the class CSR1D reuses CSR2D's methods; with CSR off the two give bit-identical particles). In a bend, cases A/B of Saldin, Schneidmiller and Yurkov (the entrance transient growing into the steady state, φ = angle × distance into the bend / L); after a bend, cases C/D of Stupakov in the γ → ∞ limit with x_R = distance past the exit / R; before the first bend, no CSR (`CSR1DGeometry`). The numerics improve on the MATLAB original where cheap: the u^{−1/3} λ′ integral is product-integrated against a piecewise-linear λ′, case D's weight is integrated per cell by Gauss–Legendre after u = t³ in the first cell, smoothing is a centred Savitzky–Golay fit, ψ is found by vectorised bisection. The bin width follows the bunch (`bin_mode`: `extent`, `quantile`, or `peak` = FWHM of the highest current peak / `bins_per_fwhm`); with `peak` and 20 bins per FWHM the BC20 1D result is converged to about −7.3 MeV (§7e, §9h).

The same kernel serves two diagnostics: the dropped-charge fraction of §3.3 (the 1D wake of (1 − T)λ against that of λ at the current geometry, both binned and smoothed alike so they add exactly, `dropped_wake_fraction`) and the frame-auto decision of §3.3.1. A `wake_1d` call on 1500 bins costs 0.1 ms; the per-kick diagnostic with its histograms of 1M particles about 0.1 s.

---

## 8. Parallelisation

The only parallel section is the wake mesh. `calculate_2D_CSR_parallel` assigns the xbins × zbins mesh points to the MPI ranks **interleaved** (rank r takes points r, r + size, r + 2·size, …), each rank evaluates `get_CSR_wake` on its points into a zero-padded full-length array, and two `Allreduce` sums assemble W and F on every rank. Every point is computed once by one rank with the same arithmetic as the serial loop, so the parallel result is bit-identical to the serial one (checked with `array_equal` at 3 and 4 ranks on the golden configuration). Until 2026-11-23 the split was into contiguous blocks gathered with `Allgatherv`; a point's cost varies 15× across the mesh (24–340 ms on the BC20 loss beam at B2RE: the adaptive node placement and the x′ refinement work where the horns are, and the core rows cost 1.7× the edge rows), and a contiguous block is one or two whole rows, so the slowest rank held 1.16× the mean load at 3 ranks and 1.27–1.30× at 6–12 ranks; interleaving brings this to 1.01–1.02× (measured on the pickled 37.74 m state, §9.5).

Everything else — tracking, deposition, the history, the mesh, the kick application, the statistics — is replicated identically on every rank; only rank 0 writes files and runs the per-kick diagnostics, and the 1D pre-pass runs on rank 0 and is broadcast. The MPICH environment `pydfcsr-mpi` must be used; the Open MPI of `environment.yml` hangs in `MPI_Init` on this machine (CLAUDE.md).

Scaling measured on BC20 (progress_facet_bc20.md §9d; 12-core Apple Silicon, 6 performance cores sharing one L2), with the old block split:

| ranks | ms per wake point | wall time of 3 kicks |
|---|---|---|
| 1 | 21.8 | 67 s |
| 3 | 29.2 | 34 s |
| 6 | 51.7 | 36 s |

Each process slows as more run, consistent with memory bandwidth (every rank reads its own copy of the history, and the integrand used to allocate some 30 temporaries of 80k doubles per call; the fused kernel of §9.5 removes the latter). Two 3-rank jobs side by side give the same total throughput as one 6-rank job. The remaining limits of the scheme are: (i) the replicated deposition and tracking of 1M particles (0.58 s per snapshot, 14 of the 238 min of the loss run, 6%) are not parallelised; (ii) the per-kick fixed cost (diagnostics, mesh and v_x fit on 1M particles) is serial; (iii) the history is replicated in memory on every rank (2 GB for 1779 snapshots at 100 × 300), which an MPI shared-memory window could remove on one node.

---

## 9. Benchmarks and run times

All runs on the same 12-core Apple Silicon machine. "Serial" means one process.

### 9.1 Single dipole, Gaussian beam

`example/input/dipole_chirp0_config.yaml`: 5 GeV, 1 nC, σ_z = σ_x = 50 µm, no chirp, 1M Hammersley particles; 0.1 m drift, 1 m / 1 rad bend, 0.5 m drift; deposition 100 × 100 (rms frame), wake mesh 10 × 30, 18 kicks (`test/quantile_frame/dipole_gauss_check.py`). The reference for the energy change is the default at twice the resolution (progress_facet_bc20.md §9f):

| variant | kicks | mean ΔE [keV] | core slice σ_E [keV] | ε_nx [µm] | W vs 2× default |
|---|---|---|---|---|---|
| default (100×100 rms grid, 10×30 mesh) | 18 | −1378.6 | 726.3 | 26.00 | 4.7% |
| default at 2× resolution (200×200, 20×60) | 18 | −1416.6 | 754.9 | 26.70 | — |
| quantile frame + charge mesh, same counts | 18 | −1411.1 | 744.4 | 26.43 | 1.4% |
| quantile, 2× resolution | 18 | −1428.1 | 757.8 | 26.79 | 1.4% |
| full BC20 settings (150×1200, 12×80 mesh, csr1d kicks) | 68 | −1429.4 | 765.1 | 26.59 | 4.3% |

Run time of the whole default run on the idle machine, serial: 94 s; with `near_adapt` 126 s (+34%); with `near_adapt` + `near_adapt_x` 138 s; the wakes are identical to 3e-4 (§9l, re-timed 2026-11-18). Per wake point 11–12 ms default, 17–19 ms with the near scan. The full-BC20-settings variant took 991 s on older code (§9f). A 200k-particle, 5 × 40-mesh version of the same case runs in 71–72 s (shipped form) and 126 s with `ibp` (progress_cancellation.md §7c).

Figure: `pyDFCSR_2D/test/benchmark_results/quantile_frame/dipole_gauss_check.png`.

### 9.2 The benchmark chicane

`example/input/chicane_lattice.yaml`: four 0.5002 m dipoles of ±0.0483 rad (R ≈ 10.36 m), drifts 5.0058 / 1.0 / 5.0058 m, total 13.31 m; beam `chicane_init_beam.yaml`, 5 GeV, 1M particles, σ_z = 200 µm, σ_x = 64 µm, z–p_z shear −180 GeV/c/m (h = −36 /m), compressing to about 6 kA (`test/chicane_2d_vs_1d.py`). 2D and 1D run on the same bmad-x tracking and the same auto schedule (489 nodes, 105 midpoint kicks), so one CSR-off run is the reference for both. The `2dP` run is the BC20 recipe (peak frame, `smoothing_sigma_z` 6.5 capped at 0.1 FWHM, charge mesh 12 × 40 with `mesh_x_quantile` 0.002); `2dC` is the same with the rms frame and a uniform mesh; `_noT` has the transverse force off; `1d_full` is the 1D model with no drift cutoff. Serial, 28.8 min per 2D run (progress.md, "The BC20 recipe on the closed benchmark chicane", 2026-11-12):

| at the exit, vs CSR off | 2dC | 2dP | 2dC_noT | 2dP_noT | 1d_full |
|---|---|---|---|---|---|
| mean energy change [keV] | −2653 | −2687 | −2622 | −2656 | −1746 |
| projected σ_E change [keV] | −270 | −318 | −330 | −373 | −303 |
| core slice σ_E change [keV] | +19.6 | +28.3 | +145.7 | +159.5 | +10.4 |
| emittance growth, η-corrected [µm] | 0.497 | 0.494 | 0.934 | 0.947 | 0.549 |
| peak current [kA] (CSR off 5.99) | 5.80 | 5.80 | 5.74 | 5.72 | 5.82 |

| energy change per element [keV] | D1 | B2 | D2 | B3 | D3 | B4 | Df |
|---|---|---|---|---|---|---|---|
| 2dC | −48.9 | −109.0 | −42.1 | −627.3 | −1428.6 | −243.8 | −151.9 |
| 2dP | −48.0 | −109.1 | −42.2 | −625.7 | −1469.1 | −239.7 | −152.3 |

The two 2D recipes agree to 1.3% in energy and 0.6% in emittance. The transverse force matters for the core slice energy spread and the emittance (the `_noT` columns), where the 2D result with the force is closer to 1D than without it. `2dPm` (the same with `xi_band_margin_degenerate`) reproduces 2dP to the printed digits (total −2687.5 vs −2687.4 keV; progress_facet_bc20.md §9m), so the passage through |τ| ≈ 1 in B2/B3 contributes nothing measurable here. Note that the 2D energy loss is 1.5× the 1D one; most of the difference is in D3, the drift after the third bend, where the compressed beam is strongly tilted.

**Rerun with the reference recipe (2026-11-24, runs `2dR`, `2dR_noT`, `1d_full`).** The same chicane with the current code and the BC20 reference integration (`near_adapt`, `near_adapt_x`, `far_adapt_x`, `xbins` 100, `ibp_check` 3; deposition and mesh as 2dP), before the `seam_ratio` floor of §5.2:

| at the exit, vs CSR off | 2dR (2D, full) | 2dR_noT (2D, no transverse force) | 1d_full | 2dP (2026-11-12) |
|---|---|---|---|---|
| mean energy change [keV] | −2672 | −2682 | −1746 | −2687 |
| projected σ_E change [keV] | −322 | −368 | −303 | −318 |
| core slice σ_E change [keV] | +28.8 | +157.8 | +10.4 | +28.3 |
| emittance growth, η-corrected [µm] | 0.490 | 0.952 | 0.549 | 0.494 |
| peak current [kA] (CSR off 5.99) | 5.79 | 5.72 | 5.82 | 5.80 |
| per element, B1 / D1 / B2 / D2 / B3 / D3 / B4 / Df [keV] | −0.9 / −49 / −108 / −41 / −624 / −1468 / −231 / −150 | −0.9 / −49 / −109 / −42 / −626 / −1459 / −242 / −154 | −3 / −66 / −8.5 / −70 / −139 / −933 / −366 / −161 | −0.9 / −48 / −109 / −42 / −626 / −1469 / −240 / −152 |

The reference recipe reproduces 2dP to 0.6% in the total energy loss and 1% in the emittance growth; the largest per-element difference is B4 (−231 vs −240 keV, 4%), the bend where the beam is shortest and most tilted. `ibp_check` raised no warning in 489 steps, so the by-parts and shipped forms agreed to 1e-3 at every checked mesh point. The 1D model agrees with 2D in B1, D2 and Df, is 4× smaller in B3–D3 (the strongly tilted compressed beam) and 1.5× larger in B4. Without the transverse force the energy loss is the same within 0.4% but the core slice energy spread is 5.5× and the emittance growth 1.9× the full-force values: the 1D emittance (0.549 µm) is bracketed by the two 2D results, and the full 2D result is the closer one (§5.7 of `talman_cancellation.md`). Run times (serial, 64 min each, with two 2D runs and the test suite sharing the machine; 2dP ran alone in 28.8 min) are not a measurement of the recipe's cost. Figures: **With `prehistory: 20` (run `2dRp`, same recipe otherwise) B1 and D1 collect −3.2 and −70 keV, matching 1D (−3.0, −66) where 2dR had −0.9 and −49: the lattice's 0.1 m D0 left the first bend without the straight-line history the 1D formula assumes (§5.2, `talman_cancellation.md` §7.3–7.6). B2, B3 and D3 are unchanged (their straight sections are covered by `prev_bend_reach`); B4 moves from −231 to −264 keV through the changed chirp; total −2727 keV, emittance growth 0.526 µm, core slice σ_E +16.9 keV.** Figures: `pyDFCSR_2D/test/benchmark_results/chicane_2d_vs_1d/closed/compare_2dP_zproj.png`, `closed/along_s.png`; full table `closed/summary.txt`.

### 9.3 FACET-II BC20, Sergei's beam

The W chicane of FACET-II, 51 m, translated from a Tao lattice (progress_facet_bc20.md §3–5; CSR-off tracking agrees with Tao to 1e-4 σ_x). The S2E beam of 20 000 particles, upsampled to 1M, enters at 1.6 kA, σ_z = 133 µm, and leaves B2RE (36.85–38.67 m) as a 35–50 kA spike of 4 µm FWHM holding 40% of the charge on a 200 µm pedestal (§7a). The reference 2D run `dfcsr_2d_peak_1M_ref` (§9n): peak frame, 100 × 300 deposition, z smoothing 6.5 bins capped at 0.1 FWHM, charge mesh 12 × 40 with `mesh_x_quantile` 0.002, `csr1d` kicks with the 20k pre-pass, `kick_kappa_refine` 4 and `kick_h_exit` 0.1 (269 kicks), `near_adapt` + `near_adapt_x`. 147 min on 6 MPICH ranks, shared with another job for the first hour (alone: about 2 h).

| | 2D reference | 2D, dense hand-made schedule (297 kicks) | 1D (pyDFCSR, peak bins) |
|---|---|---|---|
| → Q2ER (36.65 m) [MeV] | −0.426 | −0.358 | −0.537 |
| B2RE | −1.770 | −1.794 | −0.820 |
| drift + wiggler | −1.664 | −1.693 | −2.252 |
| B1RE | −4.223 | −4.267 | −3.700 |
| total | −8.034 | −7.945 | −7.308 |
| ε_nx 1st / 3rd order [µm] | 13.00 / 8.32 | 12.91 / 8.24 | 20.88 / 12.13 |
| spike slice ε [µm] | 17.3 / 7.0 | 17.4 / 6.9 | 32.6 / 13.2 |

Earlier stages of the same case document what each method choice bought: the rms grid gave −3.95 MeV (half the CSR, §7); 1D converges slowly with its bin width (−6.9 MeV at 300 bins, −7.5 at 1200, §7c); the quantile frame and charge mesh recovered the spike (§9c); the kick schedule went 440 → 220 → 116 → 269 kicks with the energy change by segment tracked each time (§9d, §9j, §9n); Tao's own 1D CSR (n_bin 128) gives −6.39 MeV in 2 s (§7b). The 2D-vs-1D differences that remain are physics of the tilted beam: B2RE 2.2×, the drift after it, and the transverse force in B1RE (§9n). Figures: `pyDFCSR_2D/test/benchmark_results/facet_bc20/bc20_wake_zproj_dfcsr_2d_peak_1M_ref.png`, `bc20_wakes_along_s_dfcsr_2d_peak_1M_ref.png`, `bc20_wake_maps_long_dfcsr_2d_peak_1M_ref.png`.

### 9.4 FACET-II BC20, Indranil's beams

Two 1M-particle beams on a BC20 variant with a 2 m exit drift (`example/BC20_Indranil`, progress_facet_bc20.md §11), run with the reference recipe above at 3 MPICH ranks each:

| | gain | loss |
|---|---|---|
| at BEGBC20 | σ_z 236 µm, 0.76 kA, long bunch | σ_z 12.6 µm, two 55 kA horns of ~1 µm FWHM |
| frame | peak (= quantile for a long bunch) | quantile (the peak window kept 20% of the charge) |
| wake mesh | 12 × 40 | 12 × 80 |
| kicks / run time | 336 / 205 min | 319 / 238 min (first run with the velocity-floor bug: 364 min) |
| 2D total [MeV], ε_nx 1st/3rd [µm] | −0.301, 43.91 / 15.71 | −22.43, 72.5 / 55.8 |
| 1D | −0.396, 44.98 / 15.93 | −27.33, 77.9 / 63.6 |
| CSR off ε_nx | 44.30 / 15.94 | 25.4 / 23.2 |

The gain run is converged in the kick schedule: 497 kicks (304 min) move no section by more than 0.009 MeV and the projected wakes agree to 2% rms (§11a). The loss beam is converged in deposition resolution (300, 600 and 1200 z bins give the first-bend wake to 2%, §11c) and, after `far_adapt_x`, in the far-region x′ sampling (§11f); its rerun with that option was in progress when this was written.

**Per-point cost by stage**, measured on pickled kick states with the same options (near_adapt, near_adapt_x, far_adapt_x), serial, 2026-11-23:

| per wake point | Sergei, B1RE 48.38 m | loss, B1RE 48.73 m | loss, B2RE 37.74 m |
|---|---|---|---|
| far region | 20 ms | 52 ms | 50 ms |
| near region | 18 ms | 67 ms | 45 ms |
| near nodes, patch, layout | 23 ms | 12 ms | 10 ms |
| total | 60 ms | 131 ms | 105 ms |
| refined x′ evaluations, far | 40 k | 110 k | 78 k |
| refined x′ evaluations, near | 33 k | 198 k | 117 k |
| uniform x′ evaluations, far / near | 80 k / 29 k | 80 k / 40 k | 80 k / 25 k |
| mesh points per kick | 480 | 960 | 960 |
| kicks in the run | 269 | 319 | 319 |

The loss beam costs 2.2× per point because its 100-fold horn edges flag most near columns and many far columns for refinement at the 4× node cap (six times the near evaluations, three times the far), 2× per kick for the 80-node mesh, and 1.2× for the kicks: about 5× per run, which is the 238 min against 147. The refinement is what removed the aliased cells of §11f; the levers for a cheaper run are `near_adapt_eps` (0.02 → 0.05 flags fewer columns) and `near_adapt_x_max` (4 → 2), to be checked against an 8× x′ reference on a pickled state before use.

---

### 9.5 Where a wake point's time goes, and what was gained (2026-11-23)

cProfile on the pickled BC20 loss state at 37.74 m (B2RE, |τ| = 4.8, 12 × 80 mesh), 24 points, 110 ms per point before the changes below:

| stage | share | notes |
|---|---|---|
| integrand evaluation (`get_CSR_integrand`) | 66% | 10 calls and 324k (x′, s′) points per wake point: far 80k + near 40k uniform, the rest refinement |
| — of which numba field interpolation | 34% | `_comoving_fields`, 17 ms per 80k points |
| — of which numpy arithmetic and python | 32% | ~40 whole-array operations, ~30 temporaries of 80k doubles per call; the per-element curvature mask loop 3.8 ms per call |
| x′ refinement bookkeeping (`_adapt_x_columns`) | 6% own | its refined integrand calls are inside the 66% |
| band location (`_retarded_xi_bands`, `_eq424`, frame lookups) | 13% | 12 fixed-point iterations × 2 branches; 6 iterations give bit-identical nodes |
| near-node scan (`_near_adapt_nodes`) | 7% | |
| polar patch | 4% | |

Changes made, each bit-identical to the previous code (W and F `array_equal` on 60 points of that state; 176 tests including the golden fixtures): the integrand arithmetic fused into one numba pass per point (`integrand_kernel.py`, `CSR_integration.integrand_kernel: numba`, default; `numpy` keeps the old path), the curvature lookup as one `searchsorted` (3.8 → 0.5 ms per call), and the x′ refinement reading the retarded density from the plain field pass instead of the integration-by-parts pass (22.3 → 17.4 ms per 80k points). Per wake point: 110 → 80 ms (−27%); the interleaved MPI split (§8) adds ~14% at 3 ranks and ~25% at 6–12. What did not pay: per-column node counts in the refinement (the common count over-refines by only 15%), and fewer band iterations (no measurable time). The floor is now the field interpolation itself (one trilinear-in-time, B-spline-in-space lookup of five fields per (x′, s′) point) at about 45% of a wake point.

Why the loss beam costs 5× Sergei's per kick (per point 131 vs 60 ms with the same options on the B1RE states; 960 vs 480 mesh points; 319 vs 269 kicks): the x′ refinement flags most columns on a two-horn beam with 100× density edges and runs them to the 4× node cap (near region 198k vs 33k refined evaluations, far 110k vs 40k), the charge mesh needs 80 z nodes for 1 µm horns, and the horns radiate from the first bend on so the pre-pass places more kicks. The knobs, to be checked against an 8× reference on a pickled state before use, are `near_adapt_eps` (0.02 → 0.05 flags fewer columns) and `near_adapt_x_max` (4 → 2).

## 10. Known limitations and open items

* **No independent check of the 2D integral.** The band-localised integral has been checked against brute-force integrals of the same formula (40 001-node columns, 64× s′ refinement, 8× x′ nodes) but not against a Liénard–Wiechert point-charge sum. The one-snapshot glitch at 19.12 m of BC20 (a plausible caustic at |τ| → 1) is converged within the band path and unconverged in the legacy path; it is logged as the first item of progress_facet_bc20.md §10.
* **W₁ varies linearly with x at the bunch ends in a bend.** On a long bunch at the first kicks of a bend, the end columns of the mesh carry a wake linear in ξ and antisymmetric between head and tail, up to several times the core wake at ±6σ_ξ where no particles are (the "frame" on the B1LE/B2LE maps). It is unchanged by every quadrature, band and deposition variant tried, and absent in a pure drift where W₁ ≡ 0; it is consistent with the u·x/2R shift of the retarded coordinate acting on the Coulomb-scale terms of the integrand before their cancellation (progress_facet_bc20.md §11e). Whether the 2D entrance theory has this gradient at this size is not verified.
* **The velocity terms have a floor.** In a pure drift, where the exact longitudinal field vanishes, the W₂ term and the source-velocity part of W₁ leave 0.003 MeV/m on the gain beam (0.8 kA, 236 µm) and 0.06–0.09 MeV/m on the loss beam, against a 20 MeV/m horn wake (§11e, §11b). The two terms cancel by parts only if ∂v_x/∂x is the derivative of the interpolated v_x, which the deposited fields satisfy to 0.1–1%. This is the accuracy limit wherever the CSR is weaker than that, i.e. the first bend of a long bunch.
* **Wake roughness in the dispersive sections.** With the comoving frame the density really varies along z on a 5–20 µm scale inside the folded ribbon of the sextupole regions (§9f); with 20 000 particles the wake there is 1.2–1.7× rougher than the old rms grid, which smoothed across the ribbon. The remedy would be anisotropic smoothing along the ribbon, or more particles.
* **Edge rows of the mesh.** `mesh_x_quantile` 0.002 puts rows at ±6σ_ξ on heavy-tailed beams; 0.01 keeps them within the charge (§11e).
* **1D guidance of the kicks** cannot see 2D-only transients (the D5 x-waist at 18.9 m, the lobes after B2RE); `kick_kappa_refine` and `kick_h_exit` cover the known ones (§9j, §9m).
* **The lattice start.** Before b15f71a observers more than one table cell before s = 0 got a zero geometry (short test lattices only); the orbit is now continued straight backwards.
* **3D** (x, y, z) is not implemented; the formulas and the loop generalise, the cost does not trivially.

---

## 11. Parameter reference

Defaults from `params.py` and `deposit_smooth.py`; the "BC20 reference" column is the recipe of progress_facet_bc20.md §9n.

### `particle_deposition`

| key | default | BC20 reference | meaning |
|---|---|---|---|
| `method` | bspline_comoving | same | `legacy` (CIC) and `bspline_fft` exist for regression only |
| `xbins`, `zbins` | 128, 128 | 100, 300 | cells of the fixed (u, w) grid |
| `xlim`, `zlim` | 5, 5 | 5, 5 | half-widths of the grid in σ_ξ, σ_z (or window units) |
| `smoothing_sigma` | 3.0 | 2.0 | Gaussian width in u, bins |
| `smoothing_sigma_z` | = smoothing_sigma | 6.5 | Gaussian width in w, bins |
| `smoothing_res_fwhm` | None | 0.1 | cap: z resolution ≤ this × FWHM of the highest peak (quantile/peak frames) |
| `smoothing_min_bins` | 1.0 | 1.0 | the cap never goes below this (sub-bin FFT Gaussians ring) |
| `frame` | rms | peak | rms / quantile / peak / auto |
| `frame_quantile`, `frame_margin` | 0.005, 0.1 | same | quantile window and its taper margin (fraction of the span) |
| `peak_halfwidth`, `peak_taper` | 3.0, 2.0 | same | peak window ± halfwidth × FWHM, taper over taper × FWHM |
| `peak_rate`, `peak_shift` | 0.2, 0.25 | same | rate limits of the peak window between snapshots |
| `auto_pedestal_max` | 0.05 | — | frame auto: dropped 1D loss/field fraction above which quantile is chosen |
| `smoothing_adaptive` | False | False | density-adaptive z smoothing (`smoothing_min_particles` 100, `smoothing_sigma_z_max` 16) |
| `poly_degree` | 1 | 1 | tilt fit; the two-root band locator needs 1 |
| `fit_zlim` | 3.0 | same | core for the tilt and v_x fits, in σ_z |
| `curvature`, `curv_zlim` | True, 4.0 | same | cubic curvature shear of the frame |
| `vx_model` | affine | same | fitted + residual velocity field |
| `velocity_threhold` | 5 | 1000 | floor ε = (1/2π)/value on the velocity ratio |
| `u_taper` | 0.5 | same | deposit taper before the ±xlim edge, in σ_ξ |
| `frame_blend` | orient | same | how the frame is blended between snapshots |

### `CSR_computation`

| key | default | BC20 reference | meaning |
|---|---|---|---|
| `compute_CSR`, `apply_CSR`, `transverse_on` | 1, 1, 1 | same | switches |
| `xbins`, `zbins` | 20, 30 | 12, 40 (80 for the loss beam) | wake mesh |
| `xlim`, `zlim` | 5, 5 | — | uniform mesh half-widths in σ |
| `mesh_mode` | uniform | charge | charge-equidistributed axes |
| `mesh_uniform_frac`, `mesh_max_ratio`, `mesh_smooth`, `mesh_pad` | 0.3, 1.5, 3.0, 0.02 | same | charge mesh monitor, grading, smoothing (in resolution lengths), pad |
| `mesh_x_quantile` | None | 0.002 | ξ axis on the [q, 1−q] span, kick clamped beyond |
| `mesh_z_quantile` | None | charge mesh: the z axis on the [q, 1−q] quantiles (+ pad) built from the particles inside, particles beyond get the edge kick; default every particle, which lets a sparse tail take nodes (0.001 in the BC20 recipe) |
| `pedestal_warn` | 0.05 | same | dropped-charge diagnostic threshold |
| `write_wakes`, `write_beam`, `workdir`, `write_name` | True, None, '.', '' | | output |

### `CSR_integration`

| key | default | meaning |
|---|---|---|
| `n_formation_length` | 1 | causal reach and history truncation in units of L_f |
| `xbins` | 200 | uniform x′ nodes per band |
| `zbins` | 200 | near-region node count when `near_cell` = 0; also caps |
| `far_zbins` | 200 | far-region nodes (2× in a bend) |
| `seam_ratio` | 2.0 | in a bend, near-region length ≥ far_cell/(near_grade·seam_ratio) so the s′ cell is continuous across the seam; 0 = old layout |
| `reach_straight`, `reach_slip_max` | 40, 5 | the straight section before the current (or last) bend's entrance, down to reach_straight·Rφ before it, integrated as its own region graded in u = s − s′ (ratio 1 + near_grade, a few dozen nodes), while its retardation (Rφ³/6 inside the bend, + xφ²/2 at x past the exit) is within reach_slip_max σ_z. Its contribution converges as U/(U + Rφ); the flat reach covered 0.9 m of it at the chicane's B1 exit and lost 40% of ⟨W⟩. 0 = off |
| `prehistory` | 0 | synthetic straight-line history [m] before the lattice start: the far region may reach to s′ = −prehistory along the straight continuation of the orbit (zero curvature), the history lookup returning the first snapshot there. Needed when the lattice starts at or just before a dipole (the 1D entrance transient assumes an infinite straight line); set ≥ the reach, ~2 m on the chicane. 0 = sources only from s′ ≥ 0 |
| `near_cell`, `near_grade` | 0.25, 0.05 | near-region cell in σ_ξ and geometric growth |
| `near_floor`, `branch_sin_min` | 20.0, 0.05 | near-region length floor (σ_z) and saturation |
| `near_patch`, `near_patch_nr`, `near_patch_nphi` | 5.0, 100, 180 | polar patch radius (σ_ξ) and nodes |
| `xi_bands`, `xi_band_margin`, `xi_band_margin_degenerate` | True, 2.0, 3.0 | band localisation, half-width factor, factor near \|τ\| = 1 |
| `tangent_bands` | True | band at the closest approach when the cone misses the axis |
| `band_jacobian_min` | 0.2 | widen oblique far bands by 1/max(\|J\|, this) |
| `causal_clip`, `far_window`, `far_cell`, `far_fine_max` | True, 1.0, 10.0, 500 | exit-transient far layout |
| `drift_reach`, `drift_snap_thin`, `drift_cutoff` | True, 0.01, 0.0 | reach back to the bend entrance after an exit; thin drift snapshots; never stop computing in a drift |
| `prev_bend_reach`, `prev_bend_grade` | True, 0.02 | the previous bend's exit tail inside a bend |
| `near_adapt`, `near_adapt_eps`, `near_adapt_scan_dz`, `near_adapt_scan_iter` | False, 0.02, 0.25, 4 | adaptive near s′ nodes (BC20 reference: on) |
| `near_adapt_max_scan`, `near_adapt_max_nodes`, `near_adapt_max_ratio` | 4000, 2000, 1.5 | their caps and grading |
| `near_adapt_x`, `near_adapt_x_rounds`, `near_adapt_x_max` | False, 4, 4 | adaptive x′ nodes on flagged near columns (BC20 reference: on) |
| `far_adapt_x` | True | the same on the far region (needs the two above) |
| `integrand_kernel` | `numba` | `numba`: the integrand arithmetic fused per point (integrand_kernel.py); `numpy`: the original whole-array path. Bit-identical; numba is ~8% faster per wake point (§9.5). |
| `transverse_terms` | all | `W1` drops the ∂ₜρ terms (diagnostic only) |
| `ibp`, `ibp_check`, `ibp_check_xbins`, `ibp_check_tol` | False, 0, 400, 1e-3 | integration by parts (robustness, not speed; §5.7) and the shipped-vs-by-parts check (3 in the BC20 recipe) |

### `step_control`

| key | default | BC20 reference | meaning |
|---|---|---|---|
| `mode` | auto (since 2026-11-06; `legacy` = uniform `step_size`) | auto | schedule construction |
| `kick_interval` | midpoint in auto mode, trailing in legacy | midpoint | kick quadrature interval |
| `kappa` | 4 | — | kicks every kappa snapshots (`kick_placement: kappa`) |
| `kick_placement` | kappa | csr1d | |
| `kick_tol`, `kick_h_max` | 0.01, 1.0 | 0.005, 0.5 | csr1d placement |
| `kick_prepass_particles` | None | 20000 | pre-pass subsample |
| `kick_kappa_refine`, `kick_h_exit` | None, None | 4, 0.1 | kicks for 2D-only transients |
| `n_sub` | — | 400 | linear-optics scan resolution of the auto schedule |

### `CSR_1D` (CSR1D runs and the pre-pass)

`nbins` 300, `bin_mode` extent / quantile / peak, `bin_quantile` 0.05, `bins_per_fwhm` 20, `max_bins` 20000, `nsmooth` 2, `kick_interp` spline, `drift_cutoff` 0.0, `record` false. The pre-pass and the BC20 1D reference use `bin_mode: peak`.

---

*Facts not verified against a primary source: the chicane beam's σ_z (200 µm) and shear are read from `chicane_init_beam.yaml` and the test header; the "about 2 h alone" run time of the BC20 reference is the log's own estimate (the measured 147 min included a shared hour). The loss beam's rerun with `far_adapt_x` had not finished when this was written; its numbers in §9.4 are from the run without it.*
