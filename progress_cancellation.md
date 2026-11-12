# Cancellation in the CSR integrand: how large, what it costs, how to remove it

Branch `cancellation_effects` (from `csr1d_mode` @ `d1bb56c`), started 2026-11-12.
Earlier work on this topic is Step 12 of `progress_comoving_interp.md` (2026-10-02, branch
`integrand_cancellation`). §5 below revises two of its conclusions.

All numbers below use the current defaults (`bspline_comoving`, `orient` blend, curvature
shear on, `tangent_bands`, `drift_reach`, `step_control: auto`). The test lattice is
`dipole_lattice.yaml`: a 0.1 m drift, an R = 1 m dipole bending 1 rad, then a 0.5 m drift. The
beam is 5 GeV, 1 nC, σx0 = σz0 = 50 µm unless a row says otherwise, 200k particles
(Hammersley). Deposition is 200×200 over ±5σ with smoothing 3 cells; integration uses
`xbins = zbins = 200` and `n_formation_length = 4`.

**Wake grid (the observers).** `CSR_computation` sets `mesh_mode: uniform`, `xbins: 5`,
`zbins: 40`, `xlim = zlim = 3`, and the grid is built by `get_CSR_mesh` at the stop position t:
- z nodes: 40 evenly over z̄ ± 3σz, i.e. a spacing of 0.154σz. σz is the beam's rms at t.
- x nodes: 5 evenly over ±3σ of x_transform = x − polyval(slope, z) − ridge, i.e. across the
  tilt-removed beam. Each node is mapped back to lab x by adding the tilt fit and the
  curvature ridge at its z, so the rows follow the tilted, curved beam.
- Observer: s = t + z_node at x_node.

All cuts here use the middle row (index 2, the beam axis):
- A, D, E, F: every 2nd z node from 4 to 34 (16 points, z/σz = −2.38 … +2.23);
- B: every 4th (8 points);
- C: every 3rd (11 points).

"Centre" means the node closest to z = 0, which is z/σz = +0.08. The L2 norms in C–F are over
these cuts.

---

## 0. Short answer

- **What the two bands are.** In the longitudinal wake they are the + and − lobes of ∂ρ/∂x on
  the two sides of the beam axis (figure in §1). On each s′ column they cancel down to a small
  residual. That residual is the physical, 1D-like ∂ρ/∂z wake, which the 2D formula recovers
  only through the retarded-time chain rule.
- **How large.** C = Σ|node contributions| / |total| is
  - 60 at the bunch centre in the bend;
  - 200–250 for a 12.5 µm thin beam;
  - ~45 past the exit;
  - ~1100 at shear 20.

  All 13 cases collapse onto **C ≈ 188·σz^(2/3)/σξ** (µm, R = 1 m) to within ±20 % (§3). Here
  σξ is the **slice** width: the rms across the tilt-removed beam, the co-moving frame's σξ.
  It is not the projected σx; for a tilted beam the two differ by up to 100×. Shear 20's
  C ≈ 1100 is simply its 8.5 µm slice width. The transverse
  integrand cancels much less: C ≈ 1.2 in the bend, ~12 past the exit, ~28 at shear 20.
- **What it costs.**
  - **Not shot noise.** The wake's scatter over random seeds is identical, to all quoted digits,
    whether the integral is written with or without the cancellation. This is because the stored
    ∂ρ/∂x is the consistent derivative of the same smoothed ρ.
  - **Quadrature accuracy.** The shipped form needs ≥ 128 x′ nodes per band. At 16–64 nodes it
    is off by 1–20 % (50–200 % for the thin beam).
  - **Robustness.** Any error in ∂ρ/∂x that does not integrate to zero across the beam is
    amplified by ~C. Clipping 0.3 % of the charge moves the shipped wake by 3 %.
- **A fix that works (prototype, diagnostic only).** Integrate the main term by parts along
  each s′ column. It reproduces the shipped wake to ~2e-6 and brings C down to 1–3 at the bunch
  centre. It reaches 0.1–1.4 % accuracy with 16–24 x′ nodes (≤ 0.6 % inside the bend), and its response to the clipped
  charge is ~q, not ~C·q.
- **Its limit.** It is singular where a column runs along a strongly tilted beam (J → 0, beam
  tilt |τ| > 1). A partition of unity handles σx = 200 µm (agreement 1.8e-4) but leaves a ~1 %
  difference at shear 20.

---

## 1. What the two bands are, and why they cancel

**The object.** The longitudinal integrand (`get_CSR_integrand`) is

```
g_z = h' K·∇'ρ_ret / R + (div v term)/R,       K = v − (v·v′) v′,     R = |r − r′|
K·∇'ρ = (K·n′) ∂ρ/∂x + (K·τ′) (∂ρ/∂z)/h′
```

K is the part of the observer's velocity perpendicular to the source's velocity. So it points
almost exactly along n′, and K·τ′ = −v_x′(v·n′) ~ 1e-5. Step 12 measured this: ∂ρ/∂z has zero
gain in the longitudinal wake. Almost all of W1 is

```
G0 · ∂ρ/∂x,         G0 = h′ (v·n′) T(R) / R          (T = the near-patch taper)
```

G0 is pure geometry and smooth. ∂ρ/∂x is positive on one side of the beam axis and negative on
the other. On the (x′, s′) integration nodes this gives two long parallel stripes of opposite
sign, one per side of the beam. These are the "two large bands":

![integrand map](pyDFCSR_2D/test/benchmark_results/cancellation_effects/G_integrand_map.png)

- Left panel: the shipped integrand, ±8e6 MeV/m per m².
- Third panel: the same integral after integration by parts (§2), on a scale 22× finer.
- Right panel, per s′ column: Σ|nodes| ≈ 10³ (blue) against the column integral, 0.1–40 (black).

**Why they cancel, physically.** Along one column (fixed s′), moving across the beam in x′
also moves the retarded point. z_ret = s′ − t + R and R changes with x′ at the rate
R_x′ = dR/dx′ = −n′·(r − r′)/R, which is ≈ ±sin(φ/2) for a source φ upstream in the bend. The
light cone therefore crosses the beam **obliquely**: the + lobe at x′ < x̄ and the − lobe at
x′ > x̄ are read from z-slices a distance ≈ R_x′·σξ apart. Here σξ is the beam's width at
fixed z, i.e. the slice width across the tilt-removed beam, which is what the two lobes span.

- If the slices were identical, the two lobes would cancel exactly. ∫∂ρ/∂x dx′ = 0, the lobes
  of a derivative.
- What survives is the density **difference** between two slices ~R_x′·σξ apart. That is a
  finite-difference estimate of ∂ρ/∂z, which is the 1D CSR wake (λ′ convolved with a smooth
  kernel).

So the 2D formula computes the 1D wake as **the small difference of two large x-lobes**.

An analogy: you could measure the slope of a hill by weighing two buckets of sand, one dug on
each side of a ridge line that runs slightly downhill, and subtracting. That works, but every
gram you mis-weigh in either bucket counts against the small difference, not the large weight.

**The scaling law this predicts.**
- Gross per column: ∫|∂ρ/∂x| dx′ = √(2/π) λ/σξ.
- Net per column: ≈ R_x′ λ′ ~ sin(φ/2) λ/σz.
- So C ~ √(2/π) σz / (σξ sin(φ/2)), with the formation angle φ ~ (24σz/R)^(1/3). That gives
  **C ∝ σz^(2/3) R^(1/3) / σξ**.

§3 checks the prediction. A free fit over all 13 cases gives C ∝ σξ^−1.08 σz^+0.63, and the law
holds with one constant, C = 188·σz^(2/3)/σξ (µm), to ±20 %. That constant is 3.4× the crude
estimate (√(2/π)·2/24^(1/3) = 0.55 in metres, i.e. 55 in µm units). The difference is the
ratio of λ to σz λ′ averaged over the formation region. The R^(1/3) has not been tested: every
case here has R = 1 m.

The key consequence: **the thinner the slice, the worse it gets.**
- The 1D limit, the usual operating point, is where the 2D formula is most ill-conditioned.
- So is a strongly chirped or dispersed beam. It can be millimetres wide in projection while its
  slice is a few µm.

**A wrong turn, recorded so it is not repeated.** §3's first version tabulated the *projected*
σx (`beam._sigma_x` = std(x)). Against it the σx scan seemed to "flatten" at σx ≥ 100 µm, and
shear 20 (σx = 922 µm, C = 1165) looked like a separate tilt effect. Against the slice width
both are on the same line (`H_collapse.png`). A fit against the projected σx gives the
meaningless σx^−0.31 σz^+1.29.

---

## 2. The fix tried: integration by parts along each s′ column

**Identity.** Along one column, the retarded point moves with x′:

- dz_ret/dx′ = R_x′
- dt_ret/dx′ = −R_x′

Let c(z, t) be the beam's ridge line: the co-moving frame's tilted axis plus the curvature
shear, from `_comoving_frame_at` and `_curv_shift_at`. Let ξ = x′ − c be the transverse
position relative to the ridge. The chain rule then gives, exactly:

```
dρ_ret/dx′ = J · ∂ρ/∂x + D
J = 1 − dc/dx′|column          (how fast the column crosses the beam; J = 1 for an untilted beam)
D = dρ_ret/dx′ at FIXED ξ      (the slice-to-slice change along the column: no lobes)
```

so per column

```
∫ G0 ∂ρ/∂x dx′ = [G0 ρ/J]_lo^hi − ∫ ∂x′(G0/J) ρ dx′ − ∫ (G0/J) D dx′
```

**What each term is.**
- The boundary term is 0 to machine precision, because the bands are ±10σξ wide and ρ is zero
  at their edges. Where two disjoint bands abut, their edge terms cancel; both are included.
- ∂x′(G0/J) is geometry and the frame only. It has no density in it.
- D is what the lobes were estimating: the derivative along the beam, times R_x′.

This is the identity that makes the cancellation analytic instead of numerical.

**Algorithm (prototype, `test/cancellation_ibp.py`; diagnostic only, nothing shipped calls it).**

```
input: observer (s, x), the shipped node meshes of get_CSR_wake (far, near, prev regions;
       per s′ column, the band x′-nodes), the near-patch taper
eps = 1e-3 · σξ                                   # finite-difference step along the column path
for each region except the polar patch, for each band mesh (xp[i,j], sp[j]):
    R, R_x  ← geometry at the nodes;   G0 ← h′ (v·n′) T(R)/R
    ρ, ρ_x  ← _comoving_fields at (xp, z_ret, t_ret)
    J  ← 1 − [c(z+eps R_x, t−eps R_x) − c(z−eps R_x, t+eps R_x)] / (2 eps)
    D  ← [ρ(xp+Δc⁺, z+eps R_x, t−eps R_x) − ρ(xp+Δc⁻, z−eps R_x, t+eps R_x)] / (2 eps)
         with Δc± = c(path±) − c(z, t), i.e. at fixed ξ
    χ  ← 1 for |J| ≤ J0, 0 for |J| ≥ 2J0, C² smootherstep between     (J0 = 0.25)
    Q  ← (1 − χ) G0 / J;   Q′ ← central difference of Q in x′ (step eps), J recomputed at x′±eps
    node integrand ← [g_z − (1−χ) G0 ρ_x]   (the shipped remainder: small terms + the χ share)
                     − Q′ ρ − Q D
    boundary(column) ← Q ρ |_last node − Q ρ |_first node
    quadrature: the SAME trapezoid weights as _integrate_xi_region
polar patch: shipped form (its C is 1–3)
```

**Invariants.**
- The total equals the shipped total up to quadrature error. This is checked at every point
  (§3 column "|ship − ibp|"); the agreement is the test that the identity is implemented right.
- J and D are differentiated with **the same c**. The identity holds for any c, so the result
  does not depend on matching the interpolant's frame exactly; c only decides how small D and
  how far from 0 J are.
- The interpolant is a C² cubic B-spline in (u, w), so the central differences are O(eps²).

**Why each choice was made.**
- *Columns rather than a 2D divergence.* The quadrature already runs per column, and per
  column the only singular point is J = 0.
- *Finite differences for J and D, rather than formulas.* A prototype has to be right before it
  is fast. Formulas exist: J = 1 − (c_z − c_t)R_x′, and D could be built from the stored ∂F/∂w.
- *A partition χ(J) rather than nothing.* For |τ| > 1 some columns run along the ridge, where
  J → 0 and 1/J blows up. With J0 = 0 the σx = 200 µm and shear-20 cases differ from the shipped
  form by 2e3–4e3×.

---

## 3. How large the cancellation is

**Per case** (`A_anatomy.txt`; mid-row z cut, 16 points (see the wake grid above); τ = beam tilt dx/dz at s):

| case | σx proj. µm | σξ slice µm | σz µm | τ | C shipped, centre / median | C by parts, centre / median | \|ship − ibp\|/\|ship\| | min\|J\| |
|---|---|---|---|---|---|---|---|---|
| in bend, s = 0.5 m | 46.7 | 43.5 | 53.8 | −0.32 | 60.2 / 53.4 | 1.16 / 1.36 | 2.3e-6 | 0.68 |
| in bend, σx0 = 12.5 µm | 14.3 | 14.3 | 50.3 | −0.03 | 203 / 179 | 1.57 / 1.66 | 6.9e-7 | 0.97 |
| past exit, s = 1.2 m | 60.4 | 53.9 | 67.3 | −0.40 | 46.9 / 42.6 | 2.65 / 3.49 | 4.9e-4 | 0.59 |
| in bend, shear 0.5 | 52.0 | 52.0 | 44.7 | −0.00 | 43.3 / 38.5 | 1.09 / 1.29 | 4.8e-6 | 0.97 |
| in bend, shear 2 | 103 | 103 | 22.4 | −0.03 | 14.1 / 12.6 | 1.15 / 1.63 | 2.4e-5 | 0.82 |
| in bend, shear 20 (xlim 8) | 922 | 8.5 | 341 | −2.70 | 1165 / 870 | 9.0 / 17.4 | **1.1e-2** | **0** |

How to read the table:
- The C left after integration by parts is the 1D-like cancellation of λ′ ± lobes along z
  (it rises to 5–15 in the bunch head, where the wake crosses zero).
- **Shear 20 is the one case not solved** (§6).
- The past-exit difference, 4.9e-4, is larger than in the bend and has not been traced yet.
- Transverse integrand, C over nodes (shipped form): median 1.18 in the bend, 11.7 past the
  exit, 28 at shear 20. The longitudinal wake is the main problem, but the transverse one is not
  free of it past the exit or at strong tilt. It is not analysed further here.

**Scaling** (`B_scaling.txt`):

![scaling](pyDFCSR_2D/test/benchmark_results/cancellation_effects/B_scaling.png)

| σx0 µm (σz0 = 50; initial, uncorrelated) | 10 | 25 | 50 | 100 | 200 |
|---|---|---|---|---|---|
| σx projected at s = 0.5 m | 12.6 | 24.5 | 46.7 | 92.1 | 183.7 |
| σξ slice at s = 0.5 m | 12.6 | 24.0 | 43.5 | 72.1 | 98.0 |
| C shipped (centre) | 246 | 108 | 60.2 | 41.7 | 42.0 |
| C by parts (centre) | 1.81 | 1.14 | 1.16 | 1.39 | 2.00 |

| σz0 µm (σx0 = 50) | 25 | 50 | 100 | 200 |
|---|---|---|---|---|
| σz at s = 0.5 m | 32.1 | 53.8 | 102.0 | 201.0 |
| σξ slice at s = 0.5 m | 36.5 | 43.5 | 45.8 | 46.4 |
| C shipped (centre) | 53.3 | 60.2 | 85.4 | 130.3 |
| C by parts (centre) | 1.36 | 1.16 | 1.13 | 1.22 |

How to read the tables:
- The figure above (`B_scaling.png`) is plotted against the *projected* σx and the *final* σz.
  That is why it seems to flatten at σx ≥ 100 µm: the dispersion in the bend tilts those beams
  (τ = −0.9, −1.66), so the projected width grows while the slice width grows much less
  (72 and 98 µm).
- The σz scan's small-σz point also has a narrower slice (36.5 µm), which flattens the naive
  σz slope to +0.41.

**All cases against the slice width** (`H_collapse.txt`, centre node):

![collapse](pyDFCSR_2D/test/benchmark_results/cancellation_effects/H_collapse.png)

- Free fit: C ∝ σξ^−1.08 σz^+0.63 (the law: −1, +2/3).
- With one constant, C = 188·σz^(2/3)/σξ (µm, R = 1 m), every case lies within 0.81–1.21 of the
  law. That includes shear 20 (σξ = 8.5 µm, σz = 341 µm, C = 1165) and shear 2 (σξ = 103 µm,
  σz = 22 µm, C = 14).
- Left panel: the same data against the projected σx does not collapse.

---

## 4. What the cancellation costs, and what it does not

### 4a. Shot noise: no cost (`D_shot_noise.txt`, `E_injected_noise.txt`)

The two forms are **the same linear functional of the deposited ρ**, because the stored ∂ρ/∂x
is ρ's spectral derivative. So the noise in ρ reaches the wake identically through either
form. In the shipped form the noise in the two lobes cancels just as the signal does.

| particles (4 pseudo-random seeds) | 20k | 200k | 1M |
|---|---|---|---|
| wake scatter, shipped form | 2.61 % | 1.06 % | 0.47 % |
| wake scatter, by parts | 2.61 % | 1.06 % | 0.47 % |

Injected noise confirms the mechanism (gain = relative wake change per unit relative array
change). The noise is white with variance ∝ ρ, like shot noise, and is smoothed and
differentiated by the deposition's own `smooth_and_differentiate`:

| injected into | gain, shipped | gain, by parts |
|---|---|---|
| ρ only | 0.01 | 0.92 |
| ∂ρ/∂x only (Step 12's test) | 0.14 | 0.00 |
| ρ, with its own derivative in ∂ρ/∂x and ∂ρ/∂w (= real shot noise) | **0.92** | **0.92** |

Each form is insensitive to the array it does not use. The physically meaningful row, the
consistent one, is identical. The gain of 0.92 says the wake is about as noisy as the density
itself, which is the 1D-like λ′ sensitivity.

So **reformulating the integral cannot reduce shot noise.** Only better density estimation can
(Development Goal 1).

### 4b. Quadrature: the shipped form needs ~128 x′ nodes per band (`C_quadrature.txt`)

![quadrature](pyDFCSR_2D/test/benchmark_results/cancellation_effects/C_quadrature.png)

Relative L2 error of the dE cut against a 400-node reference:

| xbins | 12 | 16 | 24 | 32 | 64 | 128 | 200 (default) |
|---|---|---|---|---|---|---|---|
| bend, shipped | 1.4e-1 | 1.8e-1 | 1.3e-1 | 1.7e-1 | 1.8e-2 | 1.1e-6 | 7.8e-7 |
| bend, by parts | 1.4e-2 | 1.4e-3 | 1.2e-3 | 1.2e-3 | 1.1e-4 | 1.4e-5 | 2.0e-6 |
| thin, shipped | 2.1 | 1.1 | 1.1 | 0.92 | 0.11 | 7.4e-6 | 2.6e-7 |
| thin, by parts | 1.4e-2 | 5.7e-3 | 4.4e-3 | 4.9e-3 | 2.0e-4 | 4.5e-7 | 4.5e-7 |
| past exit, shipped | 1.3e-1 | 2.3e-1 | 2.8e-1 | 2.0e-1 | 3.1e-2 | 2.6e-4 | 2.9e-4 |
| past exit, by parts | 6.4e-2 | 1.4e-2 | 7.3e-3 | 4.4e-3 | 5.9e-4 | 3.1e-4 | 3.0e-4 |

What the numbers show:
- The trapezoid rule on a smooth, compactly supported integrand converges very fast once it
  resolves the beam. That is the cliff between 64 and 128 nodes.
- Below the cliff, the shipped form's error is the per-lobe error × C, so it stays at 10–100 %.
  The by-parts form's error is the same per-node error × ~1.
- **At the default 200 nodes both forms are converged.** The cancellation is affordable today
  only because of that node count. With the by-parts form, 16–24 nodes give 0.1–0.6 % in the bend and 0.7–1.4 % past the exit.
- Accounting for the extra evaluations of the prototype (D needs two more density evaluations
  per node), that is roughly 3× fewer density evaluations at 0.1 % accuracy. With analytic J and
  D it would be ~8×. Not yet timed.

### 4c. Robustness: errors that do not integrate to zero across the beam are amplified by ~C

The lobes cancel only if the stored ∂ρ/∂x integrates to exactly zero across the beam. An error
that is itself a derivative (a shift, a rescaling, a different smoothing width) also integrates
to zero and is not amplified. **An error with a non-zero integral is amplified by ~C.** Examples:

- a density step at the grid edge, which ∂ρ/∂x does not contain;
- derivatives blended separately from the field;
- noise added to the derivative array alone.

Measured examples:

| error | size | shipped wake moves | by-parts wake moves |
|---|---|---|---|
| stored fields zeroed at \|u\| > 3 (as a too-narrow grid does) | q = 0.30 % of charge, edge 1.3 % of peak | **3.0 %** | 0.19 % |
| zeroed at \|u\| > 3.5 | q = 0.054 % | 0.93 % | 0.031 % |
| zeroed at \|u\| > 4 | q = 0.0074 % | 0.38 % | 0.0071 % |
| one 20k seed with an outlier past 5σξ | edge 5e-4 of peak | 0.28–0.54 % apart | — |
| 1M pseudo-random particles, any seed (presumably outliers at the ±5σ edge; not checked) | | 1–3e-4 apart (Hammersley: 2e-6) | — |
| uniform injected noise ending in a step at the grid edge | | gain 3.85 | gain 0.92 |

(`F_clipped_charge.txt`; the shear-0.5 case gives the same numbers to 10 %.)

How to read this:
- Removing a fraction q of the charge should change the wake by ~q, and the by-parts form does
  exactly that.
- The shipped form instead changes by ~C·(edge density) ≈ 10q: a density without the clipped
  tails, but also without the matching delta function at the cut.
- Neither form is "right" for a clipped density. The by-parts form is at least the wake of the
  clipped density. The shipped form computes the wake of something that is not a density.

These connect to known defects:

- **BC20 multi-level smoothing** (`progress_facet_bc20.md`): blending each level's own
  derivatives left them off by Σ f_k ∇w_k. That is an error with a non-zero x-integral, and it
  gave a 20k-particle thin beam a wake of **−3.9× the 1D value**.
- **±3σz spikes in B2/B3** (`integration_region.md` §8): 0.28 % of the charge clipped out of the
  u grid made a spike of 24 % of the peak, consistent with the ~10–100× amplification here.
- **Shear 20 at xlim 5**: the u-edge holds 2e-4 of the peak. The shipped form differs from the
  xlim-8 result by 1.5 %, the by-parts form by 0.5 %. The reference is the mean of the two forms
  at xlim 8, which themselves differ by 1.1 % (§6, M1).

**The by-parts total is also a free consistency monitor.** Wherever it disagrees with the
shipped total by more than the quadrature error, ρ and ∂ρ/∂x are inconsistent somewhere and C
is amplifying it.

---

## 5. Revisions to Step 12 (`progress_comoving_interp.md`)

1. **"Noise in ∂ρ/∂x reaches the wake with ~100× the gain of noise in ρ."** True for noise
   injected into ∂ρ/∂x *alone*. That noise is not a derivative, so it does not cancel. Real shot
   noise reaches ρ and ∂ρ/∂x consistently and has the same gain (0.92) in either form. **The
   cancellation does not amplify shot noise** (§4a). It amplifies quadrature error and
   inconsistency (§4b, §4c).
2. **"Integration by parts is NOT the cure."** Step 12 tried the 2D form ∫K·∇′ρ = −∫(∇′·K)ρ. It
   correctly noted that K·∇′ρ is a single directional derivative, so that form does not close.
   The column form of §2 does close, because it integrates along the path the retarded point
   actually takes, which includes the z_ret and t_ret motion. It is exact and removes the
   pointwise cancellation (C 60 → 1.2).
3. **"Since L_f ∝ σz^(1/3), slope −0.76 against L_f/σz is +1 against σz."** Arithmetic slip:
   −0.76 × (−2/3) = **+0.51**, and Step 12's own table gives σz^0.50. Against the slice width
   and the final σz the law is σz^(2/3)/σξ (fit +0.63, −1.08).
4. **Proposed control variate.** It would reduce the quadrature cost for the Gaussian part only,
   and does nothing for shot noise (point 1). §6 compares it with integration by parts.

---

## 6. Methods to fix it — discussion

| method | removes the pointwise cancellation? | shot noise | robustness to inconsistency | x′ nodes for 0.1 % | general? | effort |
|---|---|---|---|---|---|---|
| status quo: 200 x′ nodes | no, resolves it | unchanged | ×C | ~128 | yes | 0 |
| **M1 column integration by parts** (prototyped) | yes, C → 1–3 | unchanged | ×1 | 16–24 | \|τ\| < ~1 cleanly; partition needed beyond | medium |
| M2 integration by parts along ∇ξ in 2D | yes | unchanged | ×1 | expected similar | yes, incl. J = 0 | high |
| M3 Gaussian control variate | only for the fitted Gaussian | unchanged | partial | depends on non-Gaussianity | weak for spikes, horns, BC20 | medium |
| M4 remove the inconsistency sources | no | unchanged | removes the cause | ~128 | case by case | low each |
| M5 better density estimation | no | **reduced** | — | — | yes | Goal 1 |

**M1 — integration by parts along s′ columns (§2).** It is the most direct fix: the lobes are
subtracted analytically, so nothing has to cancel numerically.
- Good: exact; verified to 1e-6–1e-5 in every case with |τ| < ~1. It reuses the shipped nodes,
  bands and quadrature, so it can be switched on per region. It needs only ρ (plus a derivative
  along the beam), not ∂ρ/∂x.
- Bad: 1/J. Where a column runs along a tilted ridge (beam tilt |τ| > 1, which in a bend means
  σx·R51 ≳ σz), J crosses 0. The partition of unity fixes the singularity (σx 200 µm: 1.8e-4).
  But the χ and 1 − χ shares are themselves large and opposite near the observer, so it creates
  a local cancellation of its own. At shear 20 the forms differ by 1.1 %; resolving it needed
  2000 nodes on one column. Shear 20 is also where the shipped form's C reaches ~1000, so
  neither form is clean there.
- To ship it: analytic J and D (D from the stored ∂F/∂w and the frame's time derivative), a
  per-region switch, the totals agreement as a runtime check, then timing against the default at
  matched accuracy.

**M2 — the 2D version.** The general statement is to integrate by parts along the direction e
in the (x′, s′) plane that crosses the beam's transverse coordinate ξ: e·∇ξ = 1 and e·∇w = 0.
- Then ∂ρ/∂ξ = e·∇ρ_ret − (time term), and ∫G ∂ρ/∂ξ = −∫ρ ∇·(Ge) + boundary terms.
- The column choice e ∝ x̂′ is M1, singular at J = 0. Where J → 0 the beam is crossed by moving
  in s′ instead, and e turns toward ŝ′. ∇ξ vanishes only if both J = 0 and ξ_s′ = 0, which
  requires τ(1 + R_s′) = 0.
- Cost: boundary terms on the s′ boundaries of each region, where ρ ≠ 0 (far/near seams, causal
  edge), and a 2D divergence on meshes whose columns move. This is the general cure for strong
  chirp, the regime of the known open |τ| ≫ 1 problem (`integration_region.md` §8). Worth doing
  only if strongly tilted bunches matter.

**M3 — control variate (Step 12's proposal).** Write ρ = ρ_G + δρ, with ρ_G the fitted
Gaussian and its wake integrated analytically, or on a fine grid once.
- It reduces the cancelling amplitude by the Gaussian's share, but δρ keeps C in full.
- It does nothing for shot noise.
- It is weakest exactly where the code is now used hardest: BC20 spikes and horns, curved
  bunches.

M1 does everything M3 would, for any shape. Not recommended.

**M4 — remove the inconsistency sources directly.** This is needed whatever is done to the
integral, and it is cheap item by item:
- grid coverage (xlim large enough that the edge density is < 1e-6 of the peak, or a soft edge
  that goes to zero smoothly);
- derivatives always taken *after* blending (already done for the multi-level smoothing);
- the same rule for any future windowing or adaptive smoothing.

The M1 total is the detector: a run-time warning whenever |W_by-parts − W_shipped| exceeds the
quadrature tolerance would have caught the BC20 bug and the B2/B3 spikes.

**M5 — shot noise is a separate problem.** The wake's noise is set by the functional
(gain ≈ 0.92 relative to ρ, the λ′ sensitivity of 1D CSR), not by how the integral is written.
At 200k particles it is ~1 %, at 1M ~0.5 %. Only density estimation can reduce it.

**Recommendation.**
1. Adopt M4 now. Add the M1-vs-shipped agreement as a diagnostic (`CSR_integration.check_ibp`,
   evaluated on a few mesh points per kick).
2. Develop M1 into an option for the far and near regions with analytic J and D. Measure the
   speed-up at matched accuracy, where the expected gain is a 5–10× reduction in x′ nodes.
3. Keep the shipped form in the polar patch and wherever min|J| < J0.
4. M2 only if strongly tilted bunches (|τ| ≳ 2) become a target.

---

## 7. Files and reproduction

```bash
conda activate pydfcsr
cd pyDFCSR/pyDFCSR_2D/test
python cancellation_effects.py            # all sections; or e.g. `python cancellation_effects.py A C`
```

- `test/cancellation_lib.py`: case builder (Gaussian beam in the R = 1 m dipole; inputs
  written under `benchmark_results/cancellation_effects/inputs/` with absolute paths), and the
  shipped quadrature opened node by node. The node sums are asserted equal to `get_CSR_wake`.
- `test/cancellation_ibp.py`: the integration-by-parts prototype (§2).
- `test/cancellation_effects.py`: sections A–H.
- `test/benchmark_results/cancellation_effects/`: `A_anatomy.txt`, `B_scaling.txt`/`.png`,
  `H_collapse.txt`/`.png`,
  `C_quadrature.txt`/`.png`, `D_shot_noise.txt`, `E_injected_noise.txt`,
  `F_clipped_charge.txt`, `G_integrand_map.png`.

Runtime: A–C ~10 min, D ~15 min (1M particles), E–G ~2 min, single process.

**Not yet done:**
- the chicane and BC20 states (compressed, curved, quantile frame);
- timing of M1;
- the transverse integrand's past-exit cancellation (C ~ 12).
