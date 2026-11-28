# GPU and differentiable pyDFCSR (branch `gpu_diff`)

Plan: PyTorch, float64 everywhere; develop on the Mac CPU, production on S3DF NVIDIA GPUs (MPS has no
float64, and the ±∂ρ/∂x cancellation of the integrand needs it); gradients end-to-end with respect to
lattice/beam parameters and the initial particle coordinates. The numba path stays the reference.
Phases: P1 torch wake evaluator, P2 layout in torch, P3 differentiable pieces (deposit, geometry,
kicks, tracking), P4 end-to-end gradients, P5 S3DF.

## 1. P1: the wake evaluator in torch (2026-11-27)

**How it works.** A wake-mesh point's integral is a sum over quadrature nodes, and the nodes are
decided before any integrand is known, except for the x′ refinement, which looks at integrand
values. So the work splits in two:

- **Layout** (numpy, unchanged code): `WakeMixin.wake_layout(s, x)` returns, per s′ region, the s′
  nodes, the two disjoint x′ bands per column, the radial taper and whether the refinement applies,
  plus the polar-patch radii. It is the first half of `get_CSR_wake`, which now calls it. The numba
  path still evaluates each region as before and stays bit-identical (golden fixture, 208 tests,
  MPI = serial).
- **Evaluation** (`torch_backend/`): all live columns of all mesh points are packed into flat
  tensors (point, group, lo, width, s′, s′-trapezoid weight, taper). They are expanded to nodes in
  chunks of 2²¹, and the integrand is evaluated in one batched pass:
  - `interp.py` is a port of `interpolate3D_comoving_fields` (time bracket, frame blend, curvature
    shear, 4×4 B-spline taps on the two bracketing snapshots, all five fields per gather);
  - `integrand.py` ports `csr_integrand_kernel` and the geometry of `get_CSR_integrand`.

  Per-column sums go to points by `index_add`. The refinement repeats `_adapt_x_error` for all groups
  at once: each round evaluates the midpoints of the active columns, and the greedy selection runs in
  numpy, vectorised over the groups.

Selected by `CSR_computation.backend: torch` (`device`, `torch_chunk`). The MPI split works the same.
`ibp` is not ported (it is a diagnostic option and raises).

**Agreement with numba** (relative to the mesh's largest value):

| case | W | F | note |
|---|---|---|---|
| golden case, each kick on the same state | 4e-14 – 1e-13 | 5e-16 – 1e-14 | |
| golden + `near_adapt`, `near_adapt_x`, 40 x′ nodes | 1e-14 – 3e-14 | 5e-16 – 8e-15 | refinement reproduced |
| BC20 gain beam, s = 1.25 m (48 points) | 2e-13 | 2e-14 | |
| BC20 loss beam, s = 49.0 m (96 points) | 2e-15 | 5e-14 | |
| BC20 L1-17.6284 beam, s = 27.0 m (192 points) | 7e-7 | 4e-12 | 16 points |

The 7e-7 is a refinement choice. With the refinement off, the same state agrees to 6e-11, and with
one round to 3e-11. In rounds 2–4 a few columns sit at the selection threshold, and the round-off of
their error estimates flips them. The difference is bounded by `x_adapt_tol` (1e-3 of the region's
gross sum), so it is a tie, not an error.

Per point the integrand itself agrees only to ~1e-8 of its maximum, because at random source points
it is a difference of terms up to ~1e7 × larger (the ±∂ρ/∂x cancellation); the fields agree to
1e-15. The integrated wakes agree to 1e-13.

**Time on the Mac CPU (torch, 6 threads) against serial numba:**

| state | points | numba | torch layout (numpy) | torch evaluation |
|---|---|---|---|---|
| gain, 1.25 m | 48 | 1.8 s | 0.4 s | 6.5 s |
| L1-17.6284, 27.0 m | 192 | 7.9 s | 2.7 s | 22.3 s |
| loss, 49.0 m | 96 | 5.1 s | 1.1 s | 16.8 s |

The torch evaluation on CPU is 3–4× slower than numba, as expected: it is gather- and memory-bound,
and the GPU is the target. What matters is the **layout: 22–34% of numba's whole serial time**. On a
GPU with one CPU process it would run serially, against numba split over 6 ranks. A full BC20 kick
(1920 points) has ~27 s of layout against ~13 s for the whole numba kick at 6 ranks. So the layout has
to move to the device as well (P2), or be split over ranks feeding one GPU.

Tests: `test/test_torch_backend.py` (in conftest's list; skipped without torch). Whole golden runs
with both backends: every kick to 1e-8 (the kicks feed back into the beam), kernels on random
queries.

## 2. P2 (first part): the layout batched over mesh points (2026-11-27)

**Why the layout was slow.** On the L1-17.6284 state at 27.0 m, 88% of the layout time was the band
solve `_retarded_xi_bands`. It ran about 1,900 times for 192 points (~10 per point): the causal-edge
scans, the near-adapt column scans and the final bands of each region. Each call works on a few
hundred columns, so its cost was Python and numpy call overhead, not arithmetic.

**What changed.**
- The band solver now accepts one observer per column. Every operation in it is elementwise, so
  each column gets the result it would get with its observer alone.
- The layout functions that need band solves (causal edge, column scan, near-adapt nodes, far, near
  and extra regions, `wake_layout`) are generators that *yield* a `BandRequest` and receive the bands
  and branch records.
- `LayoutMixin._drive` answers the requests of one generator one by one. The plain methods use it,
  with the same calls in the same order as before.
- `_drive_many` runs the generators of many observers in lockstep and answers all pending requests
  with the same (n_iter, scale_bands) in one concatenated solve.
- The near-adapt grading recursion (two Python passes, ~300k `max` calls per 192 points) moved to a
  numba function.

**Results.** Layouts are bit-identical: every node and band edge of 192 + 96 points on two BC20
states (`array_equal`), the golden fixture, and MPI = serial.

| | layout of 192 points (L1-17.6284, 27.0 m) |
|---|---|
| per point (P1) | 3.1 s |
| batched final solve only | 1.8 s |
| batched scans + final solves | 0.76 s |
| + numba grading loop | 0.54 s, of which 0.36 s is the 14 band solves' arithmetic |

The numba path uses the batched layout too (`CSR2D._numba_wakes`, 256 points per batch), still
bit-identical. Whole wake meshes on four BC20 states, measured in one process each, are faster by:

| state | points | per-point layout | batched layout | speed-up |
|---|---|---|---|---|
| L1-17.6284, 31.3 m | 480 | 19.5 s | 14.4 s | 1.36× |
| L1-17.6284, 49.1 m | 480 | 48.0 s | 44.5 s | 1.08× |
| loss, 49.0 m | 240 | 12.0 s | 9.8 s | 1.23× |
| gain, 1.25 m | 120 | 4.4 s | 3.5 s | 1.26× |

For the torch backend the layout is now ~10% of numba's serial time, down from 22–34%. What remains
is mostly the band-solve arithmetic, which can move to the device (P2c).

## 3. P3: the differentiable run (2026-11-27)

**How it works.** `torch_backend.diff.CSR2DDiff` is CSR2D with a torch shadow of the beam. The numpy
CSR2D still makes every structural decision, on the detached values of the torch beam, and is
re-synchronised to them after every step:
- the schedule and element pieces;
- history retention and thinning;
- the deposition's fit masks;
- the wake mesh;
- the s′ layout, x′ bands and refinement choices.

The torch side recomputes every value with gradients:
- **Tracking:** bmadx on torch particles. Element parameters can be tensors (`params` =
  {(element, key): tensor}, key K1, K2, angle, E1, E2). p0c and mc2 are tensors as well, because the
  quadrupole's z correction takes `sqrt` of them.
- **Deposition:** `torch_backend.deposit.get_DF_rms`, a port of `get_DF` for the rms frame.
  - The cubic B-spline deposit is a custom autograd Function: its backward is the B-spline gather of
    the adjoint grid with B3′, and it saves only the coordinates.
  - The FFT smoothing and spectral derivatives use `torch.fft`.
  - The tilt, curvature and vx fits use the normal equations. The fit masks are detached.
- **Wake:** the P1 evaluator on a history built from the snapshot tensors (`history_from_tensors`).
  The observer vx comes from the torch interpolant.
- **Kick:** a torch port of `RegularGridInterpolator`'s bilinear lookup. Particles are mapped into
  the mesh with the detached tilt fit and ridge, evaluated at the differentiable coordinates. That
  is the consistent derivative of "kick = W at the particle", with the mesh nodes fixed in absolute
  coordinates.
- **Geometry:** with an angle parameter, the reference-orbit table and the element curvatures of
  the CSR integral are rebuilt in torch from cumulative arcs (`Orbit.from_lattice_params`). It
  agrees with `get_reference_traj` to 1e-14.

`structure=` replays a previous run's per-kick structure: the mesh, the kick-lookup frame and the
layout. With it, a finite difference sees exactly the map the gradient differentiates. Without it,
moving mesh and quadrature nodes add discretisation-level terms that the gradient, taken at fixed
structure, correctly leaves out.

**Forward.** On the golden case CSR2DDiff equals CSR2D to 4e-16 in the final beam.

**Gradients against central differences, structure replayed** (loss = 1e6·⟨pz⟩ + 1e3·σ_px of the
final beam):

| case | derivative | AD vs FD |
|---|---|---|
| golden (B1 + D1), 20k particles | random direction in all 6×20k initial coordinates | 2.4e-7 (h = 1e-6) |
| golden | bend angle (tracking only) | 1.3e-7 |
| D0, Q1, B1, S1, D1 | bend angle (tracking + CSR geometry) | 3.2e-6 (h = 1e-5·angle) |
| same | quad K1 | 6.1e-7 |
| same | sextupole K2 (a 1e-7 derivative) | 4.5e-5 (round-off limited) |

The finite differences converge to the AD values only linearly in h before round-off takes over
(coordinates: 1.8e-4 at h = 1e-4, 4.3e-5 at 1e-5, 2.4e-7 at 1e-6). The kick lookup is piecewise
bilinear, and particles cross mesh cells within ±h. The AD value is the derivative of the smooth
pieces.

**Cost** (golden, 20k particles, 8 kicks, Mac CPU): forward 17–19 s with the graph (9 s without),
backward 52–55 s. Memory was not an issue at this size. Checkpointing (plan §4) is needed for long
lattices and 1e5+ particles.

**Not yet differentiable:** windowed frames (quantile/peak: the window weights would be detached,
the rest ports as for rms), element lengths (they move the schedule), MPI.

Tests: `test/test_torch_diff.py`: deposition vs numpy plus `gradcheck`, orbit table vs numpy,
forward vs CSR2D, and one end-to-end directional derivative in (angle, K1, coordinates) against
central differences (1e-4).

## 4. Windowed frames and checkpointing (2026-11-27)

**Windowed frames (quantile, peak; BC20 needs them).** `get_DF_windowed` ports `_get_DF_quantile`.
Four things are taken from the numpy deposit of the same particles and treated as per-snapshot
structure:
- the window (lo, hi and the margins);
- the frame it implies (z̄, σ_z);
- the z smoothing width;
- the adaptive-smoothing blend weights (now kept on the tracker as `adaptive_levels`).

The window weight T(z) and everything else are differentiated. The T-weighted fits use the normal
equations with weights T, not √T rows, so the derivative stays finite where T = 0. The observer vx
blends the windowed source with the beam-wide fit as `CSR2D._observer_vx` does, with that fit
computed on the torch beam. The torch deposit matches numpy to ≤ 4e-14 for quantile, peak
(`smoothing_res_fwhm`) and adaptive smoothing.

Two things the comparison caught:
- **The kick lookup's ridge.** `CSR2D._mesh_ridge` holds w at the grid edge in windowed frames, and
  the torch kick did not. Tail particles then got a slightly different mesh row, which gave 6.6e-11
  in pz after the first kick. Fixed: the windowed forward run equals CSR2D to 4e-16.
- **Refinement ties under finite differences.** The x′ refinement chooses columns from integrand
  values. A finite-difference run moved those choices at the ties, so the error grew as h shrank
  (3.6e-3 at h = 1e-5, 1.4e-2 at 1e-6). The active columns of every round are now part of the replayed
  structure.

  | case (golden, `near_adapt` + `near_adapt_x`) | AD vs FD, random direction in all coordinates |
  |---|---|
  | quantile frame + charge mesh | 4.4e-7 (h = 1e-5), 5.4e-8 (h = 1e-6) |
  | peak frame | 4.1e-8 (h = 1e-6) |

**Checkpointing** (`CSR2DDiff(..., checkpoint=True)`). Each tracking step and each wake chunk is
recomputed in the backward pass instead of keeping its graph. A chunk keeps only its column range,
not the ~50 saved doubles per integrand node.

The first version gave a wrong gradient (dL/dangle −32.3 instead of −372.1): the recompute read the
observation time from the beam, which by then had moved on. The time is now bound to each chunk.
The gradients are then identical, and the backward pass is faster, since it no longer manipulates a
huge graph:

| golden, 20k particles, 8 kicks | forward | backward | peak RSS | dL/dangle |
|---|---|---|---|---|
| no checkpointing | 17.1 s | 55.1 s | 25.8 GB | −372.13299567 |
| checkpointing, chunk 2²¹ | 9.4 s | 20.4 s | 9.4 GB | −372.13299567 |
| checkpointing, chunk 2¹⁹ | 9.1 s | 20.2 s | 4.2 GB | same |
| checkpointing, chunk 2¹⁷ | 10.9 s | 23.0 s | 2.0 GB | same |

The forward pass alone peaks at 2.9–3.3 GB, with or without the graph. For gradient runs, use
`checkpoint=True` and `torch_chunk: 524288`.

Tests (`test_torch_diff.py`):
- windowed deposits against numpy;
- the forward run against CSR2D in the rms frame and in the quantile frame with a charge mesh;
- the end-to-end finite-difference check, now checkpointed;
- checkpointed and plain gradients equal to 1e-12.

## 5. Chicane gradient demo (2026-11-27)

`test/gpu_diff/chicane_gradients.py`: the 4-dipole benchmark chicane in a coarse setup that runs on
the Mac CPU:
- 20k particles from `chicane_init_beam.yaml`;
- 100 × 150 deposition, an 8 × 24 wake mesh, 60 x′ nodes;
- the auto schedule: 489 nodes, 105 kicks;
- numba forward: 174 s.

One angle θ drives all four bends and their rectangular edges (B1, B4: +θ; B2, B3: −θ). With the
differentiable run the CSR geometry follows θ as well. The chirp derivative (pz → pz + dh·z) comes
from the coordinate gradient, Σ_i ∂L/∂pz_i z_i, so it needs no extra run.

| quantity | value | ∂/∂θ (AD) | ∂/∂h (AD) |
|---|---|---|---|
| ⟨ΔE⟩ (CSR energy loss) | −2777.67 keV | −3.5423e6 keV/rad | 2.4448e3 keV·m |
| Δε_x (projected, dispersion removed) | 0.0525 nm | 46.030 nm/rad | −3.034e-2 nm·m |

Cost on the Mac CPU (torch, checkpointed, chunk 2¹⁹): forward 481 s, both backward passes 2858 s.

**Checks against central differences, structure replayed.** The structure now includes the
history's snapshot set per kick. Retention and drift thinning are decided on the beam, so a
replayed run must use the recorded set. A direct test showed it never differed here: at θ(1 ± 1e-4)
the replayed run's own selection equalled the record at all 105 kicks. So it is a safeguard, not
the cause of the first, poorer agreement.

| step | d⟨ΔE⟩/dθ | dΔε/dθ | d⟨ΔE⟩/dh | dΔε/dh |
|---|---|---|---|---|
| 1e-4 θ, 1e-3 σ_pz/σ_z | 6.3e-3 | 2.3e-2 | 4.1e-2 | (−0.0318 vs −0.0303) |
| 1e-5 θ, 1e-4 σ_pz/σ_z | 2.2e-5 | 1.3e-3 | 9.2e-3 | 5.3e-2 |
| 1e-6 θ, 1e-5 σ_pz/σ_z | 2.0e-4 | 2.3e-4 | 2.9e-3 | 4.4e-2 |

- **θ:** the finite differences converge onto the AD values. The error at h = 1e-4 was
  finite-difference error, not a missing term.
- **Chirp:** the finite differences of ⟨ΔE⟩ close in on the AD value. Those of Δε scatter around it
  on both sides (−0.0288, −0.0317, −0.0318 against −0.0303). That is the derivative of a 0.05 nm
  growth against a chirp that changes the compression, and at these steps it is limited by
  particles crossing kick-mesh cells. The initial emittance does not depend on the chirp
  (dε₀/dh = −4e-13).

**Bug found:** an integer element parameter (`E2: 0` in the chicane YAML) was passed to bmadx as a
Python int, and `tan()` refuses ints for torch particles. Every numeric parameter except NUM_STEPS
is now a tensor.

**GPU:** `environment-gpu.yml`, `test/gpu_diff/gpu_benchmark.py` and `gpu_benchmark.slurm` are
ready for an S3DF A100 node. Not run yet.

## 6. First GPU runs: RTX 5000 Ada workstation (2026-11-27)

**Setup.**
- Machine: Ubuntu 24.04, NVIDIA RTX 5000 Ada (32 GB), 32 CPU cores.
- Environment: Miniforge in `~/miniforge3`, env `pydfcsr-gpu` from `environment-gpu.yml` (torch 2.14.1, CUDA 13.4), repo installed editable (`pip install -e .`).
- This card's float64 is 1/64-rate: measured 0.90 TFLOP/s against 25.6 for float32 (4096² matmul).

**Tests on x86 Linux.**
- The golden fixture was made on the ARM Mac. Here it agrees to 4e-12 relative, not bit for bit: macOS libm and numba's arm64 code differ in the last bits. `test_defaults_bit_identical` is now exact on Darwin/arm64 and checks 1e-10 elsewhere.
- The whole-run torch-vs-numba test with the x′ refinement reached 4.3e-7 here: round-off decides the threshold ties, bounded by `x_adapt_tol`. Its tolerance is now 1e-5 for the refinement case and still 1e-8 without it.
- All other tests pass, and the new CUDA tests (plain and compiled) agree with numba to 1e-8.

**Why the plain GPU evaluation was only 4× one CPU core.** Scaling the chicane wake mesh from 6M to
113M integrand nodes left the GPU evaluation at a constant 4.3× serial numba, about 23M nodes/s. So it
is not launch overhead, and not float64 arithmetic either: about 800 operations per node is only
~20 GFLOP/s. Each of the ~300 elementwise torch operations per node writes a chunk-sized array, and
the 160-tap B-spline gather reads scattered addresses: the evaluation is bound by memory traffic.

`CSR_computation.torch_compile: true` fuses the integrand (history interpolation and kernel) with
`torch.compile`, and this gave 10×. Notes:
- a data-dependent branch in `Orbit.vectors` (`if before.any()`) broke the graph; it is now an
  unconditional `torch.where`;
- triton needs a C compiler and the CUDA headers (`c-compiler`, `cuda-driver-dev`,
  `cuda-cudart-dev`, now in `environment-gpu.yml`; `export CC=$CONDA_PREFIX/bin/gcc`);
- the first calls in a process compile (25–110 s, cached on disk afterwards).

**Benchmark** (`test/gpu_diff/gpu_benchmark.py`; chicane wake mesh in B2, 20k particles, 192 points,
106k columns; golden differentiable run, 20k particles, 8 kicks, checkpointed):

| | numba (1 core) | torch CPU (16 threads) | CUDA | CUDA compiled |
|---|---|---|---|---|
| wake mesh, total | 1.8 s | 8.05 s | 0.60 s | **0.28 s** |
| … of which evaluation | ~1.6 s | 7.8 s | 0.4 s | 0.04 s (40× one core) |
| … of which layout (numpy, CPU) | | 0.2 s | 0.2 s | 0.2 s |
| agreement with numba | | 9.6e-11 | 5.5e-10 | 7.2e-10 |
| differentiable golden: forward / backward | — | 6.5 / 20.4 s | 2.1 / 16.6 s | **1.8 / 2.8 s** |
| gradient vs CPU | | | 9e-11 | 9e-11 (dL/dangle), 6e-10 (coordinates) |

Larger meshes, plain CUDA (12×40 points with 100 x′ nodes: 26M nodes; 16×64 with 200: 113M): 4.3×
one numba core in the evaluation at every size. Compiled, on the 12×40 mesh: evaluation 1.30 → 0.13 s.

**Where it stands.**
- Gradients: a differentiable run on this GPU is 7× faster than torch on the CPU in the backward
  pass, 2.8 s against 20.4 s on the golden case. That is the case the GPU is for.
- Forward-only wakes: the compiled evaluation (40× one core) is no longer the cost; the numpy layout
  on the CPU is (0.2 of 0.28 s). numba with ~16 MPI ranks on this 32-core machine (~0.11 s,
  extrapolated linearly) is still faster than one GPU process. Next: the band solve, which is most of
  the layout, in torch on the device (P2c), or the layout split over CPU processes feeding the GPU.

## 7. The chicane gradient demo on the GPU (2026-11-27)

`chicane_gradients.py grad 20000 --device cuda --compile` on the RTX 5000 Ada (the coarse chicane of
§5: 20k particles, 489 steps, 105 kicks):

| | Mac CPU (torch, 6 threads) | RTX 5000 Ada, compiled |
|---|---|---|
| forward, differentiable | 481 s | **99 s** |
| both backward passes (⟨ΔE⟩ and Δε) | 2858 s | **227 s** (including the first-run compiling) |
| 4 finite-difference forward runs | ~40 min (2 in parallel) | ~7 min |
| whole job | ~2 h | 12.2 min |

For comparison, the serial numba forward run (no gradients) takes 236 s on this machine and 174 s on
the Mac. So one differentiable GPU forward run is 2.4× faster than a plain numba run on one core
here, and it gives the gradients.

**Results** agree with the Mac's.

| quantity | GPU | Mac (§5) |
|---|---|---|
| ⟨ΔE⟩ | −2777.05 keV | −2777.67 keV |
| d⟨ΔE⟩/dθ | −3.54216e6 keV/rad | −3.54234e6 |
| d⟨ΔE⟩/dh | 2.44476e3 keV·m | 2.44479e3 |
| dΔε/dθ | 46.037 nm/rad | 46.030 |
| dΔε/dh | −3.0348e-2 nm·m | −3.0336e-2 |

The beam itself differs slightly between the machines: distgen draws the same 20k particles to
different last bits on ARM and x86, and the initial ⟨pz⟩·E0 here is −0.026 keV.

Finite differences at the same steps as on the Mac give the same agreement: d⟨ΔE⟩/dθ 1.9e-5,
dΔε/dθ 1.3e-3, d⟨ΔE⟩/dh 9.2e-3, dΔε/dh 5.3e-2.

**GPU against numba over the whole chicane.** numba gives ⟨ΔE⟩ = −2776.91 keV here, 5e-5 below the
GPU's −2777.05 keV. Each kick's wake agrees to ~1e-9 (§6). Over 105 kicks the two runs see
round-off-different beams, and the history's drift thinning (tolerance 1%) and retention are
thresholds that can flip on round-off. Not verified; on the Mac the CPU torch run was 1e-5 from
numba.

**Fix.** A compiled backward pass reuses ("donates") its buffers, so it allows only one backward per
forward. The demo takes two gradients from one forward (`retain_graph=True`), which failed. With
`torch_compile` the evaluator now sets `torch._functorch.config.donated_buffer = False`.

## 8. Parameterised initial beams (2026-11-27)

**Why.** A gradient with respect to single particles is not useful:
- 6N unknowns;
- each particle moves the collective field by O(1/N);
- the derivative is only piecewise smooth, because the kick lookup is bilinear and the structure is
  held fixed.

The useful gradient is with respect to parameters p of the beam's **distribution**: current profile,
chirp, emittance. `torch_backend/beam_param.py` keeps the randomness fixed (a given beam, or draws
made once) and builds the particles as a differentiable function x_i = g(ξ_i; p). The loss is then a
smooth, deterministic function of p, and dL/dp = Σ_i ∂L/∂x_i ∂x_i/∂p averages over all particles.
The per-particle gradient remains only the inner link of the chain rule.

| class | what it is | parameters |
|---|---|---|
| `BeamTransform` (a) | a fixed beam (file, distgen, upstream code) with physical knobs; identity at the defaults | `z_shift`, `z_scale`, `profile_knots` (a monotonic map φ of z: φ(w) = −W + 2W·C(w)/C(W), C = ∫exp(q), q piecewise linear in 9 knots on ±4σ_z, so horns move, grow or shrink while every correlation of the beam is kept), `espread_scale` (about the beam's linear pz(z)), `chirp`, `chirp2` |
| `BeamFromParams` (b) | a beam from physical parameters on seeded fixed draws | Twiss and geometric emittance in x and y; profile `gaussian` (σ_z) or `spline` (log-current knots, sampled by inverse CDF, starting from the Gaussian); `chirp`, `chirp2`, `sigma_pz`, `pz_mean` |
| `NeuralBeam` (c) | particles = mean + scale·(ξ + MLP(ξ)), with fixed 6D normal draws ξ | the MLP weights (last layer zero-initialised: the start is the uncorrelated Gaussian of the given rms sizes) |

Usage:
```python
t = BeamTransform(input_coords)                  # (6, N) of the fixed beam
d = CSR2DDiff(input_file=cfg, coords=t.beam(), checkpoint=True); d.run()
loss(d.tp).backward()                            # -> t.params['chirp'].grad, ['profile_knots'].grad ...
```
`t.beam(p={...})` overrides knobs, for finite differences or line searches. The structure
(schedule, windows, mesh, layout) is decided from the beam the run sees. When an optimisation moves
the beam far, rerun without `structure=` to rebuild it.

**Tests** (`test/test_beam_param.py`):
- **identity:** `BeamTransform` returns the input beam to round-off;
- **knobs:** `z_scale` and `chirp` act exactly, and the profile map is monotonic (it keeps the
  particles' order);
- **statistics:** `BeamFromParams` reproduces ε, β, α, σ_z and the chirp to ≤ 2% on 2e5 particles;
- **`gradcheck`** of all three generators;
- **end to end:** the gradient of the golden run's CSR loss with respect to a direction in (chirp,
  z_scale, one profile knot), through `CSR2DDiff`, agrees with central differences to 1e-4
  (structure replayed).

Full suite: 223 passed.

**Fix:** a local `import torch._functorch.config` inside `TorchWake.__init__` made `torch` a local
name for the whole function, so the uncompiled path raised UnboundLocalError. It now imports
`from torch._functorch import config`.

## 9. P2c: the band solve on the GPU (2026-11-28)

**Why.** With the compiled evaluation the CPU-side layout was most of a wake mesh's time: 3.09 of
3.59 s on a 1920-point mesh. Two thirds of the layout is the band solve: the fixed point through the
retarded frame, Eq. 4.24 and the band widths. It is elementwise arithmetic over all columns of all
observers, batched since P2a into a few calls per mesh, so it suits the GPU.

**How.** `torch_backend/bands_t.BandSolver` ports `_retarded_xi_bands` line by line, with
`_comoving_frame_at` (and its one-snapshot history case), `_eq424` and `_curv_shift_at`. It uses
the lattice file's geometry, like the numpy layout. `LayoutMixin._drive_many` calls it when
`sim._band_solver` is set. `TorchWake` sets that only inside its own layout pass, when
`CSR_computation.torch_bands: true`, so the numba path is untouched. The rest of the layout (node
construction, the scans' density lookups) stays numpy.

**Same layouts.** Node counts are identical at every point; nodes and band edges agree to round-off:

| state | points | node counts identical | max rel diff | numpy layout | GPU bands |
|---|---|---|---|---|---|
| chicane B2, 8×24 | 192 | 192/192 | 5.9e-12 | 0.23 s | 0.08 s |
| chicane B2, 12×160 | 1920 | 1920/1920 | 1.4e-11 | 2.57 s | **0.29 s** (8.8×) |
| golden, quantile frame + `near_adapt` + x′ refinement, 12×160 | 1920 | 1920/1920 | 1.6e-13 | 5.07 s | 1.58 s (3.2×) |

With `near_adapt`, what remains is Python that runs once per point: the near-adapt node construction
(1.2 s of 1.58) and the per-request bookkeeping of the batched driver. Vectorising the near-adapt
construction over points would be the next step there.

**Whole wake mesh** (chicane B2, 12×160 = 1920 points, 100 x′ nodes):

| | time | vs one numba core | vs numba |
|---|---|---|---|
| numba, 1 core | 23.4 s | 1× | |
| GPU, compiled, numpy bands | 3.59 s (layout 3.09, eval 0.50) | 7× | 7.8e-10 |
| GPU, compiled, GPU bands | **1.05 s** (layout 0.56, eval 0.49) | **22×** | 2.1e-9 |

Test: `test_torch_backend.py::test_torch_band_solver_reproduces_the_layout` (CPU device, quantile
frame with `near_adapt` + x′ refinement: same node counts, nodes and edges to 1e-10).

## 10. Segment checkpointing (2026-11-28)

**Why.** Reverse mode needs every step's beam in the backward pass, because the tracking maps, the
deposit and the kick lookup are nonlinear in it. Checkpointing every step kept 6N doubles per step:
about 85 GB for BC20 (1771 steps, 1e6 particles), 23 GB for the chicane (489 steps). That holds
whether the inputs are particles or beam parameters.

**How.**
- **Pure operations on a tape.** The torch side of each step is now a pure operation that reads no
  mutable run state: `op_track(beam, element, ele, params)`, `op_snap(beam, window structure)` and
  `op_kick(beam, snapshots, kick record, L_kick, params)`. The wake evaluator takes the observation
  time as an argument for the same reason.
- **Forward pass** (`CSR2DDiff(segment=k)`): runs without a graph. It records these operations on a
  tape and keeps only the beam every k tracking steps plus every snapshot (fields and frame).
- **Reverse sweep** (`segment_gradient(loss_fn, coords=)`): from the end, each segment is recomputed
  with gradients from its stored start beam. Three adjoints are carried backwards: the beam's, the
  snapshots' (a kick reads snapshots made in earlier segments, so their gradients from later
  segments are added up and passed on), and the parameters'. These finally go through the
  parameters' own graphs (a bend's −θ, a `BeamTransform`) by `autograd.backward`.
- With `checkpoint=True` the segment's own steps, wake chunks and (new) deposits are checkpointed
  as well. Without the deposit checkpoint each snapshot's fits kept dozens of particle-sized
  intermediates.

**Same gradient.** On the golden case, gradients with respect to the bend angle and the
`BeamTransform` knobs (chirp, z_scale, 9 profile knots) agree with the per-step checkpointed backward
pass to 5e-13 for k = 3 and k = 1000, and the loss to 1.6e-13. A test now covers this, including a
derived parameter (2θ − θ₀) whose gradient must reach the leaf. The first 1e6 run lost its θ
gradient on exactly that: the sweep assigned gradients to the derived ±θ tensors directly instead
of passing them through `backward`.

**1e6 particles on the chicane** (`test/gpu_diff/chicane_segment.py`; RTX 5000 Ada, compiled, GPU
bands; k = 22, 24 stored beams, 105 kicks):

| | peak GPU memory | forward (no graph) | segmented gradient |
|---|---|---|---|
| deposit not checkpointed | 17.6 GB | 314 s | 184 s |
| deposit checkpointed | **9.8 GB** | 315 s | 192 s |
| checkpoint every step (estimate) | > 23 GB for the beams alone | | |

| quantity | 1e6 particles | 20k particles |
|---|---|---|
| ⟨ΔE⟩ | −2814.6 keV | −2777 |
| d⟨ΔE⟩/dθ | −3.5262e6 keV/rad | −3.542e6 |
| d⟨ΔE⟩/d chirp | 2433.5 keV·m | 2444.8 |
| d⟨ΔE⟩/d z_scale | 9.034e4 keV | |

d⟨ΔE⟩/d(profile knot) at w = −4 … +4:

| w | −4 | −3 | −2 | −1 | 0 | +1 | +2 | +3 | +4 |
|---|---|---|---|---|---|---|---|---|---|
| keV per unit | −4.9e3 | −9.8e3 | −7.2e3 | +9.2e3 | +3.4e4 | +8.4e3 | −1.0e4 | −1.3e4 | −6.3e3 |

Physically: stretching the core (positive knot near w = 0) lowers the peak current and reduces the
loss; stretching the tails compresses the core relative to them, which increases it. The gradient
pass is faster than the forward pass because the forward pass also runs the numpy structure (the
1e6-particle numba deposit and the layout on the CPU).

**Open: GPU runs are not bit-reproducible.** Two forward runs of the identical 1e6 beam (identical
Hammersley-sampled distgen output, checked) gave ⟨ΔE⟩ −2815.86 and −2814.64 keV (4e-4); the
gradients differ by 3e-5. The likely cause is CUDA's `index_add` (atomic adds in varying order),
with the round-off flipping structural thresholds (drift thinning, node counts) over the 105 kicks.
Not verified; `torch.use_deterministic_algorithms(True)` would test it.

BC20 projection (1771 steps, k ≈ 42): ~42 stored beams (2 GB) plus one 42-step segment, so roughly
10–14 GB. It fits the 32 GB card.

## 11. The near-adapt node construction batched over mesh points (2026-11-28)

**Why.** With the band solve on the GPU (§9), a 1920-point `near_adapt` layout still took 1.44 s.
1.1 s of that was `_g_near_adapt_nodes`, which runs per point: the bisection rounds of the column
scan (each a few dozen small numpy calls on a few hundred values), one density lookup per request,
and the final monitor, grading and inverse CDF. That is 10,180 generator resumptions per layout.

**How.**
- **A new request type.** With the batched construction available, a point's near region yields one
  `NearAdaptRequest` (layout.py) instead of running its own rounds. `_drive` answers it with the
  per-point generator, so the numba path is unchanged. `_drive_many` answers all points' requests
  at once with `sim._near_adapt_batch`, set by the torch evaluator together with the GPU band solver
  (`torch_bands`).
- **`torch_backend/near_adapt_t.near_adapt_batch`** runs each bisection round for all points together.
  The intervals live on flat arrays with a segment id per point. Each round needs one band solve
  (GPU) and one density lookup. The merge sorts by (segment, s′) and drops duplicates, keeping the
  old point as `np.unique` does.
- **The final steps** go through one numba kernel over all segments: np.gradient's formula, the
  grading recursion, the cumulative integral and the inverse CDF.
- **Also on the numba path:** the interval splitting inside the per-point rounds is now one
  vectorised expression instead of a Python loop over intervals. It is bit-identical (same
  operation order) and the per-point numpy layout went 5.34 → 5.11 s.

**Results** (golden case, quantile frame + `near_adapt` + x′ refinement, 12×160 = 1920 points):

| layout | time | node counts identical | max rel diff |
|---|---|---|---|
| numpy (the numba path) | 5.11 s | (reference) | |
| GPU bands, per-point near-adapt | 1.48 s | 1920/1920 | 1.6e-13 |
| GPU bands, batched near-adapt | **0.87 s** | 1920/1920 | 1.6e-13 |

- The numba path's layouts are bit-identical to before (`array_equal` on all 1920), and MPI equals
  serial.
- What remains is spread thinly over the per-point generator steps (far and base near nodes, extra
  regions) and the GPU→CPU copies of the band results, with no single hot spot left.

A BC20-type `near_adapt` kick on this GPU is now about 0.9 s of layout plus ~0.5 s of compiled
evaluation, against ~5 s of layout alone before §9. Test: the layout test of §9 now also runs the
batched near-adapt. Full suite: 225 passed.

## 12. The full BC20 run on the GPU (2026-11-28)

Indranil's L1-17.6284 beam with the recommended recipe (`run_indranil.py b176_v2`):
- 1e6 particles, 1771 steps, 315 kicks;
- a 12×160 charge mesh, `frame: auto` (it chose quantile, as on the Mac);
- `near_adapt` + x′ refinement, 100 x′ nodes, prehistory 45 m.

It ran with `backend: torch`, `device: cuda`, `torch_compile`, `torch_bands`
(`test/gpu_diff/run_bc20_gpu.py`, output in `benchmark_results/bc20_indranil/runs_gpu/`).

| | RTX 5000 Ada, 1 process | Mac, numba, 6 MPI ranks (`b176_err_fl1`) |
|---|---|---|
| run time | **44.2 min** (315 kicks: 1843 s, 5.8 s each; 1456 other steps: 804 s, 0.55 s each) | 2.6 h |
| total energy loss vs CSR off | **−23.838 MeV** | −23.834 MeV |

The CSR-off reference (`b176_off`, 4.6 min on the CPU) was rerun here. The two runs agree to
1.7e-4, the level of the GPU's run-to-run reproducibility (§10) and of the x′-refinement ties. 1D
gives −28.58 MeV.

The ~0.55 s per non-kick step is the 1e6-particle deposit and tracking, still numba and numpy on
the CPU in the plain torch-backend run. It is now 30% of the run, and moving it to the GPU (the
torch deposit of `CSR2DDiff` already exists) is the next speed-up.
