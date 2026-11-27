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
