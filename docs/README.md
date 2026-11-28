# docs

- [DFCSR_2D_methods.md](DFCSR_2D_methods.md) — how the 2D code works: the physical model (Stupakov–Tang longitudinal and Stupakov transverse wakes), the simulation loop, the comoving density estimation, the wake mesh, the band-localised integration with its adaptive node placement and polar patch, the kick schedule, the 1D model used for guidance and diagnostics, MPI, benchmarks (single dipole, benchmark chicane, FACET-II BC20 with two beams) with run times and a per-point cost breakdown, known limitations, and a parameter reference. Written as material for a journal paper; every number carries the section of the development log it comes from.

Development logs (all in this folder; figures they link live in `../pyDFCSR_2D/test/benchmark_results/` and are not committed):

- [progress.md](progress.md) — the main development log: the comoving deposition, schedules, the chicane and dipole benchmarks, the 1D model.
- [progress_facet_bc20.md](progress_facet_bc20.md) — the FACET-II BC20 port and Indranil's beams (frames, charge mesh, reach rules, x′ refinement).
- [progress_cancellation.md](progress_cancellation.md) — the ±∂ρ/∂x cancellation of the integrand, u taper, integration by parts.
- [progress_comoving_interp.md](progress_comoving_interp.md) — the co-moving history interpolation.
- [integration_region.md](integration_region.md) — the integration-region localisation.
- [progress_gpu.md](progress_gpu.md) — the torch backend, the differentiable run, GPU tooling.
- [code_notes.md](code_notes.md), [parameter_notes.md](parameter_notes.md) — comment history moved out of the code and the parameters.

- `talman_cancellation.md` — the Talman / centrifugal space-charge cancellation: the literature (Talman, Lee, Derbenev–Shiltsev, Li, Geloni et al., Stupakov 2022), what the controversy was, which formulation pyDFCSR uses (kinetic energy + full transverse Lorentz force), the chicane and BC20 results that show the cancellation, and how the code can test it quantitatively.
