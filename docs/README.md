# docs

- [DFCSR_2D_methods.md](DFCSR_2D_methods.md) — how the 2D code works: the physical model (Stupakov–Tang longitudinal and Stupakov transverse wakes), the simulation loop, the comoving density estimation, the wake mesh, the band-localised integration with its adaptive node placement and polar patch, the kick schedule, the 1D model used for guidance and diagnostics, MPI, benchmarks (single dipole, benchmark chicane, FACET-II BC20 with two beams) with run times and a per-point cost breakdown, known limitations, and a parameter reference. Written as material for a journal paper; every number carries the section of the development log it comes from.

The development logs themselves live one level above the repository (`../progress.md`, `../progress_facet_bc20.md`) and in the repository root (`progress_cancellation.md`, `progress_comoving_interp.md`, `integration_region.md`).

- `talman_cancellation.md` — the Talman / centrifugal space-charge cancellation: the literature (Talman, Lee, Derbenev–Shiltsev, Li, Geloni et al., Stupakov 2022), what the controversy was, which formulation pyDFCSR uses (kinetic energy + full transverse Lorentz force), the chicane and BC20 results that show the cancellation, and how the code can test it quantitatively.
