# FACET-II W chicane (BC20) with pyDFCSR: input files, tracking vs Bmad, and the tracking fixes

Started 2026-11-05. Git branch `csr1d_mode`, on top of commit `f33038e`. **Nothing from this work is committed yet** (see section 6).

Goal: prepare DFCSR input files (lattice and beam) for the FACET-II W chicane from Sergei Kladov's Bmad simulation. Then check that pyDFCSR with CSR off tracks the beam the same as Bmad/Tao with every collective effect off.

---

## 1. Impact on earlier simulations (read this first)

Three tracking changes were needed to match Bmad. They live in shared code (`CSR.py`, `csr1d.py`, `beams.py`), so they also change other lattices. Each was measured with CSR off, on the same particles, old behaviour vs new.

### 1a. Bend edges: the exit edge (E2) was dropped in `auto` and `manual` mode. **Earlier chicane `auto` runs are affected.**

**What the edge is.** A bend whose pole face is rotated by E1/E2 gives a thin focusing kick at that face: Δpx = x·h·tan E, which is horizontal focusing or defocusing. In the benchmark chicane the faces are cut so the edges carry the focusing that closes the dispersion. B1 has E2 = 0.0483, B2 has E1 = −0.0483, B3 has E2 = −0.0483 and B4 has E1 = 0.0483. Remove the exit faces of B1 and B3 and the chicane no longer closes.

**Why it was dropped.** The run loop tracks an element in steps and builds each step's bmad-x element with `entrance`/`exit` flags. E2 was attached ONLY to the "remainder" piece `DL_1`: the part of a step that straddles the boundary, tracked when the next element begins. That piece is skipped when it is < 1e-6 m. In `legacy` mode the nodes sit on a uniform grid, so a step usually straddles the boundary and `DL_1` exists. In `auto` and `manual` mode **every boundary is a node** (by design), so `DL_1` is always 0 and no step ever carried E2. The same happens in `legacy` whenever an element length is a multiple of step_size. Separately, the entrance edge E1 of the **first** element was never applied: the branch that adds it requires `ele_count > 0`. In the benchmark chicane B1 has E1 = 0, so this second defect did not matter there.

**The fix** (`CSR2D._ends_element`, used in `CSR.py` and `csr1d.py`): any step that ends within 1e-6 m of its element's end is built with `exit=True`. The 1e-6 is the same tolerance as the `DL_1` guard, so exactly one of the two paths applies each edge. Step 0 of the first element is built with `entrance=True`.

**Measured, CSR off, chicane_lattice.yaml + chicane_init_beam.yaml, at the chicane exit.** The table was made on a biased 1e5-particle subsample (every 10th particle; see the correction below). The old-vs-new difference is real, because both columns use the same particles, but the absolute σx is wrong:

| step mode | version | η_x [mm] | η_x′ [mrad] | σx [µm] | σz [µm] | εnx [µm] |
|---|---|---|---|---|---|---|
| legacy | old | +0.017 | 0.000 | 15.42 | 20.493 | 0.6017 |
| legacy | new | +0.017 | 0.000 | 15.42 | 20.493 | 0.6017 |
| **auto** | **old** | **−7.466** | **−1.239** | **55.38** | **18.352** | 0.6040 |
| auto | new | +0.017 | 0.000 | 15.40 | 20.493 | 0.6017 |

**Correction (2026-11-05, found by the other session).** With the full 1e6-particle beam the corrected chicane gives σx = **22.93 µm** at the exit, not 15.4 (old `auto`: 58.0 µm; η_x −7.48 → −0.000 mm; σz 18.4 → 20.49 µm). The 15.4 came from my sampling. `chicane_init_beam.yaml` uses a Hammersley quasi-random sequence, and the particles are in sequence order. Every 10th point of a Hammersley set is not a fair sample: it keeps every first and second moment (σx, σpx, σδ and the x–px correlation all identical to the full beam) but carries a strong third-order moment, ⟨px·δ²⟩ = −0.29 (in rms units) instead of 0. A random 1/10 gives 0.003 and 22.91 µm. In a chicane, x at the exit picks up chromatic terms like T_{x,px,δ}·px·δ, so a fake px–δ² correlation shows up in σx even though all the second moments are right. Lesson: **never subsample a quasi-random distgen beam by stride**; use a random choice, or regenerate with fewer particles.

Read the table this way. Old `auto` left 7.5 mm of residual dispersion, so σx at the exit was too big (58.0 vs 22.9 µm on the full beam) and σz 10% too short: the R56 changed because the beam took a different path. New `auto` matches `legacy` to the step size. In `legacy` mode the change is bit-identical (max |Δ| = 0), because every boundary there was straddled. The single dipole (`dipole_lattice.yaml`, E1 = E2 = 0) is bit-identical in both modes.

**What this means for earlier results.** Every chicane run made with `step_control: auto` or `manual` since those modes were added (commit 9507c4b, 2026-09-14) tracked a chicane that does not close. That includes `chicane_2d_vs_1d.py` (all its 2D and 1D runs use `auto`, and its fine runs use `manual`), `test_chicane_auto.py`, the chirp scans on the chicane, the D3 diagnosis and the 1D-vs-2D chicane comparisons in `progress.md` from 2026-10-29 to 11-04. `benchmark_csr1d_vs_emma.py` uses `manual` too. Its chicane case (lines 92-94) has the same edge angles, so it is affected. There it matters more. Emma's `csr_calc` transports the beam with its own linear R + T566 (the benchmark's docstring). So for the chicane case, pyDFCSR's side tracked a chicane that does not close, while Emma's closed. The benchmark normalises each code to its own CSR-off run, which cancels much of the tracking difference, but not the CSR computed on a differently compressed bunch. Its other cases have E1 = E2 = 0 and are unaffected. **1D vs 2D comparisons within one run set are still like-for-like** (both codes tracked the same wrong chicane, bit-identically). But absolute numbers (final σz, energy loss, emittance growth, the D3 drift wake of a fully compressed bunch) belong to a different chicane, and need rerunning. The CSR in D3 depends strongly on how compressed the bunch is there, so it may shift more than the tracking table suggests.

Lattices in `example/input/` with nonzero E1/E2 (so possibly affected in `auto`/`manual`): `chicane_*`, `dipole_facet_lattice.yaml`, `FACETII_BC14_lattice.yaml`, and the scan families `es_lat_*`, `kap_lat_*`, `tf_lat_*`, `sxz_lat_*`, `a2_lat_*`, `kb*_lat`, `fp_lat`, `man_lat`, `ring200_lat`, `tr_lat`, `up_lat_320`. Not checked one by one.

### 1b. Sextupole: bmad-x's integrator replaced by Bmad's

**What was wrong.** `bmadx.Sextupole` advances x by `h·px` and divides the kick by (1 + pz). In Bmad, px is the *canonical* momentum. Then the drift is x += h·px/√((1+pz)² − px² − py²), and the field's kick to the canonical momentum, Δpx = −K2·h·(x² − y²)/2, has **no** 1/(1+pz). The 1/(1+pz) belongs to the slope x′ = px/(1+pz), not to px. bmad-x mixes the two. So it is a *chromatic* error, which is exactly what a sextupole at dispersion is there to correct. Bmad's sextupole is half kick, exact drift, half kick per step. That is now `tracking.SextupoleKDK`. `Beam.track` dispatches to it; all other elements still go to `bmadx.track_element`.

```
SextupoleKDK(L, K2, NUM_STEPS=n, X_OFFSET, Y_OFFSET, TILT):
  h = L/n; k = K2*h/2
  offset into the element frame (X_OFFSET, Y_OFFSET, TILT)
  repeat n times:
    px -= k*(x² - y²)/2;  py += k*x*y        # canonical, no 1/(1+pz)
    exact bmad-x drift of length h
    px -= k*(x² - y²)/2;  py += k*x*y
  offset back to the lab frame
```

Verified against an isolated sextupole in Tao: 4e-19 difference at one step. On the BC20 sextupoles, bmad-x was up to 5e-5 m off Tao in x **per element**.

**Impact elsewhere.** Only `fodo_lattice.yaml` has a sextupole (L = 0.05, K2 = 0.5). On its beam the change is 1.4e-10 m in x (σx = 47 µm), so it is negligible. No other example lattice uses sextupoles.

### 1c. NUM_STEPS is now shared out over the pieces of an element

The YAML NUM_STEPS is for the whole element, as in Bmad. The run loop tracks an element in pieces of length DL and used to pass the full count to every piece, so the integration step depended on the CSR step. Each piece now gets max(1, round(NUM_STEPS·DL/L)). No existing example lattice sets NUM_STEPS (only the new `facet_bc20_lattice.yaml`), so nothing else changes.

### 1d. Two small fixes

* A bend tracked in one piece (entrance and exit together) used `FRINGE_AT` without defining it unless the YAML gave it: a NameError. It now defaults to `both_ends`.
* New element type `vbend` (see section 3). It is new, so it changes nothing existing.

---

## 2. Source material

Sergei Kladov's email: "I managed to correct the lattice and simulation that is based on the experimentally saved element parameters…". The attachment is in `pyDFCSR/pyDFCSR_2D/example/bmad_Wchicane_Sergei/`:

* `S2E_DAQ_E320_17418.ipynb`: the tuned DAQ simulation (machine state of DAQ E320_17418, 2026-06-19, L2 matched, W-chicane dispersion tuned with sextupole offsets)
* `S2E_default_feedback.ipynb`: the default lattice without DAQ data
* `S2E_guide.pdf`: installation, the parameters used, and the reference results
* `S2E_reference.7z`, extracted to `S2E_reference/E320_17418_2026-10-19/`: the reference run of the DAQ notebook (2026-10-19, FACET2-S2E commit 3bc1b17): `lattice.bmad` (the full machine as Tao tracked it), `beam_BEGL2F.h5`, `beam_BEGBC20.h5`, `beam_ENDBC20.h5` (2·10⁴ particles, 1.6 nC), `diagnostics.csv`, `run_settings.json` and `restart_settings.json`

The DFCSR input covers **BEGBC20 → ENDBC20** (49.084 m), starting from `beam_BEGBC20.h5`: ⟨E⟩ = 10001.07 MeV, σz = 133 µm (core 150 µm), σδ = 1.05%, εnx/εny = 4.62/4.33 µm. Sergei's run, with ISR, 1D CSR, slice SC and wakes, ends at 7.1 µm core length, 47.7 kA and εnx = 32.4 µm (guide §5.5).

---

## 3. The W chicane, and how the Bmad lattice was translated

BC20 is a four-family "W" chicane: B1 (+22.6 mrad), B2 (−32.8), B3 (+5.1), Q6E, B3 (+5.1), B2 (−32.8), B1 (+22.6), with zero net angle. 18 quads and 8 sextupoles sit at dispersion. A three-magnet **vertical** wiggler (WIGE1-3, ±2.5 mrad) sits after B2R. The 172 Bmad elements become 69 pyDFCSR elements: 6 dipoles, 3 `vbend`s, 18 quads, 8 sextupoles and 34 drifts. The length matches Bmad to < 1e-9 m.

**The reference momentum is not the beam momentum.** After Sergei's energy feedback with wakes on, Bmad's P0C in BC20 is 10116.23 MeV, but the beam is at 10001.07 MeV, 1.15% lower: the reference particle feels no wakes. He pinned the bends to the beam with `DB_FIELD`, and the quads and sextupoles with `field_master`. In Bmad terms that is G and DG with G + DG = G·(10000.10 MeV)/P0C. **All Bmad coordinates are relative to P0C.** Rebuilding strengths from the beam energy makes every magnet ~1% wrong, which is what his notebook caveat warns about. The robust way is to keep Bmad's reference exactly. The beam goes in as `style: from_file` with `energy: P0C`, so pyDFCSR's (x, px, y, py, z, pz) are Bmad's numbers, and K1, K2, G, DG are copied verbatim. `style: ParticleGroup` would not work for two reasons. It takes the mean energy as the reference. And it reads `pg.z`, which is identically 0 in a Tao beam file: Tao writes the beam at fixed s, and the longitudinal position is in t. The build script converts z = −βc(t − t_ref), which reproduces Tao's `bunch1(z)` to 1e-19 m. (The beam file's own P0C differs from the lattice's in the 17th digit; the lattice value is used.)

**Superimposed elements.** B2, B3, S1, S2, Q6E and WIGE2 are split into slaves by a trim corrector or marker at their centre (XCB2LE in B2LE, AS1EL in S1EL, …). They are merged back into their lords. Every corrector in BC20 is at 0 and every patch is zero; the build script asserts both, so a future lattice with a nonzero trim fails loudly instead of being dropped. Consecutive drifts, markers, monitors and collimators are merged into one drift. A Bmad drift name that recurs gets a `__2`, `__3` suffix.

**The vertical wiggler** becomes `type: vbend`, a new element type. It is tracked as an SBend with TILT = π/2, but it is a straight element for the reference orbit, formation length, schedule and CSR integral, which are all (x, z) only. That is what the 2D code can represent. The wiggler's own CSR is vertical and is not computed.

**Bend fringe: Bmad's `FRINGE_TYPE = Full` has no bmad-x equivalent.** Tested against Tao on an isolated B2, 40 random particles with the beam's spreads:

| bmad-x model | max error x | y | py | z |
|---|---|---|---|---|
| `hard_edge_only` (with FINT, HGAP), **used** | 1.2e-12 | 2.0e-9 | 2.4e-9 | 4.3e-11 |
| hard (FINT = 0) + soft edge | 3.1e-12 | 1.8e-10 | 2.1e-10 | 1.9e-9 |
| `linear_edge` (the bmad-x default) | 3.6e-8 | 4.9e-8 | 5.4e-8 | 5.3e-10 |

`hard_edge_only` is best in x and z, which are the planes the CSR acts in. Its y error is a small linear vertical-focusing difference: FINT·HGAP enters Bmad's hard edge differently in Full mode.

**Sextupole integration steps.** `lattice.bmad` has `num_steps = 1` for every BC20 sextupole: Bmad's default when an element has no `ds_step`. The S2 sextupoles (0.76 m, K2 = −12.4 m⁻³) at x up to 9 mm are under-resolved by that. In Tao, going from 1 step to 5 mm steps moves the beam at ENDBC20 by 34 µm (max) / 8.4 µm (rms) in x, against σx = 194 µm, and σz by 0.29 µm (0.6%). The pyDFCSR lattice uses 5 mm steps (NUM_STEPS = 25 / 50 / 76), and so does the Tao reference below. **Sergei's published BC20 numbers use one step; worth telling him.**

---

## 4. Files and how to regenerate them

| step | script (`pyDFCSR/pyDFCSR_2D/test/facet_bc20/`) | conda env | output |
|---|---|---|---|
| 1 | `export_bc20.py` | `bmad-wch` | `bc20_export.npz`: Tao beam at every element, sextupoles as-is and at 5 mm (386 MB cache, **not for git**) |
| 2 | `build_bc20_input.py` | `pydfcsr` | `example/input/facet_bc20_lattice.yaml`, `facet_bc20_beam.txt`, `facet_bc20_config.yaml`, `facet_bc20_nocsr_config.yaml` |
| 3 | `compare_tracking.py`, then `plot_tracking.py` | `pydfcsr` | `compare_tracking.txt`, `compare_tracking.npz`, `test/benchmark_results/facet_bc20/tracking_vs_bmad.png` |

**Environments.** The local `bmad` env is broken: numpy fails to load libgfortran with a "duplicate LC_RPATH" dyld error. A fresh env was created:
`conda create -n bmad-wch -c conda-forge "bmad=*=nompi*" pytao openpmd-beamphysics h5py python=3.12` (Bmad 20261031, pytao 1.4). Sergei used Bmad 20260828. bmad-x is not on PyPI, so Tao and bmad-x run in different envs and exchange data only through the `.npz`. Tao truncates long paths and this tree has spaces in it, so `export_bc20.py` copies the lattice and beam to a temporary directory first.

**YAML numbers** are written so that PyYAML reads them back as floats: `2e-05` is a string in YAML 1.1, `2.0e-05` is a float. The build script asserts the round trip.

**`facet_bc20_config.yaml`** is the CSR run: 2D, `bspline_comoving`, `step_control: auto` (2062 steps, 440 kicks, h = 10–25 mm), deposition 300², integration 200², wake mesh 12 × 40, `n_formation_length` 1. `drift_cutoff` is left at the current code default (0). The production runs (section 7) do not read it directly: `run_dfcsr_bc20.py` writes its own configs with every 2D setting pinned.

| step | script | conda env | output |
|---|---|---|---|
| 4 | `run_dfcsr_bc20.py <name>` (from `example/`) | `pydfcsr` (1D), `pydfcsr-mpi` + `mpirun -np 6` (2D) | `test/benchmark_results/facet_bc20/runs/<name>.npz`, logs, `inputs/<name>.yaml` |
| 4 | `tao_bc20.py tao_csr` / `tao_all` | `bmad-wch` | `runs/tao_csr.npz`, `runs/tao_all.npz` |
| 5 | `compare_csr_runs.py` | `pydfcsr` | `compare_csr_runs.txt`, `benchmark_results/facet_bc20/csr_along_s.png`, `csr_at_end.png`, `csr_resolution_1d.png` |

---

## 5. Result: pyDFCSR (CSR off) vs Bmad/Tao (all collective effects off)

Tao: ISR, radiation damping, CSR, space charge, wakes and apertures all off; sextupole `ds_step = 0.005`. pyDFCSR: `facet_bc20_nocsr_config.yaml` (`mode: manual`, every boundary a node). Same 20 000 particles, compared particle by particle at every element exit.

**Method.** First, element by element: each element started from Tao's own input beam and was compared with Tao's output. That separates per-element differences from accumulation. Quads, drifts, markers and zero patches agree to 1e-17. The bends agree to 1e-12 in x after choosing `hard_edge_only`. That is how the sextupole defect (1b) and the dropped-edge defect (1a) were found. Then end to end.

![tracking vs Bmad](../pyDFCSR_2D/test/benchmark_results/facet_bc20/tracking_vs_bmad.png)

| at ENDBC20 | x [m] | px | y [m] | py | z [m] | pz |
|---|---|---|---|---|---|---|
| beam rms | 1.94e-4 | 1.79e-5 | 4.87e-5 | 9.87e-6 | 5.14e-5 | 1.04e-2 |
| max \|pyDFCSR − Tao\| | 2.6e-8 | 4.7e-9 | 3.4e-7 | 2.2e-8 | 5.5e-10 | 3.9e-16 |
| rms (pyDFCSR − Tao) | 9.2e-9 | 9.3e-10 | 4.2e-8 | 2.2e-9 | 2.0e-10 | 6.8e-17 |
| for scale: Tao 1-step − Tao 5 mm sextupoles, max | 3.4e-5 | 8.8e-6 | 8.6e-6 | 1.2e-6 | 1.6e-6 | 0 |

Geometric εx (dispersion-corrected): pyDFCSR 7.55862e-10 m, Tao 7.55859e-10 m (4e-6 relative). σz: 51.43506 vs 51.43490 µm (3e-6). The (x, z) plane, which is what the CSR sees, agrees to 1e-4 of σx and 1e-5 of σz. The largest error is in y (7e-3 of σy). The figure shows where it comes from. It is 1e-9 already at B1LE, set by the Full-vs-hard fringe, and the quads then magnify it. The x error jumps by 100× at S1EL. That is the sextupole amplifying a 1e-12 input difference: the error is propagated there, not created.

**Tests.** New: `test/test_tracking_edges.py` (4 tests, 4 s). It checks that the edges are applied in `legacy` and `manual`, `SextupoleKDK` against the hand-written map, and that n steps equal n one-step pieces. Existing: `test_csr1d.py` + `test_import.py`, 124 pass, including the bit-identity of CSR1D vs CSR2D tracking (`csr1d.py` got the same edge fix). The full suite was not run.

---

## 6. Repository state

* Branch **`csr1d_mode`**, HEAD `f33038e` ("band_jacobian_min defaults to 0.2", 2026-11-05 18:08). That commit and `0a32ce9` appeared during this work. They are not from this session: another session on the same branch was active.
* **Uncommitted changes from this work:** `pyDFCSR_2D/CSR.py` (edges, NUM_STEPS, FRINGE_AT, `vbend`, SextupoleKDK import), `csr1d.py` (edges), `beams.py` (tracking dispatcher). New files: `pyDFCSR_2D/tracking.py`, `test/test_tracking_edges.py`, `test/facet_bc20/*`, `example/input/facet_bc20_*`.
* **Uncommitted changes in the same files that are NOT from this work** (from the other session): in `CSR.py`, the `curv_zlim` and `_prev_region_nodes` hunks; `deposit_smooth.py`; `params.py` (`prev_bend_grade`); `test/chicane_2d_vs_1d.py`, `test_curvature.py`, `test_frame_blend.py`; and the chicane/dipole PNGs in `benchmark_results/`. These need separating when committing.

---

## 7. Result: BC20 with CSR, pyDFCSR 1D / 2D / 2D without transverse force vs Tao 1D CSR (2026-11-06)

Runs: `dfcsr_1d`, `dfcsr_2d`, `dfcsr_2d_noT` (`run_dfcsr_bc20.py`, step_control auto, 440 kicks, all three on the same schedule), `tao_csr` (Tao, 1D CSR only) and `tao_all` (Tao with Sergei's BC20 settings: ISR, 1D CSR, slice space charge, wakes). All start from the same 20 000 particles, so the CSR effect is taken **particle by particle** as (run − CSR off). The CSR-off beam is Tao's, which pyDFCSR reproduces to 1e-4 σx (section 5). The comparison is in `test/facet_bc20/compare_csr_runs.py`; tables in `compare_csr_runs.txt`.

### 7a. What happens, in physical terms

**The CSR in BC20 is made almost entirely by a 4 µm current spike, and the spike exists only in the last 12 m.** The bunch arrives at 1.6 kA, with σz = 133 µm. Up to Q2ER (s = 36.7 m) it is still long: 4 kA, 85 µm FWHM. Inside **B2RE** the chirp rotates it to full compression. It leaves B2RE as a 35 kA spike **4 µm FWHM**, on top of a 200 µm long low-current pedestal. 30% of the charge is within ±3 µm of the peak at the B2RE exit, and 39% at ENDBC20. The spike stays at 3.5–6 µm through the drift, the wiggler and the last bend B1RE, and ends at 45–50 kA. The rms σz (51 µm at the end) describes the pedestal, not the spike. Of the CSR energy loss, 8–10% happens before B2RE, 12% in B2RE and 78–80% after it (1D numbers, both codes). B1RE alone gives about half.

**So the answer depends first on how finely each code resolves z, and only second on 1D vs 2D.** The CSR field of a spike of length ℓ scales roughly as N/ℓ^{4/3}. A code that blurs the 4 µm spike to 10 µm sees a field several times weaker. The 2D run sizes all its grids by the rms σz:
- wake mesh: ±4σz over 40 points = **10.5–11 µm spacing** after B2RE;
- deposition: 1.7 µm bins, then a 3-bin Gaussian filter, so the spike's density is seen at ~5 µm rms.

pyDFCSR 1D bins the full z extent: 300 bins = 0.7 µm at the end. Tao 1D uses 128 bins with `particle_bin_span 2` (~3 µm triangles). Neither is converged (7c).

**What the bunch-resolved plot shows.** At ENDBC20 (`csr_at_end.png`), 1D (both codes) gives up to −15 MeV in the spike and **+3.5 MeV at its head**: the head of a short spike gains energy from the spike behind it. 2D gives a smooth −8 MeV dip and no head gain at all. That is what a spike blurred to ~10 µm gives. Coarsening pyDFCSR 1D to the same resolution reproduces it (7c).

**The transverse CSR force** (2D vs 2D with `transverse_on: 0`) leaves the energy loss unchanged (−3.948 vs −3.958 MeV). It reduces the CSR-induced px kick in the spike by ~40%, and adds 0.8 µm to εnx (first-order dispersion correction; 0.06 µm after third-order correction). It also makes the bunch 0.13 µm shorter, out of 51 µm. All of this sits on top of the resolution problem, so it is indicative only.

**ISR, space charge and wakes** (`tao_all` vs `tao_csr`) change nothing visible at 10 GeV: ⟨ΔE⟩ by 0.012 MeV, εnx by 0.03 µm, σz by 0.0003 µm.

**Emittance in BC20 is dominated by chromatic aberration, not by CSR.** With CSR off, the dispersion-corrected εnx (x and px with their linear fit in δ removed, as in `twiss.py`) grows from 4.52 µm at BEGBC20 to **14.79 µm** at ENDBC20, peaking above 100 µm in the sextupole regions. Removing a third-order fit in δ instead leaves 7.19 µm: nonlinear dispersion, the T166/T266-type terms of a W chicane with σδ = 1%, accounts for most of the growth. The CSR changes are a few µm on top of this. Their sign depends on resolution: coarse 1D (40 bins) gives 11.0 µm, *below* CSR off, as 2D does (11.5 µm); fine 1D gives 20.8 µm. A guess, not tested: a smooth, z-correlated energy loss partly cancels the chromatic x–δ correlation, while the spike's sharp energy jump does not.

### 7b. Numbers

![CSR along BC20](../pyDFCSR_2D/test/benchmark_results/facet_bc20/csr_along_s.png)

![CSR along the bunch at ENDBC20](../pyDFCSR_2D/test/benchmark_results/facet_bc20/csr_at_end.png)

At ENDBC20. ⟨ΔE⟩ and rms ΔE are the CSR-induced mean and rms energy change per particle. εnx is dispersion-corrected to 1st / 3rd order in δ.

| run | σz [µm] | I_pk [kA] | ⟨ΔE⟩ [MeV] | rms ΔE [MeV] | σδ [%] | εnx 1st [µm] | εnx 3rd [µm] | run time |
|---|---|---|---|---|---|---|---|---|
| CSR off | 51.435 | 50.6 | 0 | 0 | 1.0418 | 14.79 | 7.19 | — |
| Tao 1D CSR | 51.561 | 45.4 | −6.394 | 6.81 | 0.9995 | 15.63 | 8.95 | 2 s |
| pyDFCSR 1D (300 bins) | 51.549 | 45.1 | −6.888 | 7.92 | 0.9959 | 18.76 | 10.71 | 6 s |
| pyDFCSR 2D | 51.355 | 47.1 | −3.948 | 2.95 | 1.0208 | 11.54 | 6.88 | 115 min, 6 ranks |
| pyDFCSR 2D, no transverse | 51.482 | 45.3 | −3.958 | 2.94 | 1.0207 | 10.75 | 6.82 | 115 min, 6 ranks |
| Tao, all effects | 51.561 | 45.1 | −6.382 | 6.80 | 0.9997 | 15.66 | 8.98 | 3 s |

(I_pk: 0.5 µm bins smoothed by 1 bin; it is noisy at the ±10% level. The two 2D runs ran at the same time on the 12 cores.)

CSR mean energy change per section [MeV]:

| section | Tao 1D | pyDFCSR 1D | pyDFCSR 2D | 2D no transverse |
|---|---|---|---|---|
| BEGBC20 → Q2ER exit (36.7 m, bunch long) | −0.632 | −0.547 | −0.367 | −0.368 |
| B2RE (spike forms) | −0.774 | −0.848 | −1.306 | −1.310 |
| B2RE exit → end of D1ET2B (drift, wiggler; spike) | −2.026 | −2.138 | −1.164 | −1.175 |
| B1RE (spike) | −2.963 | −3.356 | −1.111 | −1.106 |
| total | −6.394 | −6.888 | −3.948 | −3.958 |

### 7c. z-resolution check with pyDFCSR 1D (6 s per run)

`run_dfcsr_bc20.py dfcsr_1d_nb<N>` repeats the 1D run with N bins. The bin width quoted is at ENDBC20, where the z extent is 209 µm.

The figure is in 7e, together with the other bin modes.

| run | bin [µm] | → Q2ER | B2RE | drift | B1RE | total [MeV] | εnx 1st [µm] |
|---|---|---|---|---|---|---|---|
| 1D, 1200 bins | 0.17 | −0.597 | −0.895 | −2.205 | −3.792 | −7.490 | 20.75 |
| 1D, 600 | 0.35 | −0.562 | −0.870 | −2.193 | −3.680 | −7.304 | 20.38 |
| 1D, 300 (`dfcsr_1d`) | 0.70 | −0.547 | −0.848 | −2.138 | −3.356 | −6.888 | 18.76 |
| 1D, 150 | 1.39 | −0.540 | −0.803 | −1.973 | −2.684 | −6.000 | 15.44 |
| 1D, 80 | 2.61 | −0.537 | −0.716 | −1.666 | −1.741 | −4.660 | 12.10 |
| 1D, 40 | 5.22 | −0.533 | −0.568 | −1.261 | −0.866 | −3.229 | 11.02 |
| 1D, 20 | 10.4 | −0.526 | −0.397 | −0.861 | −0.368 | −2.152 | 12.07 |
| Tao 1D (n_bin 128) | | −0.632 | −0.774 | −2.026 | −2.963 | −6.394 | 15.63 |
| 2D (wake dz ≈ 11 µm) | | −0.367 | −1.306 | −1.164 | −1.111 | −3.948 | 11.54 |

What this shows:
1. **1D converges slowly, to about −7.6 MeV** (pinned down with the peak-based bin width and a shot-noise check in 7e): +6% from 300 to 600 bins, +2.5% from 600 to 1200. **Tao's n_bin 128 is ~15% low** and the 300-bin pyDFCSR 1D ~8% low. Before Q2ER the bunch is long and the result is resolution-independent (−0.53 to −0.60 MeV); the small drift with N there is particle noise in the thin bins.
2. **After B2RE, the 2D-vs-1D difference is mostly resolution.** 2D's drift + B1RE loss (−2.28 MeV) lies between 1D at 40 bins (−2.13) and at 80 bins (−3.41), consistent with its ~5 µm effective resolution. **The 2D result after B2RE therefore says nothing yet about 2D physics.**
3. **Two differences are not resolution, and are not explained yet:**
   - Before Q2ER, 2D loses 35% less than 1D (−0.37 vs −0.53 to −0.63 MeV), in a region where 1D is converged. Up to the B2LE exit the factor is 5.5 (−0.030 vs −0.165 MeV). The beam there is extremely tilted. σx reaches 1.3 mm at the B2LE exit (η ≈ 0.12 m × σδ 1%) against σz = 108 µm, so dx/dz ≈ 10 through the energy chirp. That is far outside where projecting onto z is justified, and is plausibly a real 2D effect. But the 2D settings (12 x-points in the wake mesh, the large-chirp branch of the integration region) were never validated at a tilt that large.
   - In B2RE, 2D loses **more** than converged 1D (−1.31 vs −0.90 MeV). The beam is still strongly tilted while it compresses, and the spike forms only in the second half of the bend, so the resolution problem would push the other way.

### 7d. Caveats

* The kick schedule is coarse where it matters most: 10 kicks in B1RE (1.06 m). It is the same for all pyDFCSR runs, so the comparisons are like-for-like, but it was not checked for convergence.
* Two 2D settings were not adapted to a non-Gaussian bunch: the 2D grids span ±5σz (deposition) and ±4σz (wake mesh) of the rms, and 0.1–0.5% of the charge falls outside the deposition grid all along (log warning).
* Sergei's published ENDBC20 numbers (7.1 µm core, 47.7 kA, εnx 32.4 µm) are not directly comparable. They use one-step sextupoles (section 3) and possibly another emittance definition. Not compared.

---

### 7e. 1D bin width set by the quantiles or by the current peak (`CSR_1D.bin_mode`)

**Why.** `CSR1D` rebuilds its z grid at every kick: `nbins` uniform bins from min z to max z (`bin_mode: extent`). The bin width is therefore set by the two outermost particles, not by the structure that makes the CSR. In BC20 that structure is a 4 µm spike sitting inside a 209 µm span. Two new modes, in `csr1d.py`, `bin_width()`:

* `quantile`: ds = (z(1−q) − z(q)) / nbins, q = `bin_quantile` (default 0.05). The bins resolve the central 90% of the charge, so tails and outliers no longer widen them.
* `peak`: ds = FWHM of the highest current peak / `bins_per_fwhm` (default 20). The FWHM is measured on a 1000-bin histogram of the central 99.8% of the charge, smoothed over 2 bins. Bins are coarse while the bunch is long and fine once a spike forms.

In both, the grid still covers every particle; `max_bins` (default 20 000) caps it, with a one-time warning. `bin_mode: extent` stays the default and is **bit-identical** to before (`dfcsr_1d_recheck` vs `dfcsr_1d`: identical coordinates). New tests in `test_csr1d.py`: `test_line_density_given_ds` and `test_bin_width_modes` (outlier, spike on pedestal), plus parameter validation in `test_params`. All 125 pass.

**What it shows on BC20.** In the figure, all three modes fall on **one curve of energy loss against the bin width at ENDBC20**. The resolution at the spike controls the result; how ds is chosen does not. The remaining question was whether that curve converges, or keeps falling because of shot noise. The mean CSR energy loss is quadratic in λ, so histogram noise adds a spurious loss that grows as the bins shrink. Test: the same runs on two complementary random halves of the beam (10 000 particles each, same 1.6 nC). A noise term scales as 1/N, so 2·(20k result) − (10k result) removes it.

* At bin widths ≥ 0.2 µm, the B1RE loss of the half beams equals the full beam's to 0.02–0.06 MeV: noise is negligible there. At 0.1 and 0.05 µm, the half beams lose 0.09 and 0.20 MeV more: from there on, the further growth is mostly shot noise of the 20 000 particles.
* Noise-corrected, `peak` mode gives (bin width at the end → B1RE / total, MeV): 0.39 µm → −3.78 / −7.33; 0.20 → −3.91 / −7.50; 0.10 → −3.95 / −7.56; 0.05 → −4.03 / −7.64. So **converged 1D ≈ −7.6 ± 0.1 MeV in total and −4.0 ± 0.1 MeV in B1RE**. Tao's 1D (−6.39) is 16% low, and the 2D run (−3.95) is half of it.
* `extent` with 1200 bins also reaches 0.17 µm at the end. But it over-resolves the long bunch early on (0.3–0.45 µm bins at 1.6–4 kA), and shot noise inflates the loss before Q2ER: −0.60 MeV, against −0.54 noise-corrected. `peak` keeps the bins coarse there (17 µm at B1LE for 20/FWHM) and gets −0.53 without correction.
* The two halves differ by up to ±0.1 MeV in B2RE and in the drift, systematically and at every bin width. That is not binning noise: they are two different macro-particle beams.
* `peak` with 5/FWHM is too coarse while the bunch is long (1.8 bins per σz at B1LE; the code warns).

![1D resolution scan](../pyDFCSR_2D/test/benchmark_results/facet_bc20/csr_resolution_1d.png)

**Tao bins the same way, and converges to the same answer.** Bmad's `csr_bin_particles` (`bmad/space_charge/csr_and_space_charge_mod.f90`) sets `dz_slice = (max z − min z) / (n_bin − 2 − (particle_bin_span + 1))` over the live particles: the `extent` rule, so outliers widen Tao's bins too. Each particle is a triangle of base `particle_bin_span · dz_slice`, cloud-in-cell for span 2, with no other smoothing. The line-density derivative is the difference of neighbouring bin charges at the bin edges, and the kick is interpolated linearly onto the particles. Tao's −6.39 MeV is simply Sergei's `n_bin = 128` (1.7 µm bins at the end), never refined. Scan (`tao_bc20.py tao_csr_nb<N>`, `tao_csr_span<S>`, `tao_csr_nb1024_span<S>`; table in `compare_csr_runs.txt`, curve in the figure):

| Tao 1D CSR | dz at end | → Q2ER | B2RE | drift | B1RE | total [MeV] |
|---|---|---|---|---|---|---|
| n_bin 64 | 3.5 µm | −0.627 | −0.698 | −1.701 | −1.894 | −4.920 |
| n_bin 128 (Sergei's) | 1.7 µm | −0.632 | −0.774 | −2.026 | −2.963 | −6.394 |
| n_bin 256 | 0.83 µm | −0.640 | −0.800 | −2.173 | −3.568 | −7.181 |
| n_bin 512 | 0.41 µm | −0.662 | −0.817 | −2.222 | −3.810 | −7.511 |
| n_bin 1024 | 0.20 µm | −0.714 | −0.850 | −2.235 | −3.845 | −7.644 |
| n_bin 2048 | 0.10 µm | −0.838 | −0.928 | −2.236 | −3.811 | −7.813 |
| n_bin 128, particle_bin_span 8 | 1.8 µm | −0.625 | −0.632 | −1.529 | −1.418 | −4.204 |
| pyDFCSR 1D peak 20/FWHM, noise-corrected | 0.20 µm | −0.530 | −0.856 | −2.205 | −3.908 | −7.500 |

* Refined, the two codes agree in the spike sections: Tao at n_bin 1024 gives −0.85 / −2.24 / −3.85 MeV in B2RE / drift / B1RE, against pyDFCSR's −0.86 / −2.21 / −3.91. That is within 1–3%, an independent check of the converged 1D value.
* Tao shows the same shot-noise artefact on the long bunch: at n_bin 2048, the loss before Q2ER grows from −0.63 to −0.84 MeV. A wider `particle_bin_span` widens the effective bin (span 8 at 128 bins: −4.20 MeV).
* At equal nominal bin width Tao resolves somewhat better (its curve sits left of pyDFCSR's `extent`). Part of that is pyDFCSR's 5-bin Savitzky–Golay smoothing. Without it (`dfcsr_1d_nb150_ns0`), B1RE goes from −2.68 to −2.88 MeV at 1.39 µm, against Tao's −2.96 at 1.69 µm. At 300 bins, −3.36 → −3.42 against Tao's −3.57 at 0.83 µm. The rest is likely the derivative stencil (pyDFCSR's `np.gradient` is a central difference over 2 bins, Bmad's edge difference spans 1). Not verified.
* **Not resolution:** before Q2ER, where the bunch is long and both codes are converged in bin width, Tao loses −0.63 MeV and pyDFCSR −0.53. Candidates: the kick schedule (Tao `ds_track_step` 5 cm, pyDFCSR's auto schedule), or the entrance/exit transient models. Not checked.

**Recommendation for compressed bunches:** `bin_mode: peak`, `bins_per_fwhm: 20` (40 is the limit for 20 000 particles before noise dominates). Run time is unchanged (10 s for BC20). Tables in `compare_csr_runs.txt`. Runs: `run_dfcsr_bc20.py dfcsr_1d_q<N>`, `dfcsr_1d_pk<M>`, `dfcsr_1d_<v>_half{A,B}`. The half beams are written to `runs/inputs/facet_bc20_beam_half{A,B}.txt`.

---

## 9. 2D quantile frame and charge-equidistributed wake mesh (2026-11-06)

### 9a. Why, and what it does

**The problem (section 7).** After B2RE the BC20 bunch is a ~2.7 µm FWHM current spike holding ~45% of the charge, sitting on a 200 µm pedestal. The 2D code laid all its grids out in units of the rms σz (51 µm):
- **deposition:** 300 bins over ±5σz, which is 515 µm, 2.5× the beam's full 208 µm span; then a 3-bin smoothing, so the density was seen at ~5 µm resolution;
- **wake mesh:** 40 points over ±4σz, 11 µm apart.

The spike was smeared to several times its width, and its CSR came out about half of the converged 1D value.

**The fix had to work for double horns too**, which are common. So nothing may depend on finding "the" peak: with two horns the highest one can switch from step to step, and a window around one horn would drop the other. Both new options depend only on the charge distribution:

* **`particle_deposition.frame: quantile`** (default `rms`, which is unchanged and bit-identical).
  - The fixed normalized grid is laid over the charge window [z_q, z_{1−q}] (q = `frame_quantile`, default 0.005), widened by `frame_margin` (default 0.1) of the span on each side.
  - Each particle is weighted by T(z): 1 inside, a C2 smootherstep to 0 across the margin. So the outermost ~1% of the charge is dropped as a source smoothly, not by a hard clip, which would itself radiate.
  - Quantiles are continuous in the particle coordinates, so the frame neither jitters nor jumps between horns.
  - All the frame fits (tilt, ξ̄, σξ, curvature, vx affine model) are T-weighted, so the dropped tails don't lever them either.
  - New `smoothing_sigma_z` lets z be smoothed less than x.
  - For BC20, 1200 z bins over the ~240 µm window give 0.2 µm bins, and with `smoothing_sigma_z: 1` an effective resolution of ~0.26 µm.
* **`CSR_computation.mesh_mode: charge`** (default `uniform`).
  - The same xbins × zbins tensor grid, but each axis equidistributes a monitor (de Boor): `mesh_uniform_frac` (0.3) uniform, plus the projected current smoothed over `mesh_smooth` (3) deposition resolution lengths.
  - Neighbouring spacings differ by at most `mesh_max_ratio` (1.5), and the axes span every particle.
  - Every horn or spike gets nodes in proportion to its charge. The node count is fixed, so the MPI split is unchanged.
* **Consequences handled in quantile mode:**
  - the mesh's curvature ridge is held at the window edge for tail particles;
  - the observer velocity outside the window comes from a beam-wide px fit, because there is no source density there;
  - the clip warning and the velocity floor ε are taken relative to the windowed charge f_window.
* **Diagnostics per kick** (statistics h5, group `quantile_frame`):
  - the window and f_window;
  - the 1D estimate of the dropped tails' share of the energy loss and of the field on the window (warning above `pedestal_warn` = 5%);
  - warnings if the highest peak's FWHM is less than 5× the deposition resolution or less than 3× the finest mesh spacing. The FWHM is used only for these warnings, never for the frame.

Code: `deposit_smooth.py` (`_get_DF_quantile`, `window_weight`, separate z smoothing in `smooth_and_differentiate`), `CSR.py` (`get_CSR_mesh`, `_mesh_ridge`, `_observer_vx`, `_quantile_diagnostics`, wake and statistics output), `params.py`, and the new leaf module `peakfind.py` (`peak_fwhm` moved out of `csr1d.bin_width`, bit-identical; `window_weight`; `equidistribute` and `charge_nodes`).

### 9b. Checks

* **Defaults bit-identical.** `test/golden/make_golden_2d.py` saved a 2D golden case before any edit: a tilted, chirped beam through a dipole and a drift, every kick's wake, the final density fields and the beam. `test_quantile_frame.py` reruns it with the defaults and with explicit `frame: rms` / `mesh_mode: uniform`, and gets every array identical. All 159 existing tests pass (`test_csr1d`, `test_curvature`, `test_frame_blend`, `test_ghosting`, `test_drift_reach`, `test_tangent_bands`, …). `test_deposit_smooth.py` fails at import independently of this change: it imports `deposit_smooth` as a top-level module, which its relative `.lookup` import no longer allows.
* **MPI = serial:** `test/golden/mpi_check_quantile.py`, 3 MPICH ranks, quantile + charge mode, all 36 arrays identical.
* **Unit tests:** taper (C2), equidistribution grading, charge mesh on two horns, quantile window on spike and double-horn beams, continuity of the frame under a shift, a Gaussian giving the same density as the rms frame, and a full CSR2D run in quantile/charge mode.
* **2D reduces to 1D on a thin beam** (`test/quantile_frame/validate_thin_beam.py`). A beam with σx = 1 µm, so the 1D model is exact, sits frozen in a dipole (R = 5 m, 1.5 m long, 1e6 particles, 1 nC). The 2D wake on a z line at the dipole exit is compared with `csr1d.wake_in_bend` on the same particles (0.1 µm bins):

| profile | 2D variant | ⟨W⟩ / 1D | peak \|W\| / 1D |
|---|---|---|---|
| spike (1.2 µm rms, 45%) on a 200 µm pedestal | rms frame (as BC20 ran) | 0.28 | 0.17 |
| | quantile, 600 z bins | 0.92 | 0.87 |
| | **quantile, 1200 z bins** | **0.95** | **0.95** |
| | quantile, 2400 z bins | 0.97 | 0.97 |
| | quantile 1200, all integration grids 2× | 0.95 | 0.94 |
| double horn (two 1.5 µm horns, 25% each, on a ±80 µm flat top) | rms frame | 0.37 | 0.25 |
| | **quantile, 1200 z bins** | **0.98** | **0.97** |
| | quantile, 2400 z bins | 0.98 | 0.98 |

![thin beam, spike](../pyDFCSR_2D/test/benchmark_results/quantile_frame/thin_spike.png)
![thin beam, double horn](../pyDFCSR_2D/test/benchmark_results/quantile_frame/thin_horns.png)

  What this shows:
  - **The deposition resolution was the limit, not the integration.** Doubling every integration grid (x′, s′, the near-patch radial and angular nodes) changes nothing.
  - **The result converges with z bins**, towards ~0.97 of the 1D value, which itself carries a percent or two of binning and noise error.
  - **2400 bins is past the useful limit:** shot noise appears along the flat top (±1 MeV/m) even with 1e6 particles. 1200 bins is the setting used for BC20.
  - The rms frame reproduces the BC20 deficit on this clean case: 28% of the loss for the spike.

### 9c. BC20 with the quantile frame (`dfcsr_2d_q`, `dfcsr_2d_q_noT`)

**Settings** (`run_dfcsr_bc20.py`):
- deposition: `frame: quantile`, q = 0.005, margin 0.1, 150 × 1200 bins, `smoothing_sigma: 3` in x and `smoothing_sigma_z: 1`;
- wake mesh: `mesh_mode: charge`, 12 × 80;
- everything else as the old 2D runs, including the same schedule (440 kicks).

Run time 114.7 / 108.8 min on 6 MPICH ranks, with the two runs one after the other. That is the same as the old 2D runs, which ran concurrently and so got little from MPI. Memory was 7.6–8 GB for the 6 ranks.

**What happens.**
- **The spike is resolved now.** The diagnostics (statistics h5, group `quantile_frame`) confirm it everywhere:
  - the highest peak's FWHM is at least 10.4× the deposition resolution (0.26–0.68 µm) and at least 12.5× the finest mesh spacing (0.20–5.1 µm);
  - the dropped 0.03–0.1% of the charge carries at most 0.7% of the 1D energy loss and at most 4.6% of the 1D field on the window (the maximum is in the B3 region, s ≈ 25 m), below the 5% warning level.
- **In the last bend, B1RE, the 2D loss now agrees with converged 1D** (−4.06 vs −3.91 MeV), where the old 2D had 28% of it. The bunch profile at ENDBC20 (`csr_at_end.png`) has the resolved shape: about −20 MeV in the spike and +2.5 MeV at its head, against 1D's −15 to −17 and +3.5. The old 2D gave a smooth −8 MeV dip.
- **Three differences remain, and none of them is resolution:**
  1. **B2RE, where the spike forms:** 2D loses 2.2× more than 1D (−1.87 vs −0.86 MeV). The excess *grew* once the spike was resolved (old 2D: −1.31). The beam there is extremely tilted, with σx up to 1.7 mm while the bunch shortens from 85 to 4 µm FWHM. That is outside the thin-beam validation, so a real tilted-beam 2D effect and a numerical cause are both still possible.
  2. **Drift and wiggler after B2RE:** 2D loses 15% less (−1.87 vs −2.21 MeV), with σx = 0.4–1.4 mm there. A transverse-size suppression of the exit transient is plausible.
  3. **Before Q2ER:** the 30% gap is unchanged (−0.38 vs −0.53 MeV). It is a long-bunch, high-tilt (dx/dz ≈ 10) region and was never a resolution effect.
- **The transverse CSR force matters for the emittance, not for the energy.**
  - Without it the energy loss is the same within 1% (−8.10 vs −8.18 MeV).
  - Without it, εnx at ENDBC20 is **19.8 µm**, about what 1D gives (18.8 at 300 bins, 21.1 at peak 20/FWHM). With it, εnx is **15.0 µm**, against 14.8 with CSR off.
  - Almost all of the difference is made in B1RE. There the transverse force cuts the CSR-induced px kick by ~40% (mean −10 vs −17 µrad; in the spike −22 vs −37 µrad).
  - This is the cancellation expected in a bend (Derbenev–Shiltsev / Talman): the energy a particle loses to CSR bends its orbit more tightly, and the centripetal (transverse) part of the same field acts against that.
  - So **1D codes, which apply only the energy kick, overestimate the emittance growth from the last bend**: about +5 µm (1D) against +0.2 µm (2D).
  - Before quoting this, confirm it on a validated case: a thin beam in a dipole with dispersion, against the analytic cancellation.
- **New shot noise.** At 0.26 µm resolution, 20 000 particles leave ±1 MeV and ±4 µrad wiggles in the kicks along the low-current pedestal, which the smooth old 2D did not have. The CSR-induced rms energy spread is 9.2 MeV, against 7.9–8.8 MeV for 1D. The mean loss may carry a small noise bias too. Not measured yet: it needs a half-beam test as in 1D (2 × 10 000 particles, ~2 h each).

![CSR along BC20](../pyDFCSR_2D/test/benchmark_results/facet_bc20/csr_along_s.png)
![CSR along the bunch at ENDBC20](../pyDFCSR_2D/test/benchmark_results/facet_bc20/csr_at_end.png)
![resolution](../pyDFCSR_2D/test/benchmark_results/facet_bc20/csr_resolution_1d.png)

Mean CSR energy change per section [MeV]:

| section | converged 1D (peak 20, noise-corrected) | Tao n_bin 1024 | old 2D | **2D quantile** | 2D quantile, no transverse |
|---|---|---|---|---|---|
| BEGBC20 → Q2ER | −0.530 | −0.714 | −0.367 | **−0.382** | −0.385 |
| B2RE | −0.856 | −0.850 | −1.306 | **−1.865** | −1.888 |
| drift + wiggler | −2.205 | −2.235 | −1.164 | **−1.872** | −1.916 |
| B1RE | −3.908 | −3.845 | −1.111 | **−4.058** | −3.912 |
| total | −7.50 | −7.64 | −3.95 | **−8.18** | −8.10 |

At ENDBC20 (εnx dispersion-corrected to 1st / 3rd order in δ, normalised with the mean energy as in section 7):

| run | σz [µm] | I_pk [kA] | ⟨ΔE⟩ [MeV] | rms ΔE [MeV] | εnx 1st [µm] | εnx 3rd [µm] | σx [µm] |
|---|---|---|---|---|---|---|---|
| CSR off | 51.435 | 50.6 | 0 | 0 | 14.79 | 7.19 | 194.3 |
| Tao 1D (n_bin 128) | 51.561 | 45.4 | −6.39 | 6.81 | 15.63 | 8.95 | 192.6 |
| pyDFCSR 1D (300 bins) | 51.549 | 45.1 | −6.89 | 7.92 | 18.76 | 10.71 | 193.3 |
| old 2D | 51.355 | 47.1 | −3.95 | 2.95 | 11.54 | 6.88 | 194.7 |
| **2D quantile** | 51.403 | 45.9 | **−8.18** | 9.20 | **14.99** | 9.43 | 192.2 |
| 2D quantile, no transverse | 51.550 | 43.4 | −8.10 | 9.06 | 19.78 | 11.09 | 189.0 |

**Emittance without dispersion correction, and slice emittance** (`slice_emittance.py`, `slice_emittance.png`).
- **The beam leaves BC20 with residual dispersion even with CSR off:** η_x = 16.4 mm and η′_x = 1.47 mrad at ENDBC20, which accounts for ~170 µm of the 194 µm σx. The plain projected εnx (√det cov(x, px)) is therefore mostly dispersion: 34.1 µm with CSR off.
- **The ordering is the same as dispersion-corrected:**

| run | projected εnx, uncorrected [µm] |
|---|---|
| converged 1D | 43.4 |
| 2D quantile, no transverse | 40.3 |
| CSR off | 34.1 |
| Tao 1D | 31.9 |
| 2D quantile | 31.0 |

- CSR mostly lowers η′ (to 0.5–1.0 mrad), and the betatron–dispersion cross term can shrink, so this quantity can fall below the CSR-off value. It is not a clean CSR measure.
- **Slice emittance** uses 40 equal-charge slices of 500 particles, so the spike gets 18 slices of ≲1 µm:

| run | mean slice εnx, raw / corrected [µm] | spike slices, raw / corrected [µm] |
|---|---|---|
| CSR off | 9.3 / 5.2 | 14.4 / 6.6 |
| Tao 1D | 13.1 / 6.2 | 21.3 / 8.8 |
| converged 1D | 18.3 / 8.3 | 32.1 / 13.2 |
| old 2D | 9.4 / 4.6 | 13.9 / 5.2 |
| **2D quantile** | 13.0 / 6.4 | **21.3 / 8.6** |
| 2D quantile, no transverse | 16.3 / 7.7 | 28.2 / 11.8 |

  ("corrected" removes each slice's own linear x–δ correlation.)
- **The pedestal slices (2–7 µm) are untouched by CSR. The spike's slice emittance grows in every CSR run.** In the full 2D run it grows 14.4 → 21.3 µm raw, even though the projected εnx barely moves (14.8 → 15.0). The projected value hides growth in the spike, offset by changed correlations elsewhere in the bunch. The transverse force halves the spike's growth (+6.9 vs +13.8 µm raw), the same direction as the projected result.

**The deposition window along the lattice** (`plot_window.py`):
- `window_along_s.png`: the current at every element exit with the window and grid edges at every kick, the peak current, the resolution lengths and the dropped charge.
- `window_profiles.png`: line-outs at six places, with the window shaded.

What the figures show:
- **The window follows the bunch:** z_(1−q) − z_q goes 526 → 199 µm, where the rms grid would be ±5σz = 1330 → 515 µm. The dropped charge is ≤ 0.1%, largest around B3, where the head has a thin 0.01–0.1 kA tail at the window edge.
- **Bins and resolution:** the 1200 bins are 0.53 → 0.20 µm, the deposition resolution 0.68 → 0.26 µm, and the finest wake-mesh spacing 5 → 0.25 µm.
- **The highest peak's FWHM** is ~300 µm until B2LE, 70 µm at Q2ER, and 3.3 µm after B2RE. So FWHM/resolution is ~450 early (heavily over-resolved, which gives the shot noise above) and ~13 at the end.
- The beam is saved only at element exits, so the map shows the compression inside B2RE as a step; the window lines, logged per kick, resolve it.

![window along s](../pyDFCSR_2D/test/benchmark_results/facet_bc20/window_along_s.png)
![window profiles](../pyDFCSR_2D/test/benchmark_results/facet_bc20/window_profiles.png)

**Wakes along the lattice** (`plot_bc20_wakes.py`):
- `bc20_wake_maps_long.png`, `bc20_wake_maps_trans.png`: maps of W(z, ξ) and F(z, ξ) at 21 kicks.
- `bc20_wake_maps_spike.png`: the same zoomed to ±12 µm on the spike after B2RE.
- `bc20_wake_zproj.png`: W projected on z against the old 2D and 1D at the same kicks. The projection weights the x rows independently of z, so a tilted spike comes out smeared.
- `bc20_wakes_along_s.png`: max |W| along s, and per-kick-normalised waterfalls of W(z) for 2D and 1D and of F(z) for 2D.

![wake maps](../pyDFCSR_2D/test/benchmark_results/facet_bc20/bc20_wake_maps_long.png)
![spike maps](../pyDFCSR_2D/test/benchmark_results/facet_bc20/bc20_wake_maps_spike.png)
![transverse maps](../pyDFCSR_2D/test/benchmark_results/facet_bc20/bc20_wake_maps_trans.png)
![projected wakes](../pyDFCSR_2D/test/benchmark_results/facet_bc20/bc20_wake_zproj.png)
![along s](../pyDFCSR_2D/test/benchmark_results/facet_bc20/bc20_wakes_along_s.png)

What the figures show:
- **In B1RE the 2D wake follows the spike's own tilt in (z, ξ).** The spike sits ~8 µm further back at one ξ edge than at the other. 1D sees only the projected profile, which this tilt broadens.
- **Shot noise from fixed fine z resolution.**
  - Before B2RE the bunch is 130 µm long, but it is still deposited at 0.52–0.32 µm bins (1200 bins over 1.2× the central 99% span) with 1-bin z smoothing.
  - The result is spiky projected wakes and vertical stripes in the maps, where the old 2D was smooth. The mean energy change is unaffected (−0.38 vs −0.37 MeV), but individual kicks are noisy.
  - The 1D peak mode avoided this by keeping bins coarse until a spike forms. The 2D analogue would be a per-snapshot z smoothing set by the narrowest feature, e.g. σ_z,smooth = max(1 bin, FWHM/M). The grid would stay fixed and quantile-placed; only the smoothing would adapt. Not implemented.
- **Outliers in the transverse force:**
  - In B1RE, ~14 isolated mesh cells per kick out of 960 (3 per kick in the old run) carry F up to ±1–3 MeV/m in the pedestal, where neighbours are ~0.1 MeV/m.
  - They cause the ±4.6 µrad rms px scatter in the pedestal (q − noT: pedestal mean +3.6, rms 4.6 µrad).
  - On the spike the effect is systematic: +14.4 µrad mean, 4.9 µrad rms. So the emittance cancellation is not an outlier artefact.
  - Cause not diagnosed: likely noise in ∂ρ/∂x or in the velocity term at fine resolution in the low-density pedestal.
- **A localised ±9 MeV/m diagonal feature** in the bunch tail at the first drift kick after B2RE (s = 39.98 m). It might be the exit transient of the tilted beam, or numerical. Not checked.

**Open:**
1. The B2RE excess (tilted, compressing beam): needs a validated tilted-beam case (thin beam with a large x–z tilt in a dipole, against an independent result).
2. The transverse-force emittance cancellation: confirm on a thin beam in a dipole with dispersion.
3. Shot noise in 2D at fine z resolution: a half-beam test, and adaptive z smoothing (above).
4. Before Q2ER: the long-bunch 2D-vs-1D gap.
5. The transverse-force outliers in the B1RE pedestal, and the tail feature at s = 39.98 m.

### 9d. Kicks placed where the CSR is (`step_control.kick_placement: csr1d`), and why the run was slow

**Why the 2D run took 115 min.**
- **The kicks are the cost:** 106 of the 115 min were the 440 CSR kicks. The 2062 snapshot steps took 9 min.
- **Both counts were inflated:**
  - **Steps:** the auto schedule combines its criteria as 1/h = 1/h_max + (requirements). In BC20 the transverse-size and tilt terms are never zero (dispersion changes σx everywhere), so h lands just under h_max = 50 mm, and the dyadic snap rounds it down a full factor of 2. That gives 25 mm nearly everywhere.
  - **Kicks:** one every κ = 4 snapshots, i.e. every ~10 cm along all 49 m. Before Q2ER, 333 of the 440 kicks (~75% of the time) produce ~5% of the energy loss. The 5 m drift D1ET1A alone took 41 kicks (10 min) for −0.002 MeV, while B1RE, half the loss, had 9.
- **MPI on this Mac saturates at ~3 ranks:**

| ranks | ms per wake point | wall time, 3 kicks |
|---|---|---|
| 1 | 21.8 | 67 s |
| 3 | 29.2 | 34 s |
| 6 | 51.7 | 36 s |

  - The load is balanced across ranks; each process just slows as more run. This is consistent with memory bandwidth: every rank reads its own copy of the density history, and the 6 P-cores share one 16 MB L2.
  - Pinning BLAS/OMP to 1 thread changes nothing. Each rank still keeps 2 threads busy (~190% CPU), probably MPICH's busy-waiting progress engine. Not investigated further.
  - Two 3-rank jobs side by side get the same total throughput as one 6-rank job.

**The new option.** With `step_control.kick_placement: csr1d` (auto mode only):
- A 1D pre-pass (`csr1d.csr1d_kick_profile`, a few seconds on rank 0, broadcast to the others) runs the same beam, lattice and snapshot schedule, with a kick at every snapshot and peak-mode bins. Its output is r(s) = the rms of the 1D wake over the bunch.
- `schedule.kicks_from_profile` then places the kicks at the integers of ∫ n ds, with
  n = r/(tol·R) + (|r″|/(24·tol·R))^{1/3} + 1/kick_h_max, where R = ∫ r ds:
  - the first term caps each kick's share of the total CSR at `kick_tol`;
  - the second bounds the midpoint-rule error;
  - the third caps the spacing in quiet stretches.
- Kicks are forced only at the dipole edges, where W jumps, and the midpoint intervals are clipped only there.
- The default `kappa` placement is unchanged (golden fixture identical).
- Tests: `test_kicks_from_profile`, and `test_kick_placement_csr1d_schedule` (kicks are snapshot nodes, the dipole edges are kicks, the intervals cover the lattice exactly).

**Calibration in 1D**, against a kick at every snapshot (2062 kicks, κ = 0.5); peak-mode bins throughout. Emittance errors are relative to the CSR-induced growth; "x"/"pz" are per-particle differences relative to the CSR-induced spread.

| schedule | kicks | ⟨ΔE⟩ | rms ΔE | εnx 1st | εnx 3rd | spike slice ε | pz | x |
|---|---|---|---|---|---|---|---|---|
| κ = 1 | 1089 | −0.04% | +0.04% | 0.24% | 0.18% | 0.12% | 0.57% | 0.32% |
| κ = 4 (the default) | 440 | −0.17% | +0.36% | 1.83% | 1.36% | 0.62% | 2.87% | 1.31% |
| csr1d, tol 0.02, h 1 m | 101 | −0.51% | +0.34% | 5.13% | 3.15% | 2.22% | 1.61% | 4.50% |
| csr1d, tol 0.01, h 0.5 m | 181 | −0.26% | +0.24% | 2.40% | 1.35% | 0.80% | 0.83% | 2.20% |
| **csr1d, tol 0.005, h 0.5 m** | **220** | **+0.04%** | **−0.13%** | **1.53%** | **0.76%** | **0.54%** | **0.80%** | **1.67%** |
| csr1d, tol 0.01, h 0.25 m | 273 | −0.16% | +0.19% | 1.43% | 0.88% | 0.64% | 0.54% | 0.99% |
| csr1d, tol 0.005, h 0.25 m | 308 | −0.11% | +0.12% | 1.24% | 0.76% | 0.39% | 0.46% | 0.96% |

- **Half the kicks of κ = 4 at equal or better accuracy:** tol 0.005 with h 0.5 m. Only the per-particle x error is slightly worse (1.7% vs 1.3%).
- **A 1 m cap is too coarse for x.** Energy kicks in the dispersive quad and sextupole region turn into x through dispersion, so their placement matters even where CSR is weak.

**Validation in 2D** (adaptive smoothing, 9e; 220 kicks, `dfcsr_2d_qa_kp`, against the same with κ = 4, `dfcsr_2d_qa`):
- **Agreement:**
  - total −8.637 vs −8.543 MeV (−1.1%); by section −0.049 / +0.031 / −0.035 / −0.041 MeV;
  - εnx identical (14.40 µm), εnx 3rd order −0.13 µm, spike slice emittance −1.1%, σz +0.01 µm.
- **Per particle, the spike agrees to 5–8%** (pz 5.0%, px 7.7%, x 8.4%), while **the pedestal differs by 30–35% in px and x.** In 1D the same comparison is 2–4% everywhere. This is a 2D-only difference in the pedestal, consistent with the isolated transverse-force outliers (9e), which hit different particles in each schedule. It is not a sampling error of the schedule.
- **The fixed-smoothing pair gives the same picture** (`dfcsr_2d_q_kp` vs `dfcsr_2d_q`): −1.9%, per-particle x 14%.
- **The pre-Q2ER section is ~12% lower with fewer kicks in both pairs** (−0.05 MeV, 0.6% of the total). That is where the 2D wake differs most from its 1D proxy.
- **Run time:** 116.7 vs 175.7 min with the two jobs sharing the machine (3 ranks each). On its own at 6 ranks the 220-kick run should take ~70 min: 220 × ~13 s plus 2062 × ~0.65 s.

**Recommendation for BC20:** `kick_placement: csr1d`, `kick_tol: 0.005`, `kick_h_max: 0.5`.

### 9e. Density-adaptive z smoothing (`particle_deposition.smoothing_adaptive`)

**Why.**
- The quantile frame keeps a fixed bin count, so the bins are fine (0.5 → 0.2 µm) all along BC20. Before B2RE the narrowest feature is ~300 µm wide (FWHM/resolution ≈ 450), and fixed 1-bin z smoothing there just passes shot noise through.
- At the end the low-current pedestal sits right next to the 2.7 µm spike, which needs the fine width. So a single smoothing width per snapshot cannot serve both.
- With fixed 1-bin smoothing the pedestal's ∂λ/∂z noise was 6.5× its true value at 20 000 particles.

**How.** Frame `quantile` only; default off.
- **Levels:** each snapshot is smoothed in z at `smoothing_sigma_z` × 2^k bins, up to `smoothing_sigma_z_max` (16), with one forward FFT.
- **Blending, cell by cell:** the width at a cell is the smallest one whose Gaussian kernel holds `smoothing_min_particles` (100) particles of a pilot density. The pilot takes two passes: the widest level first, then the adaptive result it implies, so a spike is not under-estimated and over-smoothed. The level map is smoothed so the weights vary slowly.
- **Derivatives are those of the blended field,** taken spectrally, and the velocity uses the same weights.
- No peak detection; double horns are handled the same as a single spike.
- Cost: ~0.35 s per snapshot on 150 × 1200 (about 30 FFTs, on every rank), ~12–24 min per BC20 run.

**A bug found and fixed on the way.** The first version blended each level's own derivatives. Then ρ and ∂ρ are inconsistent wherever the weights change, by Σ f_k ∇w_k. The CSR integrand's terms cancel against each other near the observer, and the inconsistency gave a 20 000-particle thin beam a mean wake of **−3.9×** the 1D value, with the wrong sign. Differentiating the blend fixed it. Even at 1e6 particles, where the weights are uniform, the consistent version is slightly closer to 1D.

**Thin-beam check** (`validate_thin_beam.py`; the 1D reference is always the tracked 1e6 beam):

| profile, particles | fixed 1-bin z smoothing | adaptive |
|---|---|---|
| spike, 1e6 | ⟨W⟩/1D 0.951, peak 0.95 | 0.965, 0.96 |
| spike, 20 000 | 1.078, 1.02; ±2 MeV/m pedestal noise | **1.044, 0.96; pedestal smooth** |
| double horn, 1e6 | 0.981, 0.97 | 0.983, 0.98 |
| double horn, 20 000 | 1.073, 1.08 | 1.069, 1.02 |

- At 20 000 particles, **shot noise biases the 2D energy loss upward:** +8% (spike) and +7% (horns) with fixed smoothing.
- **The adaptive smoothing removes the pedestal part of that bias** (spike: +4%) and brings the peaks back to the 1D value.
- **The remaining 4–7% comes from the noise inside the dense spike or horns themselves.** Smoothing cannot remove it without blurring the feature, so it is a limit of 20 000 macroparticles in 2D. In 1D the same particles give < 1% at 0.2 µm bins.

**BC20** (`dfcsr_2d_qa`, against `dfcsr_2d_q`, both 440 kicks):
- **Noise:**
  - roughness of the projected wake before Q2ER, median over kicks: 0.237 → **0.111**;
  - bin-to-bin noise of the px kick along the pedestal at ENDBC20: 0.75 → **0.40 µrad** (old 2D 0.13, 1D 0.21).
- **Physics:**
  - the drift + wiggler section moves from −1.87 to **−2.16 MeV**, now at converged 1D (−2.21);
  - B1RE −4.06 → −4.11, total −8.18 → −8.54 MeV (converged 1D −7.50);
  - εnx 14.99 → 14.40 µm, spike slice ε 21.3 → 20.4 µm.
- **Not fixed: the transverse-force outliers in B1RE** (14.1 → 10.9 cells per kick).
  - They are isolated single cells, up to ~100× their neighbours (worst −2.76 MeV/m among values of ~0.02).
  - They sit in the central, densest x rows and are scattered along the whole pedestal in z.
  - So they are not shot noise. Most likely a near-field problem of the transverse integrand at particular observer points (polar patch radius against the light-cone sampling, or the band/patch partition).
  - Diagnosis needs the history at a B1RE kick: a short restart at the B1RE entrance, evaluating the worst mesh points with refined near-field settings.

**Wake figures of the adaptive runs:** `plot_bc20_wakes.py <run> dfcsr_2d_q dfcsr_1d_pk20_k05`. The 1D comparison is the every-snapshot 1D run, so every 2D kick has a 1D wake at the same s. Files are `bc20_wake_maps_long_<run>.png`, `_trans_`, `_spike_`, `bc20_wake_zproj_<run>.png` and `bc20_wakes_along_s_<run>.png`, for `dfcsr_2d_qa_kp` and `dfcsr_2d_qa`. What they show:
- The projected wake before B2RE is now smooth and follows the mean of the noisy fixed-smoothing curve.
- In the drifts before Q2ER, 2D and 1D still differ in shape and sign (e.g. s = 3 m and 31 m): the known long-bunch gap.
- The tail feature at s = 39.98 m (the drift right after B2RE) survives the smoothing, so it is not noise.
- The transverse-force waterfall shows the B1RE outliers as a speckle across the pedestal that no other region has.

![wake maps, adaptive, 220 kicks](../pyDFCSR_2D/test/benchmark_results/facet_bc20/bc20_wake_maps_long_dfcsr_2d_qa_kp.png)
![spike, adaptive, 220 kicks](../pyDFCSR_2D/test/benchmark_results/facet_bc20/bc20_wake_maps_spike_dfcsr_2d_qa_kp.png)
![transverse, adaptive, 220 kicks](../pyDFCSR_2D/test/benchmark_results/facet_bc20/bc20_wake_maps_trans_dfcsr_2d_qa_kp.png)
![projected, adaptive, 220 kicks](../pyDFCSR_2D/test/benchmark_results/facet_bc20/bc20_wake_zproj_dfcsr_2d_qa_kp.png)
![along s, adaptive, 220 kicks](../pyDFCSR_2D/test/benchmark_results/facet_bc20/bc20_wakes_along_s_dfcsr_2d_qa_kp.png)

**Recommendation:** `smoothing_adaptive: true`, `smoothing_min_particles: 100`, together with the quantile frame. Tests: `test_adaptive_smoothing_spike_and_pedestal` (spike peak kept within 3%, pedestal ∂λ/∂z noise cut ≥ 3×).

### 9f. Still correct on the standard Gaussian dipole case; why the wake before B2RE stays rougher than the old grid

Committed before this section: `a9b6d2e` on `csr1d_mode` (quantile frame, charge mesh, 1D peak bins, CSR-guided kicks, adaptive smoothing; defaults bit-identical).

**The standard Gaussian dipole case** (`test/quantile_frame/dipole_gauss_check.py`): `dipole_chirp0_config.yaml`, the beam of the single-dipole study in progress.md (5 GeV, 1 nC, σz = σx = 50 µm, 1M Hammersley particles; 0.1 m drift, 1 m / 1 rad bend, 0.5 m drift).
- The wake is recorded as the particles receive it (mesh interpolated onto each particle, binned in z), so meshes of different types compare fairly.
- The reference is the default at twice the resolution (deposition 200×200, mesh 20×60).

| variant | kicks | mean ΔE [keV] | core slice σE [keV] | εnx [µm] | W vs default_fine | F vs default_fine |
|---|---|---|---|---|---|---|
| default (100×100 rms grid, 10×30 mesh), the recorded baseline | 18 | −1378.6 | 726.3 | 26.00 | 4.7% | 3.0% |
| **default_fine** (200×200, 20×60) | 18 | **−1416.6** | 754.9 | 26.70 | — | — |
| quantile frame + charge mesh, same counts as default | 18 | −1411.1 | 744.4 | 26.43 | **1.4%** | 1.3% |
| quantile, 2× resolution | 18 | −1428.1 | 757.8 | 26.79 | 1.4% | 0.6% |
| quantile, q = 0.0005 (0.1% of tails dropped) | 18 | −1401.3 | 735.6 | 26.22 | 2.8% | 1.9% |
| quantile + adaptive smoothing | 18 | −1406.9 | 744.2 | 26.43 | — | — |
| quantile + adaptive + csr1d kicks | 68 | −1402.8 | 747.5 | 26.18 | 1.8% | 1.5% |
| the full BC20 settings (150×1200, 1-bin z smoothing, adaptive, 12×80 charge mesh, csr1d kicks) | 68 | −1429.4 | 765.1 | 26.59 | 4.3% | 1.8% |

What this shows:
- **The recorded default is itself ~2.7% under-resolved in energy loss** (doubling its resolution: −1379 → −1417 keV).
- **At the same bin counts, the quantile frame is within 0.4% of the doubled default,** because its window hugs the charge and its bins are ~1.6× finer.
- **Dropping fewer tails** (q = 0.0005) spreads the bins again and lands in between. So the 1% of tails dropped by default is not a source of error here.
- **Adaptive smoothing changes nothing** on a smooth 1M-particle Gaussian.
- **CSR-guided kicks give *more* kicks here (68 vs 18):** kick_tol caps each kick's share of the total CSR, so the count follows the CSR distribution along s, not the lattice length. They stay within 2% of the fine reference.
- **The full BC20 settings land on the converged side** (−1429 keV, like the 2× quantile run at −1428). Their 0.27 µm z bins are absurdly fine for a 50 µm Gaussian, so they show small wiggles even with 1M particles: run time 991 s against 117 s for the default.
- The final core slice σE rises from ~880 keV (default) to ~950 keV in the finest runs: the same under-resolution of the default, not an error of the new options.
- **So the new options are correct on the standard case,** and slightly more accurate at equal cost.

![dipole check](../pyDFCSR_2D/test/benchmark_results/quantile_frame/dipole_gauss_check.png)

**Why the BC20 wake before B2RE is still rougher than the old rms grid.**
- On a common physical scale (rms deviation from a 20 µm smoothing, relative to the wake), the adaptive run is 1.2–1.7× rougher than the old grid:
  - B1LE … B2LE: 0.13–0.17 vs 0.10;
  - quads, sextupoles, D5: 0.05 vs 0.04;
  - B3 … Q2ER: 0.21 vs 0.18.
- **The cause is the beam, not the cap on the smoothing width.** In the dispersive section the bunch is a thin, folded ribbon in (x, z): the z spread of particles in strips of 0.2 σξ (in the comoving u) is 48–68 µm at the B2LE exit (σz 109 µm), 12–163 µm at the B3RE exit (a fold), and **5–19 µm at the Q2ER exit** (σz 82 µm).
- The comoving frame removes the linear tilt and a cubic curvature, but not the fold the sextupoles' chromatic terms create. So at fixed u the density really does vary along z on a 5–20 µm scale.
- **The old grid smoothed ~14 µm along z, across the ribbon.** Its wake was smooth because it blurred real 2D structure. With 20 000 particles, resolving the ribbon is noisy.

**A feature-limited width was tried and dropped** (`smoothing_feature_frac`, never committed):
- The z smoothing may grow to α·L, where L = 1/√(|ρ″|/ρ + (ρ′/ρ)²) is the local scale along z from an iterated pilot.
- It widens the smoothing correctly on a smooth bunch, and leaves a spike at its floor (`test_feature_limited_width`).
- But on the BC20 beams it changes little (Q2ER: 1.4–2.2 µm against 1.5 µm without), because L along z really is short there.

**The real remedies:**
- smoothing *along* the local ribbon direction (anisotropic kernels), which is a larger change;
- more macroparticles than the 20 000 of the S2E beam.

The energy loss before Q2ER is about the same in the old and new runs (−0.37 vs −0.38 to −0.45 MeV).

### 9g. 1M particles (upsampled), the current along the lattice, and the peak frame (2026-11-12)

**Upsampling** (`test/facet_bc20/upsample_beam.py`).
- Each of the 20 000 particles gets 49 clones, p + λ(q − p), with q one of its 10 nearest neighbours in the whitened 6D phase space and λ uniform in [−0.5, 0.5]. The clones stay on the beam's own local correlations, and the method adds almost no uncorrelated spread, which Gaussian jitter large enough to decorrelate the clones would add.
- Moments are kept to 0.3%. The pz residual after a cubic fit in z goes 1.92e-4 → 1.96e-4.
- **CSR off through BC20** (`nocsr_1M`), against the 20 000 beam:
  - σz agrees to 0.06% everywhere;
  - the spike's peak current and FWHM agree within the 20k histogram noise (ENDBC20: 48.1 vs 50.6 kA, 5.0 vs 5.5 µm);
  - εnx at the end is 14.90 vs 14.79 µm.
- Not physical below the 20 000-particle sampling; a real 1M S2E beam would be better.

**1D with 1M particles** (`dfcsr_1d_1M`; peak 20/FWHM bins; the 222-kick csr1d schedule; 11.9 min):

| run | → Q2ER | B2RE | drift | B1RE | total [MeV] | εnx 1st [µm] | spike slice ε raw / corr [µm] |
|---|---|---|---|---|---|---|---|
| 1D, 1M | −0.537 | −0.820 | −2.252 | **−3.700** | **−7.308** | 20.88 | 32.6 / 13.2 |
| 1D, 20k, same schedule | −0.537 | −0.849 | −2.244 | −3.899 | −7.529 | 21.06 | 32.1 / 13.2 |
| converged 1D, 20k noise-corrected (7e) | −0.530 | −0.856 | −2.205 | −3.908 | −7.500 | | |

- At 1M the loss in B1RE is 5% lower; everywhere else it agrees to 1–2%.
- So **the 20k shot-noise bias in the spike was 3–5%**, larger than the 1/N half-beam correction of 7e estimated (< 1%). Part of the gap may be the upsampling softening sub-µm spike structure; that can't be separated with an upsampled beam.

**2D with 1M particles**: stopped at s ≈ 28.5 m on request, 72 kicks recorded; partial figures `bc20_wake_maps_long_1M_partial.png`, `_trans_`, `bc20_wake_zproj_1M_partial.png`.
- **The jaggedness before B2RE does not shrink with 50× the particles.** Roughness on the common scale: B1LE–B2LE 0.19 (20k 0.17, old rms grid 0.10); quads and sextupoles 0.08 (0.05, 0.04); B3 onward 0.23 (0.22, 0.15).
- So it is not shot noise. It alternates from one mesh z-point to the next (vertical stripes in the maps), i.e. it depends on the observer position.
- **Hypothesis, not yet tested:** aliasing between a deposition resolved at about 1 bin (0.5 µm) and the coarser sampling of the retarded density by the near-field quadrature (polar patch, x′ bands) in a wide beam. The old grid's ~14 µm smoothing was coarser than that sampling.
- The z600 / z300 runs that would test it were stopped before starting.

**Current along the lattice** (`plot_current_1d.py`, from the 1D 1M line density at every kick; `current_along_s_dfcsr_1d_1M.png`):
- The FWHM of the highest peak goes 420 µm (B1LE) → 210 (B3LE) → 77 (Q2ER) → 13 (mid-B2RE) → 3.4 µm (B2RE exit) → 2.8–3.9 µm to the end.
- The peak current goes 1.3 → 3.9 → 39 → 52 kA.
- A window of ±3 FWHM with a 2-FWHM taper holds 100% of the charge until 36 m, then 48–56% after B2RE: the spike, without the 0.3–3 kA pedestal behind it.

![current along s](../pyDFCSR_2D/test/benchmark_results/facet_bc20/current_along_s_dfcsr_1d_1M.png)

**The peak frame** (`particle_deposition.frame: peak`; uncommitted):
- **Window:** full weight within the peak centre ± `peak_halfwidth` × FWHM (3), a taper over `peak_taper` × FWHM (2), never wider than the quantile window.
- **Long bunches:** they get exactly the quantile frame (the Gaussian dipole case: identical to the last digit).
- **Asymmetric margins:** a side cut by the peak window tapers over 2 FWHM; a side cut by the quantile keeps its margin. The spike sits at the head of the bunch.
- **Centre and FWHM:** `peakfind.peak_center_fwhm`. The arg-max only starts the search; the FWHM uses interpolated half-maximum crossings; the centre is the centroid within ±FWHM/2.
- **Peak identity:** the search stays within 2 half-widths of the previous centre and switches only to a peak 1.3× higher. Without that, two equal horns made the arg-max flip and the rate-limited centre drift into the gap between them: the double-horn thin beam gave 0.3% of the 1D wake.
- **Rate limits:** |Δ ln H| ≤ 0.2 and |Δ centre| ≤ 0.25 H per snapshot.
- **Shared machinery:** the charge mesh, observer vx, mesh ridge and the per-kick dropped-charge diagnostic are those of the quantile frame.
- **Tests:** `test_quantile_frame.py`, 18 pass: equal to the quantile frame on a Gaussian, keeps the spike and drops the pedestal, rate limit, continuity, one of two equal horns.
- **Thin beam** (2D must reduce to 1D; 300 z bins):

| case | quantile (1200 bins) | **peak, 1M** | peak, 20k | peak + adaptive, 20k |
|---|---|---|---|---|
| spike: ⟨W⟩ / 1D | 0.951 | **1.047** | 1.208 | 1.116 |
| spike: peak \|W\| / 1D | 0.95 | **1.003** | 1.08 | 1.006 |
| double horn: ⟨W⟩ / 1D | 0.981 | 0.502 (one horn of two, by design) | 0.565 | 0.559 |

- **At 1M the peak frame reproduces the spike's wake exactly.** The mean is ~5% above 1D: the pedestal is no longer a source, and the taper cuts the pedestal that lies under the spike.
- **At 20k the 300 bins over ±25 µm are too fine for the particle count** (+21% noise bias; adaptive smoothing +12%). The frame needs ~1M particles, or coarser bins, for a 1–3 µm spike.
- **A second significant peak is dropped.** The per-kick 1D diagnostic warns when the dropped charge carries more than `pedestal_warn` (5%) of the loss.

### 9h. BC20 with the peak frame and 1M particles (`dfcsr_2d_peak_1M`)

**Settings:**
- deposition: `frame: peak` (±3 FWHM full weight, 2-FWHM taper, intersected with the quantile window), 100 × 300 bins, 2-bin smoothing;
- wake mesh: charge mesh 12 × 80;
- kicks: csr1d placement (222); 1M upsampled beam, 6 MPICH ranks, **78 min**.

**The window** (statistics h5, `quantile_frame`):
- up to Q2ER it is the quantile window (f_window ≥ 0.99);
- inside B2RE it shrinks as the spike forms: 62.5 µm and f_window 0.66 at 38.0 m;
- from the B2RE exit on it is ~16–24 µm wide with ~6–8 µm tapers: f_window 0.49–0.56, deposition resolution 0.19–0.31 µm.
- The 1D estimate of the dropped charge's share reached **6.4% of the loss** at the end of the drift (above `pedestal_warn` = 5%) and **13% of the field** inside B2RE.

Current in the middle of each dipole, with the window as the peak frame chooses it (emulated on the 1D 1M line density; `plot_peak_window.py`):

![peak window](../pyDFCSR_2D/test/benchmark_results/facet_bc20/current_peak_window_dfcsr_1d_1M.png)

**Results** (energies in MeV, εnx dispersion-corrected 1st / 3rd order in µm, spike slices raw / slice-dispersion-corrected in µm):

| run | → Q2ER | B2RE | drift + wiggler | B1RE | total | rms ΔE | εnx | spike slice ε |
|---|---|---|---|---|---|---|---|---|
| **2D peak frame, 1M** | −0.464 | −1.793 | **−1.443** | −4.136 | **−7.835** | 8.56 | 12.95 / 8.28 | 18.2 / 7.3 |
| 2D quantile + adaptive, 20k | −0.446 | −1.846 | −2.199 | −4.146 | −8.637 | 9.55 | 14.40 / 8.97 | 20.2 / 8.2 |
| 2D quantile fixed, 20k | −0.382 | −1.865 | −1.872 | −4.058 | −8.177 | 9.20 | 14.99 / 9.42 | 21.3 / 8.6 |
| 1D, 1M | −0.537 | −0.820 | −2.252 | −3.700 | −7.308 | 8.79 | 20.88 / 12.13 | 32.6 / 13.2 |
| CSR off, 1M | | | | | | | 14.90 / 7.24 | 15.0 / 6.7 |

What happens:
- **In the bends the peak frame agrees with the quantile frame:** B2RE −1.79 vs −1.85 MeV, B1RE −4.14 vs −4.15. In B2RE 2D still loses 2.2× the 1D value.
- **In the drift and wiggler after B2RE it loses a third less:** −1.44 vs −2.20 MeV (1D −2.25). The projected wake on the spike there is ~35–40% weaker (s = 45.4 m: −0.55 vs −0.88 MeV/m).
  - **The reason is physical.** In a drift, the field on the spike is radiation emitted earlier in B2RE by charge *behind* it, which overtakes it. That charge is the pedestal: low current, but half the charge and 200 µm long. Its radiation adds up on the spike in the drift.
  - So **the assumption that the pedestal's CSR is negligible holds in the bends, but not in the drift after them.**
  - The 1D diagnostic (6.4%) underestimates this 2D effect.
  - The tail feature at 40.0 m seen with the quantile frame is gone too: it came from the pedestal as a source.
- **Emittance:** εnx 12.95 µm, below CSR off (14.90), and the spike slice emittance 18.2 µm. Both are lower than the quantile runs, consistent with the weaker drift wake.
- **Before Q2ER** the peak frame is the quantile window at 300 bins. Its wake follows the old rms grid in shape, but with isolated single-point and single-row outliers. Roughness: 0.18 / 0.06 / 0.19 against the old grid's 0.10 / 0.04 / 0.15.
- **Transverse-force outliers in B1RE:** 6.8 cells per kick, against 10.7 (20k adaptive) and 3.3 (old grid).
- Figures: `bc20_wake_maps_long_dfcsr_2d_peak_1M.png`, `_trans_`, `_spike_`, `bc20_wake_zproj_dfcsr_2d_peak_1M.png`, `bc20_wakes_along_s_dfcsr_2d_peak_1M.png`.

![projected wakes, peak frame 1M](../pyDFCSR_2D/test/benchmark_results/facet_bc20/bc20_wake_zproj_dfcsr_2d_peak_1M.png)

**What it means.** The pedestal can be dropped in a bend, where the field on the spike is local, but not in a drift after a bend, where radiation from behind catches up. Two ways to keep it:
1. a **nested deposition**: the pedestal on a coarse grid plus the spike on a fine one. This is the option set aside earlier; it keeps the pedestal's radiation at low cost.
2. a **window that does not cut the tail in drifts**, e.g. the quantile window downstream of a bend exit.

### 9i. The "isolated outliers" before Q2ER are a real narrow wake feature, not quadrature error (2026-11-12)

**What happens.** At the first kick (s = 0.42 m, B1LE) the worst mesh points all sit in one z column (iz = 5, z = −277.6 µm), 0.4–0.6 max|W| below their neighbours.
- **The input beam has a hard tail edge** at z = −284.2 µm. There are no particles below it, and the first 4 µm bin holds 55 of the 20k (about half the local density). The 1M upsample keeps the edge.
- **Early in a bend the wake is local in z.** Over the 0.42 m of history the slippage is u³/24R² ≈ 1.4 µm (R = 47 m). So the retarded source sits within ~1 µm of the observer, and the far s′ integral maps ∂λ/∂z, at the deposition resolution (~4–5 µm), straight onto the wake. A step in λ therefore gives a wake trough about one deposition resolution wide just ahead of the edge: −0.055 MeV/m at −282 µm.
- **The charge mesh is sparse there.** In the low-charge tail its z spacing is 12–16 µm, so one node (−277.6 µm) lands on the trough and its neighbours miss it. It looks like a single-point outlier, but it is a resolved feature that the mesh under-samples.
- **The second trough, at −242 µm, is the 20k beam's shot noise.** It follows a dip in the line density at −253…−245 µm: 4 µm bins go 100 → 66, 62 → 86 particles, about 3σ. The kNN upsampling copies the 20k structure above the neighbour distance, so the 1M beam (blue in the figure) has the same 5–20 µm structure as the 20k beam (grey).
- **Why the old rms grid looks smoother.** Its deposition (10σz/300 = 4.4 µm cells, 3-bin smoothing) blurs both features over ~14 µm. That is smoothing, not accuracy.

**Checks** (`test/facet_bc20/near_field_check.py`):
- The serial replay to the first kick reproduces the stored MPI wake exactly (max difference 0).
- The three rows with the worst outliers (ix = 2, 1, 4; 80 points each) were recomputed with one factor refined at a time:

| variant | W at the worst point [MeV/m] | row ix 2 roughness (W − 3-pt median)/max |
|---|---|---|
| base (run settings) | −0.04616 | 0.0950 |
| polar patch 4× nr, 4× nφ | −0.04616 | 0.0950 |
| x′ nodes 4× (800) | −0.04616 | 0.0950 |
| near s′ cells 4× finer (361 nodes) | −0.04598 | 0.0950 |
| far s′ nodes 4× (1600) | −0.04632 | 0.0954 |
| far s′ uniform 8000 nodes | −0.04633 | 0.0954 |
| x′ ribbons 3× wider | −0.04606 | 0.0934 |
| observer vx unblended at the window edge | −0.04616 | 0.0951 |
| all of patch, near s′, far 8000, ribbons | −0.04616 | 0.0936 |

- Every variant is within 0.4%. The polar patch carries ~1e-9 MeV/m there.
- **By region, at the worst point:** far −0.0318, near −0.0144 MeV/m. The neighbours' far parts are −0.0062 and −0.0006, while their near parts are smooth (−0.0119, −0.0157).
- **A sweep at 0.5 µm observer spacing** along row 2 (`plot_near_field_sweep.py`) gives a smooth curve, and the mesh nodes lie on it.
- **The 1D run at the same kick** (18 µm bins) sees the edge only as a broad −0.020 MeV/m dip at −273 µm.
- **Minor:** the near part has ~1e-3 MeV/m steps (2% of max|W|) as the observer z moves, e.g. at −368, −344 and −324 µm, from jumps in its s′ node layout. Most of them lie in front of the edge, where there is no charge.

![sweep](../pyDFCSR_2D/test/benchmark_results/facet_bc20/near_field_sweep_dfcsr_2d_peak_1M.png)

**What it means.** The roughness before Q2ER measures structure of the input beam: its hard tail edge, and the 20k shot noise frozen in by the upsampling. A finer quadrature will not remove it. The choice is:
- **keep the structure:** add nodes where |∂λ/∂z| is large as well as where λ is, a gradient term in the charge-mesh monitor;
- **remove it:** set the deposition smoothing to the resolution that is physically meant, as the old grid did implicitly;
- **get a cleaner beam:** an input with more real particles; upsampling cannot remove the 20k noise.

**More z smoothing and a 12 × 40 wake mesh, at the same first kick** (`first_kick.py`; runs `dfcsr_2d_peak_1M_sz<N>_m40` in `run_dfcsr_bc20.py`, so only the first kick is computed).
- The bins here are 2.1 µm. Effective z resolution = bin × √(sz² + 2/3).
- Mesh error is |the mesh nodes, linearly interpolated, − the wake resolved at 1 µm| along row 2, / max|W|.

| deposition z smoothing | eff. z res. | mesh | mesh error max / rms | roughness | particle mean dE/ds [MeV/m] | rms over particles |
|---|---|---|---|---|---|---|
| 2 bins (the run) | 4.6 µm | 12 × 80 | 0.49 / 0.076 | 0.063 | −2.48e-3 | 4.14e-3 |
| 2 bins | 4.6 µm | 12 × 40 | 0.55 / 0.100 | 0.108 | −2.73e-3 | 3.94e-3 |
| 4 bins | 8.6 µm | 12 × 40 | 0.30 / 0.058 | 0.074 | −2.51e-3 | 3.36e-3 |
| 6.5 bins (≈ old rms grid) | 13.8 µm | 12 × 40 | 0.17 / 0.039 | 0.044 | −2.42e-3 | 3.08e-3 |
| 10 bins | 21.1 µm | 12 × 40 | 0.08 / 0.020 | 0.020 | −2.38e-3 | 2.85e-3 |

- **Halving the mesh without more smoothing makes it worse.** Roughness 0.108, and the mean kick is aliased 10% high.
- **At 6.5 bins the shot-noise ripple is gone.** The tail edge leaves one broad −0.03 MeV/m trough instead of a −0.055 one 4 µm wide, and 40 nodes follow the field to 4% rms.
- **The particle-mean loss changes by only −2.6% (6.5 bins) and −4% (10 bins).** The smoothing removes structure, not loss.
- **The catch for a full run.** The smoothing is in bins, and the peak frame's bins shrink to 0.09–0.15 µm after B2RE. 6.5 bins is then 0.6–1.0 µm against a spike of rms ~1.2 µm, which would over-smooth the spike. It needs a physical cap, e.g. smoothing ≤ FWHM/10 of the peak.

![first kick compare](../pyDFCSR_2D/test/benchmark_results/facet_bc20/first_kick_compare.png)

**2D wake maps at this kick.** Two figures (`first_kick.py --maps`): the full ξ extent, and the same zoomed to the old grid's ±4σ. The old rms grid (20k) is shown at its nearest kick, s = 0.396 m.
- **The charge mesh puts most of its x rows in the transverse tails.** The beam has real x tails to ±12 σξ: 0.77% of the particles lie beyond 4σ, in the 20k beam as well as the 1M. The x axis spans every particle, so its 12 rows sit at ±42, 140, 256, 395, 566 and 762 µm (σξ = 60 µm).
  - Only 4 rows fall within ±2.5σ, with 84–98 µm spacing in the core.
  - The old uniform mesh had 12 rows within ±4σ at 43 µm spacing. Its particles outside ±4σ got no kick (fill 0).
- **The outer rows carry a large wake that is antisymmetric in ξ and z.** It is up to 0.05 MeV/m at |ξ| = 13σ: W ≈ ξ ∂F/∂z, the Panofsky–Wenzel x gradient of the transverse force. This is physical, but it acts on few particles.
- **Inside ±4σ**, the 6.5- and 10-bin maps match the old grid's pattern and amplitude: blue in the tail, red at the head, |W| ≲ 0.015 MeV/m away from the tail edge. The transverse force agrees in shape and peak with the old grid: 0.0087–0.0090 vs 0.0091 MeV/m.
- **Open:** the x mesh. For the core, use more x nodes (e.g. 24), a lower `mesh_uniform_frac` in x, or a uniform ±4σ core plus a few tail rows.

![first kick maps zoom](../pyDFCSR_2D/test/benchmark_results/facet_bc20/first_kick_maps_zoom.png)
![first kick maps](../pyDFCSR_2D/test/benchmark_results/facet_bc20/first_kick_maps.png)

### 9j. Full BC20 run: capped 6.5-bin z smoothing, 12 × 40 mesh with a clipped x axis, 116 kicks (2026-11-12)

**New options** (defaults off; golden fixture bit-identical; tests in `test_quantile_frame.py`, 146 pass):
- `particle_deposition.smoothing_res_fwhm`: in the quantile and peak frames, caps the z smoothing so that the effective z resolution is ≤ this fraction of the highest peak's FWHM (floor 0.5 bins). The smoothing is in bins and the bins follow the window, so without the cap 6.5 bins (14 µm on the long bunch) would be 0.6–1 µm on the spike.
- `CSR_computation.mesh_x_quantile`: the charge mesh's x axis spans [x_q, x_1−q] instead of every particle, and particles beyond it get the edge row's kick (`Beam.apply_wakes(clamp_x)`) instead of 0. At 0.002 the 12 rows span ±5.3σ with 37 µm core spacing.
- **Fewer kicks:** in 1D (`kick_count_check.py`, against a kick at every snapshot), `kick_tol 0.02, kick_h_max 0.75` gives 115 kicks: ⟨ΔE⟩ +0.26%, εnx 1st order 5.0% of the CSR change, per-particle x 3.8%. The 220-kick setting gives −0.04%, 1.5% and 1.7%.

**Run** `dfcsr_2d_peak_1M_sz6.5cap_m40_k115`: peak frame, 1M particles, `smoothing_sigma_z 6.5`, `smoothing_res_fwhm 0.1`, charge mesh 12 × 40, `mesh_x_quantile 0.002`, 116 kicks, 6 MPICH ranks, **46 min** (78 before). The time splits as follows:
- the snapshot steps take ~30 min whatever the kicks: 2062 deposits of 1M particles at 0.9 s;
- a kick costs ≈ 4 s fixed (1M-particle work: diagnostics, mesh, v_x fit, interpolation) + 9 ms per mesh point.

| run | → Q2ER | B2RE | drift + wiggler | B1RE | total | rms ΔE | εnx 1st / 3rd | spike slice ε |
|---|---|---|---|---|---|---|---|---|
| **new, 116 kicks** | −0.684 | −1.863 | −1.864 | −3.999 | **−8.409** | 8.73 | 13.05 / 8.23 | 18.5 / 7.3 |
| peak 1M, 222 kicks (9h) | −0.464 | −1.793 | −1.443 | −4.136 | −7.835 | 8.56 | 12.95 / 8.28 | 18.2 / 7.3 |
| 1D, 1M | −0.537 | −0.820 | −2.252 | −3.700 | −7.308 | 8.79 | 20.88 / 12.13 | 32.6 / 13.2 |

**What happens.**
- **The wake before Q2ER is now as smooth as the old rms grid's.** Roughness 0.094 / 0.031 / 0.154 against 0.102 / 0.039 / 0.181. In the projections it follows the old grid. Emittance and the spike slice emittance are unchanged.
- **The section differences come from two short 2D-only features in s that the 1D-guided schedule cannot see.** With fewer kicks each is sampled by 2–4 kicks. Energy change by segment, from the per-step mean energy (number of kicks in brackets):

| segment [m] | 0–18.5 | 18.5–20 | 20–36.65 | B2RE | 38.67–39.5 | 39.5–40.5 | 40.5–43.5 | 43.5–48.0 | B1RE |
|---|---|---|---|---|---|---|---|---|---|
| new (116) | −0.200 (28) | **−0.286 (2)** | −0.198 (26) | −1.863 (11) | **−1.485 (6)** | **+1.029 (4)** | +0.028 (8) | −1.268 (9) | −4.166 (22) |
| 222 kicks | −0.215 (45) | **−0.031 (3)** | −0.218 (42) | −1.793 (24) | **−1.233 (18)** | **+1.135 (11)** | +0.033 (19) | −1.306 (20) | −4.207 (40) |

  - **At 18.8–19.3 m (drift D5, the εnx minimum in the sextupole region)** the 2D wake swings to −0.45…−0.68 MeV/m and then changes sign within ~0.5 m. One 0.71 m kick at 18.91 m gives −0.37 MeV, against −0.18 + 0.10 from the 0.46 m kicks. That is the whole pre-Q2ER difference.
  - **After the B2RE exit** the 2D wake has a negative lobe (38.7–39.5 m) and then a +1.1 MeV positive lobe (39.5–40.5 m). The 1D wake does not have the second lobe, so the 1D-based schedule puts few kicks there. The lobes cancel to −0.46 MeV with 10 kicks and to −0.10 MeV with 29: that is the whole drift difference.
  - Everywhere else the two runs agree to ≤ 0.04 MeV per segment.
- **So the 222-kick results are not converged in these two segments either** (3 kicks across the 18.9 m feature). The pre-Q2ER and drift 2D-vs-1D gaps discussed earlier (7d, 9c, 9h) are partly kick sampling of 2D-only transients.

Figures: `bc20_along_s_1M.png` (⟨ΔE⟩, εnx, I_pk, FWHM at every element exit), `bc20_wake_zproj_…k115.png`, `bc20_wakes_along_s_…k115.png`, `bc20_wake_maps_long_…`, `_trans_`, `_spike_…k115.png`.

![along s 1M](../pyDFCSR_2D/test/benchmark_results/facet_bc20/bc20_along_s_1M.png)
![zproj k115](../pyDFCSR_2D/test/benchmark_results/facet_bc20/bc20_wake_zproj_dfcsr_2d_peak_1M_sz6.5cap_m40_k115.png)

**The noisy transverse force in B1RE is unconverged near-field quadrature on the pedestal** (`kick_check.py`, replay to the kick at s = 48.39 m, central mesh rows 5 and 6).
- **The pattern, in every 2D run:** in B1RE the pedestal F is as large as at the spike (0.4–2× in the peak runs and the old grid, 10× in the 20k quantile run) and almost pure cell-to-cell scatter (roughness 0.74–0.98). W is small there (2–8% of the spike). The pattern appears only once the spike exists, and also in the peak frame, where the pedestal is not deposited at all, so it is not pedestal shot noise.
- **Not the far region:** it carries none of the pedestal F (share 0.000). 4×, 16× and 40× far nodes give identical results.
- **The near region:** 4× finer near s′ cells or 4× x′ nodes move the pedestal F by ±0.03–0.06 MeV/m, the size of F at the spike (0.06–0.10). Pedestal roughness goes 0.78 → 0.85 (sp4) and → 0.75 (xp4). F at the spike column, and W everywhere, are unchanged to 1e-4.
- **Likely mechanism:** the observer in the pedestal has the spike ahead of it, inside the near region (s4 = s + 3σz, σz = 51 µm rms). A source u ahead is seen at z_ret ≈ z + 2u, so the 3 µm spike is crossed over ~1.5 µm of s′, against near cells of a few µm. The transverse integrand carries ∂ρ/∂t, i.e. the spike's z derivative, with O(1) weight (W2, W3 ~ ∂tρ/R); in W it is suppressed by (1 − β·β′).
- **Not yet confirmed:** the near s′ refinement to 16× and 64× (and a 0.5 µm observer sweep through the pedestal) was started and stopped by the user.
- **If confirmed, the fix is local:** fine s′ nodes where the retarded position crosses a narrow current peak, not everywhere.

![kick check B1RE](../pyDFCSR_2D/test/benchmark_results/facet_bc20/kick_check_dfcsr_2d_peak_1M_sz6.5cap_m40_k115_s48.38.png)

### 9k. The B1RE transverse noise is the W2 + W3 part of the force, and W2 + W3 cannot be dropped (2026-11-17)

**The decomposition.** The transverse integrand is W1 + W2 + W3 (Stupakov PRAB 25, 014401): W1 ∝ (r − r′)·(n − n′) ρ/|r − r′|³ carries the density, W2 ∝ (r − r′)·(n − n′) ∂ₜρ/|r − r′|² and W3 ∝ −(n·τ′) ∂ₜρ/|r − r′| carry its time derivative. On the arc (x = x′ = 0) the W2 and W3 geometry factors are −1/R and cos(θ/2)/R exactly, so W2 + W3 = −(1 − cos(θ/2)) ∂ₜρ/R: two O(1/R) terms whose sum is O(θ²/8R). New option `CSR_integration.transverse_terms: all | W1` (default `all`, bit-identical) to separate them.

**Runs.**
- `rerun_b1re_W1.py`: the 116-kick run replayed to the B1RE entrance (48.02 m), then every B1RE kick computed twice at the same beam state, all terms (stored) and W1 only (applied). Saved as `dfcsr_2d_peak_1M_sz6.5cap_m40_k115_b1reW1`; both forces per kick in `b1re_W1_<run>.npz`.
- `dipole_gauss_check.py W1`: the standard Gaussian dipole (R = 1 m, 1 rad, σz = σx = 50 µm), both forms at every kick, W1 applied.

**What happens in B1RE** (22 kicks, medians; pedestal = z < spike − 15 µm):

| | all terms | W1 only |
|---|---|---|
| max\|F\| at the spike [MeV/m] | 0.133 | 0.207 |
| pedestal max\|F\| / spike max\|F\| | 1.07 | 0.056 |
| pedestal roughness (rms of F − 3-node median / rms F) | 0.71 | 0.034 |

- **All of the pedestal scatter is in W2 + W3.** W1 alone is smooth everywhere (roughness 0.03) and small on the pedestal (6% of the spike). F_all − F_W1 = W2 + W3 is the speckle itself: ±0.1 MeV/m cells with no spatial correlation, as large as the force at the spike.
- **At the spike W2 + W3 is smooth and large:** it reduces the peak force by 35% (0.207 → 0.133), as a clean blue stripe on the spike column in `b1re_W1_maps.png`. So the two terms are well resolved where the charge is, and unresolved only for pedestal observers, which see the spike's ∂ₜρ ahead of them crossed over ~1.5 µm of s′ (9j).
- **Beam at ENDBC20 with W1 only in B1RE:** εnx 12.30 / 7.86 µm (1st / 3rd order) against 13.05 / 8.23; spike slice ε 16.5 / 6.5 against 18.5 / 7.3; B1RE loss −4.056 vs −3.999 MeV. Caveat: the replay is not bit-identical upstream of B1RE, because `u_taper` became default-on (a97b2a6, 2026-11-12 13:17) after the base run (08:30); the sections before B1RE differ by ≤ 0.024 MeV from that.

**Why W2 + W3 cannot be dropped: the Gaussian dipole.**

| s | where | max\|F\| all | max\|F\| W1 | rel. L2 of W2 + W3 vs F |
|---|---|---|---|---|
| 0.16–1.03 m | in the bend | 0.58 → 0.44 MeV/m | 0.57 → 0.47 | 0.10–0.20 |
| 1.10 m | bend exit | 0.23 | 0.29 | 0.24 |
| 1.17 m | 7 cm after | 0.0036 | 0.017 | 3.5 |
| 1.41 m | 31 cm after | 0.00018 | 0.0018 | 14 |

- In the bend W2 + W3 is a 10–20% correction with a definite shape (weakens the head, strengthens the tail). After the exit it cancels W1: the true force dies 60× within 7 cm, W1 alone decays slowly (n ≠ n′ for sources still in the bend). W1 only would be 5–10× too large in drifts after bends.
- Beam: energy loss unchanged (−1387 vs −1379 keV), εn 26.84 vs 26.00 µm (+3%). No noise in either form here (roughness 0.003).

![dipole transverse terms](../pyDFCSR_2D/test/benchmark_results/quantile_frame/dipole_transverse_terms.png)
![B1RE W1 maps](../pyDFCSR_2D/test/benchmark_results/facet_bc20/b1re_W1_maps.png)

**What it means.** The noise is the numerical cancellation of W2 against W3 for observers that have a narrow spike inside their near region, where the sampling of ∂ₜρ across the ~1.5 µm crossing is too coarse. The terms themselves are physical and needed. Remedies, in order of preference:
1. refine the near-region s′ nodes where the retarded position crosses a narrow current peak (local, cheap);
2. combine the two geometry factors before multiplying by ∂ₜρ, g = (r − r′)·(n − n′)/|r − r′|² − n·τ′/|r − r′|, which removes the round-off part of the cancellation but not the sampling part;
3. not: drop W2 + W3.

### 9l. Adaptive near-region s′ nodes (`CSR_integration.near_adapt`) (2026-11-17)

**What it does.** The near-region s′ nodes follow the retarded density as the column sees it, instead of a fixed cell of 0.25 σξ. For each observer the column is scanned (band location + density lookup at the band's axis point), every scan interval is split into ⌈Δz_ret/δ⌉ pieces until the retarded z moves by at most δ = `near_adapt_scan_dz` (0.25) × res_z inside the history's z support (at most `near_adapt_max_scan` = 4000 points; the support edge is located to δ), and the nodes equidistribute

  m(s′) = 1/h_fixed(s′) + |d(ρ/ρ_max)/ds′| / `near_adapt_eps` (0.1),

graded to `near_adapt_max_ratio` (1.5), at most `near_adapt_max_nodes` (2000). The first term keeps every cell at least as fine as the fixed layout, the second puts ~2/ε nodes across each rise and fall of the source. No peak finding: spikes, horns and edges are all gradients. Off by default (golden fixture bit-identical); `test_near_adapt_gaussian_close_to_fixed`.

**Why the fixed layout fails** (9j–9k): behind the observer z_ret ≈ z − u³/24R², so even a spike is a slow source and 41 µm cells are fine; ahead of it z_ret ≈ z + 2u, so a 3 µm spike is crossed in 1.5 µm of s′. At ENDBC20 (σξ 165 µm) the fixed cell is 41 µm and the polar patch (R2 = 62 µm, dr 0.6 µm) only covers observers within ~60 µm of the spike.

**B1RE kick at s = 48.39 m** (`kick_check.py`, rows 5 and 6, 80 points, plus a 0.5 µm observer sweep through the pedestal on row 5; sp64 = the fixed cells 64× finer, ~5000 nodes, the brute-force reference):

| variant | near nodes (pedestal observer) | pedestal max\|F\| / spike | rms diff from sp64_xp4 on the pedestal, / spike max | sweep noise rms [MeV/m] |
|---|---|---|---|---|
| base (fixed) | 108 | 0.80 | 0.257 | 0.025 |
| **adapt** | 146 | **0.41** | 0.139 | 0.0074 |
| sp64 | ~5000 | 0.41 | 0.140 | 0.0027 |
| adapt + 4× x′ | 146 | 0.14 | **0.020** | — |
| sp64 + 4× x′ | ~5000 | 0.10 | 0 (reference) | — |

- **The adaptive s′ layout reproduces the 64× brute force** on the mesh rows (the two curves lie on top of each other), with 146 nodes instead of 5000. The cell at the spike crossing is 0.29 µm (fixed: 44.75 µm). F at the spike column and W are unchanged to 1e-4.
- **The pedestal force halves** (0.80 → 0.41 of the spike force) and the sweep noise drops 3.4×. The remaining pedestal structure is **x′ sampling**: 4× x′ nodes take the residual from 0.139 to 0.020 of the spike force in both the adaptive and the brute-force layouts. So the same adaptivity is needed across the band.
- **The sweep still has 3× the brute force's noise** (0.0074 vs 0.0027): the adaptive node set changes from observer to observer, which modulates the result at the ~0.007 MeV/m level (7% of the spike force). A finer scan (`near_adapt_scan_dz`) or smaller ε should reduce it; not tested.

**Cost** (corrected 2026-11-18: the 117 s "default" used first was a 2026-11-12 run on older code; re-timed today on the idle machine). Gaussian dipole whole run: default 94 s, `near_adapt` 126 s (**+34%**); per wake point 11–12 ms → 17–19 ms, of which the scan is 5–7 ms (3–4 band solves, ~1.5 ms of Python overhead each; the first version bisected 16 times and cost 70–80 ms) and the extra 10–17 nodes on 71 the rest. Wakes identical (rel 3e-4). On the BC20 B1RE kick (6 ranks, wall-clock per point): base 189 ms, adapt 286 ms (+50%); pedestal observers get +35% near nodes and a 420-point scan.

![kick check adapt](../pyDFCSR_2D/test/benchmark_results/facet_bc20/kick_check_dfcsr_2d_peak_1M_sz6.5cap_m40_k115_s48.38.png)

**Adaptive x′ nodes (`near_adapt_x`, 2026-11-18).** z_ret depends on x′ as well (dz_ret/dx′ ≈ Δx/|r − r′|, O(1) for a source ahead), so the spike is crossed *within the band* over ~6 µm of x′ against the uniform 16 µm cells (band ≈ 3 mm / 200). After the uniform evaluation of a near column, every column whose retarded density jumps by more than ε·ρ_max between neighbouring x′ nodes is re-evaluated on x′ nodes equidistributing 1/h_uniform + |d(ρ/ρ_max)/dx′|/ε (2 rounds, ≤ 4 × xbins nodes); all flagged columns get the same count, so it is one more rectangular evaluation. The first "adapt_x" check (log_kick_check4) had not exercised it, the harness called the integrator without the flag; redone (log_kick_check5):

| variant | pedestal cells: rms diff from the 64× s′ + 4× x′ brute force, / spike max\|F\| | same for W | sweep noise rms [MeV/m] | ms per point (6 ranks) |
|---|---|---|---|---|
| fixed | 0.257 | 0.0046 | 0.025 | 176 |
| near_adapt | 0.154 | 0.0035 | 0.0066 | 287 |
| **near_adapt + near_adapt_x** | **0.055** | **0.0008** | 0.0060 | 326 |
| brute force (sp64_xp4) | 0 | 0 | 0.0016 | 15 100 |

- On the pedestal observer at z = −55 µm, 77 of the 144 near columns were flagged and re-evaluated with 120 x′ nodes (17 700 extra integrand evaluations, +13% per point over `near_adapt`).
- The remaining 0.055 is mostly the observer-to-observer jitter of the adaptive node sets (sweep noise 0.0060 vs 0.0016 for the brute force), not a systematic bias: the mesh-row curves lie on the brute-force curve to within that scatter.
- Dipole (idle machine, whole run): default 94 s, `near_adapt` 126 s, + `near_adapt_x` 138 s; wakes identical (3e-4).

**Full B1RE with both options** (`rerun_b1re_W1.py adapt`: replay to 48.02 m, then each of the 22 B1RE kicks computed with the fixed layout (stored) and the adaptive one (applied); 51 min on 6 ranks; saved as `…_b1reAdapt`, forces in `b1re_adapt_<run>.npz`):

| B1RE, medians over 22 kicks | fixed layout | adaptive s′ + x′ |
|---|---|---|
| max\|F\| at the spike [MeV/m] | 0.133 | 0.132 |
| pedestal max\|F\| / spike | 1.14 | 0.22 |
| pedestal rms F [MeV/m] | 0.027 | 0.0082 |

- The pedestal force drops 3.3× in rms and the spike force is unchanged; the difference map (fixed − adaptive) is the speckle itself. In the last three kicks (49.01–49.08 m, the B1RE exit) the pedestal still carries 0.6–0.9 of the spike force with the adaptive nodes: either a real exit feature or residual jitter, not separated yet.
- **End beam:** εnx 13.31 / 8.45 µm (1st / 3rd order), spike slice ε 18.4 / 7.3, B1RE loss −4.046 MeV, total −8.478. The W1-only replay (same upstream code) gave 12.30 / 7.86, 16.5 / 6.5, −4.056; the base run (older upstream code, `u_taper` off) 13.05 / 8.23, 18.5 / 7.3, −3.999. So removing the pedestal speckle changes the emittance by ≲ 2% and the spike slice emittance not at all: the noise was on the pedestal and acted on particles that carry little of the emittance. The W1-only run's lower emittance came from dropping the physical W2 + W3 at the spike (−35% of the spike force), not from removing noise.

![B1RE adapt maps](../pyDFCSR_2D/test/benchmark_results/facet_bc20/b1re_adapt_maps.png)

**Jitter study on the B1RE kick (2026-11-19, `log_kick_check6`).** Two of the four variants were void: the check script cast the float factors to int, so ε became 0 (the finite guard then fell back to the fixed layout, identical to `base`) and δ became 0. Fixed. The δ → 0 variant is still informative: a scan refined to 2756 points gave a pedestal residual of **0.101** of the spike force against **0.055** for the standard scan (420 points), with 134 vs 144 nodes. So the residual is sensitivity to the node set itself, not scan resolution: the 64× *uniform* layout is smoother (sweep noise 0.0016) at a coarser cell (0.64 vs 0.28 µm) because uniform nodes alias smoothly as the crossing moves with the observer, while adaptive sets jump. Two code crashes on the way (0/0 in the s′ and x′ gradient monitors from coincident nodes) are guarded (f867bbf, fad18ca).

Further jitter work moves to a minutes-scale test bed, `quantile_frame/jitter_spike_pedestal.py`: the thin-beam spike-on-pedestal profile with σx = 100 µm (fixed near cell 25 µm against a 0.6 µm crossing) in the 5 m dipole, one history, then the pedestal sweep for every quadrature variant against the 64× + 4× x′ reference.

**Test-bed result** (`jitter_spike_pedestal.py`, 1e6 particles, R = 5 m, frozen beam; pedestal sweep at 0.5 µm, / spike max\|F\| = 0.098 MeV/m; "noise" = rms of F minus its 5 µm smoothing, "diff" = rms(F − sp64_xp4)):

| variant | axis: noise / diff | +0.5 σx: noise / diff | ms per observer |
|---|---|---|---|
| fixed | 0.025 / **0.096** | 0.013 / 0.020 | 17 |
| near_adapt | 0.0036 / **0.0010** | 0.0085 / 0.0077 | 25 |
| + near_adapt_x | 0.0036 / 0.0014 | 0.0085 / 0.0076 | 31 |
| + ε 0.05 | 0.0036 / 0.0009 | 0.0087 / 0.0047 | 34 |
| + δ 0.1 res_z | 0.0036 / 0.0014 | 0.0085 / 0.0065 | 34 |
| brute force (64× s′, 4× x′) | 0.0036 / 0 | 0.0084 / 0 | 361 |

- The fixed layout reproduces the BC20 failure (10% of the spike force on the pedestal, 0.6 µm crossing in 25 µm cells). The adaptive layout matches the brute force to **0.1%** on axis and 0.8% off axis, and the "noise" is identical in every variant including the reference, i.e. it is real z structure. **No jitter here.**
- So the 5.5% residual in B1RE comes from what this case lacks: the dispersed, tilted and curved BC20 beam (two band branches, curvature shear), or a history that changes along the retarded path (the beam compresses). To study it on the real B1RE state without replays, `dump_kick_state.py` pickles the whole CSR2D at the kick once (~200 MB); experiments then take seconds. The brute-force sweep on the same observers is already stored (`kick_check_…_s48.39.npz`, `sweep_sp64_xp4`).

![jitter test bed](../pyDFCSR_2D/test/benchmark_results/quantile_frame/jitter_spike_pedestal.png)

**Resolved on the pickled B1RE state (2026-11-19, `b1re_state_experiments.py`; the pickle reproduces the stored replay sweep exactly).** The "jitter" was two things: **the brute-force reference was not converged in x′**, and **ε = 0.1 is too coarse on this beam**.
- x′ convergence of the 64×-s′ reference at fixed pedestal observers: with 400 / 800 / 1600 / 3200 x′ nodes the force at z − z_spike = −90 µm is −0.0039 / **+0.0100** / −0.0010 / +0.0018 MeV/m. The 800-node reference had an aliasing bump of 0.1 spike-F there (the one the adaptive layouts "missed"). Converged, the pedestal force is smooth: 1–2% of the spike force on the far pedestal, 5% at −30 µm, 11% at −15 µm. The new reference is `adapt_e02_xp16` (ε 0.02, 3200 x′), which agrees with the 64×/3200 brute force to 0.8% of the spike force at every 10th observer (`b1re_state_sp64_xp16_sparse.npz`).
- Against it (pedestal sweep, row 5; rms(F − ref) / spike max|F|; ms per observer serial):

| variant | diff | ms/obs |
|---|---|---|
| fixed layout | 0.349 | 15 |
| near_adapt, ε 0.1 | 0.098 | 28 |
| + near_adapt_x, ε 0.1 | 0.061 | 33 |
| ε 0.05 | 0.040 | 38 |
| **ε 0.02** | **0.015** | 55 |
| ε 0.02, 4 x′ rounds | **0.0046** | 80 |
| ε 0.02, base x′ 400 | 0.0015 | 85 |
| ε 0.01 | 0.0073 | 95 |
| 64× s′ + 3200 x′ | (0.008) | 5600 |

- **Full 12 × 40 mesh at this kick** (serial): fixed 9.6 s, ε 0.1 16 s, ε 0.02 23 s, ε 0.02 + 4 rounds 27 s. Pedestal F roughness / spike: 0.099 → 0.030 → 0.0056 → 0.0023; pedestal rms F / spike converges to 0.027 (its true value) from 0.199 (fixed) and 0.056 (ε 0.1). W and the spike force are unchanged throughout.
- So the scheme converges and no smoothing of the monitor is needed; the ε 0.1 default was simply too loose for a tilted, two-branch retarded density (the frozen test-bed beam was fine at 0.1). **Defaults changed to ε = 0.02 and 4 x′ rounds** (they only act when `near_adapt` is on). The full B1RE run above (`…_b1reAdapt`) used ε 0.1; its applied pedestal force still carried ~5% of the spike force in residual noise, which the emittance did not notice.
- Cost of the new defaults on the Gaussian dipole (idle machine, whole run): default 94 s, `near_adapt` ε 0.1 126 s, `near_adapt` + `near_adapt_x` at ε 0.1 138 s, **at ε 0.02 with 4 x′ rounds 166 s (+77%)**; wakes identical (3e-4), εn 26.0028 vs 26.0022 µm.
- Caveat on the earlier numbers in this section (and in `jitter_spike_pedestal.py`): "rms difference from sp64_xp4" used the 800-x′ reference; the 0.055 / 0.139 / 0.257 residuals are overstated by its own aliasing. The ordering of the variants was right, the absolute values were not.

![B1RE state experiments](../pyDFCSR_2D/test/benchmark_results/facet_bc20/b1re_state_experiments.png)

**Cheaper scan (2026-11-19).** `near_adapt_scan_iter`: the column scan's band solve with 4 or 6 fixed-point iterations gives nodes identical to 12 (F and W to the last bit on the B1RE pedestal sweep); 3 iterations differ at 1e-7 MeV/m, 2 at 3e-4. Default 4: 63 → 55 ms per observer (−13%). The rest of the adaptive cost is the extra nodes, so reusing the integration's own band solve (≤ 10% more) was not pursued.

**The B1RE-exit pedestal force (2026-11-19, pickled state at the last kick, s = 49.08 m; `b1re_state_experiments.py` with `B1RE_S=49.07`).** The 0.6–0.9 × spike-force pedestal values at the exit kicks of the ε = 0.1 run were residual error, not an exit feature. On the pickled exit state, pedestal sweep on row 5 (rms(F − converged reference) / spike max|F| = 0.097 MeV/m):

| variant | diff | ms/obs |
|---|---|---|
| fixed layout | 0.172 | 31 |
| near_adapt + x, ε 0.1, 2 rounds (old defaults) | 0.069 | 64 |
| ε 0.02, 4 rounds (defaults) | **0.0041** | 150 |
| reference ε 0.02, 3200 x′ | 0 | 349 |

Converged, the pedestal force at the exit is as small as mid-B1RE: +0.0017 MeV/m on the far pedestal (2% of the spike force) at both kicks, rising smoothly only within 20 µm of the spike. Figure `b1re_state_experiments_s49.07.png`.

**Next for this item:** the far-region x′ sampling in the exit transient (drift after B2RE) was not checked at this level.

**Next:** a kick-convergence check in 2D over 18–20 m and 38.6–41 m: a kick at every snapshot there, using `step_control` manual kicks or a local `kick_h_max`. Then either make the schedule see these features (a 2D refinement pass, or a finer `kick_h_max` in the few metres after each bend exit), or confirm whether they are physical.

### 9m. Kick convergence of the two 2D-only transients (2026-11-19, `rerun_kick_dense.py`)

**What was run.** The 116-kick schedule plus a kick at every snapshot (25 mm) inside [18.0, 20.0] m and [38.6, 41.0] m: 297 kicks, 115 min on 6 ranks (shared for the first hour). Same deposition and fixed quadrature as the 116-kick run (`…_kdense`).

**Energy change per segment [MeV] (kicks in the segment):**

| run | 0–18 | **18–20** | 20–36.65 | B2RE | **38.7–39.5** | **39.5–40.5** | 40.5–43.5 | 43.5–48 | B1RE | total |
|---|---|---|---|---|---|---|---|---|---|---|
| dense | −0.132 (27) | **−0.009 (100)** | −0.217 (26) | −1.794 (13) | **−1.325 (33)** | **+1.035 (40)** | +0.024 (27) | −1.260 (9) | −4.267 (22) | **−7.945** |
| 116 kicks | −0.126 | −0.360 (3) | −0.198 | −1.863 | −1.485 (6) | +1.029 (4) | +0.028 | −1.268 | −4.166 | −8.409 |
| 222 kicks | −0.158 | −0.088 (4) | −0.218 | −1.793 | −1.233 (18) | +1.135 (11) | +0.033 | −1.306 | −4.207 | −7.835 |

- **18–20 m (drift D5, the sextupole region):** converged to **−0.009 MeV**. The wake there swings to ±0.7 MeV/m and changes sign within 0.5 m, so its integral nearly cancels; 3 kicks gave −0.36 MeV, 4 kicks −0.09. **The whole "pre-Q2ER 2D excess" of the 116-kick run was this aliasing**: dense −0.358 MeV to Q2ER against 1D −0.537, i.e. 2D loses a third *less* than 1D there, as the old rms-grid runs had said (−0.37 … −0.38).
- **After the B2RE exit:** the negative lobe (38.7–39.5 m) and the positive one (39.5–40.5 m) net to **−0.29 MeV** converged; 116 kicks gave −0.46, 222 kicks −0.10. Drift + wiggler: dense −1.693, 116 −1.864, 222 −1.443, 1D −2.252.
- **Totals:** dense −7.945; the 116- and 222-kick runs bracket it (−8.41, −7.84) with opposite errors of 0.1–0.5 MeV. B1RE differs by 0.1 MeV because the beam arriving there differs.
- **Beam:** εnx 12.91 / 8.24 µm (116: 13.05 / 8.23; 222: 12.95 / 8.28); **spike slice emittance 17.4 / 6.9 µm against 18.5 / 7.3 and 18.2 / 7.3**, a 6% change from the schedule alone, so the post-B2RE transient is felt by the spike.

**What the dense kicks show** (`bc20_kdense_windows.png`): in D5 the projected wake is a smooth, broad feature, negative (−0.3 MeV/m charge-weighted) at 18.5–18.9 m turning positive (+0.3) at 19.0–19.2 m, i.e. an antisymmetric transient across the x-waist, plus **two single-kick glitches**: at s = 19.11 m the wake jumps to +0.7 MeV/m for one snapshot and at 19.29 m to −0.23, each spanning the whole bunch in z. One-snapshot features across all z are not a beam effect; they look like a band-location switch (the two Eq 4.24 branches at |τ| ≈ 1). In the dense run they weigh 25 mm each and are harmless; in a sparse schedule a kick landing on one would apply it over ~0.7 m, 0.5 MeV. After the B2RE exit the dense charge-weighted wake oscillates kick to kick by ±0.05 MeV/m between 39.3 and 39.7 m, a sawtooth that also points to a discretisation in the far-region layout (causal edge or fine window) rather than physics.

**The refined placement (2026-11-20).** Three additions to `kick_placement: csr1d`, all opt-in:
- `kick_prepass_particles` (20 000): the 1D pre-pass runs on a random subsample of the beam. Its cost was the tracking of 1M particles (6 min); the profile only needs the shape of r(s). Now 13 s including the construction.
- `kick_kappa_refine` (4): a kick per κ steps wherever the auto step profile refines the steps (h_eff below 0.75 of its median, which is where the beam's size or tilt changes fast: the D5 waist, the sextupole waists, the edges). This is the old κ rule applied only where the CSR-off scan says the beam is changing; the profile had marked 18.87–19.39 m with 41 refined steps all along.
- `kick_h_exit` (0.1 m): kick spacing ≤ 10 cm for R·φ past each bend exit, the exit wake's decay length (R·φ = the bend length for a sector bend: 2 m after B2RE).

On BC20 this gives **269 kicks** (116 before): 10 across the D5 feature (3 before), 29 in the 2 m after B2RE (10), 40 in B1RE (22), 60 before 18 m (27, from the sextupole-region waists and the B1LE/B2LE exits). A first attempt with "refined = h_eff < h_max" gave 616 kicks, because the dyadic snap puts the quiet stretches one level below h_max everywhere; hence the median-relative criterion. Tests: `test_kicks_from_profile_refinement_terms`.

**The 19.123 m glitch, examined on a pickled state (2026-11-20, `glitch_1911.py`).** The state at 19.110 m was pickled (dense schedule) and advanced one snapshot by hand (`step_state`: track, deposit, rebuild the interpolant); it reproduces the dense run's stored wake at 19.123 m to 7e-5 MeV/m. Findings:
- The glitch sits exactly at the branch degeneracy: τ = −1.003 (sin 2α = −1.000); one snapshot earlier τ = −1.153. At 19.123 the second Eq 4.24 branch is dead on every column (live fractions 0.15 / 0.00 against 0.25 / 0.16 at 19.110), and the step is inside a sextupole (SextupoleKDK), i.e. the D5 x-waist of the dispersed, chirped beam, where the tilt sweeps through 45° at 0.15 per 12.5 mm.
- **Within the band path the glitch is converged, so it is not a quadrature artefact of that path.** Full mesh, ⟨W⟩: 19.110 m +0.230 (default margin 2) → +0.377 (margins 6–24, 200–1600 x′, all within 0.2%); 19.123 m +0.762 → +0.838. At three mesh points, margin 24 with 800 x′, 8× finer near s′, 4× far nodes and a 2× patch changes the default values by ≤ 5% at the core and head, 20% at the tail. The factor-2.2 jump between the two snapshots survives every refinement.
- **The legacy fixed-region integration is not a usable reference at this tilt**: at the same three points it gives +0.96 / +0.17 / +0.50 with 200 nodes, +0.32 / −0.13 / +0.38 with 800, +0.28 / +0.46 / +0.36 with 1600; its apparent smoothness through the glitch (scan: +0.166, +0.156, +0.137 …) is unconverged noise.
- **Reading:** at τ = ±1 the retarded light cone is tangent to the 45°-tilted ribbon (tan 2α → ∞, the band Jacobian → 0): a caustic of the retarded geometry, where a long stretch of the ribbon is seen at once. A sharp enhancement of the 2D wake while the tilt crosses 45° is therefore plausible physics, not established. Its energy content is negligible (+0.76 MeV/m × 12.5 mm ≈ +0.01 MeV) and the sign-changing D5 transient integrates to ~0; the transverse kick there was not examined. An independent check needs a Liénard–Wiechert sum at one observer, the open item from progress.md.
- **A real defect found on the way, fixed:** the default ribbon margin (2) under-covers the source where |sin 2α| → 1: −40% at 19.110 m, −10% at 19.123 m, converged by margin 6. `CSR_integration.xi_band_margin_degenerate` (default 3) multiplies the margin by 3, ramped in with a smoothstep between |sin 2α| = 0.8 and 0.95 (`CSR2D.degenerate_margin_factor`). On the pickled states: 19.110 m +0.230 → +0.378, 19.123 m +0.762 → +0.838, both equal to the margin-6…24 converged values; B1RE (|sin 2α| = 0.05) unchanged to the last bit; golden fixture bit-identical. It changes results only while the tilt passes through |τ| ≈ 0.5–2. On the closed chicane (`2dPm` vs `2dP`, 27.7 min): total −2687.5 vs −2687.4 keV, every element and every beam quantity identical to the printed digits (B4 −239.8 vs −239.7), so the chicane's passage through |τ| ≈ 1 in B2/B3 contributes nothing measurable there. The fix matters only for a tilt crossing that coincides with strong CSR; in BC20 it is the D5 waist, at the 0.01 MeV level.

**What it means.** The 1D-guided placement (`kick_placement: csr1d`) was blind to wake features that exist only in 2D; here they sit in a drift far from any bend (18.9 m, at the x-waist in D5) and in the 2 m after the B2RE exit. Whether the 18.9 m feature is physical (the folded ribbon passing through a waist) or a band-location artefact at |τ| ≈ 1 is not established; its net energy is ~0 either way. The schedule needs either a 2D refinement pass (a cheap 2D wake proxy on a coarse mesh at every snapshot) or, pragmatically, a finer `kick_h_max` within ~2 m after each bend exit and through the sextupole section. Figure: `bc20_kdense_windows.png`.

### 9n. The reference run: hybrid kick placement + adaptive near nodes (`dfcsr_2d_peak_1M_ref`, 2026-11-20)

Settings: peak frame, 1M beam, z smoothing 6.5 bins capped at 0.1 FWHM, charge mesh 12 × 40 with x on the 0.2% quantiles, `kick_placement csr1d` with the 20k pre-pass + `kick_kappa_refine 4` + `kick_h_exit 0.1` (269 kicks), `near_adapt` + `near_adapt_x` (ε 0.02, 4 x′ rounds). 147 min on 6 ranks, shared with the glitch replay for the first hour (alone: ~2 h).

| segment [MeV] (kicks) | 0–18 | 18–20 | 20–36.65 | B2RE | 38.7–39.5 | 39.5–40.5 | 40.5–43.5 | 43.5–48 | B1RE | total |
|---|---|---|---|---|---|---|---|---|---|---|
| **ref** | −0.169 (60) | −0.020 (13) | −0.238 (63) | −1.770 (25) | −1.229 (18) | +0.942 (11) | −0.001 (19) | −1.325 (20) | −4.223 (40) | **−8.034** |
| dense windows (297) | −0.132 (27) | −0.009 (100) | −0.217 (26) | −1.794 (13) | −1.325 (33) | +1.035 (40) | +0.024 (27) | −1.260 (9) | −4.267 (22) | −7.945 |
| 116 kicks | −0.126 | −0.360 | −0.198 | −1.863 | −1.485 | +1.029 | +0.028 | −1.268 | −4.166 | −8.409 |
| 222 kicks | −0.158 | −0.088 | −0.218 | −1.793 | −1.233 | +1.135 | +0.033 | −1.306 | −4.207 | −7.835 |

| | ref | dense | 116 | 222 | 1D |
|---|---|---|---|---|---|
| → Q2ER | −0.426 | −0.358 | −0.684 | −0.464 | −0.537 |
| drift + wiggler | −1.664 | −1.693 | −1.864 | −1.443 | −2.252 |
| total | −8.034 | −7.945 | −8.409 | −7.835 | −7.308 |
| εnx 1st / 3rd [µm] | 13.00 / 8.32 | 12.91 / 8.24 | 13.05 / 8.23 | 12.95 / 8.28 | 20.88 / 12.13 |
| spike slice ε [µm] | 17.3 / 7.0 | 17.4 / 6.9 | 18.5 / 7.3 | 18.2 / 7.3 | 32.6 / 13.2 |

- **The rule-based schedule reproduces the hand-made dense run** where that one was dense: 18–20 m −0.020 vs −0.009 (13 kicks against 100), the two post-B2RE lobes −1.229 / +0.942 vs −1.325 / +1.035 (29 kicks against 73; the net −0.29 is the same), B1RE −4.22 vs −4.27, εnx and the spike slice emittance within 1%.
- **The remaining differences are elsewhere:** 0–18 m −0.169 vs −0.132 and 20–36.65 m −0.238 vs −0.217 (the ref has 60 + 63 kicks there, the dense run 27 + 26, so the ref is the better-sampled one), and 43.5–48 m −1.325 vs −1.260 (20 vs 9 kicks). Totals −8.03 vs −7.95; the 116- and 222-kick runs bracket both.
- The adaptive near nodes change the energy loss by nothing visible (B1RE −4.22 against −4.27 / −4.17 / −4.21 of the fixed-layout runs); they clean the transverse force on the pedestal (9l).
- **This is the reference BC20 2D result for now: −8.03 MeV total, εnx 13.0 / 8.3 µm, spike slice 17.3 / 7.0 µm**, against 1D −7.31 MeV, 20.9 / 12.1, 32.6 / 13.2. The 2D-vs-1D differences are physics of the tilted beam (B2RE 2.2×, the drift after it, the transverse force in B1RE), with two caveats still open: the band-margin under-coverage near |τ| = 1 (9m, D5) and the unverified caustic there.

Figures: `bc20_wake_zproj_dfcsr_2d_peak_1M_ref.png`, `bc20_wakes_along_s_dfcsr_2d_peak_1M_ref.png`, `bc20_wake_maps_long/trans/spike_dfcsr_2d_peak_1M_ref.png`.

## 11. Indranil's beams on his BC20 lattice variant (2026-11-21)

Two 1M-particle beams at BEGBC20 (`example/BC20_Indranil/`, openPMD h5 in a flat layout the installed pmd_beamphysics cannot read: converted to bmad-x text by `test/bc20_indranil/convert_beams.py`, P0C = mean p) on `lattice_bc20w_c_0.060000_000000.yaml` (the W chicane with a 2 m exit drift, no wiggler, length 51.08 m), tracked with the reference recipe of 9n (`run_indranil.py`; 2D at 3 MPICH ranks each, both at once):

| | gain | loss |
|---|---|---|
| N, Q, P0C | 999 796, 1.600 nC, 9.826 GeV/c | 999 775, 1.600 nC, 9.827 GeV/c |
| σz, span | 236 µm, −433…+494 µm | 12.6 µm, −19…+46 µm |
| I_pk at BEGBC20 | 0.76 kA | 42 kA (FWHM 1.1 µm, 21% of the charge within ±3 FWHM) |
| chirp h, dx/dz | +44 /m, +0.17 | −91 /m, −0.05 |
| deposition | peak frame (= the quantile window for a long bunch) | **quantile frame**: the peak window kept 20% of the charge and the 1D estimate put 85% of the loss in the dropped 80% (first-kick warning, `log_loss_peakframe_aborted`) |
| wake mesh | charge 12 × 40 | charge **12 × 80**: with 40 nodes the finest spacing (0.79 µm) was coarser than FWHM/3 of the peak (warning) |
| kicks | 336 | 319 |

### 11a. gain (205 min)

The bunch stays long: it compresses only 2×, to σz 166 µm, I_pk 1.6 kA, FWHM 253 µm at the end, so the CSR is weak throughout.

| | → q2er | b2re | drift | b1re | exit drift | total [MeV] | εnx 1st / 3rd [µm] |
|---|---|---|---|---|---|---|---|
| 2D | −0.113 | −0.083 | −0.069 | −0.015 | −0.020 | **−0.301** | 43.91 / 15.71 |
| 1D | −0.191 | −0.022 | −0.127 | −0.007 | −0.049 | −0.396 | 44.98 / 15.93 |
| CSR off | | | | | | | 44.30 / 15.94 |

- **Smooth:** roughness of the projected wake (rms deviation from a 20 µm smoothing / rms W) 0.02–0.06 in the first 37 m, 0.10 in b2re, 0.18 in b1re (11 kicks; the wake there is 0.05 MeV/m), 0.08 in the exit drift. No isolated outliers in W; the transverse maps are smooth except faint checkerboard cells in the outermost ξ rows at b1re (tails of the x distribution).
- **2D vs 1D** differ in shape at every kick (the long, tilted ribbon: dx/dz 0.17 at the entrance, growing with the dispersion), as for Sergei's beam before Q2ER; the net loss is 25% smaller in 2D. CSR changes the emittance by −1% (2D) / +1.5% (1D).
- Figures: `benchmark_results/bc20_indranil/wake_maps_long_gain.png`, `wake_maps_trans_gain.png`, `wake_zproj_gain.png`, `wakes_along_s_gain.png`.
- **Kick convergence** (`gain_k2`: kick_tol 0.0025, h_exit 0.05 → 497 kicks, 304 min; `compare_kicks.py gain`): the sections move by ≤ 0.009 MeV (→Q2ER −0.113 → −0.122, total −0.301 → −0.310), εnx 43.91/15.71 → 44.00/15.73 µm. The projected wakes agree to 2.0% rms (median over 271 matched kicks, 1.9% in the first 10 m, 0.5% in B2RE, 0.4% in B1RE); the few kicks above 30% are where the wake is ≤ 0.2 MeV/m and changing fast (D9 at 18.9 m — the x-waist transient also seen on Sergei's beam — and the 29.8–30.3 m sextupole region), where the compared kicks sit 1.5–3 cm apart. Converged at the reference schedule.

### 11b. loss: the velocity floor blew up on a sharp edge (fixed, 2026-11-21)

The first loss run (364 min, 319 kicks; outputs kept as `loss_vxbug.npz`, `log_loss_vxbug.txt`) was wrong from its second kick and diverged at the end: max |W| 372 MeV/m at 0.119 m, **7654 MeV/m at 0.148 m** (the 1D wake there is ±40 MeV/m), decaying to ~240 by 0.7 m, sane (15–20 MeV/m, matching 1D in shape) from B1LE's exit to 49.5 m, then 30 → 864 → 1171 → 965 → **7e20 MeV/m at 50.10 m** in the exit drift; 12% of the particles ended with |δ| > 0.5.

**The beam.** Two 55 kA horns, at the tail (z = −17.8 µm) and the head (+16.9 µm), each ~1 µm FWHM with a sharp outer edge: the head horn falls 100× within 0.5 µm (2300 → 20 particles per 0.02 µm). The horns hold 11% and 8% of the charge; 25% is within ±5 µm of the tail horn. The beam is also clumpy below the cell scale (Fano factor 11 in 12 × 0.26 µm cells, 1.5 in 3 × 0.05 µm), i.e. ~10 siblings per clump — an upsampled distribution. The 2D/1D disagreement was not noise: a 20k subsample gave 18 MeV/m at the same kick.

**Mechanism** (pickled state at 0.148 m, `dump_state.py`, `state_outliers.py`): the outliers sit 1–2 µm AHEAD of the head horn's edge, come entirely from the far region at u = 3–20 mm behind the observer at x′ ≈ x (the straight-line part of the retarded geometry, z′ ≈ z), are unchanged by 4× far columns, 4× x′ nodes, margin, Jacobian floor, tangent bands, near_adapt or `ibp`, and vanish with `ignore_vx` (−7654 → +4.95 MeV/m). So it is the W2 term, ρ · div v. The interpolated ∂vx/∂x along the band reached 7e5 /m (physical: 0.05 /m) at retarded times inside one snapshot interval: the stored `data_vx_u_interp` of the 0.1187 m snapshot had max 2005 (normal 2e-4), two other snapshots 0.48 and 2.5e-3. The z smoothing was 6.5 bins capped at 0.1 FWHM = 0.084 µm = 0.32 bins, floored at **0.5 bins**. A 0.5-bin Gaussian applied in Fourier space is not band-limited (response 0.29 at Nyquist), its kernel rings, and the smoothed density of the 100× edge overshoots to **−9e-5 … −2.3e-4** (peak 0.52): comparable to the velocity floor ε = (1/2π)/velocity_threhold = 1.6e-4. Where ρ + ε → 0, vx = ρvx/(ρ + ε) and the mask (ρ/(ρ + ε))³ (negative and unbounded for −ε < ρ < 0) both diverge. Finer u cells made it worse (320 bins: 6e5 MeV/m, ρ_min −2.3e-4 crosses −ε); 1 bin of z smoothing removed it (0.3 FWHM → 1.39 bins: 12.8 MeV/m); the 20k beam escaped only because its negative overshoot happened not to reach −ε at the sampled cells. In the exit drift the same thing recurred (FWHM 4 µm, window 150 µm → 0.5 µm bins, cap 0.4 µm ≈ 0.5 bins again).

**Fix** (commit 8a66afa): `velocity_mask()` and the floor use max(ρ, 0) — bit-identical wherever the smoothed density is non-negative — and the cap's floor is `smoothing_min_bins` (default 1.0, was 0.5). The 0.148 m kick is then 12.8 MeV/m max with 100 u bins and 13.1 with 320 (cells agree to 2–4%); the drift control (same beam, pure 2.2 m drift, states at 0.20 and 0.58 m) gives 0.06–0.09 MeV/m max against the 20 MeV/m horn wake in the bend — the velocity-term residual of 11e (an earlier 0.3 m-drift control, 0.19 MeV/m, was contaminated by the geometry bug of 11e). The gain run and Sergei's reference are unaffected (their z smoothing never went below 1.6 bins; 163 tests incl. the golden fixtures pass).

**The end-of-lattice blow-up did not reproduce on a clean beam.** A replay with the CSR off before 46 m (the beam enters B1RE without the early damage; `dump_state.py loss 49.6,50.09 46.0`, old code) gives 4.8 and 2.6 MeV/m max at 49.62 and 50.10 m, no corrupted snapshot (ρ_min −0.32 ε, all |∂vx/∂u| ≤ 1e-4), `ignore_vx` changes the worst cells by 5%. So the 1e20 at 50.10 m came through the beam the first blow-up had damaged (12% of the particles off by > 1 MeV already at 0.15 m): where a cell's ρ overshoot reaches −ε is a matter of chance at the 0.3–0.6 ε margin this beam leaves, and the same mechanism, not a second one, is the only candidate. The rerun with the fix decides.

The roughness metric of 11a (20 µm smoothing) is meaningless for 1 µm horns and is not quoted for loss.

### 11c. loss with the fix (238 min, 319 kicks)

No blow-up: max |W| over all 319 kicks is 22.8 MeV/m (at 0.53 m), the B1LE entrance transient grows smoothly (0 → 7 → 12.5 → 14 → 18 MeV/m over the first five kicks), every particle ends with |δ| < 0.02. The longitudinal maps are smooth at every landmark kick, including the exit drift (the first run's 1e20 was a consequence of the early damage, 11b).

| | → q2er | b2re | drift | b1re | exit drift | total [MeV] | rms ΔE [MeV] | εnx 1st / 3rd [µm] | horn slice εn [µm] |
|---|---|---|---|---|---|---|---|---|---|
| 2D | −8.71 | −4.35 | −3.38 | −3.47 | −2.52 | **−22.43** | 16.8 | 72.5 / 55.8 | 89.7 / 29.0 |
| 1D | −13.46 | −3.16 | −3.68 | −3.79 | −3.24 | −27.33 | 18.2 | 77.9 / 63.6 | 88.3 / 32.0 |
| CSR off | | | | | | | | 25.4 / 23.2 | 6.0 / 5.5 |

Final beam: σz 22.8 µm, I_pk 65 kA (2D) / 55 kA (off), FWHM 4 µm (8 µm off).

- **From B2RE on, 2D and 1D agree** in shape and to 10–40% in amplitude at the horn (B2RE −4.35 vs −3.16 MeV, B1RE −3.47 vs −3.79, exit drift −2.52 vs −3.24; the horn's W is −17 to −20 MeV/m in 2D, −25 to −30 in 1D). CSR triples the projected emittance in both models (25 → 72–78 µm): the horns carry 20% of the charge and get 20 MeV/m kicks with a 1077 /m chirp inside them.
- **In the first 10 m the 2D is much weaker than 1D** (→Q2ER −8.7 vs −13.5 MeV; B1LE: −0.9 vs −2.2, D1: −2.2 vs −4.6; the first-kick transient 12.5 vs 41 MeV/m). Here the horns are 0.84 µm FWHM and the deposition (300 bins over the 78 µm window, 1 bin of smoothing) resolves them at 0.34 µm, FWHM/2.5 — below the 5× the code warns at, and only 2.5× better than the rms grid that lost half of Sergei's CSR (§9). **It is not resolution** (`zres_check.py`, states pickled at 0.534 m with 600 and 1200 deposition z bins; the cap then sets the smoothing to 2.2 / 4.7 bins so both resolve 0.18 µm instead of 0.34): the projected wake at the tail horn is −11.03 / −11.01 / −11.00 MeV/m for 300 / 600 / 1200 bins, the head-horn maximum +5.8 / +6.0 / +6.0, the rms over z 2.73 / 2.78 / 2.78 — converged to 2%, against 1D −22.9 / +9.4 / 2.99 (`zres_check_loss.png`). The gap is 2D physics: the horns are 66 µm wide in x, wider than the transverse coherence scale (σz² R)^{1/3} = 32 µm of a 0.84 µm feature at R = 47 m, so the 1D (zero-width) model overestimates their steady-state wake, as it did for Sergei's beam before Q2ER (§9). By B2RE the horns have stretched to ~4 µm, (σz² R)^{1/3} ≈ 90 µm exceeds the x width, and the two models agree.
- **Transverse force**: smooth on the ribbon; the outermost ξ rows (0.2% quantile edges, little charge) show single-cell speckle in B2RE and the last 20 cm of B1RE (isolated cells of 0.7–1.8 MeV/m against 0.25 on the horn at 48.88–49.06 m). Same family as the §9k pedestal sampling; the longitudinal wake at those cells is clean. Open.
- Figures: `benchmark_results/bc20_indranil/wake_maps_long_loss.png`, `wake_maps_trans_loss.png`, `wake_maps_spike_loss.png`, `wake_zproj_loss.png`, `wakes_along_s_loss.png`.

### 11d. `frame: auto` (2026-11-23, commit 427ec3e)

The frame choice is now made by the code from the initial beam, once, before the first deposit. CSR2D previews the peak window (±3 FWHM around the highest current peak, the same `_peak_window` the peak frame uses, without touching its rate limiter), and asks the 1D model what the charge that window would drop contributes to the wake at the exit of the **first bend** — the steady-state wake developed over the whole bend, R from the lattice — with `csr1d.dropped_wake_fraction`, the arithmetic the per-kick pedestal diagnostic has used all along (now shared). If the dropped part carries more than `auto_pedestal_max` (0.05) of the energy loss or of the field on the window, the pedestal — or a second horn — matters and the frame is `quantile`; otherwise `peak`. The decision is printed as `[frame auto]`, kept for the run, and written to statistics.h5 (`quantile_frame` attrs `frame`, `frame_requested`, `auto_decision`).

| beam at the entrance | peak window keeps | dropped part carries (1D loss / field) | choice |
|---|---|---|---|
| Indranil gain (long bunch) | 100% | 0.0% / 0.1% | peak (= quantile for a long bunch) |
| Indranil loss (double horn) | 24.1% | 57.9% / 10.0% | quantile |
| Sergei BC20 1M (long at BEGBC20, spike forms in the chicane) | 100% | 0.0% / 0.1% | peak |

Cost: 2 s with 1M particles (histogram + 1D wake; `wake_1d` itself is 0.1 ms on 1500 bins). A resolved `peak` run is bit-identical to `frame: peak` (test on the golden config); `frame: peak`/`quantile` runs are unchanged (167 tests).

**Why not re-decide every step.** The per-kick diagnostic already evaluates the same dropped-charge fractions at every kick, at ~0.1 s per kick against 30–60 s for the 2D kick, so watching costs nothing; it now names `frame: quantile` as the fix if a run that auto-chose `peak` meets a pedestal that matters later (Sergei's beam in the drift after B2RE, §9h, is exactly this case: the initial beam cannot foresee it). Switching the frame mid-run is a different matter: (i) the grid shape is fixed for the run (ring buffer, numba kernel), so widening the window to the whole bunch divides the resolution by the window ratio — 5–10× on a spike beam — exactly where the spike's own wake still needs it; the bins would have to be chosen for the quantile window from the start, which forfeits the peak frame's only advantage; (ii) the history lookup blends neighbouring snapshots' frames in time, and a window that jumps between two snapshots injects a step into the retarded density (the velocity floor was made snapshot-independent for the same reason). A rate-limited, continuous widening (like `peak_rate`/`peak_shift`) is possible but buys nothing over (i). So: decide once, watch every kick, warn loudly.

### 11e. The ±300 µm "clipping" of the B1LE/B2LE maps (2026-11-23)

On long bunches (Indranil gain, Sergei's BC20 reference) the W maps in B1LE and B2LE show a rectangular frame: the outermost ξ rows, and the first/last z columns, carry wakes several times the core's and of the opposite sign, so the colour scale is set by the border and the core looks flat, "clipped" at the second row from the edge (±300 µm for a 120 µm beam with the 0.2% quantile rows at ±2.7σξ; ±6σξ on the gain beam, whose x tails are heavy). Diagnosed on pickled first-kick states (`edge_rows.py`, drift controls `gain_drift2`, `loss_drift2` on a 2.2 m drift).

**What it is not.** The pattern is unchanged by 4× wider bands, 4× far columns, 4× x′ nodes, near_adapt off, band_jacobian_min 0, integration by parts, the rms frame instead of the peak window, deposition out to ±8σξ (160 bins) instead of ±5, 2× the x smoothing, the velocity floor at 1e-6 instead of 1e-3, and the observer's own vx set to zero. The outer rows and the end columns are mesh cells that hold **no particles** (a 0.5σξ × 0.1σz box around them is empty), so the beam never feels these values beyond the 0.2% of particles that get the clamped edge-row kick.

**What it is.** Two ingredients, separated by switching the velocity terms off (which zeroes W2 and the source-vx part of W1):
- In the bends the end columns' ξ-slope is in **W1 with velocities off**: at 0.57 m in B1LE the tail column runs linearly from −0.020 (ξ = −6.6σξ) to +0.006 (+5.9σξ) while the core tail cell is −0.0006 MeV/m; the head column has the opposite slope. In a pure drift W1 is identically zero (v = v′ kills it algebraically), so this is a bend-geometry term: the retarded coordinate of a source behind by u, seen by an observer at radial offset x, is z′ − z ≈ Δx²/2u + u(x + x′)/2R − u³/24R²; the middle term shifts z′ by u·x/2R (3 µm at 6σξ, u = 0.5 m), and the Coulomb-scale terms of the integrand (0.4–4 MeV/m before their cancellation, gross/net 18 at the corner, 1600 in the core) vary on the σz scale, so the shift leaves ∂(gross)/∂z × u x/2R ≈ 0.003–0.03 MeV/m, linear in x and antisymmetric between head and tail — what is seen. It is the converged value of the Stupakov–Tang integral for this density; whether the 2D entrance-transient theory has this x-gradient at this size is the LW check of §10 item 0.
- In the chicane interior (B2LE, 11.36 m, |τ| = 3) and in the middle columns of B1LE the border excess is the **velocity term** W2 = ρ·∇·v (plus the vx′ part of W1): at the B2LE tail column the velocity-off W1 is smooth (−0.049 → −0.064 over 14σξ) and W2 adds +0.035 at the bottom rows and −0.03 at the top. The pure-drift controls, where the exact longitudinal field vanishes, put the floor of these terms at 0.003 MeV/m (gain) and 0.06–0.09 (loss), i.e. 0.3–0.5% of the loss beam's horn wake but comparable to the whole B1LE wake of the gain beam (0.003 in the core).

**A real bug found on the way** (commit b15f71a): `interpolate1D` extrapolates one table cell below the lattice start and returns zero further down, so an observer more than one cell before s = 0 — the bunch tail at the first kick, if the table is fine (a 0.3 m test lattice: 0.15 mm cells) — had the lab position (0, 0) and normal (0, 0); the transverse integrand reached 1e16 and the test beam got Δpx = 5e-4 (30× its divergence) at s = 0.1 mm. BC20's table (25 mm cells) was never hit. The orbit is now continued straight backwards along the initial tangent; inside the table nothing changes (175 tests, golden fixtures bit-identical).

**Consequences.** (i) The border is cosmetic for the beam but it hides the core in the normalised maps; `mesh_x_quantile` 0.002 puts rows at ±6σξ on heavy-tailed beams — 0.01 (rows within ~2.5σξ, the rest clamped) keeps the mesh where the charge is. (ii) The velocity-term floor (0.003 MeV/m for a 0.8 kA, 236 µm bunch) is the accuracy limit of the 2D wake wherever CSR is weaker than that: the first bend of a long bunch. It is not shot noise (same with 20k particles), not the mask, not the quadrature; the W1(vx′) ↔ W2 cancellation by parts needs ∂vx/∂x consistent with the interpolated vx, which the deposited fields are not at the 0.1–1% level. (iii) The x-slope of W1 at the bunch ends in a bend should be checked against the 2D entrance theory (open, with §10 item 0).

### 11f. The loss beam's transverse force was not converged: far-region x′ aliasing (2026-11-23, commit ea4f41f)

In B2RE and B1RE of the loss run the largest |F| on the mesh is almost always one isolated cell, 0.1–1.8 MeV/m against a 3×3 median near zero, where the force on the horn is 0.1–0.25; the cells sit in the tail columns 50–100 µm behind the horns, at any row, and the speckle (rms of F − 3×3 median over max|F|) is 7–14% in the outer rows. Sergei's reference run has 2–4% and no isolated cells: this is the double horn's doing.

**Mechanism** (pickled state at 37.74 m, B2RE ½, `f_speckle.py`): the six most isolated cells are 100% far region (near, prev, patch contribute < 1e-3). With W1 only they are smooth (−0.003), so it is the ∂ρ/∂t terms. Doubling the far region's x′ nodes flips their sign, 4× and 8× bring them to ±0.005: aliasing. An observer in the tail sees a horn through the oblique branch of the band (|τ| = 4.8 here); the retarded z′ crosses the 1 µm horn in a few µm of x′, while the far region's uniform x′ cell is band width / 200 ≈ 10 µm. The near region has had `near_adapt_x` for this since §9l; the far region had not. near_adapt_eps/4, more x′ rounds (near only), 4× far columns do nothing or little; margin ×2 moves them (different nodes, same aliasing).

**Fix.** `far_adapt_x` (default True, gated by near_adapt and near_adapt_x): the same column refinement — flag the far columns whose retarded density jumps by more than eps·ρmax between neighbouring x′ nodes, re-evaluate them on x′ nodes equidistributing 1/h + |dρ/dx′|/eps, up to 4 rounds — now runs on the far region. Whole 12 × 80 mesh at 37.74 m against an 8× x′ reference:

| | wall time | F speckle | max |ΔF| vs 8× | max |ΔW| vs 8× |
|---|---|---|---|---|
| shipped | 125 s | 0.073 | 0.121 = 134% of the horn's F | 2.24 = 34% of the horn's W |
| far_adapt_x | 145 s | 0.025 | 0.009 = 10% | 0.064 = 1.0% |
| 8× x′ nodes | 327 s | 0.026 | — | — |

So the longitudinal wake had aliased cells too (one cell off by 2.2 MeV/m); the projected W of 11c was not affected visibly, but the kick on the particles in those cells was. The loss run is being repeated with far_adapt_x (started 2026-11-23 12:48, ~4 h); the earlier outputs are kept as `loss_nofaradapt.npz` / `log_loss_nofaradapt.txt`.

**Rerun with far_adapt_x (305 min, 319 kicks; shared the machine with the profiling experiments and still ran under the sockets provider, 11h).** The transverse maps are clean: the B2RE/B1RE speckle (rms of F − 3×3 median over max|F|) is 1–3.5% in every row group, against 7–14% in the outer rows before; the largest |F| of every kick is on the horn (0.12–0.18 MeV/m) and the "most isolated" cell is the horn's own structure (e.g. 49.00 m: F 0.169 against a 3×3 median 0.149), not a 0.7–1.8 MeV/m spike in the tail. The isolated cells at the B1RE exit of the first run are gone too, which settles that they were the beam damaged by the B2RE aliasing, not the B1RE quadrature. Longitudinal: max |W| over all kicks 22.7 MeV/m, no outliers, all particles finite.

| | → q2er | b2re | drift | b1re | exit | total [MeV] | εnx 1st / 3rd [µm] |
|---|---|---|---|---|---|---|---|
| without far_adapt_x (11c) | −8.713 | −4.349 | −3.384 | −3.468 | −2.518 | −22.431 | 72.51 / 55.77 |
| with far_adapt_x | −8.728 | −4.341 | −3.366 | −3.430 | −2.517 | −22.382 | 72.44 / 55.67 |
| 1D | −13.462 | −3.157 | −3.682 | −3.789 | −3.238 | −27.329 | 77.95 / 63.63 |

The collective numbers move by at most 0.04 MeV per section and 0.1% in emittance: the aliased cells carried little charge. Particle by particle the difference is what the aliasing did: rms ΔE 0.16 MeV, 1861 particles off by more than 1 MeV (up to 2.4 MeV), rms Δpx 5.7e-7 (up to 8e-6, i.e. the full divergence of a horn particle). The 11c tables stay valid for the means; the per-particle kicks of the loss run are now trustworthy. Figures refreshed in `benchmark_results/bc20_indranil/` (`wake_maps_trans_loss.png` etc.); the first-run outputs are `loss_nofaradapt.npz`.

**B1RE is a different story.** On the replay states at 48.73 and 49.03 m (CSR switched on at 46 m, i.e. a beam WITHOUT the B2RE kicks of the real run) the mesh is already converged: F speckle 0.016 with and without far_adapt_x, max |ΔF| 1% of the horn's F against the 8× reference, and the "most isolated" cells are the horn's own structure (F 0.17 against a median 0.15, unchanged by any variant); far_adapt_x costs nothing there (no far column is flagged: |τ| = 0.24, the band is not oblique). The real run's isolated B1RE-exit cells (0.7–1.8 MeV/m at 48.88–49.06 m, outer rows) therefore do not come from the B1RE quadrature; the candidate is the beam the B2RE aliasing produced — isolated wrong kicks of ±1–2 MeV/m × 6 cm on small groups of particles, which then sit off the ribbon as clumps of a few hundred particles when the mesh's outer rows are built. The rerun decides.

### 11g. Profile and speedups (2026-11-23, commit 354cd3d)

cProfile on the pickled loss state at 37.74 m (B2RE), 110 ms per wake point: the integrand evaluation is 66% (numba field interpolation 34%, numpy arithmetic + python 32%: ~40 whole-array operations, ~30 temporaries of 80k doubles per call, and a per-element curvature mask loop of 3.8 ms per call), the x′ refinement doubles the integrand work, band location 13%, near-node scan 7%, patch 4%. Per-point cost varies 15× over the mesh (24–340 ms), so the contiguous MPI blocks left the slowest rank at 1.16× the mean (3 ranks) and 1.27–1.30× (6–12).

Changes, each bit-identical to the previous code (60 points of that state `array_equal`; 176 tests incl. the golden fixtures; MPI vs serial `array_equal` at 3 and 4 ranks): interleaved mesh assignment with Allreduce (1.01–1.02× imbalance); the integrand arithmetic fused into a numba kernel (`integrand_kernel`, default `numba`); the curvature lookup by `searchsorted`; the x′ refinement reading ρ from the plain field pass (17.4 vs 22.3 ms per 80k points). Per wake point 110 → 80 ms (−27%), plus ~14% from the balance at 3 ranks. Not worth it: per-column refinement counts (15%), fewer band iterations (6 are bit-identical to 12 but no time). The floor is the field interpolation (45% now). Documented in docs/DFCSR_2D_methods.md §8, §9.5.

### 11h. The MPI "saturation at 3 ranks" was the libfabric sockets provider (2026-11-23)

Every rank of a loss run consumed 2.98 cores over 20 s during a kick while the wake computation is single-threaded (0.98 cores on a pickled state in the pydfcsr environment, which has no MPI). Per phase (`resource.getrusage` around each method of the loop, serial golden run): the kick itself shows 1.44 user + 1.46 system cores; the 1M-particle gain beam: 1.31 + 1.60. Thread-limiting variables (VECLIB, OMP, NUMBA, OPENBLAS, NUMEXPR) and the macOS allocator cache change nothing; `sample` shows the extra threads to be libfabric's `sock_conn_listener_thread` and `sock_ep_cm_thread` in `kevent`. Merely importing mpi4py in the pydfcsr-mpi environment makes a serial process run at 2.75 cores; with `FI_PROVIDER=tcp` it is 0.99 and the wake point is 10% faster (90 → 78–80 ms). The golden run under mpirun: 3 ranks 48 → 31 s, 6 ranks 38 → 19 s, 12 ranks 17 s (0.95 / 0.81 / 0.53 cores per rank, the last two limited by the replicated serial part of that small run). CSR.py now sets `FI_PROVIDER=tcp` on macOS before importing mpi4py when the user has not set it (commit c136fe8). The §9d "memory bandwidth" explanation of the 3-rank saturation is withdrawn.

The loss rerun of 11f started before this fix (12:48, sockets provider) and is running at the old pace.

**Threads inside a rank (2577798):** the field and integrand kernels compiled twice from one source, serial and `parallel=True` (not cached — a cached entry is shared with the serial twin and the parallel one silently loads serial code), dispatched to the parallel twin for ≥ 20k points when `CSR_integration.threads` > 1. The 80k-point field pass goes 19 → 9.6 ms at 4 threads, but a wake point goes 80 → 85–95 ms (cpu/wall 1.3–1.5) with the omp and workqueue layers and passive waiting alike: ~10 large calls among hundreds of small ones plus python; the fork/join and the serial remainder eat the gain. Kept as an option, default 1. Ranks over mesh points, now at 0.95 cores each, are the way on this machine.

**Particle decomposition (next commit):** `step_control.particle_split` (tracking per rank slice + Allgatherv, bit-identical) and `deposit_split` (histograms per slice + Allreduce, 1e-16 differences). On the 1M-particle gain beam at 6 ranks, per step: tracking 262 → 326 ms, deposition 574 → 592 ms — no gain, because the conda MPICH has no shared-memory path for local ranks: a 48 MB Allgatherv costs 120 ms with tcp (390 MB/s; 6.5 s with sockets, 7 MB/s), and the deposition time is in the full-beam fits and FFTs, not in the histograms. Both options default off; they are for nodes with a real shared-memory transport.

**Shared-memory history (item 4): started, not finished.** An MPI-3 shared window holding the ring, written by the node leader, read by all (would save (ranks − 1) × 2 GB at BC20). The 6-rank golden test hung with all ranks waiting in MPI before the first run even reached the shared path; cause not found. The hunks are saved in `docs/wip_shared_history_hunks.txt` (untracked) and removed from the code; paused on request.

### 11i. The ±∂ρ/∂x′ cancellation revisited: by parts buys accuracy, not speed, here (2026-11-23)

Whole 12 × 80 mesh on the B2RE loss state against the 8× x′ reference (`ibp_nodes.py`):

| form | x′ nodes | time | max |ΔW| / horn W | rms ΔW / horn W | max |ΔF| / horn F |
|---|---|---|---|---|---|
| shipped + refinement | 200 | 79 s | 0.97% | 0.13% | 10% |
| shipped + refinement | 100 | 68 s | 1.35% | 0.19% | 10% |
| shipped + refinement | 50 | 66 s | 17.8% | 0.92% | 70% |
| ibp as shipped (refinement off) | 200 | 73 s | 34% | 2.3% | 134% |
| ibp + refinement (new) | 200 / 100 / 50 / 25 | 117 / 90 / 78 / 65 s | 0.97 / 1.34 / 17.8 / 41% | 0.13 / 0.19 / 0.81 / 4.1% | 10 / 10 / 70 / 166% |

- `ibp` used to switch the x′ refinement off; on a horn beam that brought the aliasing back (34% at 200 nodes). The integrator now lets both coexist: refined columns keep the shipped form on their own nodes, the smooth columns take the by-parts integrand (default path bit-identical, 161 tests).
- With that, by parts reproduces the shipped result exactly where the shipped result is converged, and is better only where the shipped form is already poor (25 nodes: 41% vs 237%). The node count is set by the horns, i.e. by the refinement, not by the cancellation, so the by-parts form cannot reduce it; and its geometry evaluations cost +40%. On this beam it buys robustness against ρ/∂ρ inconsistency, not speed.
- **100 uniform x′ nodes instead of 200 is the cheap win:** 1.35% instead of 0.97% at the worst cell, 0.19% rms, −15% time. Not changed as the default (the dipole and chicane validations were done at 200); recommended for the BC20 runs.
- **Decision (2026-11-24):** `ibp` stays in the code as a robustness option and a check, off by default; `ibp_check: 3` added to the BC20 reference recipe (`run_dfcsr_bc20.PIN_2D_INTEG`, inherited by the Indranil runs); the methods document §5.7 and CLAUDE.md state the verdict.
  Exercised on the pickled states (0.3–0.5 s per check), the first numbers were 5.4e-4 (loss B2RE), 2.2e-3 (loss 0.148 m) and 4.1e-2 (gain B1LE 0.57 m) and looked alarming. They were the check's own fault, twice: its "shipped" reference excluded the x′ refinement while the by-parts total included it, and the relative measure was taken against three centre-row points where W ≈ 0.002 against a mesh maximum of 0.02. With both fixed (082426b: the reference carries the refined columns; the warning is judged against the mesh's largest |W|) the two forms agree to 2e-6, 0.0 and 2e-5 on the three states, and the whole-mesh comparison agrees to < 1e-4 of max|W| at 200 nodes. The Indranil results are unaffected; the W1 ξ-slope of §11e is therefore NOT a form error.
- **The velocity-term floor is not a vx / ∂vx/∂x inconsistency.** The proposed flux-form by-parts treatment of the vx′∂ρ/∂x′ + ρ∂vx′/∂x′ terms rests on ∂vx/∂x being inconsistent with the interpolated vx. Test: replace the stored ∂vx/∂x by a central difference of the interpolated vx along x′ (δ = 1e-3 σξ), i.e. a derivative exactly consistent with vx. Pure-drift residual on 25 mesh cells: gain at 0.25 m 0.0029 → 0.0066 MeV/m (max), rms 0.0012 → 0.0019; at 1.0 m 0.0025 → 0.0051; loss drift 0.049 → 0.049; bend (gain 0.57 m) 0.0219 → 0.0220. A consistent derivative does not lower the floor, so the flux form would not either: not built. The floor is elsewhere (candidates: the time blend of vx between snapshots, or a genuine property of the W₁+W₂ integral with a finite-σ velocity field in a straight line); deciding needs the independent Liénard–Wiechert reference (§10 item 0).

### 11j. Density-adaptive far-region columns: tested, not good enough (2026-11-24)

`CSR_integration.far_adapt` (off by default): the near region's column scan and gradient monitor applied to the far region inside a bend, from a uniform base of 2 × `far_adapt_base` columns, capped at the fixed 400. Whole 12 × 80 mesh at 100 x′ nodes against the fixed layout (`far_adapt_s.py`):

| state | base | far columns | time | max |ΔW| / horn W | rms | max |ΔF| / horn F |
|---|---|---|---|---|---|---|
| loss B2RE 37.74 | 50 / 25 | 183 / 133 | 92% / 87% | **18.3% / 35.9%** | 2.1 / 2.7% | 17 / 36% |
| loss B1RE 48.73 | 50 / 25 | 181 / 131 | 92% / 88% | 3.0% / 5.1% | 0.34 / 0.53% | 7.9 / 8.3% |
| gain B1LE 0.57 | 50 / 25 | 106 / 56 | 56% / 47% | 0.48% / 1.0% | 0.12 / 0.33% | 1.2 / 4.4% |
| loss 0.148 | 50 / 25 | 174 / 125 | 88% / 82% | 0.28% / 0.58% | 0.03 / 0.05% | 2.8 / 2.8% |

The monitor follows the retarded density at the band axis along s′. On a horn beam that is not where the far region's structure is: it is the oblique crossing of the horn across x′ at each column, the thing `far_adapt_x` refines, and the s′ spacing must stay fine wherever that crossing happens, which the axis density does not reveal. So the columns it removes are the wrong ones at B2RE, and the time saved on the remaining columns is small because the refined columns dominate the cost anyway (8–12%). On the smooth gain beam it works as intended (0.5% for 44% less time) — there the far region has no structure to miss. A monitor built from the x′-refinement flags of a coarse first pass would be the next design; not pursued, because the far region's cost on horn beams is in the refined columns, which no s′ placement removes. Option kept, off, documented as such.

Also in this step: `xbins` 200 → 100 in the BC20 reference recipe (11i).

## 10. Next

0. **OPEN (needs a new tool): an independent Liénard–Wiechert check of the 2D wake at high tilt.** Everything done on the pickled D5 states (9m) refines the quadrature of one integrand; the ×2.2 jump of W at τ = ±1 (19.123 m) is converged within the band path and the legacy path cannot judge it. The only independent formulation is a direct Liénard–Wiechert sum: 10⁴–10⁵ macro-particles, each particle's retarded field at the observer from its *trajectory* (t − t_ret = |r − r_j(t_ret)|/c solved per particle), longitudinal projection = W. Needs (a) recording a subsample's coordinates at every snapshot over the retained history (the code keeps density snapshots only), (b) a regularisation of the point-charge 1/|r − r′| singularity with a convergence study in the kernel width (as CSRtrack's sub-bunches), (c) for the transverse force the renormalised form or longitudinal only. ~2 days. It would also settle the older open item below (2a): the 2D wake size in the chicane's B2/B3 and BC20's B2RE (2.2× 1D), which has never had an independent check.
1. Rerun the affected chicane results (section 1a) with the edge fix. At minimum: `chicane_2d_vs_1d.py` and the energy loss / emittance / D3 numbers quoted in `progress.md`.
2. ~~Run the 2D CSR~~ and ~~Tao with Sergei's settings~~: done, section 7. Open from it:
   a. ~~A resolved 2D run~~: done with the quantile frame and charge mesh (section 9). Open from it: the B2RE excess, the transverse-force emittance cancellation, and 2D shot noise (9c).
   b. ~~Tao `n_bin` convergence~~: done (7e). Tao converges to the same −7.6 MeV. Open: the 0.1 MeV Tao-vs-pyDFCSR difference before Q2ER (long bunch, not resolution).
   c. The 2D-vs-1D difference before Q2ER and in B2RE (beam tilt dx/dz ≈ 10): check the 2D wake-mesh x resolution and the integration-region branch at that tilt.
4. Tell Sergei about the one-step sextupoles.

### 11k. n_formation_length is converged at 1; the far-region CELL SIZE is the knob (2026-11-24)

Indranil reported that he had to raise `n_formation_length` to 8 and scale the far-region s′ bins to converge the energy loss. Tested on the pickled loss-beam states at B2RE mid (s = 37.74, L_f 1.98 m) and B1RE 2/3 (s = 48.73, L_f 1.82 m), reference recipe, 12 × 80 charge mesh (`test/bc20_indranil/nfl_check.py`). The far region in a bend is a flat `2·far_zbins` uniform columns, so raising n_fl at fixed `far_zbins` coarsens the far cells by the same factor. Two series:

| B2RE mid | reach | far cell | <W> vs reference | max local |W − W_ref| / max|W| |
|---|---|---|---|---|
| n_fl 1, far_zbins 200 (reference) | 1.98 m | 5.0 mm | — | — |
| n_fl 1.5, far_zbins 200 | 2.97 m | 7.4 mm | −0.23% | 36% |
| n_fl 2, far_zbins 200 | 3.96 m | 9.9 mm | −1.35% | 22% |
| n_fl 3, far_zbins 200 | 5.93 m | 14.8 mm | −2.73% | 31% |
| n_fl 1, far_zbins 100 | 1.98 m | 9.9 mm | −1.35% | 22% (at z = −9 µm, the horn) |
| n_fl 1, far_zbins 400 | 1.98 m | 2.5 mm | +0.14% | 18% (at z = −65 µm, tail) |
| n_fl 2, far_zbins 400 | 3.96 m | 5.0 mm | −0.01% | 3.5% |
| n_fl 3, far_zbins 600 | 5.93 m | 5.0 mm | +0.03% | 3.5% |

B1RE 2/3: n_fl 1→2 at fixed 5 mm cell: <W> +0.02%, local 1.7%; far_zbins 200→400 at n_fl 1: +0.02%, local 2.7%.

Mechanism: the reach is converged at one formation length (doubling and tripling it at constant cell size moves the mean by 0.01–0.03%). What moves the wake is the far cell size: a 10 mm cell steps the retarded z by ~1.3 µm per column at the bend entrance (slippage uφ²/2 per unit source distance), comparable to the 1.1 µm horn FWHM, and the horn then aliases (−1.35%, 22% locally at the horn). Raising n_fl at fixed `far_zbins` reproduces exactly that error, and scaling the bins back up removes it — so "convergence in n_fl" is really convergence in the far cell size. The 5 mm reference cell is converged for the mean to 0.14% (vs 2.5 mm); the residual 18% local difference sits in the low-charge tail at z = −65 µm, where the far columns sweep the pedestal edge.

**Addendum (same day): the far-cell sensitivity was the far/near SEAM, not the horn width — fixed by `seam_ratio`.** The far-region column integrals at the worst mesh point (tail row z = −65 µm, outer x = −1.26 mm; `far_columns.py`, [far_column_integrand_b2re.png](../pyDFCSR_2D/test/benchmark_results/bc20_indranil/far_column_integrand_b2re.png)) put the entire 400-vs-3200-column difference in the first 3 cm behind the observer, not at the bend entrance: the far-region W there went +2.30 → +1.08 → −0.68 MeV/m for 400 / 800 / 3200 columns. The near region for that observer was 3 mm long: d = (10σx + x − x̄)|cos 2a|/|sin 2a| shrinks on the near-x side of the beam at high tilt (σx 229 µm, x − x̄ = −0.9 mm, |cos 2a/sin 2a| ≈ 5), so the graded near cells (0.15 mm at the seam) handed over to the flat 5 mm far cells, 33× coarser, where the column integral still varied on a 1–10 mm scale (|r − r′| ~ the transverse offset; panel 3 of the figure: |I|/|dI/ds′| = 1e-3–1e-2 m for u < 3 cm). The horn row was affected too: its core rows moved 10% between 400 and 800 far columns.

Fix (`Integration_params.seam_ratio`, default 2.0, 0 = old layout bit-identical): in a bend, d is floored at far_cell/(near_grade·seam_ratio), i.e. the near region keeps grading until its cell has reached half the far cell (49 mm on BC20, +45 near columns). Not applied past a bend exit, where the far layout is itself graded at near_grade from the causal edge. Row test (`seam_check.py`), B2RE, W [MeV/m] at the 12 x rows, 400 vs 800 far columns:

| | tail row, outer 3 x rows | horn row, outer 3 x rows | horn row core (x = +232 µm) |
|---|---|---|---|
| as is, 400 / 800 | +0.93 −0.93 +0.34 / −0.23 −0.28 +0.11 | +0.62 +0.04 +0.07 / +0.30 +0.13 +0.01 | −0.452 / −0.409 |
| seam 2, 400 / 800 | +0.185 +0.006 −0.003 / identical to 3 decimals | +0.263 +0.168 −0.004 / identical | −0.416 / −0.416 |

Whole 12 × 80 mesh (`seam_mesh.py`), far_zbins 200 vs 400 (max local |ΔW| / max|W|, |ΔF| / max|F|, mean <W>), time per mesh:

| state | as is: 200 vs 400 | seam 2: 200 vs 400 | seam 2 vs as is (both 200) | time |
|---|---|---|---|---|
| loss B2RE 37.74 | 17.7% / 14.3% / 0.14% | 0.49% / 0.95% / 0.03% | +0.28% mean, 19.9% local (the edge rows) | 49 → 56 s |
| loss B1RE 48.73 | 2.7% / 6.6% / 0.02% | 2.1% / 1.3% / 0.02% | +0.06% mean, 2.9% local | 67 → 63 s |
| gain B2LE 11.36 | 5.2% / 1.8% (mean ≈ 0) | 3.6% / 0.65% | 7.9% local | 32 → 34 s |

Golden Gaussian dipole (test/golden, 0.3 m bend + 0.2 m drift): in the bend the far-count convergence (200 vs 800) improves from 1–2e-3 to ≤1.6e-4 of max|dE| and the two converged layouts agree to 1.3e-4; the drift kicks are unchanged except through the beam (3e-7). The remaining 2–3.6% local far-count sensitivity at B1RE / B2LE is at the entrance end of the far region (the horn's retarded z stepping ~0.16 µm per 5 mm column there, plus x′-refinement phase), not the seam. `test_quantile_frame.py` pins the old layout with `seam_ratio: 0` and adds `test_seam_ratio_extends_near_region_in_bend_only`.

### 11l. The benchmark chicane with the reference recipe (2026-11-24)

Runs `2dR`, `2dR_noT`, `1d_full` of `test/chicane_2d_vs_1d.py` (same tracking, beam and 489-node auto schedule as 2dP; integration = BC20 reference: near_adapt, near_adapt_x, far_adapt_x, xbins 100, ibp_check 3; before seam_ratio). Full table in `benchmark_results/chicane_2d_vs_1d/closed/summary.txt`, summary and interpretation in `docs/DFCSR_2D_methods.md` §9.2. Headline: 2dR reproduces 2dP to 0.6% in energy loss (−2672 vs −2687 keV) and 1% in emittance growth (0.490 vs 0.494 µm); without the transverse force the emittance growth is 0.952 µm and the core slice σE +158 vs +29 keV; 1D gives −1746 keV and 0.549 µm. No ibp_check warning in 489 steps. Both 2D runs took 64 min serial while sharing the machine with each other and with the state tests, so the time is not the recipe's cost (2dP alone: 28.8 min).

### 11m. Chicane wake figures; the first bend needs pre-history and a reach of 2 L_f (2026-11-24)

`test/chicane_wake_plots.py` (maps, z/x projections, along-s, 2D-vs-1D table) and `test/chicane_prehistory.py`; write-up with tables in `docs/talman_cancellation.md` §7. Findings: (1) in B1 the 2D wake was half of 1D because the lattice has 0.1 m of drift before B1 (no sources before s′ = 0; the straight-line history is retarded by ≈ Rφ³/6 and decays as 1/u) and because the 0.8 m reach covered only 0.3 m of the straight section from the exit. With D0 = 2 m and n_formation_length 2 (= 4) the peak wake matches 1D at mid-bend and is within 8% at the exit; ⟨W⟩ −0.0146 vs 1D −0.0212 at the exit, −0.051 vs −0.039 0.3 m after. **So n_fl = 1 is NOT converged at the exit of a short, strong first bend** (§11k tested only B2RE/B1RE, small φ) — the first-bend part of Indranil's observation is real; the needed reach is ≈ arc + 2Rφ. BC20 lattices starting at B1LE understate the first bend. (2) F(ξ) in bends is a bell with the curvature sign, 3.3–4.9× the plain centripetal 2κQλ/(4πε₀), i.e. the log-enhanced Talman force; zero in drifts. (3) 2D vs 1D: 30–100× inside B2/B3 (1D entrance transient vs 2D immediate), 1.4–1.7× at the exits, 2–4× late in D3, −20% at B4 exit/Df. Open item §10.0 stands.

**§11m addendum — `CSR_integration.prehistory` implemented (same day).** Synthetic straight-line history before the lattice start: `_layout_bounds` clamps s3 and s1 at −prehistory instead of 0, `get_CSR_integrand` zeroes the curvature for s′ < 0, the straight orbit comes from `_lattice_vectors` and the first snapshot from the interpolant's existing t-clamp (lookup_vec and interpolate3D both clamp k ≥ 0, α ≥ 0; `_check_retention` already requires the t = 0 snapshot while the reach is clamped). Chicane B1 with the benchmark lattice + `prehistory 2.0`, `n_fl 2`: ⟨W⟩ at the exit −0.0133 vs −0.0146 with a real 2 m drift (−0.0039 without, 1D −0.0212); peak wake 0.078 vs 0.079 (1D 0.086); 3.1 min vs 6.0. Unit test `test_prehistory_extends_the_reach_below_the_lattice_start`; default 0 keeps the golden case bit-identical. Table in talman_cancellation.md §7.4.

### 11n. The straight section before a bend: `reach_straight` (2026-11-24)

The first-bend deficit of §11m was only partly the missing history: on the pickled B1-exit state of the chicane (synthetic history, Rφ = 0.5 m) the straight-section contribution converges as U/(U + Rφ) — ⟨W⟩ −0.0103 (0.9 m of straight, the flat reach), −0.0139 (2.5 m), −0.0150 (4 m), −0.0163 (10 m), −0.01675 (20 m, converged); 1D −0.0212. The "n_fl 2 = n_fl 4" agreement in §11m was both clipped at the 2 m of drift. Note L_f in the code uses 5σz, so the flat reach at B1 was 1.37 m, not the 0.8 m I quoted. Fix: `CSR_integration.reach_straight` (default 40 Rφ, `reach_slip_max` 2σz): the straight section [entrance − 40Rφ, entrance) as a separate region graded in u (ratio 1 + near_grade; 56 nodes for 20 m, same time per mesh as without), in the bend and in the drift after it while the slip (Rφ³/6, + xφ²/2 past the exit) is < 2σz; the regular layout is untouched. Converged first-bend exit: peak 0.0855 vs 1D 0.0861, mean −0.0168 vs −0.0212 (2D 21% below 1D — physics, not boundary). Golden dipole unchanged (its exit slip is 2.5σz); BC20 B2RE mid (slip 1.9σz) would fire but a doubled reach there moves ⟨W⟩ 0.01%. Tables in talman_cancellation.md §7.5; `test_straight_section_reach_in_the_first_bend`. Full chicane rerun `2dRp` (prehistory 20 + rule) launched 14:35 for the §7 tables.

**§11n addendum.** `reach_slip_max` 2 → 5 (with 2 the rule switched off 0.18 m past the B1 exit while 1D still grew: ⟨W⟩ −0.020 vs −0.038; with 5: −0.033). The region is bounded by the previous bend's exit and by the retained history (the first version reached 36 m back through BC20 and clamped onto the oldest snapshot: +0.235% garbage at B2RE); since prev_bend_reach already integrates from the previous bend's entrance, the rule acts only on the first bend and its drift (B2RE state bit-identical). Full first-bend run with the defaults + prehistory 20: exit peak 0.0855 vs 1D 0.0861, mean −0.0167 vs −0.0212; 0.3 m after the exit 2D is 37% above 1D. Chicane `2dRp` (prehistory 20) restarted with these defaults.

### 11o. Indranil's "charge mesh over-samples the edges in quantile mode": the z axis over every particle (2026-11-24)

Checked on the pickled states (`/tmp/mesh_edges.py` → `test/bc20_indranil/mesh_edges.py`). Not the x axis: `mesh_x_quantile` clips the tail particles onto the quantile values, which is a spike in the monitor's histogram, but the monitor is smoothed over 3 deposition resolution lengths (120 µm on 41 µm cells) and the clipped and dropped variants give the same x nodes to 0.01 µm. It is the z axis: `charge_nodes` spans [min z, max z] of every particle (+2% pad) with a 30% uniform share of the nodes and a ×1.5 grading, so a sparse tail takes nodes. Loss beam at s = 0.15 (quantile frame, 12.6 µm core, 0.1% of the charge out to +49 µm): axis [−18.7, +50.1] µm, 15 of the 80 cells with < 0.05% of the charge, 3.6 µm apart in empty space, median core cell 0.77 µm. With the particles outside the 0.1–99.9% quantiles dropped (not clipped): axis [−17.9, +26.2], 2 empty cells, median cell 0.59 µm. Gain B2LE (peak frame): 1 → 0 empty cells; loss B2RE: 2 → 2 (no tail there).

Fix: `CSR_computation.mesh_z_quantile` (default None = every particle, bit-identical; 0.001 in the BC20 recipe `Q_MESH`), the z twin of `mesh_x_quantile`: axis on the quantile span built from the particles inside, particles beyond get the edge column's kick (`apply_wakes clamp_z`). `test_mesh_z_quantile_drops_the_sparse_tail`.

Effect on the kicks (same 2D wake evaluated on both meshes, then interpolated to the particles): B2RE 0.4% of the kick rms (no tail); first bend s = 0.15: 12% of the kick rms and the mean kick −0.0150 → −0.0101 MeV/m (both small against the 2.2 MeV/m rms) — i.e. at 1 µm horns and 0.6–0.8 µm mesh cells the per-particle kick is NOT converged in the mesh's z resolution. Convergence check against 160 and 320 z nodes below.

Convergence of the per-particle kick in the mesh's z resolution, loss beam s = 0.15 (1.1 µm FWHM horns), the same 2D wake evaluated on each mesh (`mesh_z_conv.py`), against 320 nodes with `mesh_z_quantile` 0.001:

| mesh | median dz | min dz | time / mesh | kick rms | rms(kick − ref) | mean kick |
|---|---|---|---|---|---|---|
| 80 nodes, every particle (as run) | 0.77 µm | 0.29 µm | 65 s | 2.156 | 18.0% of the kick rms | −0.0150 |
| 80, q = 0.001 | 0.59 | 0.25 | 77 s | 2.234 | 11.8% | −0.0101 |
| 160, q = 0.001 | 0.28 | 0.11 | 159 s | 2.367 | 2.6% | −0.0038 |
| 320, q = 0.001 (reference) | 0.14 | 0.06 | 318 s | 2.406 | — | −0.0020 |

**So with µm horns the 80-node charge mesh is not converged**: the horn's kick is 10% low in rms and 18% off particle by particle; the FWHM/3 finest-cell criterion the first-kick warning uses is too loose, FWHM/10 (160 nodes here, 2.6%) is what it takes. The loss run's first 10 m (horns 0.8–1.3 µm) carry this error; from B2RE on (σz 20 µm body, 1.2 µm peak on a broad pedestal) the 80-node mesh changes the kick by 0.4% when the tail is dropped, but the resolution check there has not been done. Recommendation for Indranil's loss beam: `zbins: 160` with `mesh_z_quantile: 0.001` (+100% mesh time in the first bends; the mesh time is the whole kick time). Not yet rerun.

**§11n addendum 2 — chicane `2dRp` (prehistory 20 + reach rule) done (104 min, machine shared):** B1/D1 energy −3.2/−70 keV = 1D (−3.0/−66), was −0.9/−49; B2, B3, D3 unchanged (prev_bend_reach covers their drifts); B4 −264 vs −231 keV through the changed chirp (1D −366); total −2727 keV (+2%), emittance growth 0.526 µm (was 0.490; 1D 0.549), core slice σE +16.9 keV (was +28.8; 1D +10.4). Tables: talman_cancellation.md §7.1/7.6, methods §9.2.


### 11p. Indranil v2 reruns; the straight history diverges in the drift (2026-11-24)

v2 = prehistory 45 m (40 Rφ of B1LE, where the lattice starts) + straight-section reach + mesh_z_quantile 0.001, loss with 160 z nodes. Launched 15:37; machine thrashing (Cisco Secure Client socket-filter extension leaked 19 GB) until ~16:40; both runs killed by the 22:18 reboot at s ≈ 25.4 m (157 kicks). They also ran without FI_PROVIDER=tcp (run_dfcsr_bc20 imports mpi4py before pyDFCSR_2D): each rank 54% kernel time. Fixed in the script; relaunched 22:52 (loss, 6 ranks) and 23:24 (gain, 3 ranks, after the fix below).

First 25 m of the interrupted runs (`b1le_v1_v2.py`), ∫⟨W⟩ds [MeV]: loss B1LE v1 −0.50, v2 −1.35, 1D −2.05; D1..Q1EL v1 −2.30, v2 −2.72, 1D −2.98; B2LE v1 −1.43, v2 −1.36, 1D −2.90. Peak wake inside B1LE: v2 12–13 MeV/m vs v1 6–11 (1D 17–43; the 1D peak is the 0.2 µm-binned horn). Gain: an artefact — 2.4 MeV/m 0.2–0.5 m past B1LE against 0.04 in 1D, from the straight-section rule in the drift, where it grows as ln(reach) (talman_cancellation.md §7.7 table). Rule restricted to observers inside the bend; gain state 0.19 m past the exit back to peak 0.046 MeV/m. The first chicane 2dRp's D1 had the same artefact: rerun started 23:24. Scripts: `gain_drift_artefact.py`, `straight_reach_divergence.py`, `b1le_v1_v2.py`.

Chicane 2dRp rerun (rule inside bends only) done, 100 min: B1 −3.2 keV (1D −3.0), D1 −56 (2dR −49, artefact run −70, 1D −66), B2–D3 unchanged, B4 −264 (same as the artefact run → from B1 via the compression, not D1), total −2709 keV, emittance growth 0.534 µm, core slice σE +17.1 keV. Tables in talman_cancellation.md §7.1/7.6.

**gain_v2 done** (143 min at 3 ranks, 336 kicks; v1 took 205 min with the kernel-spinning MPI). Per section vs CSR off [MeV] (→Q2ER / B2RE / drift / B1RE / D32 / total): v2 −0.138 / −0.090 / −0.068 / −0.014 / −0.019 / **−0.329**; v1 −0.113 / −0.083 / −0.069 / −0.015 / −0.020 / −0.301; 1D −0.191 / −0.022 / −0.127 / −0.007 / −0.049 / −0.396. εnx 1st/3rd: v2 43.88/15.72, v1 43.91/15.71, 1D 44.98/15.93, off 44.30/15.94 µm. First 12 m [keV] (`early_sections.py`): B1LE v2 −10.8, v1 −3.9, 1D −0.9; B1LE exit → B2LE −41.0 / −31.4 / −59.9; B2LE −5.4 / −2.4 / −7.2. Inside B1LE (pickled state at s = 0.90, `gain_b1le_terms.py`): ⟨W⟩ v2 −0.0051, prehistory 0 −0.0034, 1D −0.0019 MeV/m; peak 0.061 / 0.022 / 0.041. The velocity (W2, ρ·div v) terms carry −0.002 of the 2D mean (with them off: −0.0030 / −0.0011) and almost none of the peak — the beam is converging, and W2 is the potential-energy term that 1D does not have (talman_cancellation.md §5). The straight-section reach converges inside B1LE (`gain_b1le_reach.py`: ±0.0025 MeV/m from 0 to 170 m of straight). Open: the 2D peak inside B1LE is 1.5× the 1D one although σx = 144 µm ≪ (σz²R)^{1/3} = 1.4 mm; in the chicane's B1 (also thin) it matched to 1%. Figures: `bc20_indranil/wake_*_gain_v2.png`, `wakes_along_s_gain_v2.png`.

**loss_v2 done** (190.9 min at 6 ranks, 319 kicks, 160 z nodes; v1 305 min at 3 ranks with the kernel-spinning MPI). Per section vs CSR off [MeV] (→Q2ER / B2RE / drift / B1RE / D32 / total): v2 −9.872 / −4.272 / −3.445 / −3.565 / −2.678 / **−23.831**; v1 −8.728 / −4.341 / −3.366 / −3.430 / −2.517 / −22.382; 1D −13.462 / −3.157 / −3.682 / −3.789 / −3.238 / −27.329. rms ΔE 17.07 / 16.82 / 18.15 MeV. εnx 1st/3rd: v2 72.07/55.85, v1 72.44/55.67, 1D 77.95/63.63, off 25.42/23.24 µm. Spike-slice εnx 91.9/27.6 (v1 89.7/29.0). Final peak current 68.2 kA (off 55.4), FWHM 3.5 µm. First 12 m [keV] (`early_sections.py`): **B1LE v2 −1879, v1 −911, 1D −2176** (the first-bend gap to 1D from 58% to 14%); B1LE exit → B2LE −2958 / −2838 / −2947 (2D = 1D); B2LE −1806 / −1956 / −2954. So the first-bend deficit in v1 was mostly the missing straight history (+ the under-resolved 80-node mesh); B2LE remains 2D < 1D by 39% (the horn x-width effect of §11c). Downstream of Q2ER v2 and v1 agree within 0.16 MeV per section. Roughness metric (rms deviation from a 20 µm smoothing) is the same in v1 and v2 by section (0.8–5): it measures the µm horn structure, not noise. Figures `bc20_indranil/wake_*_loss_v2.png`, `wakes_along_s_loss_v2.png`.

### 11q. Indranil's FL scan, far cell fixed: three mechanisms, none of them "missing formation lengths" in a bend (2026-11-26)

Indranil reran with far_zbins = 200·FL (far cell fixed): the total loss still rose by 3.5–12% from FL 1 to FL 6 on four beams, mostly after B3RE (q4er, D24), in B1RE (−0.94 MeV on L1-17.6284) and in D31/D32 (`example/BC20_Indranil/fl_convergence.pdf`, `fl_per_element.pdf`). Studied on his L1-17.6284 beam (converted to `beam_L1-17.6284_bmadx.txt`, recipe v2 + frame auto), CSR-off kick states dumped with the FL-6 history retention (`dump_state.py`, now also writes states before s_start), FL 1 vs 6 on the same state with `fl_decomp.py` (per-source-element attribution of the change):

1. **Drift after a bend pair — a real omission at FL 1.** 1.5 m past B3RE (s = 27.0): ⟨W⟩ +0.066 (FL 1) → −0.188 (FL 2 = 4 = 6) MeV/m; sources D17 −0.10, B3LE −0.08, Q6E −0.05, D16 −0.02 (the B3LE arc and the 0.8 m between B3LE and B3RE). 6.3 m past (31.3): −18%, same sources, converged by FL 4. In a drift the reach stopped at the last bend's entrance (drift_reach), and its history before it was discarded, so B3LE was never seen downstream of B3RE. **Fix (48bbf02, default on): `drift_causal_reach`** — the sources below the regular drift layout get their own region from the causal edge (graded in u at 0.5%, ~100 columns), bounded by s′ = 0 (never the synthetic prehistory, talman 7.7) and by the retention, which now keeps the previous bend from n_fl·L_f before its entrance. Against a uniform FL-6 brute force (causal clip off, far_zbins ×4, 400 x′ nodes):

| state | old layout FL 1 | drift_causal_reach FL 1 | same, FL 6 | uniform FL 6 reference |
|---|---|---|---|---|
| 27.0 (1.5 m past B3RE) | +0.0683 | −0.1852 | −0.1851 | −0.1852 |
| 31.3 (6.3 m past B3RE) | +0.4211 | +0.3467 | +0.3468 | +0.3471 |
| 46.2 (D31, 7.5 m past B2RE) | −0.4030 | −0.4030 | −0.4029 | −0.4029 |
| 49.6 (0.56 m past B1RE) | −1.0722 | −1.0854 | −1.0854 | −1.1178 |
| 50.5 (1.43 m past B1RE) | −0.5095 | −0.5250 | −0.5250 | −0.5250 |

The 3% at 49.6 is the exit-transient fine window (far_fine_max 500; 4000 gives −1.1034 against −1.1049 for the uniform layout at FL 1 — the remaining 1.2% to the FL-6 reference is real reach beyond 1 L_f). Not changed (4× the cost at that state).

2. **Large n_fl breaks the drift layout — an artefact of FL 6 itself.** 1.43 m past B1RE the old layout gave −0.867 MeV/m at FL 6 against −0.525 for the uniform reference: with a 12 m reach the exit-transient causal scan finds B2RE's tail alive, the fine window lands in D31, and B1RE's arc is integrated on 7–12 cm graded cells. Part of Indranil's FL-6 extra loss in D32 is this. With drift_causal_reach the regular drift layout always has the FL-1 reach, so FL no longer moves it (FL 1 = FL 6 in the table).

3. **B1RE: x′ under-resolution near |τ| = 1, masked differently by n_fl.** Inside B1RE (τ = −1.17 at 48.38, −0.96 at 48.73) the change is attributed to B1RE's own sources, not to reach. One observer: recipe (100 x′ + adaptive) FL 1 −13.30, FL 6 −12.24; adaptive off: 100 nodes −13.64, 400 −12.0025, 1600 −12.0102 at FL 1 and −12.0028 / −12.0106 at FL 6 — converged, FL-independent. The adaptive x′ refinement flags columns against ρ_max over the whole region, and a longer region (FL 6) changes which columns get refined. Mesh means (`b1re_mean_xnodes.py`):

| | recipe FL 1 / FL 6 | 400 + adaptive FL 1 / FL 6 | 400 uniform FL 1 / FL 6 | 1600 uniform FL 1 / FL 6 |
|---|---|---|---|---|
| B1RE 48.38 | −1.774 / −1.131 | −0.997 / −1.007 | −0.980 / −0.986 | −0.979 / −0.985 |
| B1RE 48.73 | −3.297 / −2.996 | −2.977 / −2.972 | −2.964 / −2.959 | −2.964 / −2.959 |

Full-mesh x′ check against 1600 uniform nodes (`x_nodes_check.py`; max local error / mean W error, % of max |ref|):

| x′ nodes | B1RE 48.38 (this beam): W max / mean / F max | loss beam B2RE 37.74 (horns): W max / F max | time (B1RE) |
|---|---|---|---|
| 100 + adaptive (recipe) | 6.7 / 18 / 6.5 | 0.66 / 0.83 | 161 s |
| 200 + adaptive | 2.6 / 7.5 / 1.2 | 0.42 / 0.40 | 197 s |
| 400 + adaptive | 1.3 / 2.4 / 1.1 | 0.23 / 0.28 | 257 s |
| 400 uniform | 0.8 / 0.1 / 15 | 0.54 / 6.1 | 90 s |
| 800 uniform | 0.3 / 0.05 / 7.8 | 0.07 / 0.44 | 159 s |

So the §11i choice of 100 x′ nodes was wrong for a strongly compressed beam crossing |τ| ≈ 1 in B1RE: W needs ~400 nodes there (the adaptive refinement only reacts to density jumps, not to the geometry), F on horn beams needs the refinement. Recommendation: `xbins: 400` with the adaptive refinement (+60% at B1RE). Confirmation runs `b176_fix_fl1` / `b176_fix_fl6` (full BC20, drift_causal_reach, 400 x′ nodes, 6 ranks each) launched 01:33 Nov 26.

Inside a bend away from |τ| ≈ 1 (loss beam B1RE 2/3, B2RE mid, §11k) FL 1 → 6 at fixed cell is still 0.01–0.03%. So, for Indranil: keep n_formation_length = 1; the three sources of his FL dependence were the missing previous bend in drifts (fixed), the drift layout breaking at large FL (fixed), and x′ nodes in B1RE (use 400).

**Confirmation runs done** (`fl_per_element.py b176_fix_fl1 b176_fix_fl6`; recipe v2 + drift_causal_reach + 400 x′ nodes, 6 ranks each, 6.0 h and 7.6 h): total FL 1 −23.807, FL 6 −24.018 MeV, **+0.89%** (Indranil's scan with the old code: +11.76%). Per section FL 6 − FL 1 [MeV]: B1LE −0.001, D1 +0.001, B2LE 0.000, B3LE–B3RE −0.168, D18–D24 −0.039, B2RE −0.002, D30 +0.028, D31 −0.034, B1RE +0.009, D32 −0.004. B1RE is now FL-independent (was −0.94 MeV extra at FL 6). The remaining −0.17 MeV is inside B3RE: at FL 6 the in-bend reach covers the straight before B3LE (q5el, D14, D15), which prev_bend_reach (from the previous bend's entrance) does not. Not fixed; 0.7% of the total. For comparison Indranil's FL 1 −21.89 (old code, 100 x′) and FL 6 ≈ −24.46. `xbins: 400` now in run_dfcsr_bc20.PIN_2D_INTEG and in bc20_indranil_recommended.yaml.


### 11r. Correction to 11q item 3, and error-based x′ refinement (`x_adapt: error`, default) (2026-11-26)

**Correction.** The B1RE n_fl dependence was not the geometry near |τ| = 1 as such. At FL 1 the sources of D31 and B2RE's tail, seen from B1RE, sit in the `prev_bend_reach` region, which was integrated on `xbins` uniform x′ nodes with **no refinement**; at FL 6 the same sources fall inside the regular far region, which is refined. `fl_decomp.py` put the refinement correction into the observer's own element (its "remainder" bucket), which is why 11q attributed the change to B1RE's own sources. With the refinement also applied in the previous-bend, straight-section and drift-causal regions (gated by `far_adapt_x`), 100 x′ nodes + the old density test give FL 1 −1.136 / FL 6 −1.131 at 48.38 (was −1.774 / −1.131). The 400-node advice of 11q is superseded.

**Error-based refinement.** The density test (re-evaluate a column if ρ jumps by > 2% of the region's ρ_max between nodes) does not see integrand variation from the geometry and the ±∂ρ/∂x′ cancellation, and its threshold depends on the region. `x_adapt: error` estimates each column's x′ quadrature error from the integrand itself: xbins nodes, then the midpoints (trapezoid T_n → T_2n, error |T_2n − T_n|/3, W and F), then repeated halving (≤ `x_adapt_rounds` 4) of the columns with the largest s′-weighted errors until what is left is below `x_adapt_tol` (1e-3) of the region's gross Σ w_j|T_j|. Against 1600 uniform nodes (`x_nodes_check.py`, full mesh; FL 6 pickle):

| | B1RE 48.38: W max / mean W / F max | time | loss B2RE horns: W max / F max | time |
|---|---|---|---|---|
| 100 + density (old recipe) | 6.7 / 18 / 6.5% | 95 s | 0.66 / 0.83% | 26 s |
| 400 + density (11q advice) | 1.3 / 2.4 / 1.1% | 144 s | 0.23 / 0.28% | 37 s |
| 100 + error 1e-3 | 0.06 / 0.05 / 1.8% | 53 s | 0.06 / 0.21% | 14 s |
| 1600 uniform (reference) | — | 191 s | — | 66 s |

FL 1 vs FL 6 mesh means with all regions refined (`b1re_mean_xnodes.py`): 48.38 error-mode −0.9802 / −0.9861 against 1600 uniform −0.9794 / −0.9852; 48.73 −2.9641 / −2.9597 against −2.9636 / −2.9593 (the 0.2–0.6% FL difference is in the reference too: real reach). Time 5–6 s (FL 1) vs 11 s for the old recipe. Made the default; the BC20 recipe and bc20_indranil_recommended.yaml back to `xbins: 100`. The 1.8% F max at B1RE does not improve at tol 3e-4 (not x′). `test_x_adapt_error_converges_and_is_default`.

**Full runs with the new default** (`b176_err_fl1` / `b176_err_fl6`: 100 x′ nodes + `x_adapt: error`, drift_causal_reach, 6 ranks each, the two sharing the machine): total FL 1 −23.834, FL 6 −24.042 MeV (+0.88%, the same −0.17 MeV from the straight before B3LE seen inside B3RE as with 400 nodes; B1RE +0.014). Against the 400-node + density run at FL 1 (`b176_fix_fl1`): −23.834 vs −23.807 MeV (0.11%), every section within 0.015 MeV. **Run time 2.6 h (FL 1) and 3.2 h (FL 6) against 6.0 h and 7.6 h** — 2.3× faster. 1D (b176_1d, 8.5 min): −28.58 MeV. Figures (`plot_indranil.py b176 b176_fix_fl1 b176_err_fl1`): `bc20_indranil/wakes_along_s_b176_err_fl1.png`, `wake_zproj_…`, `wake_maps_long/trans/spike_…`. Along the lattice the new default and the 400-node run coincide in max|W(z)|; 1D peaks 2–3× higher inside the bends (it resolves the µm horn at 20 bins/FWHM; 2D is limited by the horn's x-width, §11c), 2D shows its known transients in the sextupole sections (≈9 MeV/m at 18 m, ≈5 at 29 m, where 1D decays smoothly to 0.2), and 2D falls off faster than 1D in the drifts after B1LE and B1RE. Plot fixes: lattice strip for lower-case/split element names, 1D drawn on its own kicks (its csr1d schedule differs by a few kicks), labels.
