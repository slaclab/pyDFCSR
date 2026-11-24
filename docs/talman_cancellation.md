# The Talman cancellation and what pyDFCSR says about it

*2026-11-23. Literature part compiled from the primary sources listed at the end (every reference was checked against the publisher, arXiv, INSPIRE or JACoW record; the two Stupakov papers are in `reference/`). The code part refers to runs recorded in `progress_facet_bc20.md` §7–9 and `progress.md` ("The BC20 recipe on the closed benchmark chicane").*

## 1. The question

A relativistic bunch on a straight line has almost no self-force: the electric and magnetic forces cancel to O(1/γ²). In a bend they do not. Talman (1986) pointed out a transverse self-force of order λ/R, independent of γ, that he called the centrifugal space-charge force, and argued it could matter for storage rings. Whether this force has any effect on the beam, or is cancelled by an equally large energy effect, was argued over for two decades. The argument matters for a CSR code because the two sides amount to two different prescriptions for what to compute and apply to the particles, and a code that mixes them double-counts or omits a term of order λ/R.

## 2. Talman's force

For a line charge of density λ on a circle of radius R, in the median plane at radial offset x, the unshielded transverse self-force (Gaussian units, Lee's Eq. 42) is

$$F \equiv -(E_r + B_z)_b = \frac{2\lambda}{R}\Big[1+\ln\frac{8R}{x}+\dots\Big],$$

centrifugal, γ-independent, and logarithmically singular as x → 0 for a zero-thickness beam, with gradient ∂F/∂x ≈ 2λ/(R x). For a bunch (Derbenev and Shiltsev 1996, Eq. 48) the same structure appears as F_Talman = eA_s/R with A_s ≈ Neλ(s) ln[(Rσ_s²)^{2/3}/σ_⊥²], i.e. of order e²Nλ/R times the logarithm of the "overtaking distance" r_o ~ (σ_s²R)^{1/3} over the transverse size. The force is large and local: it builds up within millimetres of a magnet entrance, far shorter than the longitudinal formation length (24σ_z R²)^{1/3}.

## 3. The cancellation

**Lee (1990)** showed, for a coasting beam, that the force is not what drives the betatron motion. A particle doing a betatron oscillation also oscillates in kinetic energy, because it moves through the beam's own potential, and in curved geometry that energy change enters the radial equation at first order:

$$\delta\gamma = -\frac{e}{mc^2}E_r\,\delta r,\qquad \ddot{\delta r} = -\Big[\frac{c^2}{R^2}-\frac{e}{\gamma m}\Big(\frac{E_r}{r}+\frac{\partial E_r}{\partial r}+\frac{\partial B_z}{\partial r}\Big)\Big]\delta r.$$

The E_r/r term nearly cancels the gradient of the centrifugal force; the correct focusing quantity is ∂²(A_θ − φ)/∂z², of order λ/R² instead of λ/(Rx). What survives is that the *total* energy E = γmc² − eφ, not the kinetic energy, relates the radius to the bending field, which gives small tune and chromaticity shifts.

**Derbenev and Shiltsev (1996)** did the same for a short bunch with retarded potentials, through a canonical substitution Ĉ_x = C_x + (e/2iE)A_0 − …, Ê = E + eA_0, after which the dynamics is driven only by V_0 = e(A_0 − A_z) = e(φ − β·A). The Talman force is "cancelled by the effect of the particle energies' deviation under the influence of the transverse electric field, and therefore does not lead to emittance growth". What remains transverse is a *centripetal* force F_0 = −2Ne²λ(s)/R with no logarithm, plus focusing gradients ∝ Ne²(R²σ_s⁵)^{−1/3}.

**Li and Derbenev (2002, 2003, 2005)** wrote this as a general statement. In the radial equation of motion, the Lorentz force contains eβ_sA_s/r (the centrifugal space-charge force) and the kinetic-energy term contains −eΦ/r; both are dominated by the same local, logarithmically singular Coulomb integral, and for β → 1, A_s ≈ Φ, so G_c = e(A_s − Φ)/r is negligible for σ_z ≪ R. The dynamics is then naturally written in (x, x′, z, δ_H) with δ_H = [(E + eΦ) − E_0]/E_0 the canonical energy offset, driven by "effective" forces from the interaction Lagrangian L_int = −e(Φ − β·A): long-range, free of the logarithm, with the steady-state impedances Z_v^eff ∝ (ik)^{1/3}R^{−2/3} (the CSR wake) and Z_x^eff = −2/R (the Derbenev–Shiltsev centripetal force). Two consequences Li drew: a spectrometer after a dispersive section sees the canonical, not the kinetic, energy spread; and the extra *kinetic* energy spread a bunch acquires from its own potential while compressing (the Bane–Chao term) does not by itself cause emittance growth in the following bend, because eΦ cancels against F^CSCF.

**Geloni, Saldin, Schneidmiller and Yurkov (2003)** objected ("Misconceptions regarding the cancellation of self-forces…", DESY 03-165). Their points: G_c ≈ 0 is trivially true in steady state and uninteresting; the head–tail part of the transverse force (sources ahead of the particle, about twice the tail–head part) is local, with a formation length of half a bunch length, while Φ has a much longer formation length, so no effective cancellation takes place in a transient; and the position-dependent singular piece reappears as eΦ(0)/R in the initial condition, which cannot be removed by a choice of coordinates. Their conclusion was not that the canonical formulation is wrong but that it is "not useful": a code should compute the Lorentz force and the kinetic energy separately, as TraFiC4 does.

**Li and Derbenev (2005)** answered point by point: the head–tail part is included (it is the head–tail part of F^CSCF, and eΦ/r has the identical head–tail piece); the "sudden jump" of F at a magnet entrance is the 1/r factor switching on in both terms; and the uncancelled eΦ(0) acts as an initial canonical energy offset, computed once, which to first order in the optics causes no emittance growth through an achromat. Their summary: the canonical formulation "is completely equivalent to the Lorentz-force approach with the cancellation implicitly taken care of. Here TraFiC4 is a good example."

**Stupakov (2022)** settled the practical question. With s as time and the canonical pair (t, −H), H = γmc² + eφ, the transverse equation is

$$\frac{d^2x}{ds^2} = -\kappa^2 x + \kappa\frac{\Delta H - e\phi}{\gamma mc^2} + \frac{e}{\gamma mc^2}(E_x - B_y),$$

and although φ, E_x and B_y each diverge on a line charge, the *renormalized force* 𝓕 = e(−κφ + E_x − B_y) is finite: the −κeφ piece of the dispersion term absorbs the logarithmic κeA_s piece of the Lorentz force. On a circular orbit in steady state 𝓕 → −2eκλ, the Derbenev–Shiltsev centripetal force. That is the Lee/Li cancellation in one line. But the renormalized scheme needs ΔH = ΔE + eφ as an initial condition at the magnet entrance, and φ for a bunch arriving from a straight section involves ln σ_⊥ and diverges for a line charge, so "it cannot be used even in the simplest problem of a beam passing through a single bending magnet" — Geloni's eΦ(0) objection in different words. Stupakov therefore reverts to the kinetic energy, dΔE/ds = eτ·E, with the full transverse Lorentz force, Eq. 27 of his paper,

$$\frac{1}{e}F_\perp = \frac{1}{c}\int \frac{d^3r'\,\partial_{t_{\rm ret}}\rho}{|r'-r|^2}\big[(r-r')\cdot(n-n') - |r'-r|\,(n\cdot\tau')\big] + \int \frac{d^3r'\,\rho}{|r'-r|^3}\,(r-r')\cdot(n-n'),$$

which converges in 3D and in 2D and whose steady-state result (his Eq. 42) contains the Talman logarithm ln(σ_z^{2/3}/κ^{1/3}σ_⊥) explicitly, because this is the real force that must accompany the kinetic energy.

## 4. Status

The controversy is resolved in substance. All parties agree the two formulations are mathematically equivalent given consistent initial conditions; the dispute was about usefulness and about the initial potential energy. The consensus is:

- The physically acting transverse force in a bend is centrifugal, γ-independent, ∝ (e²Nλ/R) ln(r_o/σ_⊥), local, with a head–tail part. Combined with the kinetic-energy dispersion term, its logarithmic piece drops out of the dynamics on a closed orbit, leaving the centripetal −2eκλ and the Derbenev–Shiltsev focusing gradients. The initial potential energy survives as a "pseudo energy spread" (Li 2014): zero effect to first order in an achromat, but measurable in a spectrometer.
- The Talman force per se does not drive emittance growth. The residual transverse effects are modest: 2–3× below the dispersive growth from the longitudinal wake (Derbenev–Shiltsev, TTF), an order of magnitude smaller transverse than longitudinal wakes (Cai 2021).
- A code must be internally consistent in one of two ways: (i) kinetic ΔE from v·E, including the compression terms, **plus** the full transverse Lorentz force with a finite σ_⊥ (TraFiC4, CSRtrack 3D, Stupakov–Tang 2021 with Stupakov 2022); or (ii) canonical ΔH plus the renormalized 𝓕, usable only where φ at the start is known, i.e. recirculating or long-orbit problems. Mixing them — kinetic energy with potential-energy terms but no transverse force, or a 1D longitudinal wake plus −2eκλ — double-counts or omits eφ/R. Elegant's CSRCSBEND is a 1D energy kick with no transverse self-force; CSRtrack's projected mode likewise, its sub-bunch modes are scheme (i).

No published work isolates "chicane emittance with and without the transverse CSR force" as a controlled numerical benchmark; the closest are Li's argument against Bane–Chao-type emittance growth, the Derbenev–Shiltsev TTF estimate, and Cai's order-of-magnitude statement. This is a gap a paper from this code can fill.

## 5. Where pyDFCSR stands

**The code is scheme (i), consistently.** The longitudinal kick is W = v·E, the rate of change of *kinetic* energy, from Stupakov–Tang 2021: W₁ (the ∂ρ terms), W₂ = the compression term ρ∇·v, W₃ (∂_tβ, dropped). No potential-energy or canonical term is added. The transverse kick is Stupakov 2022 Eq. 27 in 2D (thesis Eq. 4.8), with ∂_t ρ replaced through the continuity equation: in the code's naming, W1 is the γ-independent Coulomb term ρ(r−r′)·(n−n′)/|r−r′|³ — the Talman force — and W2 + W3 are the ∂_tρ (radiative) terms. The code never adds −κeφ. Both forces are computed from the same deposited ρ(x, z, t) history, so the eΦ-type piece of the kinetic energy change and the eA_s-type piece of the transverse force are, by construction, the same integral with the same resolution — which is exactly the consistency the cancellation needs.

**The cancellation is visible in the runs.** Two results already in hand:

1. *The benchmark chicane* (5 GeV, 1 nC, σ_z 200 µm compressing to 6 kA; `test/chicane_2d_vs_1d.py`, runs `2dP`, `2dP_noT`, `1d_full`, same tracking and schedule):

   | at the exit, vs CSR off | 2D full | 2D, transverse W1 (Coulomb) only | 2D without transverse force | 1D |
   |---|---|---|---|---|
   | mean energy change [keV] | −2687 | −2704 | −2656 | −1746 |
   | core slice σ_E change [keV] | +28 | +37 | +160 | +10 |
   | emittance growth, η-corrected [µm] | 0.494 | 0.490 | 0.947 | 0.549 |

   The 2D kinetic energy kick contains the potential-energy (compression, W₂) part that the 1D radiative wake does not have; applied alone it gives a core slice energy spread 16× the 1D one and twice the emittance growth. Adding the transverse force — whose Coulomb part is the counterpart of that potential energy — brings the emittance back to within 10% of 1D, while the kinetic slice energy spread stays at +28 keV. That is Li's statement in numbers: the potential energy shows up in the kinetic energy spread but not in the emittance, because eΦ/R cancels against F^CSCF.

2. *FACET-II BC20, Sergei's beam* (progress_facet_bc20.md §9c): without the transverse force the energy loss is the same within 1% (−8.10 vs −8.18 MeV) but the final εnx is 19.8 µm, about what 1D gives (18.8–21.1); with it, 15.0 µm against 14.8 with CSR off. Almost all of the difference arises in the last bend, where the transverse force cuts the CSR-induced px kick by ~40% (−10 vs −17 µrad mean; −22 vs −37 µrad in the spike). A 1D code, applying only the energy kick, overestimates the emittance growth from the last bend by about +5 µm against +0.2 µm.

So "the 2D result converges to the 1D result when the transverse force is included" is not a coincidence: 1D applies the radiative energy kick alone, which is the canonical-formulation answer (scheme ii without the −2eκλ remainder); 2D applies kinetic energy plus the full Lorentz force, which is scheme (i); the cancellation makes the two agree on the emittance to the extent that the centripetal remainder and the radiative transverse terms are small, and differ exactly where the potential energy is real — the kinetic slice energy spread.

## 6. How the code can address the controversy

The code has everything needed to test the cancellation quantitatively rather than argue it, because it can evaluate both members of G_c on the same density:

1. **Isolate the Talman part — done.** `CSR_integration.transverse_terms: W1` keeps only the Coulomb term of the transverse force. The chicane run `2dP_W1` (2026-11-23, table above) gives an emittance growth of 0.490 µm against 0.494 with the full force and 0.947 without it: the Coulomb (Talman) term alone does all of the cancelling. The radiative W2 + W3 terms change the emittance by 1% and the core slice energy spread from +37 to +28 keV, i.e. they are the small, separate transverse radiation effect, as Derbenev–Shiltsev and Cai estimate. This is the first direct numerical demonstration of the cancellation on a bunch compressor that I am aware of.

2. **Compute the potential.** φ at the particles is the same retarded integral with the kernel 1/|r − r′| and ρ (a "W₀" term) — the machinery of `get_CSR_integrand` with one more output. With it the code can evaluate, per particle and per kick, G_c = e(A_s − Φ)/r ≈ (W1-transverse force) − κeφ, and show directly how small it is, how it behaves at a magnet entrance (Geloni's objection), and how its head–tail part behaves. This is the controlled demonstration the literature lacks.

3. **Track both energies.** Keep ΔH = ΔE + eφ alongside ΔE. Li's prediction is that the dispersive motion follows ΔH while a spectrometer sees ΔE; the code can produce both, which is what an experiment at FACET-II with a known compressed beam would compare against.

4. **The initial-condition problem.** Stupakov's objection to the renormalized 1D scheme is that φ at a bend entrance diverges for a line charge. In 2D with a finite σ_⊥ it does not; the code can compute φ(0) and show how large the "pseudo energy spread" it implies is for BC20, and whether it is visible in the final slice energy spread (the +28 keV above is the chicane's version).

5. **A clean benchmark.** A thin Gaussian beam in a single dipole with dispersion, against the analytic cancellation (Lee's coasting-beam limit for a long bunch; Derbenev–Shiltsev's −2eκλ for a short one): the code's full-force emittance growth should match the analytic centripetal remainder, while the no-transverse-force run should show the spurious growth from the potential-energy kick. This was flagged in §9c as the validation to do before quoting the BC20 result.

Items 1 and 5 are runs of the existing code (half an hour and a few minutes respectively); 2–4 need one new integrand output and some bookkeeping.

## References

- R. Talman, Phys. Rev. Lett. 56, 1429 (1986). doi:10.1103/PhysRevLett.56.1429
- E. P. Lee, Part. Accel. 25, 241 (1990).
- B. E. Carlsten and T. O. Raubenheimer, Phys. Rev. E 51, 1453 (1995); B. E. Carlsten, Phys. Rev. E 54, 838 (1996).
- Ya. S. Derbenev and V. D. Shiltsev, SLAC-PUB-7181 / FERMILAB-TM-1974 (1996). doi:10.2172/251656
- R. Li, "Applications of the cancellation effect in CSR studies", PAC 2003, TOAC009; R. Li and Ya. S. Derbenev, "Discussions on the cancellation effect on a circular orbit", PAC 2005, TPAT019; R. Li, Phys. Rev. ST Accel. Beams 11, 024401 (2008); R. Li, arXiv:1401.2868 (2014). (R. Li and Ya. S. Derbenev, JLAB-TN-02-054 (2002), not available online.)
- K. Bane and A. Chao, Phys. Rev. ST Accel. Beams 5, 104401 (2002).
- G. Geloni et al., Nucl. Instrum. Meth. A 522, 230 (2004) [arXiv:physics/0205001]; DESY 03-044 [physics/0304033]; "Misconceptions regarding the cancellation of self-forces in the transverse equation of motion for an electron in a bunch", DESY 03-165 [physics/0310133] (2003).
- R. Talman, Phys. Rev. ST Accel. Beams 7, 100701 (2004).
- G. Stupakov and J. Tang, Phys. Rev. Accel. Beams 24, 020701 (2021) (`reference/longitudinal_wake_theory.pdf`).
- G. Stupakov, Phys. Rev. Accel. Beams 25, 014401 (2022) (`reference/transverse_wake_theory.pdf`); G. Stupakov, arXiv:1901.10745.
- Y. Cai and Y. Ding, Phys. Rev. Accel. Beams 23, 014402 (2020); Y. Cai, Phys. Rev. Accel. Beams 24, 064402 (2021).
- M. Dohlus and T. Limberg, "CSRtrack: faster calculation of 3-D CSR effects", FEL 2004, MOCOS05; A. Kabel, M. Dohlus, T. Limberg, Nucl. Instrum. Meth. A 455, 185 (2000) (TraFiC4).
- M. Borland, Phys. Rev. ST Accel. Beams 4, 070701 (2001) (Elegant CSRCSBEND).
- C. Huang, T. J. T. Kwan, B. E. Carlsten, Phys. Rev. ST Accel. Beams 16, 010701 (2013). (No paper by these authors on a "lack of Talman cancellation" could be found; that attribution is unverified.)
- A. D. Brynes et al., New J. Phys. 20, 073035 (2018).

## 7. The benchmark chicane in figures (2026-11-24)

Runs `2dR` (2D, full force), `2dR_noT` (2D, transverse force off) and `1d_full` (1D, no drift cutoff) of `pyDFCSR_2D/test/chicane_2d_vs_1d.py`: 5 GeV, 1 nC, σ_z 200 µm → 20 µm, four 0.5 m dipoles of 48.3 mrad (R = 10.36 m), same bmad-x tracking and 489-node auto schedule (105 kicks), the BC20 reference integration recipe, before the `seam_ratio` floor. Figures by `pyDFCSR_2D/test/chicane_wake_plots.py`, in `pyDFCSR_2D/test/benchmark_results/chicane_2d_vs_1d/closed/` (not in git):

- [wake_maps_long_2dR.png](../pyDFCSR_2D/test/benchmark_results/chicane_2d_vs_1d/closed/wake_maps_long_2dR.png) — W(ξ, z) at 17 positions (each panel normalised to its own max, range in MeV/m)
- [wake_maps_trans_2dR.png](../pyDFCSR_2D/test/benchmark_results/chicane_2d_vs_1d/closed/wake_maps_trans_2dR.png) — F(ξ, z) at the same positions
- [wake_zproj_2dR.png](../pyDFCSR_2D/test/benchmark_results/chicane_2d_vs_1d/closed/wake_zproj_2dR.png) — W projected on z (charge-weighted over ξ), 2D with and without the transverse force, and 1D at the same kick
- [wake_xproj_2dR.png](../pyDFCSR_2D/test/benchmark_results/chicane_2d_vs_1d/closed/wake_xproj_2dR.png) — W(ξ) and F(ξ) projected on the transverse axis (charge-weighted over z)
- [wakes_along_s_2dR.png](../pyDFCSR_2D/test/benchmark_results/chicane_2d_vs_1d/closed/wakes_along_s_2dR.png) — along the lattice: max |W|, the charge-weighted mean −⟨W⟩ (= energy loss rate) for 2D / 2D no-T / 1D, the normalised W(z), 1D W(z), F(z) and F(ξ) at every kick
- [prehistory_B1_2.png](../pyDFCSR_2D/test/benchmark_results/chicane_2d_vs_1d/closed/prehistory_B1_2.png) — the first bend with 0.1 m and with 2 m of straight line before it (§7.4)

### 7.1 Exit and per-element tables

| at the exit, vs CSR off | 2D full (2dR) | 2D, no transverse force (2dR_noT) | 1D (1d_full) |
|---|---|---|---|
| mean energy change [keV] | −2672 | −2682 | −1746 |
| projected σ_E change [keV] | −322 | −368 | −303 |
| core slice σ_E change [keV] | +28.8 | +157.8 | +10.4 |
| emittance growth, η-corrected [µm] | 0.490 | 0.952 | 0.549 |
| peak current [kA] (CSR off 5.99) | 5.79 | 5.72 | 5.82 |

| energy change per element [keV] | B1 | D1 | B2 | D2 | B3 | D3 | B4 | Df |
|---|---|---|---|---|---|---|---|---|
| 2D full | −0.9 | −49 | −108 | −41 | −624 | −1468 | −231 | −150 |
| 2D, no transverse force | −0.9 | −49 | −109 | −42 | −626 | −1459 | −242 | −154 |
| 1D | −3.0 | −66 | −8.5 | −70 | −139 | −933 | −366 | −161 |

### 7.2 The projected wake against 1D, kick by kick

Charge-weighted means and peaks of the projected wakes at the picked kicks (`wake_table_2dR.txt`), MeV/m:

| position | s [m] | ⟨W⟩ 2D | ⟨W⟩ 2D no-T | ⟨W⟩ 1D | 2D/1D | max\|W\| 2D | max\|W\| 1D | max\|F\| 2D | ⟨F⟩ 2D |
|---|---|---|---|---|---|---|---|---|---|
| B1 mid | 0.34 | −0.0012 | −0.0012 | −0.0006 | 2.0 | 0.012 | 0.022 | 0.016 | +0.010 |
| B1 exit | 0.60 | −0.0039 | −0.0039 | −0.0212 | 0.18 | 0.042 | 0.086 | 0.009 | +0.006 |
| D1 +0.3 m | 0.90 | −0.029 | −0.029 | −0.039 | 0.74 | 0.085 | 0.064 | 0.0007 | +0.0004 |
| D1 mid | 3.02 | −0.0062 | −0.0062 | −0.0094 | 0.66 | 0.014 | 0.013 | 0.0000 | 0 |
| B2 mid | 5.85 | −0.165 | −0.167 | −0.0015 | 109 | 0.27 | 0.038 | 0.026 | −0.016 |
| B2 exit | 6.11 | −0.174 | −0.177 | −0.101 | 1.7 | 0.34 | 0.24 | 0.013 | −0.009 |
| D2 mid | 6.69 | −0.026 | −0.027 | −0.056 | 0.46 | 0.067 | 0.081 | 0.0006 | −0.0002 |
| B3 mid | 7.35 | −0.69 | −0.70 | −0.022 | 32 | 1.14 | 0.23 | 0.046 | −0.029 |
| B3 exit | 7.61 | −1.91 | −2.02 | −1.36 | 1.4 | 3.12 | 2.27 | 0.046 | −0.031 |
| D3 +0.3 m | 7.94 | −0.44 | −0.44 | −0.56 | 0.79 | 0.80 | 0.82 | 0.003 | −0.001 |
| D3 30% | 9.06 | −0.196 | −0.195 | −0.171 | 1.15 | 0.32 | 0.24 | 0.0002 | 0 |
| D3 60% | 10.56 | −0.158 | −0.158 | −0.089 | 1.8 | 0.25 | 0.13 | 0.0000 | 0 |
| D3 end | 12.30 | −0.233 | −0.227 | −0.057 | 4.1 | 0.32 | 0.080 | 0.0000 | 0 |
| B4 1/3 | 12.81 | +0.266 | +0.261 | −0.218 | −1.2 | 2.09 | 1.73 | 0.170 | +0.119 |
| B4 2/3 | 12.97 | −0.92 | −1.00 | −1.39 | 0.66 | 1.87 | 2.19 | 0.150 | +0.102 |
| B4 exit | 13.11 | −0.96 | −1.00 | −1.22 | 0.79 | 1.84 | 2.01 | 0.037 | +0.024 |
| Df end | 13.31 | −0.54 | −0.56 | −0.68 | 0.80 | 0.94 | 1.00 | 0.003 | +0.001 |

What the figures show:

1. **Shape.** Wherever the beam is long and untilted (B1, D1, D2) the 2D projected wake has the 1D shape (the deceleration–acceleration pair of a Gaussian) at a different amplitude. In B2–B3 and D3 the 2D wake is a single deep trough at the compressed core with a small positive head; the 1D wake has the same trough but, inside B2 and B3, is 30–100× smaller, because the 1D entrance transient builds up over L_f = 0.8 m, longer than the 0.5 m bend, while the 2D wake is already large at the entrance. The 2D wake inside B2 and B3 (and the 2–4× excess over 1D late in D3) is the open item of progress_facet_bc20.md §10.0: it is converged within the 2D formulation and has never had an independent (Liénard–Wiechert) check.
2. **Transverse structure.** W(ξ) in a bend is a linear slope across the beam (the W₁ term linear in x, the ξ-dependence of the retarded z′), and F(ξ) is a bell centred on the charge, with the sign of the bend (+ in B1 and B4, − in B2 and B3) and essentially zero in the drifts (10⁻⁵ MeV/m against 10⁻² in the bends). The transverse force in this chicane is therefore the in-bend space-charge force with the curvature sign, not a radiative transient.
3. **The mean transverse force is the Talman force.** At B1 mid the force on the beam centre is 0.0115 MeV/m. The plain centripetal Derbenev–Shiltsev value 2κQλ_peak/(4πε₀) with λ_peak = 1/(√(2π)σ_z) = 1995 m⁻¹, κ = 1/10.36 m is 0.0035 MeV/m, a factor 3.3 lower; at B4 1/3 (6 kA, λ_peak ≈ 2·10⁴ m⁻¹) the code gives 0.17 MeV/m against 0.035, a factor 4.9. Both ratios are of the size of the logarithm ln(σ_z^{2/3}/(κ^{1/3}σ_⊥)) ≈ 4.8 (σ_z 200 µm, σ_⊥ 64 µm) that Stupakov's Eq. 42 contains, i.e. the code applies the log-enhanced Coulomb (Talman) force, scheme (i) of §4, and the cancellation of §5 is what keeps that large force from showing up in the emittance. A quantitative comparison with the Eq. 42 prefactor is still to be done.
4. **With and without the transverse force** the projected W is the same to the line width everywhere except inside B4 (−0.92 vs −1.00 MeV/m at 2/3), where the transverse kicks accumulated in B3–D3 have changed the beam's x–z correlation; the energy loss differs by 0.4% in total. The force changes the emittance (0.49 vs 0.95 µm) through x′, not through W.
5. **Where 2D equals 1D.** In the drifts 0.3 m past each bend exit the two agree to 20–25% in ⟨W⟩; at the end of the lattice (B4 exit, Df) 2D is 20% below 1D in both ⟨W⟩ and max |W|. At the exit of the compressing bend B3, 2D is 1.4× 1D.

### 7.3 The first bend: a lattice-start artefact, and the reach

In B1 the 2D wake is half the 1D one (max |W| 0.042 vs 0.086 MeV/m at the exit; ⟨W⟩ −0.0039 vs −0.0212). Two things truncate the 2D integral there and neither is physics:

- **The lattice start.** The benchmark has 0.1 m of drift before B1, and the 2D integral has no sources before s′ = 0 (the beam is first deposited there), whereas the 1D entrance transient assumes the beam arrived along an infinite straight line. A source at distance u before the bend entrance, seen from arc angle φ, is retarded by ≈ Rφ³/6 − R²φ⁴/(8u): the whole straight-line history contributes at nearly the same retarded z and its contribution decays only as 1/u.
- **The reach.** Even with the history present, the far region reaches back only n_formation_length × L_f from s3 (L_f is evaluated with 5σ_z in the code, 1.37 m here), so from the B1 exit it covers 0.9 m of the straight section, and the straight section's contribution decays only as 1/u (§7.5).

`chicane_prehistory.py` reruns the first bend with D0 = 2 m (the lattice shifted by 1.9 m, everything else identical) and n_formation_length 1, 2 and 4:

| s in B1 [m] | ⟨W⟩ 2dR (D0 0.1, n_fl 1) | D0 2 m, n_fl 1 | D0 2 m, n_fl 2 | D0 2 m, n_fl 4 | ⟨W⟩ 1D | max\|W\| 2dR | D0 2 m, n_fl 2 | 1D |
|---|---|---|---|---|---|---|---|---|
| 0.34 (mid) | −0.0012 | −0.0016 | −0.0016 | −0.0016 | −0.0006 | 0.012 | 0.022 | 0.022 |
| 0.55 | −0.0019 | −0.0066 | −0.0083 | −0.0083 | −0.0136 | 0.034 | 0.066 | 0.074 |
| 0.60 (exit) | −0.0039 | −0.0116 | −0.0146 | −0.0146 | −0.0212 | 0.042 | 0.079 | 0.086 |
| 0.66 | −0.0064 | −0.0150 | −0.0197 | −0.0197 | −0.0283 | 0.043 | 0.080 | 0.086 |
| 0.78 | −0.0097 | −0.0196 | −0.0271 | −0.0271 | −0.0375 | 0.042 | 0.076 | 0.076 |
| 0.90 | −0.029 | −0.042 | −0.051 | −0.051 | −0.039 | 0.085 | 0.098 | 0.064 |

With the straight-line history present and a reach of two formation lengths (converged: 2 and 4 agree to the printed digits) the 2D peak wake matches 1D at B1 mid and is within 8% at the exit; the mean energy loss rate is 30% below 1D at the exit and 30% above it 0.3 m after. So: (i) a 2D benchmark needs the straight section before the first bend (the 0.1 m D0 of this lattice, and BC20 lattices that start at the first dipole — Indranil's starts at B1LE — understate the first bend); (ii) `n_formation_length` = 1 is not enough at the exit of a short bend with a large angle, because the straight-section contribution converges only as 1/u. "2 and 4 agree" above is an artefact of the 2 m of drift available: both were clipped there. §7.5 measures the convergence properly and gives the fix. This is the first-bend component of Indranil's observation (§11k of progress_facet_bc20.md found the reach converged at 1 only at B2RE/B1RE, deep in the lattice, where φ is small). The per-element B1 numbers of §7.1 (−0.9 vs −3.0 keV) and the D1 entrance are therefore not a 2D-vs-1D comparison; the rest of the lattice is unaffected (the reach test at constant far cell in §11k).

### 7.4 The synthetic pre-history (`CSR_integration.prehistory`, 2026-11-24)

Implemented after §7.3: with `prehistory: L` (metres) the far and near regions may reach down to s′ = −L along the straight continuation of the reference orbit (already provided by `_lattice_vectors`), the curvature is set to zero there, and the history lookup returns the first snapshot for t < 0 (the interpolant's existing clamp), i.e. the beam is assumed to have travelled the last L metres unchanged. No tracking, no deposition and no kicks are spent on the straight section. Same first-bend test as §7.3, benchmark lattice as is (0.1 m before B1), `prehistory: 2.0`, `n_formation_length: 2` (run `2dR_pre2_nfl2`, 3.1 min against 6.0 for the real 2 m drift):

| s in B1 [m] | ⟨W⟩ no pre-history (2dR) | real 2 m drift, n_fl 2 | synthetic 2 m, n_fl 2 | ⟨W⟩ 1D | max\|W\| 2dR | real drift | synthetic | 1D |
|---|---|---|---|---|---|---|---|---|
| 0.34 (mid) | −0.0012 | −0.0016 | −0.0014 | −0.0006 | 0.012 | 0.022 | 0.021 | 0.022 |
| 0.55 | −0.0019 | −0.0083 | −0.0079 | −0.0136 | 0.034 | 0.066 | 0.065 | 0.074 |
| 0.60 (exit) | −0.0039 | −0.0146 | −0.0133 | −0.0212 | 0.042 | 0.079 | 0.078 | 0.086 |
| 0.66 | −0.0064 | −0.0197 | −0.0191 | −0.0283 | 0.043 | 0.080 | 0.080 | 0.086 |
| 0.78 | −0.0097 | −0.0271 | −0.0270 | −0.0375 | 0.042 | 0.076 | 0.077 | 0.076 |
| 0.90 | −0.029 | −0.051 | −0.047 | −0.039 | 0.085 | 0.098 | 0.086 | 0.064 |

The synthetic history reproduces the real-drift run to 2% in the peak wake and 4–9% in the mean (the real-drift beam is not the benchmark beam at B1: it has drifted 1.9 m further, so its transverse size differs; the synthetic run keeps the benchmark beam exactly). Figure: [prehistory_B1_pre2_nfl2.png](../pyDFCSR_2D/test/benchmark_results/chicane_2d_vs_1d/closed/prehistory_B1_pre2_nfl2.png). Default 0 (bit-identical to before); for a lattice starting at or near its first dipole use `prehistory` ≥ the reach together with `n_formation_length: 2`.


### 7.5 How far back the straight section matters, and `reach_straight` (2026-11-24)

On the pickled B1-exit state of the synthetic-history run (a = 0.5 m into the bend, Rφ = 0.5 m, σ_z 201 µm, flat reach 1.41 m = 0.9 m of straight; 1D at the same kick: ⟨W⟩ −0.0212, max |W| 0.0861 MeV/m) the straight section was extended at the fixed far cell (2 mm):

| straight section covered [m] | 0.9 | 1.0 | 1.5 | 2.5 | 4.0 | 6.0 | 10.0 |
|---|---|---|---|---|---|---|---|
| ⟨W⟩ [MeV/m] | −0.0103 | −0.0109 | −0.0123 | −0.0139 | −0.0150 | −0.0157 | −0.0163 |
| max \|W\| [MeV/m] | 0.0689 | 0.0705 | 0.0747 | 0.0788 | 0.0815 | 0.0831 | 0.0844 |
| far nodes | 400 | 438 | 584 | 876 | 1313 | 1897 | 3065 |

The straight-section contribution converges as U/(U + Rφ): 20 Rφ for 95%, and the far node count grows linearly. Its sources are far from the observer and retarded by nearly the same Rφ³/6, so the column integral is a smooth ~1/u² in u = s − s′, which geometric grading resolves with a few dozen nodes. `CSR_integration.reach_straight` (default 40, in units of Rφ) integrates [entrance − reach_straight·Rφ, entrance) as its own region, graded in u with ratio 1 + near_grade, below the regular layout (which is unchanged), inside the bend and in the drift after it, while the straight sources are retarded within `reach_slip_max` (5) σ_z of the bunch — inside the bend Rφ³/6, past the exit plus xφ²/2, so it switches itself off within ~Rφ past the exit and never fires deep in a long bend. With a 2 σ_z cap it switched off 0.18 m past the B1 exit while 1D was still growing (⟨W⟩ −0.020 against 1D −0.038 at 0.78 m; with 5 σ_z −0.033). The region never goes below the previous bend's exit: `prev_bend_reach` already integrates everything from the previous bend's entrance, so in a chicane or in BC20 the rule acts only on the first bend and the drift after it (checked on the BC20 B2RE state: bit-identical). Same state:

| reach_straight [Rφ] | 0 | 5 | 10 | 20 | 40 | 100 (clipped at 20 m) |
|---|---|---|---|---|---|---|
| straight covered [m] | 0.9 | 2.5 | 5 | 10 | 20 | 20 |
| extension nodes | 0 | 17 | 29 | 43 | 56 | 75 |
| ⟨W⟩ [MeV/m] | −0.0103 | −0.0138 | −0.0154 | −0.0163 | −0.01675 | −0.01675 |
| max \|W\| [MeV/m] | 0.0689 | 0.0787 | 0.0824 | 0.0844 | 0.0855 | 0.0855 |
| time per mesh [s] | 16 | 17 | 17 | 17 | 17 | 17 |

The graded region reproduces the fixed-cell result (−0.0163 at 10 m with 43 nodes against 3065) at no measurable cost, and converges at 20 m of straight. The converged first-bend exit wake is then: peak 0.0855 against 0.0861 in 1D (0.7% apart), mean energy loss rate −0.0168 against −0.0212 (2D 21% below 1D). The peak agrees with 1D; the mean does not, and that residual is a 2D-versus-1D difference in the entrance transient, not a boundary artefact. The full first-bend run with the defaults and `prehistory: 20` (`2dR_pre20_rs40_slip5`, 3.3 min) gives, against 1D:

| s [m] | 0.34 (B1 mid) | 0.55 | 0.60 (exit) | 0.66 | 0.78 | 0.90 |
|---|---|---|---|---|---|---|
| ⟨W⟩ 2D / 1D [MeV/m] | −0.0015 / −0.0006 | −0.0100 / −0.0136 | −0.0167 / −0.0212 | −0.0239 / −0.0283 | −0.0333 / −0.0375 | −0.0530 / −0.0386 |
| max \|W\| 2D / 1D | 0.0218 / 0.0216 | 0.0725 / 0.0735 | 0.0855 / 0.0861 | 0.0886 / 0.0855 | 0.0859 / 0.0764 | 0.0869 / 0.0640 |

against −0.0039 / 0.042 at the exit in the original 2dR run. Inside the bend the 2D peak wake is the 1D one to 1%; the mean is 20% below 1D at the exit, and 0.3 m after the exit 2D is 37% above 1D in the mean and in the peak — the drift after a bend is where the two formulations differ (§7.2, item 5). With `prehistory` ≥ reach_straight·Rφ the synthetic history supplies the straight section when the lattice has none. Default on (bit-identical where it does not fire, which includes the golden dipole: its 0.3 m bend has an exit slip of 2.5 σ_z); `reach_straight: 0` turns it off.
