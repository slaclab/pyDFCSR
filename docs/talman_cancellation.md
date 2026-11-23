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
