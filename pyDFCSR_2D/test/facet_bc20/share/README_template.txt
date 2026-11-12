FACET-II W chicane (BC20), BEGBC20 -> ENDBC20, 49.0837 m
=========================================================

Source: Sergei Kladov's tuned FACET-II start-to-end simulation of DAQ E320_17418
(Bmad/Tao, FACET2-S2E commit 3bc1b17, reference run 2026-10-19). Translated to the pyDFCSR
(DFCSR) input format and checked against Tao particle by particle with all collective
effects off (agreement 1e-4 of sigma_x and 1e-5 of sigma_z at ENDBC20).

Files
-----
beam_BEGBC20.txt        20 000 macroparticles at the start of the lattice (6 columns)
beam_ENDBC20_nocsr.txt  the same particles at the end of the lattice, tracked with every
                        collective effect OFF (Tao) -- same particle order, for a
                        particle-by-particle check of the transport
lattice_dfcsr.yaml      the lattice in pyDFCSR format
lattice_table.txt       the same lattice as a flat table (one element per row, with s)
optics_nocsr.txt        beam moments, bunch length, beam sizes, Twiss and dispersion at
                        BEGBC20 and at the exit of every element, collective effects OFF

Particle coordinates (both beam files)
--------------------------------------
Bmad phase-space coordinates, relative to the reference momentum
    P0C = 10116.23403001603 MeV   (electrons; m c^2 = 0.51099895069 MeV)
  col 1  x   [m]
  col 2  px  = P_x / P0          (canonical momentum, NOT x' = dx/ds; x' = px/(1+pz) in a drift)
  col 3  y   [m]
  col 4  py  = P_y / P0
  col 5  z   [m]  = -beta c (t - t_ref) at fixed s. z > 0 is the bunch HEAD (ahead of the
                    reference particle). All particles are given at the same s.
  col 6  pz  = (P - P0) / P0
Total charge 1.6 nC, 20 000 equal-weight macroparticles (8.0e-14 C each).
The beam's mean energy (10001.07 MeV) is 1.14% BELOW P0C: <pz> = -1.138e-2. That is not an
error: the reference momentum is Bmad's design value after Sergei's energy feedback with wakes
on, and all magnet strengths are normalised to P0C. Please keep P0C as the reference; using the
mean beam energy instead makes every magnet ~1% wrong.
At BEGBC20: sigma_z = 132.9 um, sigma_pz = 1.04%, eps_nx / eps_ny = 4.57 / 4.33 um
(dispersion-corrected, normalised with P0C, as in optics_nocsr.txt).

Lattice conventions (lattice_dfcsr.yaml, lattice_table.txt)
------------------------------------------------------------
Everything follows Bmad's conventions and the strengths are Bmad's numbers, unchanged.
Element order is the order in the file; s runs from 0 at BEGBC20.
  dipole     horizontal sector bend (Bmad SBEND). L = arc length [m], angle [rad] of the
             reference orbit (= G*L). G = 1/rho of the reference orbit [1/m]; DG [1/m] is the
             field error, i.e. the actual field is B = (G + DG) * P0/e (Sergei pinned the bends
             to the beam energy this way). E1, E2 = entrance / exit pole-face rotations [rad],
             relative to the sector faces. FINT, HGAP (FINTX, HGAPX at the exit) = fringe-field
             integral and half gap [m], as in Bmad. Bmad used FRINGE_TYPE = Full; pyDFCSR uses a
             hard-edge fringe with FINT/HGAP (the difference is ~1e-12 m in x, 2e-9 m in y).
  vbend      the vertical wiggler WIGE1-3 (+-2.5 mrad): a sector bend rotated by
             TILT = pi/2 about the beam axis, otherwise as 'dipole'. It deflects in y.
  quad       K1 = (1/(B rho)) dB_y/dx [1/m^2], B rho = P0/e; K1 > 0 focuses horizontally.
  sextupole  K2 = (1/(B rho)) d^2B_y/dx^2 [1/m^3]. X_OFFSET [m] is a horizontal offset of the
             magnet (Bmad: the particle sees x - X_OFFSET); S1/S2 carry offsets of 0.6-2 mm,
             which is how the W-chicane dispersion was tuned.
  drift      L only.
  NUM_STEPS  integration steps for the whole element (quads and sextupoles). The sextupoles
             use 5 mm steps; this matters: with one step per sextupole (Bmad's default in the
             original lattice) the beam at ENDBC20 is off by up to 34 um in x.
  nsep, step_size: pyDFCSR-only settings, ignore.
Many Bmad elements were merged: drifts, markers, monitors, collimators, zero-strength correctors
and zero patches between magnets are one drift here; split bends/sextupoles are recombined.
All correctors in BC20 are at zero.

Reference data (optics_nocsr.txt)
---------------------------------
Bmad/Tao tracking with CSR, ISR, space charge and wakes all off. One row per element exit
(element names as in lattice_table.txt). Means and rms of the 20 000 particles, and Twiss /
dispersion / emittance from the beam covariance: eta = <x pz>/<pz^2>, etap = <px pz>/<pz^2>,
and beta, alpha, emit from the (x, px) covariance with that linear correlation removed
(geometric emit; emit_n = emit * P0C / mc^2). They are beam-derived quantities, not
design-lattice functions, and px is canonical. At ENDBC20: sigma_x = 194.3 um,
sigma_z = 51.43 um, sigma_y = 48.7 um, eps_nx / eps_ny = 14.96 / 7.30 um.
