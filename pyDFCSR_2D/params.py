from .tools import full_path
#class Interpolation_params:
#
#    def __init__(self, input_dic = {}):
#        self.configure_params(**input_dic)

#    def configure_params(self, xbins=500, zbins=500, xlim=10, zlim=10, re_interpolate_threshold=2):
#        self.xbins = xbins
#        self.zbins = zbins
#        self.xlim = xlim
#        self.zlim = zlim
#        self.re_interpolate_threshold = re_interpolate_threshold


class Integration_params:
    # Todo: Maybe not necessary. Can just be a dictionary with an additional function to parse default values.
    def __init__(self, input_dic = {}):
        self.configure_params(**input_dic)

    def configure_params(self, n_formation_length = 1, zbins = 200, xbins = 200,
                         xi_bands = True, xi_band_margin = 2.0, xi_band_margin_degenerate = 3.0, near_patch = 5.0,
                         near_patch_nr = 100, near_patch_nphi = 180,
                         near_cell = 0.25, far_zbins = 200, near_grade = 0.05,
                         branch_sin_min = 0.05, near_floor = 20.0,
                         causal_clip = True, far_window = 1.0, far_cell = 10.0,
                         drift_cutoff = 0.0, tangent_bands = True,
                         drift_reach = True, drift_snap_thin = 0.01,
                         far_fine_max = 500, band_jacobian_min = 0.2,
                         prev_bend_reach = True, prev_bend_grade = 0.02,
                         ibp = False, ibp_j0 = 0.25, ibp_check = 0, ibp_check_xbins = 400,
                         ibp_check_tol = 1e-3, ibp_check_zone_max = 1.0,
                         transverse_terms = 'all', near_adapt = False, near_adapt_eps = 0.02,
                         near_adapt_scan_dz = 0.25, near_adapt_max_scan = 4000,
                         near_adapt_max_nodes = 2000, near_adapt_max_ratio = 1.5,
                         near_adapt_x = False, near_adapt_x_rounds = 4, near_adapt_x_max = 4,
                         near_adapt_scan_iter = 4, far_adapt_x = True, integrand_kernel = 'numba',
                         threads = 1, threads_min_points = 20000, far_adapt = False, far_adapt_base = 50,
                         seam_ratio = 2.0, prehistory = 0.0, reach_straight = 40.0, reach_slip_max = 5.0,
                         drift_causal_reach = True, drift_causal_grade = 0.005, drift_causal_scan = 128,
                         x_adapt = 'error', x_adapt_tol = 1e-3, x_adapt_rounds = 4):
        # Default 1 since 2026-11-06 (was 4): what every validated chicane run used, now that
        # drift_reach and prev_bend_reach cover the bends explicitly. NOT yet checked against
        # 4 with the current defaults -- a 1 vs 2 vs 4 comparison was started and paused
        # (progress.md, 2026-11-06).
        self.n_formation_length = n_formation_length
        self.zbins = zbins
        self.xbins = xbins
        # Longitudinal nodes per s' region, instead of `zbins` for every region.
        #
        # near_cell is the target cell of the NEAR region (s3, s4) in units of
        # sigma_xi; far_zbins is a flat count for the two far regions. Setting
        # near_cell = 0 restores the old behaviour of zbins everywhere.
        #
        # Why: the three regions differ in length by ~100x, and the near one carries
        # the 1/|r-r'| singularity and the steep integrand. Its required cell is set
        # by the transverse support width sigma_xi -- about 1 sigma_xi for 1%
        # accuracy, measured to collapse across tilt amplification 66x and 167x --
        # while its LENGTH grows with tilt as d ~ N sigma_x / |tan 2 alpha|. So the
        # near node count must scale as sigma_x/sigma_xi. Expressed this way a single
        # value works across regimes; expressed as a flat zbins it needs retuning
        # (zbins = 400 gives 0.5% error at shear 20 but 4.3% at shear 50).
        # The far regions need very few nodes: the far edge s1 is invariant to
        # machine precision, because 1/|r-r'| has already damped the integrand there.
        self.near_cell = near_cell
        self.far_zbins = far_zbins
        # Geometric growth rate of the near-region longitudinal cell with distance
        # from the observation point: du = max(near_cell*sigma_xi, near_grade*u),
        # u = |s - s'|. 0 gives the uniform grid.
        #
        # Why: once u exceeds the transverse offsets, |r - r'| ~ u and the per-column
        # contribution falls as 1/u, so equal contributions come from equal
        # LOGARITHMIC intervals. A uniform grid is then under-resolved at small u
        # (where accuracy is set) and over-resolved at large u (where the cost is),
        # forcing node count ~ u_max/sigma_xi. Grading makes it ~ln(u_max/u*), which
        # grows only logarithmically with tilt. It is the longitudinal counterpart of
        # the polar patch: there r dr absorbs the 1/r, here d(ln u) does.
        #
        # Only active when near_cell != 0, so it cannot affect the flat-zbins path.
        self.near_grade = near_grade
        # Threshold on |sin 2a| below which the two Eq 4.22 localization branches are
        # treated as ONE band, and the near-region reach (in sigma_z) used there.
        #
        # thesis 4.4.2: x2 = x - (s-s')tan(2a) degenerates to x2 = x at a = 0 AND at
        # a = +-pi/2, i.e. the chirp branch merges into the narrow one in both limits.
        # sin 2a = 2 tau/(1 + tau^2) and cos 2a = (1 - tau^2)/(1 + tau^2) are bounded
        # and pole-free, and |sin 2a| is small at BOTH limits (0.0998 at tau = 20,
        # 0.0124 at tau = 161, 0.0998 at tau = 0.05), so one test catches both.
        # tan 2a must NOT be used: it has a pole at |tau| = 1.
        #
        # Threshold comes from geometry, not tuning: the branches are unresolvable once
        # their separation over the region falls below one band width,
        #     |tan 2a| * L < 2 * xi_band_margin * xlim * sigma_xi
        # which gives ~0.045 for L = 5.6 mm, sigma_xi = 12.5 um and ~0.019 for
        # L = 12.9 mm. 0.05 is the conservative single-pass value.
        # It also saturates the near-region extent, since in the degenerate case
        # d = N sigma_x/|tan 2a| is not merely large but meaningless (there is only one
        # band). Without that saturation the near region swung
        # 100 mm -> 808 mm -> 50.9 um across a longitudinal waist.
        self.branch_sin_min = branch_sin_min
        # Minimum UPSTREAM reach of the near region, in sigma_z.
        #
        # d = (10 sigma_x + x - xmean)|cos 2a|/|sin 2a| sizes the near region from where
        # the CHIRP band exits the beam, and that goes to zero at |tau| = 1 (alpha = 45
        # deg) where cos 2a -> 0. The chirp band really does exit at once there, but the
        # near region also has to hold the NARROW band and the neighbourhood of the
        # 1/|r-r'| pole, neither of which cares about the chirp geometry.
        #
        # Without the floor, at shear 50 / 0.80 m into a 1 rad dipole (tau = -0.999,
        # cos 2a = 8.8e-4) the upstream reach collapsed to 15 um against sigma_z =
        # 1744 um: 1-2 of 88 graded nodes on the causal side, the pole 15 um from the
        # region edge, and region 2's uniform 1.74 mm cells left to resolve it. The wake
        # went visibly blocky.
        #
        # 20 is not tuned -- it is the s3 = s - 20 sigma_z that the deleted
        # |tan theta| <= 1 branch used to provide.
        self.near_floor = near_floor
        # --- the FAR region: causal clipping and the fine window -------------------
        #
        # causal_clip applies ONLY in a drift after a bend, i.e. to the exit transient.
        # Inside the bend the causal edge and the n_formation_length*L_f reach agree --
        # measured, the edge sits at u ~ 1.0 L_f (0.994 at shear 0, 0.783 at shear 20) --
        # so clipping there would buy nothing and cost a band solve per mesh point. Past
        # the exit the edge moves to u ~ 3 L_f and the two diverge.
        #
        # causal_clip: start the far region at the CAUSAL EDGE -- the lowest s' whose
        # retarded band still has nonzero width -- instead of at
        # s1 = s3 - n_formation_length*L_f. Below that edge the retarded source has
        # slipped outside the bunch and every column is identically zero, so the nodes
        # there compute nothing. It also removes a real defect: with
        # s2 = s3 - 200 sigma_z still negative while s1 clamps at 0, region 1 came out
        # INVERTED (s1 > s2) near the lattice start, and ~400 of 597 far nodes landed
        # at s' < 0 where interpolate1D silently returns 0.
        #
        # far_window / far_cell: the far integrand is NOT spread over the region. At a
        # bend-exit transient 10-99% of it sits in a band ~0.3 L_f wide just above the
        # causal edge, at u/L_f ~ 2.7-3.1 -- i.e. FAR from s' = s, which is exactly
        # where grading by distance from s puts its coarsest cells. Measured 0.5 m past
        # a 1 rad bend, plain log grading left a 30 mm cell spanning the entire
        # contribution and was 400x worse than the old 3-region split. So the far
        # region is uniform at far_cell*sigma_z over far_window*L_f above the edge, and
        # graded by near_grade above that.
        #
        # far_window is in L_f, not sigma_z, because the band sits at fixed u/L_f
        # independent of position. Calibrated on the single dipole (rel L2 of the far
        # region against a 40000-node reference, 3-region split = baseline):
        #
        #     window    s=0.50    s=1.20    s=1.30    s=1.60    nodes
        #     0.5 L_f  9.8e-04   2.3e-03   4.9e-03   1.9e-02   246-272
        #     1.0 L_f  1.5e-04   2.7e-06   5.5e-06   1.8e-05   347-416
        #     2.0 L_f  1.7e-04   8.7e-05   1.8e-04   1.8e-05   336-701
        #     3 region 9.2e-05   2.9e-04   3.8e-04   5.0e-04   400
        #
        # 1.0 is best or tied everywhere at no more nodes than the 3-region split it
        # replaces. It is a genuine optimum, not a saturation: 0.5 under-covers the
        # band, and 2.0 spends so many nodes on the flat part that it crowds out the
        # grading near s' = s and gets WORSE while costing up to 701 nodes.
        #
        # far_cell is deliberately coarse. Refining it 20 -> 2 sigma_z at fixed window
        # changed the error by 0.1% (2.108e-01 -> 2.111e-01) while tripling the nodes:
        # the integrand is smooth across the band, so the accuracy is set by WHERE the
        # fine nodes are, not how fine they are.
        self.causal_clip = causal_clip
        self.far_window = far_window
        # Upper bound on the number of fine-window nodes (0 = no bound). The cell is in
        # sigma_z but the band it resolves scales with L_f, so a short bunch over-resolves
        # it: 3300 nodes per kick in the chicane's D3 (sigma_z 19 um). Default 500 since
        # 2026-11-04, calibrated against a 3-6x finer reference (rel L2):
        #     case                     default (far_cell 10)   cap 500
        #     D1 0.49 m past B1        1.07e-3 (539 nodes)     1.25e-3
        #     D1 2.4 m / D2 6.34 m     7.6e-6 / 2.5e-6         1.5e-5 / 8.8e-6
        #     D3, four positions       <= 3.0e-4 (~3300)       <= 3.0e-4 (~550), 4.5x faster
        #     1 rad dipole exit drift  <= 4e-5 (387-419)       unchanged (cap not reached)
        # A uniformly coarser far_cell does NOT work: far_cell 100 gave 5e-2 in D1.
        self.far_fine_max = far_fine_max
        # Scale each FAR x' band by 1/|du/dx'| at its centre, floored at this value (0 =
        # off, the old fixed +-xlim*margin*sigma_xi). See CSR2D._retarded_xi_bands.
        # Default 0.2 since 2026-11-05: the D3 far integral was 8-18% low without it, and
        # matches a +-40 sigma_xi reference to <= 0.3% with it; B1-B4, D1, D2 change
        # <= 0.1%; no run-time cost (D3 maps 42-47 -> 32-38 ms/pt, same node count).
        self.band_jacobian_min = band_jacobian_min
        # Inside a bend, also integrate the PREVIOUS bend's exit tail (needs drift_reach):
        # an extra far region from the previous bend's entrance up to this bend's own
        # reach, laid out like an exit transient. Off by default.
        # Default True since 2026-11-05 (closed chicane: B4 energy -172 -> -244 keV, emittance
        # growth 0.455 -> 0.497 um vs 1D 0.549, +6% run time).
        self.prev_bend_reach = prev_bend_reach
        # log-grading ratio of that region's s' nodes (CSR2D._prev_region_nodes)
        self.prev_bend_grade = prev_bend_grade
        # Longitudinal integrand integrated by parts along each s' column (bspline_comoving
        # only; progress_cancellation.md §2, §7). The main term G0 * d rho/dx has +- lobes on
        # the two sides of the beam that cancel by C ~ 188 sigma_z^(2/3)/sigma_xi (um, R = 1 m),
        # 60-1000 in practice, so the shipped form needs >= 128 x' nodes per band and amplifies
        # any rho / d rho inconsistency by ~C. Along a column the retarded point moves, and
        #     d rho_ret/dx' = J rho_x + D,   J = 1 - A R_x,   D = B R_x
        # (interp3D.interpolate3D_comoving_ibp), so
        #     int G0 rho_x dx' = [Q rho] - int Q' rho dx' - int Q D dx',   Q = G0 / J.
        # Same integral, no cancellation; the boundary term is ~0 because the bands are
        # +-10 sigma_xi wide. The transverse kick and the polar patch are unchanged.
        self.ibp = bool(ibp)
        # J -> 0 where a column runs along a tilted beam's ridge (tilt |tau| > 1, or curved
        # tails steeper than 1). A column keeps the shipped form whole if |J| <= 2 ibp_j0 on any
        # node with charge (exact: the identity holds per column). A partition inside the column
        # (chi(J) = 1 for |J| <= ibp_j0, 0 beyond 2 ibp_j0) was tried first and fails at high
        # tilt: its ramp is too steep in x' (chicane B3/D3, progress_cancellation.md §8); chi now
        # only guards 1/J on empty nodes.
        self.ibp_j0 = float(ibp_j0)
        # Consistency monitor: per kick, evaluate ibp_check points of the wake mesh's centre row
        # in BOTH forms at ibp_check_xbins x' nodes and warn once if they differ by more than
        # ibp_check_tol (relative L2). They are the same integral, so a difference is a rho /
        # d rho/dx inconsistency that the shipped form amplifies by ~C. 400 nodes, not 200:
        # where the curved tails or the tilt are steeper than 1 the partition zone is active and
        # the by-parts form converges more slowly than the shipped one (dipole with CSR applied,
        # 0.5 m past the exit: 2.0e-3 at 200 nodes, 2.2e-4 at 400, 1.5e-5 at 800; a larger
        # ibp_j0 does not help). ibp_check_log also records the zone's share of the main term's
        # sum|node| in columns kept in the shipped form; a point whose share exceeds
        # ibp_check_zone_max is skipped and counted
        # (default 1: never). Under |tau| >> 1 the check's own floor rises (shear 20: 1.1e-2).
        # 0 = off.
        self.ibp_check = int(ibp_check)
        self.ibp_check_xbins = int(ibp_check_xbins)
        self.ibp_check_tol = float(ibp_check_tol)
        self.ibp_check_zone_max = float(ibp_check_zone_max)
        self.far_cell = far_cell
        # Stop computing wakes once the beam is more than drift_cutoff * R*phi past the
        # bend it just left, in units of that bend's own R*phi. 0 disables the cutoff.
        #
        # OFF by default (0) since 2026-11-04, the author's decision. It was ON at 1.0
        # because the far-drift 2D wake was untrustworthy, not because it was negligible.
        # That wake was two defects -- dropped near columns (tangent_bands) and a reach
        # that lost the bend (drift_reach) -- both fixed and on by default, and with
        # far_fine_max the far drift costs ~+20% run time on the chicane (26.4 vs 22.0
        # min). The 1D calibration of what the cutoff removes on the chicane, as % of
        # no cutoff:  cutoff [R phi]   0.25  0.5   1     2     4     6
        #             energy loss      49    57    67    77    86    92
        #             emittance growth 27    35    49    62    77    87
        # so no fixed cutoff is safe there: past ~1 R*phi the exit wake is the head term
        # ~A lam/(2d), which falls only as 1/d. Set it > 0 only as a deliberate speed-up.
        # The rest of this note is the history from when it was on.
        #
        # What it discards. On the chicane, with each kick weighted by its arc length
        # L_kick (drift kicks span 0.25 m against 0.005 m in a bend, so weighting by kick
        # COUNT understates them ~50x -- an error that was made once here):
        #
        #     region                              integrated mean dE   share
        #     inside the four dipoles                     -0.41911     55.1 %
        #     drift, within 1 R*phi of an exit            -0.30228     39.8 %
        #     drift, beyond 1 R*phi                       -0.03857      5.1 %
        #
        # and 90.5% of that 5.1% is ONE kick, at s = 8.317 (0.711 m past the B3 exit),
        # whose mean dE is 70x its two neighbours and whose wake swings -5.19 -> +1.06
        # across a single z cell. Excluding it, the entire far-drift region carries 0.48%.
        #
        # That 0.48% is what the 2D code computes there, NOT what the physics puts there
        # (progress.md, top TODO, 2026-10-29). The 2D far-drift wake is the broken part:
        # past ~1.9 R*phi in D3 it turns bipolar across x, its x-average goes positive, and
        # it GROWS toward B4 instead of decaying. A 1D run (Emma csr_calc, same particles,
        # same 1 R*phi cutoff rule) puts a large share of the chicane's CSR effect in the
        # region this cutoff removes, almost all of it in D3:
        #
        #     chicane exit, 1D                  full drift   far drift alone   share
        #     mean CSR energy loss                -1721 keV         -638 keV    37 %
        #     emittance growth (eta-corrected)    +0.516 um       +0.297 um     58 %
        #     CSR-added slice sigma_E               45.6 keV        19.9 keV    44 %
        #
        # The bunch is fully compressed (19 um) after B3, and the exit wake decays only as
        # 1/(phi + 2d/R): still -0.05 MeV/m at the B4 entrance, 4.5 m on, acting while the
        # dispersion is being closed. So this cutoff removes a wrong 2D wake AND a large real
        # effect. 2D chicane results with it ON are missing that effect, and with it OFF they
        # include the bad wake, which produced a spurious 8x slice-energy-spread excess over
        # 1D. With the cutoff applied to BOTH codes, the emittance growth agrees within 4%
        # (2D 0.211 vs 1D 0.220 um). Report the cut share whenever this is on.
        #
        # The defect is NOT repaired by the 11u far-region rebuild: measured at the same
        # position, peak 5.1864 -> 5.1412 and roughness 4.601 -> 4.720, i.e. unchanged.
        # Run-wide roughness is likewise unchanged (mean 1.577 -> 1.581, 70 of 105 maps
        # above 1.0 both ways). An earlier claim here that the rebuild fixed it was an
        # artefact of comparing two different kick positions.
        #
        # Enabling it is NOT a small perturbation, and its sign is NOT predictable from the
        # discarded share. Measured end-to-end on the chicane, cutoff 0 -> 1.0: energy loss
        # -1.6475 -> -1.8474 MeV, i.e. 12% LARGER, and emittance growth 1.304x -> 1.207x.
        # Along the lattice the cutoff first reduces the loss as expected (+151 keV by
        # s = 8.4) and then reverses sign through the last bend, because the beam entering
        # B4 differs -- sigma_x 3.6% smaller, norm_emit_x 5.8% larger -- so B4 radiates
        # differently. Suppressing a drift wake changes the beam, and a chicane amplifies
        # that. Set drift_cutoff = 0 to integrate the drifts in full.
        #
        # NOTE on the threshold: abs() is required on R_rec. R_rec = L/angle carries the
        # sign of the bend, so a chicane's negative-angle dipoles gave a NEGATIVE threshold
        # and blocked their entire downstream drift from the first step -- B2 and B3 kept
        # one kick each, sitting exactly at the exit face, so their exit transients could
        # not be plotted. Fixed; the arc length |R*phi| is what the scale means.
        #
        # The scale is R*phi, not a multiple of L_f: the exit decay is geometric. Eq. 10
        # of Stupakov & Emma gives W ~ 1/(phi + 2 d/R), halving at d = R*phi/2, and the
        # retarded slippage saturates at the same point (both cross at d/R = 0.5 phi). An
        # earlier commented-out version of this cutoff used 3*formation_length and carried
        # a Todo saying the formation length was wrong there; it was, which is presumably
        # why it was never enabled. Measured decay on the 1 rad test dipole, on-axis peak
        # relative to the exit face:
        #     d/(R phi)   0.1     0.2     0.4     0.5
        #     amplitude   0.338   0.173   0.073   0.054
        # faster than Eq. 10 predicts (0.338 vs 0.833 at 0.1) because Eq. 10 assumes
        # phi << 1 -- so R*phi is conservative for a strong bend and much less so for a
        # weak one. Skipped kicks are COUNTED and reported once when this is enabled.
        self.drift_cutoff = drift_cutoff
        # What to do with an s' column where the retarded light cone does not cross
        # the beam AXIS (Eq 4.24 discriminant < 0).
        #   False  the branch is treated as absent and the column dropped.
        #   True   the band is placed at the column's closest approach to the axis
        #          (CSR2D._eq424) and kept if it lands inside the ribbon.
        # The axis is a zero-width line; the beam is not. Near the observer the
        # column traces xi_ret(y) = xi + y - tau y^2 / (2 (s - s')), y = x' - x, whose
        # extremum is xi + (s - s')/(2 tau): for tau * xi_obs < 0 there is no root
        # while s - s' < 2 |tau xi_obs|, although the observer sits inside the beam.
        # Measured in the chicane D3 (tau = -6.9, s = 12.30 m): observers at
        # xi = +0.37 and +1.1 sigma_xi lost every near column with s - s' < 140 and
        # 440 um (2|tau xi| = 177, 527 um), the ones at xi < 0 none, and the lost
        # columns carried the self-field (v_x) work, so the wake jumped by up to
        # +1 MeV/m across xi = 0.
        # Default True since 2026-11-04, after validation on the chicane, the single
        # dipole and the FODO W2 test (progress.md). Results recorded before that
        # date were computed with False; set it explicitly to reproduce them.
        self.tangent_bands = tangent_bands
        # Integration reach in a drift after a bend.
        #   False  s' >= s3 - n_formation_length * L_f, with L_f the in-bend value of the
        #          last bend. The bend leaves the reach ~1.7 R*phi past the exit at
        #          n_fl = 1, and the D3 wake then turns positive (progress.md 2026-11-03).
        #   True   the far edge goes back to the entrance of the last bend, the causal-
        #          edge scan trims it, and the history is kept back to the retarded time
        #          of that edge.
        # Default True since 2026-11-04 (chicane D3, progress.md "The D3 reach fix").
        self.drift_reach = drift_reach
        # History thinning in a drift after a bend, to keep drift_reach cheap: a new
        # snapshot is pushed only if the beam frame (sigma_xi, sigma_z, tilt) has moved by
        # more than this fraction since the last one kept (or the step is a kick, or
        # inside a bend). 0 keeps every snapshot. The interpolant is non-uniform in t
        # already (lookup_hybrid), so dropped snapshots cost only interpolation accuracy.
        # Default 0.01 since 2026-11-04: D3 history 1.4 GB -> 461 MB, wake change <= 0.13%;
        # 0.01..0.1 give the same map to 0.2%.
        self.drift_snap_thin = drift_snap_thin
        # Place the transverse integration nodes on the retarded density ribbon
        # (in the tilt-removed xi frame) instead of on a rectangle in lab x'.
        # Only meaningful for the bspline_fft deposition, whose grid is sized by
        # sigma_xi while the legacy x' bands are sized by sigma_x -- at high tilt
        # those differ by sigma_x/sigma_xi and the quadrature misses the beam.
        self.xi_bands = xi_bands
        # Widen the located ribbon by this factor, to absorb the error in the
        # single-pass estimate of where t_ret places the beam.
        self.xi_band_margin = xi_band_margin
        # Near the branch degeneracy |sin 2a| -> 1 (frame tilt |tau| -> 1, the two Eq 4.24 roots
        # merge and the band Jacobian -> 0) the default margin under-covers the retarded source:
        # BC20 D5 x-waist, tau = -1.15: -40% of <W>, converged by margin 6 (progress_facet_bc20 9m).
        # The margin is multiplied by this factor, ramped in smoothly between |sin 2a| = 0.8 and
        # 0.95 (CSR2D.degenerate_margin_factor). 1 (or 0) disables.
        self.xi_band_margin_degenerate = float(xi_band_margin_degenerate)
        # Radius (in sigma_xi) of the polar patch used for the near field around
        # the singular point x' = x, s' = s, where the integrand goes as 1/|r-r'|.
        # 0 disables the patch. See Stupakov, PRAB 25, 014401 (2022) Sec. IV, which
        # splits out the same |s'-s| < ds region analytically.
        self.near_patch = near_patch
        # Patch resolution is deliberately independent of xbins/zbins: the patch is a
        # different geometry, and coupling them would make it impossible to refine
        # the Cartesian mesh and the patch separately when checking convergence.
        self.near_patch_nr = near_patch_nr
        self.near_patch_nphi = near_patch_nphi
        # Transverse integrand: 'all' = W1 + W2 + W3 (Stupakov PRAB 25, 014401), 'W1' = the density
        # term only. W2 and W3 both carry d rho/dt at O(1/R) and cancel to O(theta^2/8R) on the
        # arc, so their sum is a small residual of two large terms; it is what goes noisy where a
        # narrow spike sits in the near region of a pedestal observer (BC20 B1RE, progress_facet_bc20 9j).
        if transverse_terms not in ('all', 'W1'):
            raise ValueError(f"CSR_integration.transverse_terms must be 'all' or 'W1', got {transverse_terms!r}")
        self.transverse_terms = transverse_terms
        # near_adapt: add near-region s' nodes where the RETARDED density varies along the column
        # (CSR2D._near_adapt_nodes). The fixed layout (near_cell, near_grade) resolves a source that
        # varies on the sigma_z scale; ahead of the observer z_ret moves at 2 ds', so a feature of
        # width w in z is crossed in w/2 of s', and a 3 um spike in a 41 um cell is sampled by at
        # most one node (BC20 B1RE pedestal: the transverse d rho/dt terms go to speckle,
        # progress_facet_bc20 9j-9k). The column is scanned (band location + density lookup),
        # refined until z_ret changes by <= near_adapt_scan_dz * res_z per interval inside the
        # history's z support (at most near_adapt_max_scan points), and the nodes equidistribute
        # 1/h_fixed + |d(rho/rho_max)/ds'| / near_adapt_eps, graded to near_adapt_max_ratio, at most
        # near_adapt_max_nodes. No peak finding: horns, spikes and edges are all just gradients.
        # eps 0.02 (not 0.1): on the tilted, two-branch BC20 B1RE state 0.1 left 6% of the spike
        # force as residual on the pedestal, 0.02 with 4 x' rounds 0.5% (progress_facet_bc20 9l).
        self.near_adapt = bool(near_adapt)
        self.near_adapt_eps = float(near_adapt_eps)
        self.near_adapt_scan_dz = float(near_adapt_scan_dz)
        self.near_adapt_max_scan = int(near_adapt_max_scan)
        self.near_adapt_max_nodes = int(near_adapt_max_nodes)
        self.near_adapt_max_ratio = float(near_adapt_max_ratio)
        # near_adapt_x: the same for the x' nodes of the near-region columns. z_ret also depends on
        # x' (d z_ret / d x' ~ dx / |r - r'|, O(1) for a source ahead), so a narrow feature is
        # crossed within the band over ~w / (dz_ret/dx') of x' against a uniform x' cell of
        # band width / xbins (BC20 B1RE: 6 um against 16 um). After the uniform evaluation, every
        # column whose retarded density jumps by more than near_adapt_eps * rho_max between
        # neighbouring x' nodes is re-evaluated on x' nodes that equidistribute
        # 1/h_uniform + |d(rho/rho_max)/dx'| / near_adapt_eps (at most near_adapt_x_max * xbins
        # nodes), near_adapt_x_rounds times. Needs near_adapt; not combined with ibp.
        self.near_adapt_x = bool(near_adapt_x)
        self.near_adapt_x_rounds = int(near_adapt_x_rounds)
        self.near_adapt_x_max = int(near_adapt_x_max)
        # far_adapt_x: the same x' refinement on the FAR region's columns. A horn 1 um wide seen
        # through the oblique branch of the band from far behind is crossed in a few um of x'
        # against a uniform cell of band width / xbins (BC20 loss, B2RE: 10 um), and the
        # transverse integrand (1/|r-r'|^2, d rho/dt) aliases it into isolated cells of F up to
        # 10x the force on the horn, and W into cells off by 34% of the horn's W (progress_facet_bc20.md
        # 11f). On the B2RE state far_adapt_x reproduces an 8x x' reference to 10% of the horn F and 1%
        # of the horn W at +16% run time (8x: +160%). Default True; inert unless near_adapt and
        # near_adapt_x are on (they are off by default).
        self.far_adapt_x = bool(far_adapt_x)
        # integrand_kernel: 'numba' (default; integrand_kernel.py, the arithmetic fused per point, no
        # temporaries) or 'numpy' (the original whole-array arithmetic). Bit-identical to each other
        # (checked on the BC20 B2RE state and the golden fixtures); numba is ~8% faster per wake point.
        if integrand_kernel not in ('numpy', 'numba'):
            raise ValueError(f"integrand_kernel must be 'numpy' or 'numba', got {integrand_kernel!r}")
        self.integrand_kernel = integrand_kernel
        # threads: numba threads per process for the field interpolation and the integrand kernel on
        # point sets of at least threads_min_points (smaller calls stay serial). 1 = the serial
        # kernels, bit-identical to before; the parallel ones are the same source compiled with
        # parallel=True and give identical numbers (independent iterations, no reductions).
        self.threads = int(threads)
        # far_adapt: density-adaptive s' columns on the far region inside a bend (the near-adapt scan
        # and monitor from a 2 * far_adapt_base uniform base, capped at the fixed 2 * far_zbins).
        # Needs near_adapt. Measured 2026-11-24 (progress_facet_bc20.md 11j): the monitor follows the
        # density at the band axis along s', which is NOT where a horn beam's far-region structure
        # is -- that is the oblique crossing across x', handled by far_adapt_x -- so at base 50 it
        # kept 180 of 400 columns and was off by 18% of the horn's W at B2RE, 3% at B1RE, for
        # 8-12% less time; on the smooth gain beam 0.5% for 44% less time. Off by default; usable
        # only on beams without sharp features.
        self.far_adapt = bool(far_adapt)
        self.far_adapt_base = int(far_adapt_base)
        # seam_ratio: continuity of the s' cell across the far/near seam. The near region's
        # graded cell is near_grade*u; the far region's (in a bend) is a flat
        # n_formation_length*L_f / (2 far_zbins), 5 mm on BC20. The near-region length d is
        # set by the chirp-band exit (plus near_floor), which for an observer on the
        # beam's near-x side (x - xmean < 0 at high tilt) was 3 mm -- so the 0.15 mm cells
        # at the end of the near region handed over to 5 mm cells, 33x coarser, while the
        # column integral still varied on a 1-10 mm scale (|r - r'| ~ the transverse
        # offset). W at the outer x rows was +0.93 / -0.93 / +0.34 MeV/m at 400 far
        # columns and -0.23 / -0.28 / +0.11 at 800; the horn row's core moved 10%.
        # With d >= far_cell / (near_grade * seam_ratio) the graded cell reaches
        # far_cell / seam_ratio before the seam, and 400 and 800 far columns agree to
        # 1e-3 MeV/m on every row for +45 near nodes (progress_facet_bc20.md 11k).
        # 0 disables it (bit-identical to the layout before 2026-11-24).
        self.seam_ratio = float(seam_ratio)
        # prehistory [m]: synthetic straight-line history before the lattice start. The beam is
        # first deposited at s = 0, and without this the retarded integral has no sources at
        # s' < 0, while the 1D entrance transient assumes the beam arrived along an infinite
        # straight line. A source at distance u before the first bend's entrance, seen from arc
        # angle phi, is retarded by ~ R phi^3/6 - R^2 phi^4/(8u): the whole straight history
        # contributes at nearly one retarded z and decays only as 1/u, so a lattice starting at
        # (or 0.1 m before) the first dipole understates its wake 2x at the exit (chicane B1,
        # talman_cancellation.md 7.3). With prehistory = L > 0 the far region may reach down to
        # s' = -L along the straight continuation of the reference orbit (_lattice_vectors),
        # with zero curvature there, and the history lookup returns the FIRST snapshot for
        # t < 0 (the interpolant's clamp), i.e. the beam is assumed to have travelled the last
        # L metres unchanged. Set it to at least the reach (n_formation_length * L_f, or ~2 R phi
        # for a short strong bend) -- the drift test needed 2 m on the chicane. 0 = off (bit-identical).
        self.prehistory = float(prehistory)
        # Straight-section reach for the entrance transient (talman_cancellation.md 7.3-7.5).
        # Inside a bend, a = R phi past the entrance, the straight section the beam came along is
        # a source retarded by ~ R phi^3/6 (nearly the same for all of it) whose contribution
        # decays only as 1/u with the distance u before the entrance, so the fraction beyond U is
        # ~ R phi/(U + R phi): U = 20 R phi captures 95%, 40 R phi 98% (the default; measured
        # converged to 1e-4 between 40 and 100 with 56 and 75 nodes). Measured on the chicane's B1 exit state
        # (R phi = 0.5 m): <W> -0.0103 with the 0.9 m of straight the flat reach covered, -0.0139
        # at 2.5 m, -0.0150 at 4 m, -0.0163 at 10 m (1D: -0.0212; the peak wake converges to the
        # 1D one, 0.084 vs 0.086). The rule: integrate [-reach_straight * R phi, entrance) as its
        # own region, geometrically graded in u (ratio 1 + near_grade, ~40 nodes for 10 m against
        # 3000 at the fixed far cell), inside the bend and in the drift after it, while the
        # straight sources are still retarded within reach_slip_max * sigma_z of the bunch
        # (R phi^3/6 inside, + x phi^2/2 at x past the exit; beyond that their retarded point has
        # left the bunch). 5 sigma_z: with 2 the rule switched off 0.18 m past the chicane's B1
        # exit while 1D still grew (<W> -0.020 vs -0.038 there; with 5: -0.033). The region never
        # goes below the previous bend's exit (prev_bend_reach owns that, and covers the whole
        # section between two bends already, so in a chicane or BC20 the rule only acts on the
        # FIRST bend) nor below the retained history unless that is the synthetic pre-history.
        # Needs the history: a lattice starting at its first bend also needs prehistory.
        # reach_straight = 0 disables the rule (bit-identical).
        self.reach_straight = float(reach_straight)
        # drift_causal_reach: in the drift after a bend, the regular layout has the n_formation_length = 1
        # reach (back to that bend's entrance through drift_reach) whatever n_formation_length is;
        # everything earlier -- the previous bend, the straight between the two -- is integrated as
        # its own region from the causal edge (the lowest s' whose retarded band still meets the bunch,
        # _causal_edge with drift_causal_scan samples) up to there, geometrically graded in u = s - s' at
        # 1 + drift_causal_grade. Indranil's L1-17.6284 beam 1.5 m past B3RE: the B3LE arc and the 0.8 m
        # between B3LE and B3RE changed <W> from +0.066 to -0.188 MeV/m and were dropped at n_fl = 1 (the
        # reach stopped at B3RE's entrance and its history was discarded); the region reproduces the
        # n_fl = 6 brute-force value to 0.1% with ~100 columns. It is bounded below by s' = 0 (never the
        # synthetic prehistory, whose straight-line term diverges seen from a drift, talman 7.7) and by
        # the retained history; the retention keeps the previous bend from n_formation_length * L_f before
        # its entrance. Large n_formation_length in a drift used to move the exit-transient fine window off
        # the last bend's arc (65% wrong 1.4 m past B1RE at n_fl = 6); the fixed drift reach removes that.
        # False = the old drift layout (bit-identical).
        self.drift_causal_reach = bool(drift_causal_reach)
        # x_adapt: how near_adapt_x / far_adapt_x refine the x' nodes of a column.
        #   'density'  re-evaluate columns where the retarded DENSITY jumps by > near_adapt_eps * rho_max
        #              between nodes (rho_max over the whole region), nodes following the density gradient.
        #   'error'    estimate each column's x' quadrature error from the integrand itself: the column is
        #              evaluated on xbins nodes and on the midpoints (trapezoid T_n -> T_2n, error ~ |T_2n -
        #              T_n|/3), for W and F; the columns with the largest s'-weighted errors are halved again
        #              (up to x_adapt_rounds times) until the error left in the unrefined ones is below
        #              x_adapt_tol of the region's gross sum_j w_j |T_j| (each component). One criterion for the
        #              horn aliasing the density test was made for AND the geometry the density test cannot
        #              see: in B1RE near |tau| = 1 of Indranil's L1-17.6284 beam the density test left the mean
        #              wake 18% off at 100 nodes (progress_facet_bc20.md 11q-r). Default since 2026-11-26:
        #              on that state 100 nodes + 'error' match a 1600-node reference to 0.1% in W at 0.4x the
        #              time of 100 + 'density', and on the loss beam's B2RE horns W 0.06% / F 0.2% at 0.5x.
        #              Applies to every far-type region (far, previous bend, straight section, drift causal).
        self.x_adapt = str(x_adapt)
        if self.x_adapt not in ('density', 'error'):
            raise ValueError(f"CSR_integration.x_adapt must be 'density' or 'error', got {x_adapt!r}")
        self.x_adapt_tol = float(x_adapt_tol)
        self.x_adapt_rounds = int(x_adapt_rounds)
        self.drift_causal_grade = float(drift_causal_grade)
        self.drift_causal_scan = int(drift_causal_scan)
        self.reach_slip_max = float(reach_slip_max)
        self.threads_min_points = int(threads_min_points)
        # fixed-point iterations of the band solve in the column scan (the integration's own solve
        # keeps 12). On the BC20 B1RE state 4 and 6 give nodes identical to 12 to the last bit, 3
        # differs at 1e-7 MeV/m; 4 saves 13% per observer (progress_facet_bc20 9l).
        self.near_adapt_scan_iter = int(near_adapt_scan_iter)


class CSR_params:
    # Todo: Maybe not necessary. Can just be a dictionary with an additional function to parse default values.
    def __init__(self, input_dic = {}):
        self.configure_params(**input_dic)

    def configure_params(self, workdir = '.', apply_CSR = 1, compute_CSR = 1,
                         transverse_on = 1, xbins = 20, zbins = 30, xlim = 5, zlim = 5, write_beam = None, write_wakes = True, write_name = '',
                         mesh_mode = 'uniform', mesh_uniform_frac = 0.3, mesh_max_ratio = 1.5,
                         mesh_smooth = 3.0, mesh_pad = 0.02, pedestal_warn = 0.05,
                         mesh_x_quantile = None, mesh_z_quantile = None):
        # mesh_mode 'uniform': xbins x zbins evenly over +-xlim, +-zlim rms (the original mesh).
        # mesh_mode 'charge': the same node counts, each axis equidistributing
        #   mesh_uniform_frac / L + (1 - mesh_uniform_frac) * (projected charge, smoothed over
        #   mesh_smooth deposition resolution lengths), over every particle widened by mesh_pad
        #   of the span, neighbouring spacings within mesh_max_ratio (peakfind.charge_nodes).
        #   Every horn or spike gets nodes in proportion to its charge; xlim/zlim are unused.
        # mesh_x_quantile (mesh_mode 'charge'): the x axis spans [x_q, x_(1-q)] of the transformed x
        #   instead of every particle, and particles outside get the kick of the edge row (clamped,
        #   not 0). A beam with sparse transverse tails (BC20: to +-12 sigma) otherwise puts most
        #   of a 12-row mesh in the tails, 4 rows within +-2.5 sigma. None: every particle.
        # pedestal_warn: with particle_deposition.frame 'quantile', warn when the 1D estimate
        #   of the dropped tails' wake exceeds this fraction of the full wake.
        if mesh_mode not in ('uniform', 'charge'):
            raise ValueError(f"CSR_computation.mesh_mode must be 'uniform' or 'charge', got {mesh_mode!r}")
        if not 0.0 <= float(mesh_uniform_frac) <= 1.0:
            raise ValueError(f'CSR_computation.mesh_uniform_frac must be in [0, 1], got {mesh_uniform_frac}')
        if not float(mesh_max_ratio) > 1.0:
            raise ValueError(f'CSR_computation.mesh_max_ratio must be > 1, got {mesh_max_ratio}')
        self.mesh_mode = mesh_mode
        self.mesh_uniform_frac = float(mesh_uniform_frac)
        self.mesh_max_ratio = float(mesh_max_ratio)
        self.mesh_smooth = float(mesh_smooth)
        self.mesh_pad = float(mesh_pad)
        if mesh_x_quantile is not None and not 0.0 < float(mesh_x_quantile) < 0.5:
            raise ValueError(f'CSR_computation.mesh_x_quantile must be in (0, 0.5), got {mesh_x_quantile}')
        self.mesh_x_quantile = None if mesh_x_quantile is None else float(mesh_x_quantile)
        # mesh_z_quantile: like mesh_x_quantile for the charge mesh's z axis. By default the z axis
        # spans every particle (+ mesh_pad): a sparse tail then takes nodes -- Indranil's loss beam at
        # s = 0.15 (quantile frame, 12.6 um core, 0.1% of the charge out to +49 um) had 15 of its 80 z
        # cells carrying < 0.05% of the charge, 3.6 um apart in empty space, and the core cells 30%
        # coarser than needed (progress_facet_bc20.md 11o). With q the axis spans the [q, 1-q]
        # quantiles (+ pad) and particles beyond it get the edge column's kick (clamped), as in x.
        # None = every particle (bit-identical); 0.001 in the BC20 recipe.
        self.mesh_z_quantile = None if mesh_z_quantile is None else float(mesh_z_quantile)
        self.pedestal_warn = float(pedestal_warn)
        self.compute_CSR = compute_CSR
        self.apply_CSR = apply_CSR
        self.transverse_on = transverse_on
        self.xbins = xbins
        self.zbins = zbins
        self.xlim = xlim
        self.zlim = zlim
        self.write_beam = write_beam
        self.write_wakes = write_wakes
        self.workdir = full_path(workdir)
        self.write_name = write_name


