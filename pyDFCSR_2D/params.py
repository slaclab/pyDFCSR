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
                         xi_bands = True, xi_band_margin = 2.0, near_patch = 5.0,
                         near_patch_nr = 100, near_patch_nphi = 180,
                         near_cell = 0.25, far_zbins = 200, near_grade = 0.05,
                         branch_sin_min = 0.05, near_floor = 20.0,
                         causal_clip = True, far_window = 1.0, far_cell = 10.0,
                         drift_cutoff = 0.0, tangent_bands = True,
                         drift_reach = True, drift_snap_thin = 0.01,
                         far_fine_max = 500, band_jacobian_min = 0.2,
                         prev_bend_reach = True, prev_bend_grade = 0.02):
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


class CSR_params:
    # Todo: Maybe not necessary. Can just be a dictionary with an additional function to parse default values.
    def __init__(self, input_dic = {}):
        self.configure_params(**input_dic)

    def configure_params(self, workdir = '.', apply_CSR = 1, compute_CSR = 1,
                         transverse_on = 1, xbins = 20, zbins = 30, xlim = 5, zlim = 5, write_beam = None, write_wakes = True, write_name = ''):
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


