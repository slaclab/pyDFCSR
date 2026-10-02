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

    def configure_params(self, n_formation_length = 4, zbins = 200, xbins = 200,
                         xi_bands = True, xi_band_margin = 2.0, near_patch = 5.0,
                         near_patch_nr = 100, near_patch_nphi = 180,
                         near_cell = 0.5, far_zbins = 200, near_grade = 0.05,
                         branch_sin_min = 0.05, near_floor = 20.0,
                         causal_clip = True, far_window = 1.0, far_cell = 10.0,
                         drift_cutoff = 1.0):
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
        self.far_cell = far_cell
        # Stop computing wakes once the beam is more than drift_cutoff * R*phi past the
        # bend it just left, in units of that bend's own R*phi. 0 disables the cutoff.
        #
        # ON by default at 1.0, which is the author's decision and rests on the far-drift
        # wake being untrustworthy there rather than on it being negligible. The numbers
        # below are what that costs and what it removes; both matter.
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
        # So the cutoff removes ~0.5% of real signal plus one map that is visibly wrong.
        #
        # The defect is NOT repaired by the 11u far-region rebuild: measured at the same
        # position, peak 5.1864 -> 5.1412 and roughness 4.601 -> 4.720, i.e. unchanged.
        # Run-wide roughness is likewise unchanged (mean 1.577 -> 1.581, 70 of 105 maps
        # above 1.0 both ways). An earlier claim here that the rebuild fixed it was an
        # artefact of comparing two different kick positions.
        #
        # Enabling it is NOT a small perturbation, because suppressing the wake in a drift
        # also changes the beam entering the next bend and the chicane amplifies that:
        # measured end-to-end, energy loss -1.6475 -> -1.3825 MeV (16%) and emittance
        # growth 1.306x -> 1.117x, against the 5.1% direct share. Set drift_cutoff = 0 to
        # integrate the drifts in full, which reproduces the pre-11v behaviour.
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


