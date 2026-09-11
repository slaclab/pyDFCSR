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
                         branch_sin_min = 0.05):
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


