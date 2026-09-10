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
                         near_cell = 0.5, far_zbins = 200):
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


