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
                         near_patch_nr = 100, near_patch_nphi = 180):
        self.n_formation_length = n_formation_length
        self.zbins = zbins
        self.xbins = xbins
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


