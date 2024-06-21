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

    def configure_params(self, n_formation_length = 4, zbins = 200, xbins = 200):
        self.n_formation_length = n_formation_length
        self.zbins = zbins
        self.xbins = xbins

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


