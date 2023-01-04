from tools import *
class Interpolation_params:
    #Todo: Maybe not necessary. Can just be a dictionary with an additional function to parse default values.
    def __init__(self, input_dic = {}):
        self.configure_params()

    def configure_params(self, xbins=500, zbins=500, xlim=10, zlim=10, re_interpolate_threshold=2):
        self.xbins = xbins
        self.zbins = zbins
        self.xlim = xlim
        self.zlim = zlim
        self.re_interpolate_threshold = re_interpolate_threshold


class Integration_params:
    # Todo: Maybe not necessary. Can just be a dictionary with an additional function to parse default values.
    def __init__(self, input_dic = {}):
        self.configure_params(**input_dic)

    def configure_params(self, n_formation_length = 4,
                         zlim_end = 10, zlim_mid1 = 100, zlim_mid2 = 200,
                         zbins_1 = 500, zbins_2 = 500, zbins_3 = 500,
                         xlim = 10, xbins = 1000):
        self.n_formation_length = n_formation_length
        self.zlim_end = zlim_end
        self.zlim_mid1 = zlim_mid1
        self.zlim_mid2 = zlim_mid2
        self.zbins_1 = zbins_1
        self.zbins_2 = zbins_2
        self.zbins_3 = zbins_3
        self.xlim = xlim
        self.xbins = xbins


class CSR_params:
    # Todo: Maybe not necessary. Can just be a dictionary with an additional function to parse default values.
    def __init__(self, input_dic = {}):
        self.configure_params(**input_dic)

    def configure_params(self, workdir = '.', apply_CSR = 1,
                         transverse_on = 1, xbins = 20, zbins = 30, xlim = 5, zlim = 5, write_beam = True, write_wakes = True):
        self.apply_CSR = apply_CSR
        self.transverse_on = transverse_on
        self.xbins = xbins
        self.zbins = zbins
        self.xlim = xlim
        self.zlim = zlim
        self.write_beam = write_beam
        self.write_wakes = write_wakes
        self.workdir = full_path(workdir)


