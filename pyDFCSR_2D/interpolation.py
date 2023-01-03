class Interpolation:
    def __init__(self, input_dic = {}):
        self.configure_params()

    def configure_params(self, xbins=500, zbins=500, xlim=10, zlim=10, re_interpolate_threshold=2):
        self.xbins = xbins
        self.zbins = zbins
        self.xlim = xlim
        self.zlim = zlim
        self.re_interpolate_threshold = re_interpolate_threshold
