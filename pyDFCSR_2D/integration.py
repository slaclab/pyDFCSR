class Integration:
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

