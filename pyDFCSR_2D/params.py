from .tools import full_path


class Integration_params:
    def __init__(self, input_dic = {}):
        self.configure_params(**input_dic)

    def configure_params(self, n_formation_length = 4, zbins = 200, xbins = 200):
        self.n_formation_length = n_formation_length
        self.zbins = zbins
        self.xbins = xbins


class CSR_params:
    def __init__(self, input_dic = {}):
        self.configure_params(**input_dic)

    def configure_params(self, workdir='.', apply_CSR=1, compute_CSR=1,
                         transverse_on=1,
                         xbins=20, zbins=30, xlim=5.0, zlim=5.0,
                         kick_xlim=None, kick_zlim=None,
                         write_beam=None, write_wakes=True, write_name='',
                         restart=None,
                         stop_step=None):

        self.compute_CSR   = int(compute_CSR)
        self.apply_CSR     = int(apply_CSR)
        self.transverse_on = int(transverse_on)

        self.xbins = int(xbins)
        self.zbins = int(zbins)
        self.xlim  = float(xlim)
        self.zlim  = float(zlim)

        self.kick_xlim = float(self.xlim if kick_xlim is None else kick_xlim)
        self.kick_zlim = float(self.zlim if kick_zlim is None else kick_zlim)

        self.write_beam  = write_beam
        self.write_wakes = bool(write_wakes)
        self.workdir     = full_path(workdir)
        self.write_name  = str(write_name)

        self.restart  = restart
        self.stop_step = None if stop_step is None else int(stop_step)