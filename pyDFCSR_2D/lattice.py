import numpy as np

from .schedule import build_legacy, build_manual, build_auto
from .yaml_parser import parse_yaml

def get_referece_traj(lattice_config, Nsample = 5000, Ndim = 2):
    """
    A function to get the reference trajectory of partices with given lattice configuration
    :param lattice_config: dictionary
           Nsample: the number of points in each element to calculate the traj
           Ndim:  the dimension of the trajectory
    :return:
      s : longitudindal coordinates (Nsamle,)
      coords:  interpolant of coordinate of the trajectory, array (Nsample, Ndim). coord[:,0] = x, coord[:,1] = y
      tau_vec, n_vec: interpolant of tangential and normal vectors along the trajectory, array (Nsamp, Ndim).
                        tau_vec[:, 0] x component. tau_vec[:, 1] y component
      rho: bending radius (1/R) of the trajectory, array (Nsample,)
      distance: (Nelement,): distance[i] is the distance from the lattice entrance to the end of ith element
    """
    # Element keys, selected BY NAME rather than by position. This used to be
    # `list(lattice_config.keys())[1:]`, i.e. "everything after the first key", which assumed
    # step_size came first. yaml.dump sorts keys alphabetically by default, which moves step_size
    # to the END and made get_referece_traj die with
    # "TypeError: 'float' object is not subscriptable". It bit twice during this work.
    ele_keys = [k for k in lattice_config if k != 'step_size']
    Nelement = len(ele_keys)
    distance = np.zeros(Nelement)        # distance[i] is the distance between the entrance and the end of ith element
    rho = np.zeros(Nelement)
    nsep = np.zeros(Nelement)
    #s = np.zeros(Nelement*Nsample + 1)
    count = 0
    for key in ele_keys:
        current_element = lattice_config[key]
        L = current_element['L']
        nsep[count] = current_element['nsep']
        if current_element['type'] == 'dipole':
            angle = current_element['angle']
            rho[count] = angle/L

        if count == 0:
            distance[count] = L
            #s[count*Nsample: (count + 1)*Nsample + 1] = np.linspace(0, L, Nsample + 1)
        else:
            distance[count] = L + distance[count - 1]
            # this is to make sure that the edge of each elements is sampled in s
            #s[count*Nsample + 1: (count + 1)*Nsample + 1] = np.linspace(distance[count - 1], distance[count], Nsample + 1)[1:]
        count += 1
    #s[-1] = distance[-1]

    #Todo: I change the definition of s to be equidistant. However, it cannot garanteed that the edge of each element is included.
    L_lattice = distance[-1]     # total length of the lattice
    s = np.linspace(0, L_lattice, Nsample)                     # coordinates along s to calculate the reference traj
    interval = np.mean(np.diff(s))
    coords = np.zeros((Nsample, Ndim))
    tau_vec = np.zeros((Nsample, Ndim))
    n_vec = np.zeros((Nsample, Ndim))

    #rho = np.zeros(Nelement*Nsample + 1)

    element_infer = 0      # pointer to which element we are in
    count = 1
    theta_0 = 0  # te angle between the traj tangential and x axis
    tau_vec[0, 0] = np.cos(theta_0)
    tau_vec[0, 1] = np.sin(theta_0)

    # Todo: High Priority! check the sign of n_vec
    n_vec[0, 0] = np.sin(theta_0)
    n_vec[0, 1] = -1 * np.cos(theta_0)
    keys = list(lattice_config.keys())[1:]
    for st in s[1:]:
        if st > distance[element_infer]:
            element_infer += 1


        delta_s = s[count] - s[count - 1]
        ele_name = keys[element_infer]


        # for dipole
        if lattice_config[ele_name]['type'] == 'dipole':
            L = lattice_config[ele_name]['L']
            angle = lattice_config[ele_name]['angle']
            phi = delta_s/L*angle
            r = L/angle

            # seems to be stupid. Todo
            x0 = coords[count - 1, 0] - r*np.sin(theta_0)
            y0 = coords[count - 1, 1] + r*np.cos(theta_0)

            coords[count, 0] = x0 + r*np.sin(phi + theta_0)
            coords[count, 1] = y0 - r*np.cos(phi + theta_0)
            #rho[count]  = angle/L

        # for drift, quad, sext
        else:
            phi = 0
            coords[count, 0] = coords[count - 1, 0] + delta_s *np.cos(theta_0)
            coords[count, 1] = coords[count - 1, 1] + delta_s *np.sin(theta_0)
            #rho[count] = 0

        # Todo: check other elements (quad, sextupole)

        theta_0 += phi


        tau_vec[count, 0] = np.cos(theta_0)
        tau_vec[count, 1] = np.sin(theta_0)

        # Todo: High Priority! check the sign of n_vec
        n_vec[count, 0] = np.sin(theta_0)
        n_vec[count, 1] = -1*np.cos(theta_0)

        count += 1




    return s, rho, distance, nsep, coords, n_vec, tau_vec

class Lattice():
    """
    lattice class to read lattice file and get information like reference trajectory
    maybe install a pointer for the position of the current beam
    """

    def __init__(self, input_lattice, step_control=None, sigma0=None):

        assert 'lattice_input_file' in input_lattice, 'Error in parsing lattice: must include the keyword <lattice_input_file>'
        self.lattice_input_file = input_lattice['lattice_input_file']

        lattice_config = parse_yaml(self.lattice_input_file)
        self.check_input(lattice_config)
        self.lattice_config = lattice_config
        self.step_control = step_control or input_lattice.get('step_control', None)
        # initial 6x6 beam moments, needed only by mode 'auto' to run the linear-optics
        # scan. The beam is constructed before the Lattice in CSR2D.parse_input, so it
        # is available here.
        self.sigma0 = sigma0
        self._Nelement = len([k for k in lattice_config if k != 'step_size'])
        self.get_ref_traj()
        self.get_steps()

        self.build_interpolant()
        self.current_element = None           # pointer of the element where the beam is now in.

    def check_input(self, input):
        # Todo: check input for lattice
        assert 'step_size' in input, f'Required input parameter step_size to {self.__class__.__name__}.__init__(**kwargs) was not found.'
    def get_ref_traj(self, Nsample = 2000):
        self.s, self.rho, self.distance, self.nsep, self.coords, self.n_vec, self.tau_vec = get_referece_traj(lattice_config = self.lattice_config, Nsample = Nsample)

        self._lattice_length = self.distance[-1]

    def build_interpolant(self):
        self.min_x, self.max_x = self.s[0], self.s[-1]
        self.delta_x = (self.max_x - self.min_x) / (self.s.shape[0] - 1)
        #self.F_x_ref = RegularGridInterpolator(points=(self.s,), values=self.coords[:, 0], method='linear',bounds_error = False)
        #self.F_y_ref = RegularGridInterpolator(points=(self.s,), values=self.coords[:, 1], method='linear',bounds_error = False)
        #self.F_n_vec_x = RegularGridInterpolator(points=(self.s,), values=self.n_vec[:, 0], method='linear',bounds_error = False)
        #self.F_n_vec_y = RegularGridInterpolator(points=(self.s,), values=self.n_vec[:, 1], method='linear',bounds_error = False)
        #self.F_tau_vec_x = RegularGridInterpolator(points=(self.s,), values=self.tau_vec[:, 0], method='linear',bounds_error = False)
        #self.F_tau_vec_y = RegularGridInterpolator(points=(self.s,), values=self.tau_vec[:, 1], method='linear',bounds_error = False)
        #self.F_rho = RegularGridInterpolator(points = (self.s,), values = self.rho, method = 'nearest',bounds_error = False)

    def get_steps(self):
        """
        Build the step schedule and expose the legacy attributes derived from it.

        `mode: legacy` (the default) reproduces the historical node set, kick cadence and
        steps_per_element exactly -- verified against a verbatim copy of the old algorithm on 120
        randomised lattices, including the cases where np.arange overshoots the lattice end.
        """
        self.step_size = self.lattice_config['step_size']
        cfg = self.step_control or {}
        mode = cfg.get('mode', 'legacy')

        kick_interval = cfg.get('kick_interval', 'trailing')
        if kick_interval not in ('trailing', 'midpoint'):
            raise ValueError(f"kick_interval must be 'trailing' or 'midpoint', got "
                             f"'{kick_interval}'")

        if mode == 'legacy':
            self.schedule = build_legacy(self.distance, self.nsep, self.lattice_length,
                                         self.step_size, self.Nelement,
                                         kick_interval=kick_interval)
        elif mode == 'auto':
            if self.sigma0 is None:
                raise ValueError("step_control mode 'auto' needs the initial beam moments; "
                                 "they are passed from CSR2D.parse_input")
            from .waist import propagate_frame, formation_length_profile
            s_scan, sz, sx, tau, sxi, rho = propagate_frame(
                self.lattice_config, self.sigma0, n_sub=cfg.get('n_sub', 400))
            L_f = formation_length_profile(self.lattice_config, s_scan, sz, rho)
            self.schedule = build_auto(
                self.lattice_config, self.distance, self.lattice_length, self.Nelement,
                s_scan, sz, rho, L_f, step_size=self.step_size,
                sxi_scan=sxi, tau_scan=tau, sx_scan=sx,
                kick_interval=kick_interval, nsep=self.nsep,
                m_steps=cfg.get('m_steps'), m_steps_xi=cfg.get('m_steps_xi'),
                tau_frac=cfg.get('tau_frac'), edge_steps=cfg.get('edge_steps'),
                kappa=cfg.get('kappa'), h_min=cfg.get('h_min'), h_max=cfg.get('h_max'),
                r_floor=cfg.get('r_floor'),
                dyadic=cfg.get('dyadic'),
                force_nodes=cfg.get('force_nodes'))
        elif mode == 'manual':
            self.schedule = build_manual(
                self.lattice_config, self.distance, self.lattice_length, self.Nelement,
                default_steps=cfg.get('default_steps'),
                default_kick_every=cfg.get('default_kick_every', 1),
                step_size=self.step_size, kick_interval=kick_interval, nsep=self.nsep)
        else:
            raise NotImplementedError(
                f"step_control mode '{mode}' is not implemented yet; use 'legacy'")

        # legacy-facing attributes, all derived from the schedule so there is one source of truth
        self._positions_record = self.schedule.s_nodes
        self._total_steps = self.schedule.n_nodes
        self.steps_per_element = self.schedule.steps_per_element(self.Nelement)
        self._CSR_steps_index = self.schedule.kick_indices
        self._CSR_steps_count = len(self._CSR_steps_index)


    @property
    def lattice_length(self):
        return self._lattice_length

    @property
    def CSR_steps_index(self):
        return self._CSR_steps_index







    @property
    def total_steps(self):
        return self._total_steps

    @property
    def steps_record(self):
        return self._positions_record

    @property
    def Nelement(self):
        return self._Nelement

    def update(self, ele_name):
        self.current_element = ele_name

