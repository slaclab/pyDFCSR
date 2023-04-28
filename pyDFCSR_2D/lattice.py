import numpy as np
from scipy.interpolate import RegularGridInterpolator
from yaml_parser import *
def get_referece_traj(lattice_config, Nsample = 1000, Ndim = 2):
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
    Nelement = len(lattice_config) - 1
    distance = np.zeros(Nelement)        # distance[i] is the distance between the entrance and the end of ith element
    rho = np.zeros(Nelement)
    nsep = np.zeros(Nelement)
    #s = np.zeros(Nelement*Nsample + 1)
    count = 0
    for key in list(lattice_config.keys())[1:]:
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

    def __init__(self, input_lattice):

        assert 'lattice_input_file' in input_lattice, f'Error in parsing lattice: must include the keyword <lattice_input_file>'
        self.lattice_input_file = input_lattice['lattice_input_file']

        lattice_config = parse_yaml(self.lattice_input_file)
        self.check_input(lattice_config)
        self.lattice_config = lattice_config
        self._Nelement = len(lattice_config) - 1
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
        self.step_size = self.lattice_config['step_size']
        self._positions_record = np.arange(0, self.lattice_length, self.step_size)
        self._total_steps = len(self._positions_record)
        self._CSR_steps_index = np.array([])                   # the index of total_steps where the CSR will be computed
        self.steps_per_element = np.zeros((self.Nelement,), dtype = int)
        count = 0
        prev_ind = 0
        for d in self.distance:
            ind = np.searchsorted(self._positions_record, d, side = 'right')    #a[ind-1]<= d<a[ind], a is positions_record
            nsep_t = self.nsep[count]
            new_index = np.arange(prev_ind, ind, nsep_t)
            self._CSR_steps_index = np.append(self._CSR_steps_index, new_index)
            if count == 0:
                self.steps_per_element[count] = ind - prev_ind - 1 # s = 0
            else:
                self.steps_per_element[count] = ind - prev_ind
            count += 1
            prev_ind = ind

        self._CSR_steps_count = len(self._CSR_steps_index)



    @property
    def lattice_length(self):
        return self._lattice_length

    @property
    def CSR_steps_index(self):
        return self._CSR_steps_index

    @property
    def total_steps(self):
        return self.total_steps

    @property
    def CSR_steps_count(self):
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

    @property
    def lattice_length(self):
        return self._lattice_length

    def update(self, ele_name):
        self.current_element = ele_name

