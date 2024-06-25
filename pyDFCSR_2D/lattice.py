import numpy as np
from .yaml_parser import parse_yaml

def get_referece_traj(lattice_config, Nsample = 5000, Ndim = 2):
    """
    A function to get the reference trajectory of partices with given lattice configuration
    :param lattice_config: dictionary
           Nsample: the number of points in each element to calculate the traj
           Ndim:  the dimension of the trajectory
    :return:
      s : longitudindal coordinates (Nsample,)
      coords:  interpolant of coordinate of the trajectory, array (Nsample, Ndim). coord[:,0] = x, coord[:,1] = y
      tau_vec, n_vec: interpolant of tangential and normal vectors along the trajectory, array (Nsamp, Ndim).
                        tau_vec[:, 0] x component. tau_vec[:, 1] y component
      rho: bending radius (1/R) of the trajectory, array (Nsample,)
      nsep: array, number of steps to compute CSR wake for each lattice element
      distance: (Nelement,): distance[i] is the distance from the lattice entrance to the end of ith lattice element
    """

    # The number of lattice elements
    Nelement = len(lattice_config) - 1

    # distance[i] is the distance between the entrance and the end of ith lattice element
    distance = np.zeros(Nelement)

    rho = np.zeros(Nelement)
    nsep = np.zeros(Nelement)
    #s = np.zeros(Nelement*Nsample + 1)

    # Loop through each lattice element (ie drift, quad, drift, etc)
    for count, key in enumerate(list(lattice_config.keys())[1:]):
        current_element = lattice_config[key]

        # length of the current element
        L = current_element['L']

        # Number of steps to compute CSR wake within current element
        nsep[count] = current_element['nsep']

        # Need to account for reference trajectory curving through dipole
        if current_element['type'] == 'dipole':
            angle = current_element['angle']
            rho[count] = angle/L

        # Populate the distance array
        if count == 0:
            distance[count] = L
            #s[count*Nsample: (count + 1)*Nsample + 1] = np.linspace(0, L, Nsample + 1)

        else:
            distance[count] = L + distance[count - 1]
            # this is to make sure that the edge of each elements is sampled in s
            #s[count*Nsample + 1: (count + 1)*Nsample + 1] = np.linspace(distance[count - 1], distance[count], Nsample + 1)[1:]
    #s[-1] = distance[-1]

    #Todo: I change the definition of s to be equidistant. However, it cannot garanteed that the edge of each element is included.
    # total length of the lattice 
    L_lattice = distance[-1]

    # Initialize reference trajectory arrays
    # coordinates along s to calculate the reference traj 
    s = np.linspace(0, L_lattice, Nsample)
    interval = np.mean(np.diff(s))
    coords = np.zeros((Nsample, Ndim))
    tau_vec = np.zeros((Nsample, Ndim))
    n_vec = np.zeros((Nsample, Ndim))

    #rho = np.zeros(Nelement*Nsample + 1)

    # Populate initial values for reference trajectory vectors
    theta_0 = 0  # te angle between the traj tangential and x axis
    tau_vec[0, 0] = np.cos(theta_0)
    tau_vec[0, 1] = np.sin(theta_0)
    # Todo: High Priority! check the sign of n_vec
    n_vec[0, 0] = np.sin(theta_0)
    n_vec[0, 1] = -1 * np.cos(theta_0)

    # All lattice elements
    keys = list(lattice_config.keys())[1:]

    # Define counter variables
    element_infer = 0      # pointer to which element we are in
    count = 1

    # For each reference trajectory point (RTP)
    for st in s[1:]:
        # Check to see if current RTP is outside current lattice element
        if st > distance[element_infer]:
            element_infer += 1

        # Change in s from previous RTP (should be uniform rn)
        delta_s = s[count] - s[count - 1]
        
        # The name of the current lattice element
        ele_name = keys[element_infer]

        # Populate the RTP depending on the type of the current element
        # for dipole (account for curve)
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

        # Update our current angle
        theta_0 += phi

        # Populate the reference trajectory vectors
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
        """
        Parameters:
            input_lattice: a one element dictionary with key 'lattice_input_file' and value with pathname.yaml (string)
        Returns:
            instance of Lattice
        """

        # Make sure that the input_lattice has the input_file path name
        assert 'lattice_input_file' in input_lattice, 'Error in parsing lattice: must include the keyword <lattice_input_file>'
        self.lattice_input_file = input_lattice['lattice_input_file']

        # Create lattice dictionary from pathname.yaml
        lattice_config = parse_yaml(self.lattice_input_file)
        
        # Verify that all necessary parameters are present in settings dictionary
        self.check_input(lattice_config)
        self.lattice_config = lattice_config
        
        # Number of lattice elements
        self._Nelement = len(lattice_config) - 1

        # Compute the reference trajectory of the beam path
        self.get_ref_traj()

        # Compute the location of each step where CSR will be caclualted
        self.get_steps()

        # Compute quick interpolant of x positions
        self.build_interpolant()
        self.current_element = None           # pointer of the element where the beam is now in.

    def check_input(self, input):
        # Todo: check input for lattice
        assert 'step_size' in input, f'Required input parameter step_size to {self.__class__.__name__}.__init__(**kwargs) was not found.'

    def get_ref_traj(self, Nsample = 2000):
        # Calls the function to compute reference trajectory
        self.s, self.rho, self.distance, self.nsep, self.coords, self.n_vec, self.tau_vec = get_referece_traj(lattice_config = self.lattice_config, Nsample = Nsample)
        self._lattice_length = self.distance[-1]

    def build_interpolant(self):
        # Computes min, max, and average change of x along reference trajectory
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
        Calcuates the locations on the reference trajectory (the steps) where the CSR wake will be computed
        """

        # The distance between each location
        self.step_size = self.lattice_config['step_size']

        #Todo: Deal with the endpoint
        # Array with nth element containing the nth step position
        self._positions_record = np.arange(0, self.lattice_length + self.step_size/2, self.step_size)
        self._total_steps = len(self._positions_record)
        
        # ???
        self._CSR_steps_index = np.array([])                   

        # Initialize with Nelement zeros, records how many steps are inside each lattice element
        self.steps_per_element = np.zeros((self.Nelement,), dtype = int)

        # The lattice element's step position from the prevoius loop iteration
        prev_ind = 0

        # Populate _CSR_steps_index and steps_per_element
        # Loop through each element in lattice - d will be the distance from lattice entrance to end of the element
        for count, d in enumerate(self.distance):
            # Find where the lattice element ends in the _positions array
            ind = np.searchsorted(self._positions_record, d, side = 'right')    #a[ind-1]<= d<a[ind], a is positions_record

            # The number of steps in this lattice element
            nsep_t = self.nsep[count]

            # Create a smaller array for this specific lattice element
            new_index = np.arange(prev_ind, ind, nsep_t)

            # Append new_index to the overall array
            self._CSR_steps_index = np.append(self._CSR_steps_index, new_index)

            # Populate steps_per_element
            if count == 0:
                self.steps_per_element[count] = ind - prev_ind - 1 # s = 0
            else:
                self.steps_per_element[count] = ind - prev_ind

            prev_ind = ind

        # ???
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

