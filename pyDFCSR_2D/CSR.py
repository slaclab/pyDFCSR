import os
import time
from bmadx import  Drift, SBend, Quadrupole, Sextupole
from .tools import dict2hdf5
import h5py
import numpy as np
from mpi4py import MPI

from .beams import Beam
# from .deposit import histogram_cic_1d, histogram_cic_2d
from .deposit import DF_tracker
from .interp1D import interpolate1D
from .interp3D import interpolate3D
from .lattice import Lattice  # , get_referece_traj
from .params import Integration_params, CSR_params
# from .physical_constants import c, e, qe, me, MC2
from .r_gen6 import r_gen6
#from line_profiler_pycharm import profile
# from .tools import (find_nearest_ind, full_path, isotime, plot_2D_contour,
#                     plot_surface)
from .tools import full_path, isotime
from .twiss_R import twiss_R
from .yaml_parser import parse_yaml

class CSR2D:
    """
    The main class to calculate 2D CSR
    """

    def __init__(self, input_file, parallel = False):
        """
        Creates an instance of the CSR2D class
        Parameters:
            input_file: assumed to be a configuration in YMAL file format that can be converted to dict
            parallel: boolean, indicate if parallel processing should be used for computations
        Returns:
            instance of CSR2D
        """

        self.timestamp = isotime()

        # Given an input file, parse it into a dictionary and create instances of Beam, Lattice, DF_Tracker,
        # CSR_integration, and CSR_computation
        self.parse_input(input_file)
        self.input_file = input_file

        self.formation_length = None

        # 'process' the initial beam (just read the doctring for more info)
        self.initialization()

        self.prefix = f'{self.CSR_params.write_name}-{self.timestamp}'

        # If we are using parallel computing, initialize MPI
        if parallel:
            self.init_MPI()
        else:
            self.parallel = False

    def parse_input(self, input_file):
        """
        Given an input_file, we parse it into a dictionary. Defines many class attributes and creates
        instances of Beam, Lattice, DF_Tracker, CSR_integration, and CSR_computation.
        Parameters:
            input_file: configuration YAML file data, assumed to be either type string indicating
                        file path or type stream
        """

        # Convert yaml data into dict
        input = parse_yaml(input_file)

        # Make sure that our dictionary has the necessary elements for a configuration
        self.check_input_consistency(input)
        self.input = input

        # Create Beam and Lattice instances (input required)
        self.beam = Beam(input['input_beam'])
        self.lattice = Lattice(input['input_lattice'])

        # Creates DF_tracker, Integration_params, and CSR_params instances (input not required)
        if 'particle_deposition' in input:
            self.DF_tracker = DF_tracker(input['particle_deposition'])
        else:
            self.DF_tracker = DF_tracker()

        if 'CSR_integration' in input:
            self.integration_params = Integration_params(input['CSR_integration'])
        else:
            self.integration_params = Integration_params()

        if 'CSR_computation' in input:
            self.CSR_params = CSR_params(input['CSR_computation'])
        else:
            self.CSR_params = CSR_params()

    def initialization(self):
        """
        'Deposit' the initial beam. This consists of creating the initial DF (attribute already initialized),
        logging the DF, and initializing the statistics dictionary.
        :return:
        """

        # Give the DF_tracker the inital beam distribution so that it computes initial beam characteristics
        self.DF_tracker.get_DF(x=self.beam.x, z=self.beam.z, px=self.beam.px, t=self.beam.position)

        # Log initial beam characteristics to the DF_tracker log
        self.DF_tracker.append_DF()
        self.DF_tracker.append_interpolant(formation_length=float('inf'),
                                           n_formation_length=self.integration_params.n_formation_length)

        #TODO: add more flexible unit conversion, for both charge and energy
        self.CSR_scaling = 8.98755e3 * self.beam.charge # charge in C (8.98755e-6 MeV/m for 1nC/m^2)
        self.init_statistics()

    def init_statistics(self):
        """
        Initialize the statitics dictionary, a class attribute that stores the twiss of the beam at each
        step (also a dictionary) and other various statistics. All statistics are a np array which have an
        element for each 'step' in the lattice.
        """

        # The number of steps in the lattice
        Nstep = self.lattice.total_steps

        # Preallocate size based upon Nstep of np arrays for speed
        self.statistics = {}
        self.statistics['twiss'] = {'alpha_x': np.zeros(Nstep),
                                    'beta_x': np.zeros(Nstep),
                                    'gamma_x': np.zeros(Nstep),
                                    'emit_x': np.zeros(Nstep),
                                    'eta_x': np.zeros(Nstep),
                                    'etap_x': np.zeros(Nstep),
                                    'norm_emit_x': np.zeros(Nstep),
                                    'alpha_y': np.zeros(Nstep),
                                    'beta_y': np.zeros(Nstep),
                                    'gamma_y': np.zeros(Nstep),
                                    'emit_y': np.zeros(Nstep),
                                    'eta_y': np.zeros(Nstep),
                                    'etap_y': np.zeros(Nstep),
                                    'norm_emit_y': np.zeros(Nstep)}

        self.statistics['slope'] = np.zeros((Nstep, 2))
        self.statistics['sigma_x'] = np.zeros(Nstep)
        self.statistics['sigma_z'] = np.zeros(Nstep)
        self.statistics['sigma_energy'] = np.zeros(Nstep)
        self.statistics['mean_x']  = np.zeros(Nstep)
        self.statistics['mean_z'] = np.zeros(Nstep)
        self.statistics['mean_energy'] = np.zeros(Nstep)

        # Populate first step of statistics with the initial beam stats
        self.update_statistics(step = 0)

        # ??? Why are these here?
        self.inbend = False
        self.afterbend = False
        self.R_rec = None
        self.phi_rec = None

    def init_MPI(self):
        """
        Set up for parallel processing
        """
        self.parallel = True
        comm = MPI.COMM_WORLD
        self.rank = comm.Get_rank()
        mpi_size = comm.Get_size()
        work_size = self.CSR_params.xbins * self.CSR_params.zbins
        ave, res = divmod(work_size, mpi_size)
        self.count = [ave + 1 if p < res else ave for p in range(mpi_size)]
        displ = [sum(self.count[:p]) for p in range(mpi_size)]
        self.displ = np.array(displ)

    def check_input_consistency(self, input):
        """
        Checks to make sure that the dictionary we are using for our configuration has the correct format
        Parameters:
            input: the dictionary in question
        Returns:
            returns nothing if the dictionary has the correct format, if not asserts what is wrong
        """
        # TODO: need modification if dipole_config.yaml format changed
        # Must be given a beam and a lattice
        self.required_inputs = ['input_beam', 'input_lattice']

        allowed_params = self.required_inputs + ['particle_deposition', 'distribution_interpolation', 'CSR_integration',
                                                 'CSR_computation']

        # Make sure all keys in input are allowed
        for input_param in input:
            assert input_param in allowed_params, f'Incorrect param given to {self.__class__.__name__}.__init__(**kwargs): {input_param}\nAllowed params: {allowed_params}'

        # Make sure all required parameters are specified
        for req in self.required_inputs:
            assert req in input, f'Required input parameter {req} to {self.__class__.__name__}.__init__(**kwargs) was not found.'

    def get_formation_length(self, R, sigma_z, phi = 0.0, inbend=True):
        """
        Computes the formation length of the nominal path
        Parameters:
            sigma_z: RMS beam size (in x direction)
            R: distance from observation point to retarded point of emittance
            phi: radius of the beam from origin (only used if in curve)
            inbend: boolean, indicates if nominal path is bending
        Returns:
            Nothing... TODO change this?

        See Fig 18.1 in Classical Mechanics and Electromagnetism
        in Accelerator Physics for reference
        """
        if inbend:
            self.formation_length = (24 * (R ** 2) * sigma_z) ** (1 / 3)
        else:
            self.formation_length = (3*R**2*phi**4)/(4*(-6*sigma_z + R*phi**3))

    def get_bmadx_element(self, ele, DL, entrance = False, exit = False):
        """
        Creates a bmadx_element object from the given element parameters
        Parameters:
            ele: dictionary, a lattice element
            DL: distance of the element (if many steps in an element this may be smaller than the entire element)
            entrance, exit: booleans, if the current slice of the lattice element contains an entrance or exit
                            relevant only for dipoles
        Returns:
            element: bmadx_element object
        """
        L = self.lattice.lattice_config[ele]['L']
        type = self.lattice.lattice_config[ele]['type']

        # For each element type bmadx has a different class
        if type == 'dipole':
            # The degree through which the dipole curves the nominal trajectory
            angle = self.lattice.lattice_config[ele]['angle']

            # E1 and E2 are the entrance and exit angles respectively
            E1 = self.lattice.lattice_config[ele]['E1']
            E2 = self.lattice.lattice_config[ele]['E2']

            # Radius of curvature
            G = angle/L

            # If the dipole element we want to create includes an entrance and/or exit it is constructed slightly differently
            if entrance and exit:
                element = SBend(L = DL, P0C = self.beam.init_energy, G = G, E1 = E1, E2 = E2)

            elif entrance:
                element = SBend(L=DL, P0C=self.beam.init_energy, G=G, E1=E1, E2=0.0)

            elif exit:
                element = SBend(L=DL, P0C=self.beam.init_energy, G=G, E1=0.0, E2=E2)

            else:
                element = SBend(L=DL, P0C=self.beam.init_energy, G=G, E1=0.0, E2=0.0)

        elif type == 'drift':
            element = Drift(L = DL)

        elif type == 'quad':
            K1 = self.lattice.lattice_config[ele]['strength']
            element = Quadrupole(L=DL, K1=K1)

        elif type == 'sextupole':
            K2 = self.lattice.lattice_config[ele]['strength']
            element = Sextupole(L=DL, K2=K2)

        return element

#    @profile
    def run(self, stop_time = None, debug = False):
        """
        Computes the CSR wake using the configurations setup in __init__
        """

        if (not self.parallel) or (self.rank == 0):
            print('Starting the DFCSR run')

        DL = self.lattice.step_size

        # Define numerous loop variables:
        step_count = 1          # current step, note that it starts at 1
        ele_count = 0           # current element
        skip_ele = False        # skip current element - occurs when entering a new element sometimes
        self.inbend = False     # does the current element have a bend in the reference trajectory?
        self.afterbend = False  # did the previous element have a bend in the reference trajectory?
        self.formation_length = 0.0 # initialize formation length

        # Compute the CSR wake in each element of the lattice, ele is a dictionary representing a lattice element
        for ele in list(self.lattice.lattice_config.keys())[1:]:

            # Update the lattice with the current element
            self.lattice.update(ele)

            # TODO: add sextupole, maybe Bmad Tracking?
            # ---------------load current lattice paramaters--------------- #
            # Pre-process the lattice params
            L = self.lattice.lattice_config[ele]['L']           # length of the lattice element
            type = self.lattice.lattice_config[ele]['type']     # type of lattice element
            steps = self.lattice.steps_per_element[ele_count]   # number of steps in this lattice element
            R = float('inf')                                    # radius of curvature of nominal trajectory

            # When the current step is not in the same element as the previous, we must propagate the beam through the
            # remainder of the previous element
            if (not skip_ele) and ele_count > 0:
                # The distance that remains in the previous element
                DL_1 = self.lattice.distance[ele_count - 1] - self.beam.position

                #TODO: Bmadx seems to have some problems when DL is very small
                if DL_1 > 1.0e-6:
                    # Create bmadx element object for the previous element
                    element = self.get_bmadx_element(ele=ele_prev, type=type_prev, DL=DL_1, exit=True)

                    # Propagate
                    self.beam.track(element, DL_1, update_step=False)

                else:
                    DL_1 = 0.0

            # If there are no steps inside an element (one step over the whole element),
            # we propagate the beam through it all at once
            if steps == 0:
                skip_ele = True
                element = self.get_bmadx_element(ele=ele, type=type, DL=L, exit=True, entrance = True)
                self.beam.track(element, L, update_step=False)

            # ---------------initialize lattice element characteristic variables--------------- #
            #  Dipole case must be handled separately becuase of bend
            if type == 'dipole':
                angle = self.lattice.lattice_config[ele]['angle']
                R = L / angle # In dipole this is finite

                self.inbend = True
                self.afterbend = True

                self.R_rec = R
                self.phi_rec = angle

                self.get_formation_length(R=R, sigma_z=5*self.beam.sigma_z, inbend = True)

            else:
                self.inbend = False

                if self.afterbend:
                    # TODO: Verify the formation length in the drift
                    #self.get_formation_length(R=self.R_rec, sigma_z=5*self.beam.sigma_z, phi = self.phi_rec, inbend=False)
                    self.get_formation_length(R=self.R_rec, sigma_z=5 * self.beam.sigma_z, inbend=True)

                else:  # if it is the first drift in the lattice
                    self.formation_length += L

            # ---------------CSR wake tracking--------------- #
            distance_in_current_ele = 0.0

            # loop through all steps in the current element
            for step in range(steps):
                time0  = time.time()

                # _______________Propagate the beam through the slice of the current element_______________ #
                # For the first step we must propagate the beam only through the current element
                # At this point, the beam has already been propagated through the entirety of the previous element
                if (step == 0) and (ele_count > 0):
                    # Compute the length of the remainder of the step
                    DL_2 = self.lattice._positions_record[step_count] - self.lattice.distance[ele_count - 1]

                    # Propagate the beam through the first step
                    element = self.get_bmadx_element(ele = ele, DL = DL_2, entrance = True)
                    self.beam.track(element, DL_2)
                    distance_in_current_ele += DL_2

                    # Reset the flag
                    skip_ele = False

                # For all steps after the first
                else:
                    # Create bmadx element and propagate the beam through it for one step
                    element = self.get_bmadx_element(ele = ele, DL = DL)
                    self.beam.track(element, DL)

                    # Record distance travelled
                    distance_in_current_ele += DL

                # _______________Compute CSR Wakefield_______________ #
                if debug or self.CSR_params.compute_CSR:
                    # Input what what our beam looks like currently to get the density functions
                    self.DF_tracker.get_DF(x=self.beam.x, z=self.beam.z, px=self.beam.px, t=self.beam.position)

                    # Append the density functions to the log
                    self.DF_tracker.append_DF()

                    # Append 3D matrix for interpolation with the new DFs by interpolation
                    #self.get_formation_length(R=R, sigma_z=self.beam.sigma_z)
                    self.DF_tracker.append_interpolant(formation_length=self.formation_length,
                                                       n_formation_length=self.integration_params.n_formation_length)

                    # Build interpolant based on the 3D matrix
                    self.DF_tracker.build_interpolant()

                # If beam is in an after-bend drift and away from the previous bend for more than n*formation_length, stop calculating wakes
                #TODO: formation length not correct here
                #if  self.afterbend and (not self.inbend) and distance_in_current_ele > 3*self.formation_length:
                #    CSR_blocker = True
                #    if (not self.parallel) or (self.rank == 0):
                #        print("Far away from a bending magnet, stopping calculating CSR")

                #else:
                #    CSR_blocker = False
                CSR_blocker = False

                # Some debugging checks
                if self.CSR_params.compute_CSR and (not CSR_blocker):

                    # If we should compute the CSR for this step (some elements require less precision and we can skip
                    # every nsep[ele_count] steps)
                    if step % self.lattice.nsep[ele_count] == 0:

                        # Calculate CSR mesh given beam shape
                        self.get_CSR_mesh()

                        # Calculate CSR on the mesh
                        if self.parallel:
                            self.calculate_2D_CSR_parallel()
                        else:
                            self.calculate_2D_CSR()

                        # Apply CSR kick to the beam
                        if self.CSR_params.apply_CSR:
                            self.beam.apply_wakes(self.dE_dct, self.x_kick,
                                              self.CSR_xrange_transformed, self.CSR_zrange, DL*self.lattice.nsep[ele_count],
                                                  self.CSR_params.transverse_on)

                        # Record data to file if requested
                        if (self.CSR_params.write_beam == 'all' or
                                (isinstance(self.CSR_params.write_beam, list) and (step_count in self.CSR_params.write_beam))):
                            self.dump_beam(label = step_count)

                        if self.CSR_params.write_wakes:
                            self.write_wakes()

                # Record statistics at each step
                self.update_statistics(step = step_count)

                if not self.parallel or self.rank == 0:
                    print("Finish step {}, s = {},  in {} seconds".format(step_count, self.beam.position, time.time() - time0))

                step_count += 1

                if stop_time and self.beam.position > stop_time:
                    return

            # Redefine some loop variables for the next iteration
            ele_prev = ele
            type_prev = type

            ele_count += 1

        # Record data to file
        self.dump_beam(label='end')
        self.write_statistics()


    def get_CSR_mesh(self):
        """
        Calculates the mesh of observation points. Achieves this by linearly transfering the particle distribution
        to be virtical, creating a mesh on the transformed distribution, and the reversing the original transformation.
        (xmesh, zmesh) TWO 1D arrays representiong (x, z) coordinates on a linear transformed mesh
        """
        # Remove any xz tilt (chrip) from the dataset, (rotates the distirbution function to be virtical in x direction)
        x_transform = self.beam.x_transform

        # The x/z slant of the beam distribution function
        p = self.beam.slope

        # Compute the new std and mean of the transformed beam
        sig_x = np.std(x_transform)
        mean_x = np.mean(x_transform)
        sig_z = self.beam.sigma_z
        mean_z = self.beam.mean_z

        # The number of std away from the mean to make the mesh
        xlim = self.CSR_params.xlim
        zlim = self.CSR_params.zlim
        xbins = self.CSR_params.xbins
        zbins = self.CSR_params.zbins

        # Create 2 1D arrays which define the mesh boundaries
        zrange = np.linspace(mean_z - zlim * sig_z, mean_z + zlim * sig_z, zbins)
        xrange = np.linspace(mean_x - xlim * sig_x, mean_x + xlim * sig_x, xbins)

        # TODO: check the order
        # Create mesh
        xmesh_transform, zmesh = np.meshgrid(xrange, zrange, indexing='ij')

        # Make 2D arrays 1D. Once we reverse the transformation we won't really have a clear
        # "left, right, up, down" view of the mesh from the indexing anymore
        xmesh_transform = xmesh_transform.flatten()
        zmesh = zmesh.flatten()

        # Reverse the transformation made
        xmesh = xmesh_transform +  np.polyval(p, zmesh)

        # Assign our computed variables to their respective class attribute
        self.CSR_xmesh = xmesh
        self.CSR_zmesh = zmesh
        self.CSR_zrange = zrange
        self.CSR_xrange_transformed = xrange

#    @profile
    def calculate_2D_CSR(self):
        """
        Computes the energy change (dE_kick) and the momentum change (x_kick) for each mesh element on the
        CSR mesh.
        """
        # The total number of mesh elements
        N = self.CSR_params.xbins*self.CSR_params.zbins

        # Initialize the energy and momenta change arrays
        self.dE_dct = np.zeros((N,))
        self.x_kick = np.zeros((N,))

        # Record the time for benchmarking
        start_time = time.time()

        # Loop over all mesh elements
        for i in range(N):
            # Get the s and x position of the mesh element relative to the nominal path
            s = self.beam.position + self.CSR_zmesh[i]
            x = self.CSR_xmesh[i]

            # Compute the CSR_wake for the element's position
            self.dE_dct[i], self.x_kick[i] = self.get_CSR_wake(s,x)

        # Reshape csr arrarys to be 2D
        self.dE_dct = self.dE_dct.reshape((self.CSR_params.xbins, self.CSR_params.zbins))
        self.x_kick = self.x_kick.reshape((self.CSR_params.xbins, self.CSR_params.zbins))

    def calculate_2D_CSR_parallel(self):
        """
        Uses parallel processing to do what the calculate_2D_CSR method above does
        """
        work_size= self.CSR_params.xbins * self.CSR_params.zbins
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        start = int(self.displ[rank])
        local_size = int(self.count[rank])

        self.dE_dct = np.zeros((work_size,))
        self.x_kick = np.zeros((work_size,))

        dE_dct_local = np.zeros((local_size,))
        x_kick_local = np.zeros(local_size, )

        start_time = time.time()
        for i in range(local_size):
            k  = start + i
            # if i == 210:
            #    print(i)

            # if i%int(N//10) == 0:
            #    print('Complete', str(np.round(i/N*100,2)), '%')

            s = self.beam.position + self.CSR_zmesh[k]
            x = self.CSR_xmesh[k]

            dE_dct_local[i], x_kick_local[i] = self.get_CSR_wake(s,x)

        comm.Allgatherv(dE_dct_local, [self.dE_dct, self.count, self.displ, MPI.DOUBLE])
        comm.Allgatherv(x_kick_local, [self.x_kick, self.count, self.displ, MPI.DOUBLE])

        self.dE_dct = self.dE_dct.reshape((self.CSR_params.xbins, self.CSR_params.zbins))
        self.x_kick = self.x_kick.reshape((self.CSR_params.xbins, self.CSR_params.zbins))

#    @profile
    def get_CSR_wake(self, s, x, debug = False):
        """
        Computes the CSR wake at a specific position
        Parameters:
            s, x: position coordinates
        """
        # The current beam position
        t = self.beam.position

        # std of beam distribution in x and z
        sigma_z = self.beam._sigma_z
        sigma_x = self.beam._sigma_x

        # The slope of the beam's distribution x/z
        tan_theta = self.beam._slope[0]

        # The virtical x distance from the point to the center line of the distribution
        x0 = (s-t)*self.beam._slope[0]

        # Average x position of the DF
        xmean = self.beam._mean_x

        ######### For Debug ##########################################################
        if np.abs(tan_theta) <= 1:  # if theta <45 degre, the chirp band can be ignored. theta is the angle in z-x plane
            ignore_vx = False
        else:
            ignore_vx = False
        ############################################################################

        chirp_band = False

        # If the x/z distribution is slanted no more than 45 degree from virtical, ignore chirp band
        # This is the result of isosceles triangle geometry... in this case we need to integrate over fewer regions
        if np.abs(tan_theta) <= 1:
            s2 = s - 500 * sigma_z
            s3 = s - 20*sigma_z
            s4 = s + 5 * sigma_z
            x1_w = x0 - 20 * sigma_x
            x2_w = x0 + 20 * sigma_x

            x1_n = x0 - 10 * sigma_x
            x2_n = x0 + 10 * sigma_x

        # If there is a chrip band then there are more integration regions we have to consider
        else:
            chirp_band = True

            # Area 1: accounts for chirp band
            # Area 2: accounts for region around singularity
            # Area 3 and 4: accounts for main band

            # Construct the integration regions in a way that depends on  which way the band is tiled
            if tan_theta > 0:
                tan_alpha = -2 * tan_theta / (1 - tan_theta ** 2)  # alpha = pi - 2 theta, tan_alpha > 0
                d = (10 * sigma_x + xmean - x) / tan_alpha

                s4 = s + 3 * sigma_z
                s3 = np.max((0, s - d))
                s2 = s3 - 200 * sigma_z

                # area 1
                x1_l = x + 0.1 * sigma_x
                x1_r = x + 10 * sigma_x

                # area 2
                x2_l = x - 3 * sigma_x
                x2_r = x1_l

                # area 3
                x3_l = x0 - 5 * sigma_x
                x3_r = x0 + 5 * sigma_x

                x4_l = x0 - 20 * sigma_x
                x4_r = x0 + 20 * sigma_x

            else:
                tan_alpha = 2 * tan_theta / (1 - tan_theta ** 2)
                d = -(xmean - x - 10 * sigma_x) / tan_alpha

                s4 = s + 3 * sigma_z
                s3 = np.max((0, s - d))
                s2 = s3 - 200 * sigma_z

                # area 1
                x1_l = x - 10 * sigma_x
                x1_r = x - 1 * sigma_x

                # area 2
                x2_l = x1_r
                x2_r = x + 3 *sigma_x

                # area 3
                x3_l = x0 - 5 * sigma_x
                x3_r = x0 + 5 * sigma_x

                x4_l = x0 - 20 * sigma_x
                x4_r = x0 + 20 * sigma_x

        # Make sure that we cap the smallest intergration bound at s=0, where the nominal path begins
        s1 = np.max((0, s2 - self.integration_params.n_formation_length * self.formation_length))

        # ------------------ Perform integration over all areas ------------------ #
        if chirp_band:
            # Initalize the integration meshes with the desired number of bins
            sp1 = np.linspace(s1, s2, self.integration_params.zbins)
            sp2 = np.linspace(s2, s3, self.integration_params.zbins)
            sp3 = np.linspace(s3, s4, self.integration_params.zbins)
            xp1 = np.linspace(x1_l, x1_r, self.integration_params.xbins)
            xp2 = np.linspace(x2_l, x2_r, self.integration_params.xbins)
            xp3 = np.linspace(x3_l, x3_r, self.integration_params.xbins)
            xp4 = np.linspace(x4_l, x4_r, 2*self.integration_params.xbins)

            [xp_mesh1, sp_mesh1] = np.meshgrid(xp4, sp1, indexing='ij')
            [xp_mesh2, sp_mesh2] = np.meshgrid(xp3, sp2, indexing = 'ij')
            [xp_mesh3, sp_mesh3] = np.meshgrid(xp1, sp3, indexing='ij')
            [xp_mesh4, sp_mesh4] = np.meshgrid(xp2, sp3, indexing='ij')

            # Compute integrands using trapedzoid integration
            CSR_integrand_z1, CSR_integrand_x1 = self.get_CSR_integrand(s=s, t=t, x=x, xp=xp_mesh1, sp=sp_mesh1, ignore_vx = ignore_vx)
            dE_dct1 = -self.CSR_scaling * np.trapz(y=np.trapz(y=CSR_integrand_z1, x=xp4, axis=0), x=sp1)
            x_kick1 = self.CSR_scaling * np.trapz(y=np.trapz(y=CSR_integrand_x1, x=xp4, axis=0), x=sp1)

            CSR_integrand_z2, CSR_integrand_x2 = self.get_CSR_integrand(s=s, t=t, x=x, xp=xp_mesh2, sp=sp_mesh2, ignore_vx = ignore_vx)
            dE_dct2 = -self.CSR_scaling * np.trapz(y=np.trapz(y=CSR_integrand_z2, x=xp3, axis=0), x=sp2)
            x_kick2 = self.CSR_scaling * np.trapz(y=np.trapz(y=CSR_integrand_x2, x=xp3, axis=0), x=sp2)

            CSR_integrand_z3, CSR_integrand_x3 = self.get_CSR_integrand(s=s, t=t, x=x, xp=xp_mesh3, sp=sp_mesh3, ignore_vx = ignore_vx)
            dE_dct3 = -self.CSR_scaling * np.trapz(y=np.trapz(y=CSR_integrand_z3, x=xp1, axis=0), x=sp3)
            x_kick3 = self.CSR_scaling * np.trapz(y=np.trapz(y=CSR_integrand_x3, x=xp1, axis=0), x=sp3)

            CSR_integrand_z4, CSR_integrand_x4 = self.get_CSR_integrand(s=s, t=t, x=x, xp=xp_mesh4, sp=sp_mesh4, ignore_vx = ignore_vx)
            dE_dct4 = -self.CSR_scaling * np.trapz(y=np.trapz(y=CSR_integrand_z4, x=xp2, axis=0), x=sp3)
            x_kick4 = self.CSR_scaling * np.trapz(y=np.trapz(y=CSR_integrand_x4, x=xp2, axis=0), x=sp3)

            if debug:
                return xp1, xp2, xp3, xp4, sp1, sp2, sp3,  CSR_integrand_z1, CSR_integrand_x1, CSR_integrand_z2, CSR_integrand_x2, CSR_integrand_z3, CSR_integrand_x3, CSR_integrand_z4, CSR_integrand_x4
            else:
                return dE_dct1 + dE_dct2 + dE_dct3 + dE_dct4, x_kick1 + x_kick2 + x_kick3 + x_kick4

        else:
            # Initalize the integration meshes with the desired number of bins
            sp1 = np.linspace(s1, s2, self.integration_params.zbins)
            sp2 = np.linspace(s2, s3, self.integration_params.zbins)
            sp3 = np.linspace(s3, s4, self.integration_params.zbins)
            xp_w = np.linspace(x1_w, x2_w, 2*self.integration_params.xbins)
            xp_n = np.linspace(x1_n, x2_n, self.integration_params.xbins)

            [xp_mesh1, sp_mesh1] = np.meshgrid(xp_w, sp1, indexing='ij')
            [xp_mesh2, sp_mesh2] = np.meshgrid(xp_n, sp2, indexing='ij')
            [xp_mesh3, sp_mesh3] = np.meshgrid(xp_n, sp3, indexing='ij')

            # Compute integrands using trapedzoid integration
            CSR_integrand_z1, CSR_integrand_x1 = self.get_CSR_integrand(s=s, t=t, x=x, xp=xp_mesh1, sp=sp_mesh1, ignore_vx = ignore_vx)
            dE_dct1 = -self.CSR_scaling * np.trapz(y=np.trapz(y=CSR_integrand_z1, x=xp_w, axis=0), x=sp1)
            x_kick1 = self.CSR_scaling * np.trapz(y=np.trapz(y=CSR_integrand_x1, x=xp_w, axis=0), x=sp1)

            CSR_integrand_z2, CSR_integrand_x2 = self.get_CSR_integrand(s=s, t=t, x=x, xp=xp_mesh2, sp=sp_mesh2, ignore_vx = ignore_vx)
            dE_dct2 = -self.CSR_scaling * np.trapz(y=np.trapz(y=CSR_integrand_z2, x=xp_n, axis=0), x=sp2)
            x_kick2 = self.CSR_scaling * np.trapz(y=np.trapz(y=CSR_integrand_x2, x=xp_n, axis=0), x=sp2)

            CSR_integrand_z3, CSR_integrand_x3 = self.get_CSR_integrand(s=s, t=t, x=x, xp=xp_mesh3, sp=sp_mesh3, ignore_vx = ignore_vx)
            dE_dct3 = -self.CSR_scaling * np.trapz(y=np.trapz(y=CSR_integrand_z3, x=xp_n, axis=0), x=sp3)
            x_kick3 = self.CSR_scaling * np.trapz(y=np.trapz(y=CSR_integrand_x3, x=xp_n, axis=0), x=sp3)

            if debug:
                return xp_w, xp_n,  sp1, sp2, sp3, CSR_integrand_z1, CSR_integrand_x1,CSR_integrand_z2, CSR_integrand_x2,CSR_integrand_z3, CSR_integrand_x3
            else:
                return dE_dct1 + dE_dct2 + dE_dct3, x_kick1 + x_kick2 + x_kick3

    def get_CSR_integrand(self, s, x, t, sp, xp, ignore_vx = False):
        """
        Computes the integrand function for a specific point in the distribution function over the inputed integration area
        Parameters:
            s, x: the coordinate of the point for which to compute the integrand
            t: the current time
            sp, xp: the mesh grid of the integration
        """
        # Interpolate the velocity of the beam at this point
        vx = interpolate3D(xval=np.array([t]), yval=np.array([x]), zval=np.array([s-t]),
                             data=self.DF_tracker.data_vx_interp,
                             min_x=self.DF_tracker.min_x, min_y=self.DF_tracker.min_y,
                             min_z=self.DF_tracker.min_z,
                             delta_x=self.DF_tracker.delta_x, delta_y=self.DF_tracker.delta_y,
                             delta_z=self.DF_tracker.delta_z)[0]

        # Flatten the meshgrid
        sp_flat = sp.ravel()
        xp_flat = xp.ravel()

        # Get x & x prime and y & y prime in the lab frame
        # the 'prime' variables are the integration mesh grid converted to lab frame coordinates
        X0_s = interpolate1D(xval = np.array([s]), data = self.lattice.coords[:, 0], min_x = self.lattice.min_x,
                             delta_x = self.lattice.delta_x)[0]
        X0_sp = interpolate1D(xval = sp_flat, data = self.lattice.coords[:, 0], min_x = self.lattice.min_x,
                              delta_x = self.lattice.delta_x)
        Y0_s = interpolate1D(xval = np.array([s]), data = self.lattice.coords[:, 1], min_x = self.lattice.min_x,
                             delta_x = self.lattice.delta_x)[0]
        Y0_sp = interpolate1D(xval = sp_flat, data = self.lattice.coords[:, 1], min_x = self.lattice.min_x,
                              delta_x = self.lattice.delta_x)

        # Do the same for the normal and tangential vectors
        n_vec_s_x = interpolate1D(xval = np.array([s]), data = self.lattice.n_vec[:, 0], min_x = self.lattice.min_x,
                                  delta_x = self.lattice.delta_x)[0]
        n_vec_sp_x =interpolate1D(xval = sp_flat, data = self.lattice.n_vec[:, 0], min_x = self.lattice.min_x,
                                  delta_x = self.lattice.delta_x)
        n_vec_s_y = interpolate1D(xval=np.array([s]), data=self.lattice.n_vec[:, 1], min_x=self.lattice.min_x,
                                  delta_x=self.lattice.delta_x)[0]
        n_vec_sp_y = interpolate1D(xval=sp_flat, data=self.lattice.n_vec[:, 1], min_x=self.lattice.min_x,
                                   delta_x=self.lattice.delta_x)
        tau_vec_s_x = interpolate1D(xval=np.array([s]), data=self.lattice.tau_vec[:, 0], min_x=self.lattice.min_x,
                                  delta_x=self.lattice.delta_x)[0]
        tau_vec_sp_x = interpolate1D(xval=sp_flat, data=self.lattice.tau_vec[:, 0], min_x=self.lattice.min_x,
                                   delta_x=self.lattice.delta_x)
        tau_vec_s_y = interpolate1D(xval=np.array([s]), data=self.lattice.tau_vec[:, 1], min_x=self.lattice.min_x,
                                  delta_x=self.lattice.delta_x)[0]
        tau_vec_sp_y = interpolate1D(xval=sp_flat, data=self.lattice.tau_vec[:, 1], min_x=self.lattice.min_x,
                                   delta_x=self.lattice.delta_x)

        # Comupte the magnitude of the different between r and rp (r prime) vectors
        r_minus_rp_x = X0_s - X0_sp + x * n_vec_s_x - xp_flat * n_vec_sp_x
        r_minus_rp_y = Y0_s - Y0_sp + x * n_vec_s_y - xp_flat * n_vec_sp_y
        r_minus_rp = np.sqrt(r_minus_rp_x**2 + r_minus_rp_y**2)

        # The effective radius from the lab frame origin for each lattice step
        rho_sp = np.zeros(sp_flat.shape)

        # Populate rho_sp
        for count in range(self.lattice.Nelement):
            if count == 0:
                rho_sp[sp_flat < self.lattice.distance[count]] = self.lattice.rho[count]
            else:
                rho_sp[(sp_flat < self.lattice.distance[count]) & (sp_flat >= self.lattice.distance[count - 1])] = self.lattice.rho[count]

        # Retarded time array for each point in meshgrid
        t_ret = t - r_minus_rp

        # Compute various vector and scalar valued functions inside the integrand
        density_ret = interpolate3D(xval = t_ret, yval = xp_flat, zval = sp_flat - t_ret,
                                  data = self.DF_tracker.data_density_interp,
                                  min_x = self.DF_tracker.min_x, min_y = self.DF_tracker.min_y,  min_z = self.DF_tracker.min_z,
                                  delta_x = self.DF_tracker.delta_x, delta_y = self.DF_tracker.delta_y, delta_z = self.DF_tracker.delta_z)

        density_x_ret = interpolate3D(xval=t_ret, yval=xp_flat, zval=sp_flat - t_ret,
                                  data=self.DF_tracker.data_density_x_interp,
                                  min_x=self.DF_tracker.min_x, min_y=self.DF_tracker.min_y, min_z=self.DF_tracker.min_z,
                                  delta_x=self.DF_tracker.delta_x, delta_y=self.DF_tracker.delta_y,
                                  delta_z=self.DF_tracker.delta_z)

        density_z_ret = interpolate3D(xval=t_ret, yval=xp_flat, zval=sp_flat - t_ret,
                                    data=self.DF_tracker.data_density_z_interp,
                                    min_x=self.DF_tracker.min_x, min_y=self.DF_tracker.min_y,
                                    min_z=self.DF_tracker.min_z,
                                    delta_x=self.DF_tracker.delta_x, delta_y=self.DF_tracker.delta_y,
                                    delta_z=self.DF_tracker.delta_z)

        vx_ret = interpolate3D(xval=t_ret, yval=xp_flat, zval=sp_flat - t_ret,
                                    data=self.DF_tracker.data_vx_interp,
                                    min_x=self.DF_tracker.min_x, min_y=self.DF_tracker.min_y,
                                    min_z=self.DF_tracker.min_z,
                                    delta_x=self.DF_tracker.delta_x, delta_y=self.DF_tracker.delta_y,
                                    delta_z=self.DF_tracker.delta_z)

        vx_x_ret = interpolate3D(xval=t_ret, yval=xp_flat, zval=sp_flat - t_ret,
                             data=self.DF_tracker.data_vx_x_interp,
                             min_x=self.DF_tracker.min_x, min_y=self.DF_tracker.min_y,
                             min_z=self.DF_tracker.min_z,
                             delta_x=self.DF_tracker.delta_x, delta_y=self.DF_tracker.delta_y,
                             delta_z=self.DF_tracker.delta_z)

        ## TODO: More accurate vx, maybe add vs
        vs = 1
        vs_ret = 1
        vs_s_ret = 0
        vx_t = 0
        vs_t = 0
        #vx = 0
        #vx_x_ret = 0
        #vx_ret = 0

        if ignore_vx:
            vx = 0
            vx_x_ret = 0
            vx_ret = 0

        # Accounts for transfer to the lab frame
        scale_term =  1 + xp_flat*rho_sp

        # Compute velocity in the lab frame for current time and retarded
        velocity_x = vs * tau_vec_s_x + vx * n_vec_s_x
        velocity_y = vs * tau_vec_s_y + vx * n_vec_s_y
        velocity_ret_x = vs_ret * tau_vec_sp_x + vx_ret * n_vec_sp_x
        velocity_ret_y = vs_ret * tau_vec_sp_y + vx_ret * n_vec_sp_y

        #velocity_partial_t_x = vs_t * tau_vec_sp_x + vx_t * n_vec_sp_x
        #velocity_partial_t_y = vs_t * tau_vec_sp_y + vx_t * n_vec_sp_y

        nabla_density_ret_x = density_x_ret  * n_vec_sp_x + density_z_ret / scale_term * tau_vec_sp_x
        nabla_density_ret_y = density_x_ret * n_vec_sp_y + density_z_ret / scale_term * tau_vec_sp_y

        div_velocity = vs_s_ret + vx_x_ret

        # TODO: Consider using general form
        ## general form
        # part1: beta dot beta prime
        part1 = velocity_x * velocity_ret_x + velocity_y * velocity_ret_y

        # Some numerators found in the longitudinal wake integrals
        CSR_numerator1 = scale_term * ((velocity_x - part1 * velocity_ret_x) * nabla_density_ret_x  + \
                          (velocity_y - part1 * velocity_ret_y)*nabla_density_ret_y)
        CSR_numerator2 = -scale_term * part1 * density_ret * div_velocity
        #CSR_numerator3 = scale_term * density_ret * (velocity_partial_t_x * velocity_x + velocity_partial_t_y * velocity_y)

        #CSR_denominator = r_minus_rp

        #self.CSR_integrand = CSR_numerator1/CSR_denominator + (CSR_numerator2 + CSR_numerator3)/CSR_denominator
        CSR_integrand_z = CSR_numerator1 /r_minus_rp + (CSR_numerator2) / r_minus_rp

        n_minus_np_x = n_vec_s_x - n_vec_sp_x
        n_minus_np_y = n_vec_s_y - n_vec_sp_y

        # part1: (r-r')(n - n')
        part1 = r_minus_rp_x * n_minus_np_x + r_minus_rp_y * n_minus_np_y

        #part2: n tau'
        part2 = n_vec_s_x * tau_vec_sp_x + n_vec_s_y * tau_vec_sp_y

        # part3: partial density/partial t_ret
        partial_density = - (velocity_ret_x * nabla_density_ret_x + velocity_ret_y * nabla_density_ret_y) - \
                          density_ret * div_velocity

        # Three integrands for logitudinal wake
        W1 = scale_term * part1 / (r_minus_rp * r_minus_rp * r_minus_rp) * density_ret
        W2 = scale_term * part1 / (r_minus_rp * r_minus_rp) * partial_density
        W3 = -scale_term * part2 / r_minus_rp * partial_density

        CSR_integrand_x = W1 + W2 + W3
        CSR_integrand_x = CSR_integrand_x.reshape(xp.shape)
        CSR_integrand_z = CSR_integrand_z.reshape(xp.shape)

        return CSR_integrand_z, CSR_integrand_x

    def dump_beam(self, label):
        """
        Record beam in the particle_group format
        """
        if self.parallel and self.rank != 0:
            return

        path = full_path(self.CSR_params.workdir)
        filename = os.path.join(path, f'{self.prefix}-particles-{label}.h5')

        if os.path.isfile(filename):
            os.remove(filename)
            print("Existing file " + filename + " deleted.")

        print("Beam at position {} is written to {}".format(self.beam.position, filename))

        self.beam.particle_group.write(filename)

    def write_wakes(self):
        """
        Writes the CSR wake data for the current step to a h5 file
        """
        if self.parallel and self.rank != 0:
            return

        path = full_path(self.CSR_params.workdir)

        filename = os.path.join(path, f'{self.prefix}-wakes.h5')

        if self.beam.step == 1:
            if os.path.isfile(filename):
                os.remove(filename)
                print("Existing file " + filename + " deleted.")
            print("Wakes written to ", filename)

        with h5py.File(filename, 'a') as hf:
            step = self.beam.step
            groupname = 'step_' + str(step)
            g = hf.create_group(groupname)
            g.attrs['step'] = step
            g.attrs['position']  = self.beam.position
            g.attrs['mean_gamma'] = self.beam.init_gamma
            g.attrs['beam_energy'] = self.beam.init_energy
            g.attrs['element'] = self.lattice.current_element
            g.attrs['charge'] = self.beam.charge
            g1 = g.create_group('longitudinal')
            g1.attrs['unit'] = 'MeV/m'
            g1.create_dataset('x_grids', data = self.CSR_xmesh.reshape(self.dE_dct.shape))
            g1.create_dataset('z_grids', data = self.CSR_zmesh.reshape(self.dE_dct.shape))
            g1.create_dataset('dE_dct', data = self.dE_dct)
            g2  = g.create_group('transverse')
            g2.attrs['unit'] = 'MeV/m'
            g2.create_dataset('x_grids', data = self.CSR_xmesh.reshape(self.dE_dct.shape))
            g2.create_dataset('z_grids', data = self.CSR_zmesh.reshape(self.dE_dct.shape))
            g2.create_dataset('xkicks', data = self.x_kick)

#    @profile
    def update_statistics(self, step):
        """
        Updates the statistics dictionary with the current step's beam characteristics
        """
        twiss = self.beam.twiss
        self.statistics['twiss']['alpha_x'][step] = twiss['alpha_x']
        self.statistics['twiss']['beta_x'][step] = twiss['beta_x']
        self.statistics['twiss']['gamma_x'][step] = twiss['gamma_x']
        self.statistics['twiss']['emit_x'][step] = twiss['emit_x']
        self.statistics['twiss']['eta_x'][step] = twiss['eta_x']
        self.statistics['twiss']['etap_x'][step] = twiss['etap_x']
        self.statistics['twiss']['norm_emit_x'][step] = twiss['norm_emit_x']
        self.statistics['twiss']['alpha_y'][step] = twiss['alpha_y']
        self.statistics['twiss']['beta_y'][step] = twiss['beta_y']
        self.statistics['twiss']['gamma_y'][step] = twiss['gamma_y']
        self.statistics['twiss']['emit_y'][step] = twiss['emit_y']
        self.statistics['twiss']['eta_y'][step] = twiss['eta_y']
        self.statistics['twiss']['etap_y'][step] = twiss['etap_y']
        self.statistics['twiss']['norm_emit_y'][step] = twiss['norm_emit_y']
        self.statistics['slope'][step, :] = self.beam._slope
        self.statistics['sigma_x'][step] = self.beam._sigma_x
        self.statistics['sigma_z'][step] = self.beam._sigma_z
        self.statistics['sigma_energy'][step] = self.beam.sigma_energy
        self.statistics['mean_x'][step] = self.beam._mean_x
        self.statistics['mean_z'][step] = self.beam._mean_z
        self.statistics['mean_energy'][step] = self.beam.mean_energy

    def write_statistics(self):
        """
        Writes the statistics dictionary to a h5 file
        """
        if self.parallel and self.rank != 0:
            return

        path = full_path(self.CSR_params.workdir)

        filename = os.path.join(path, f'{self.prefix}-statistics.h5')

        if os.path.isfile(filename):
            os.remove(filename)
            print("Existing file " + filename + " deleted.")
        print("Statistics written to ", filename)

        with h5py.File(filename, 'w') as hf:
            hf.create_dataset(name = 'step_positions', data = self.lattice.steps_record, shape = self.lattice.steps_record.shape)
            hf.create_dataset(name='coords', data=self.lattice.coords)
            hf.create_dataset(name='n_vec', data=self.lattice.n_vec)
            hf.create_dataset(name='tau_vec', data=self.lattice.tau_vec)
            dict2hdf5(hf, self.statistics)
