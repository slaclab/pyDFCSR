# Import standard library modules
import math

# Import third-party modules
import numpy as np
import matplotlib.pyplot as plt
from distgen import Generator
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import linregress
from scipy.optimize import minimize
from skimage.measure import EllipseModel
from bmadx import Particle, track_element
from pmd_beamphysics import ParticleGroup

# Import modules specific to this package
from interfaces import  openpmd_to_bmadx_particles, bmadx_particles_to_openpmd
from physical_constants import MC2
from twiss import twiss_from_bmadx_particles
from histogram_functions import histogram_cic_2d

"""
Module Name: beam.py

Contains the Beam class
"""

class Beam():
    """
    Beam class to initialize, track, and apply wakes
    """
    def __init__(self, input_beam):
        """
        Initalizes instance of Beam using the settings defined in input_beam. Input beam may be in 3 different allowed
        formats. Regardless of the formatt, 3 class attributes are defined: _charge, _init_energy, and particle.
        Parameters:
            input_beam: dictionary of beam settings
        Returns:
            instance of Beam
        """

        # Verify that the input beam has the correct format
        self.check_inputs(input_beam)
        self.input_beam_config = input_beam

        # Indicates how the beam settings are stored
        self.style = input_beam['style']

        # Create a bmadx Particle instance using beam settings
        # There are 3 ways beam settings can be stored
        # 1: from a .dat file path inside the input_beam dictionary
        # 2: from a YAML distgen file path inside the input_beam dictionary
        # 3: from a h5 file in particlegroup format

        if self.style == 'from_file':
            filename = input_beam['beamfile']

            ## Read bmadx coords
            coords = np.loadtxt(filename)
            assert coords.shape[1] == 6, f'Error: input beam must have 6 dimension, but get {coords.shape[1]} instead'

            self._charge = input_beam['charge']
            self._init_energy = input_beam['energy']

            # Keep track of both BmadX particle format (for tracking) and Particle Group format (for calculating twiss).
            self.particle = Particle(*coords.T, 0, self._init_energy, MC2)   #BmadX Particle
            #self.particleGroup = bmadx_particles_to_openpmd(self.particle)  # Particle Group

        elif  self.style == 'distgen':
            filename = input_beam['distgen_input_file']
            # Generates a particle distribution based upon the settings in the distgen_input_file
            gen = Generator(filename)
            gen.run()
            pg = gen.particles
            self._charge = pg['charge']
            self._init_energy = np.mean(pg['energy'])

            self.particle = openpmd_to_bmadx_particles(pg, self._init_energy, 0.0, MC2)   #Bmad X particle

        else:
            ParticleGroup_h5 = input_beam['ParticleGroup_h5']
            pg = ParticleGroup(ParticleGroup_h5)

            self._charge = pg['charge']
            self._init_energy = np.mean(pg['energy'])

            self.particle = openpmd_to_bmadx_particles(pg, self._init_energy, 0.0, MC2)  # Bmad X particle

        # unchanged, initial energy and gamma
        self._init_gamma = self._init_energy/MC2

        # Used during CSR wake computations
        self.position = 0
        self.step = 0

        self.update_status()

    def check_inputs(self, input_beam):
        """
        Checks to make sure that the dictionary we are using for our inital beam settings has the correct format.
        Parameters:
            input_beam: the dictionary in question
        Returns:
            nothing if the dictionary has the correct format, if not asserts what is wrong
        """

        # The input_beam must have a style key, indicating in what format the beam parameters are stored in
        assert 'style' in input_beam, 'ERROR: input_beam must have keyword <style>'

        # The beam parameters can be stored either in another YAML file,
        if input_beam['style'] == 'from_file':
            self.required_inputs = ['style', 'beamfile', 'charge','energy']
        elif input_beam['style'] == 'distgen':
            self.required_inputs = ['style', 'distgen_input_file']
        elif input_beam['style'] == 'ParticleGroup':
            self.required_inputs = ['style', 'ParticleGroup_h5']
        else:
            raise Exception("input beam parsing Error: invalid input style")

        allowed_params = self.required_inputs + ['verbose']
        for input_param in input_beam:
            assert input_param in allowed_params, f'Incorrect param given to {self.__class__.__name__}.__init__(**kwargs): {input_param}\nAllowed params: {allowed_params}'

        # Make sure all required parameters are specified
        for req in self.required_inputs:
            assert req in input_beam, f'Required input parameter {req} to {self.__class__.__name__}.__init__(**kwargs) was not found.'

    def update_status(self):
        """
        Updates the internal status attributes of the object based on the current state of other related attributes
        """
        self._sigma_x = self.sigma_x
        self._sigma_z = self.sigma_z
        self._slope = self.linear_fit.slope
        self._mean_x = self.mean_x
        self._mean_z = self.mean_z

    def track(self, bmadx_elements):
        """
        Propagates the beam from one step to another in the lattice. May pass through multiple elements
        Parameters:
            lattice: Lattice object
            s_init: s value of start of step
            s_final: s value at end of step
        """
        # Track the particle through all bmadx ojects in this step
        for element in bmadx_elements:
            # Use bmadx to move the particle object
            self.particle = track_element(self.particle, element)
            self.position += element.L

        # Update our step counter
        self.step += 1

        # Round the position
        self.position = round(self.position, 10)

        # Updates the internal values
        self.update_status()

    def apply_wakes(self, dE_vals, x_kick_vals, csr_params, t2_csr, R_inv_csr, step_size, p_sd, o_sd):
        """
        Apply the CSR wake to the current position of the beam
        Paramters:
            dE_vals, x_kick_vals: arrays corresponding to the energy and momentum change of each csr mesh element
            CSR_param: dict detailing CSR mesh characteristics
            t2_csr, R_inv_csr: 2 transformation matrices used to construct the CSR mesh
            step_size: the distance between the slices for which CSR is computed
        """
        # Unpack CSR_params
        (transverse_on,
         pbins,
         obins,
         plim,
         olim) = csr_params["transverse_on"], csr_params["pbins"], csr_params["obins"], csr_params["plim"], csr_params["olim"]
        
        # Compute the std wrt to the parallel and orthogonal directions
        #p_sd, o_sd = self.get_std_wrt_linear_fit(tilt_angle)

        # The dimensions of the CSR mesh (non rotated)
        o_dim = np.linspace(-olim*o_sd, olim*o_sd, obins)
        p_dim = np.linspace(-plim*p_sd, plim*p_sd, pbins)

        # Convert energy from J/m to eV/init_energy
        dE_vals = step_size * dE_vals * 1e6 / self.init_energy  # self.energy in eV

        # Create an interpolator that will transfer the CSR wake from the CSR mesh to the beam's particles
        dE_interp = RegularGridInterpolator((o_dim, p_dim), dE_vals, fill_value=0.0, bounds_error=False)

        # Put the beam macro particle positions in the space where the grid is rotated
        particle_positions = (((np.stack((self.z, self.x))).T - t2_csr) @ R_inv_csr.T)

        # Swap the columns of particle positions to match ij indexing
        particle_positions = particle_positions[:, [1, 0]]

        # Apply the interpolator to populate the change in momentum for all particles in the beam
        dE_per_particle = dE_interp(particle_positions)

        # Apply longitudinal kick, note that since the electrons are moving at near the speed of light,
        # change in momentum is roughly equal to change in energy
        pz_new = self.particle.pz + dE_per_particle

        # Use the same process as above to apply the transverse wake
        x_kick_vals = step_size * x_kick_vals * 1e6 / self.init_energy
        x_kick_interp = RegularGridInterpolator((o_dim, p_dim), x_kick_vals, fill_value=0.0, bounds_error=False)
        x_kick_per_particle = x_kick_interp(particle_positions)
        px_new = self.particle.px + x_kick_per_particle

        # Update the particle object with the new energy and momentum values
        self.particle = Particle(self.particle.x, px_new,
                                 self.particle.y, self.particle.py,
                                 self.particle.z, pz_new,
                                 self.particle.s, self.particle.p0c, self.particle.mc2)

        self.update_status()


    def get_tilt_angle2(self):
        """
        Estimates the tilt angle of the beam using Jingyi's idea
        First bins the beam distribution into a histogram, then allocates one point
        per bin with adaquent particle count and does a linear fit.
        """
        
        # Number of bins for historgram is determined by particle number
        num_bins = math.ceil(np.sqrt(len(self.x))/10) # on average, 1 bin per 100 particles

        print(num_bins)
        
        # Create the histogram
        density = histogram_cic_2d(self.z, self.x, np.ones(len(self.x)),  num_bins, self.mean_z-self.sigma_z*3, self.mean_z+self.sigma_z*3, num_bins, self.mean_x-self.sigma_x*3, self.mean_x+self.sigma_x*3)
        print(density)

        # Set values of histogram equal to one when enough points are in the bin
        points = np.argwhere(num_bins > 100) # The coordinates of the points where this is true

        # Now we need to apply transformation from coordinate space to real space
        print(points)

    def get_tilt_angle(self):

        # Fit an ellipse to the data, this is useful to see if the beam is round
        def fit_ellipse(x, y):
            model = EllipseModel()
            data = np.column_stack([x, y])
            model.estimate(data)
            xc, yc, a, b, theta = model.params
            return xc, yc, a, b, theta
        
        xc, yc, a, b, theta = fit_ellipse(self.z, self.x)

        # If the beam is round then we set the tilt angle to zero
        if (a/b > 0.9) and (a/b < 1.10):
            return 0.0
            
        # If the beam is not round then we use a different method to compute tilt angle
        else:
            #return 0.0
            return theta

    def get_tilt_angle3(self):
        p = np.polyfit(self.z, self.x, deg=1)
        return(math.atan(p[0]))
        
    def get_std_wrt_linear_fit(self, theta):
        """
        Finds the standard deviation of the beam wrt to the line of best fit and wrt the line
        orthonormal to the line of best fit.
        Returns:
            parallel_std: standard deviation along the line 
            ortho_std: standard deviation along the orthonormal line
        """
        
        # Center the distribution about the origin
        centered_x = self.x - self.mean_x
        centered_z = self.z - self.mean_z

        # Compute the tilt angle and its sin and cos
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        # Project all points onto the line of best fit
        proj_z_line = (cos_theta**2)*centered_z + (sin_theta*cos_theta)*centered_x
        proj_x_line = (sin_theta*cos_theta)*centered_z + (sin_theta**2)*centered_x

        # Compute the tilt angle of the line or:w
        # othogonal to the best fit line
        ortho_theta = theta + math.pi*0.5
        cos_o_theta = np.cos(ortho_theta)
        sin_o_theta = np.sin(ortho_theta)

        # Project all points onto the line orthonormal to the line of best fit
        proj_z_ortho = (cos_o_theta**2)*centered_z + (sin_o_theta*cos_o_theta)*centered_x
        proj_x_ortho = (sin_o_theta*cos_o_theta)*centered_z + (sin_o_theta**2)*centered_x 

        # Compute standard deviation along the line
        parallel_std = np.std(np.sqrt(proj_x_line**2 + proj_z_line**2) * np.sign(proj_z_line))

        # Compute standard deviation along the orthonormal line
        ortho_std = np.std(np.sqrt(proj_x_ortho**2 + proj_z_ortho**2) * np.sign(proj_z_ortho))
        
        return parallel_std, ortho_std
    
    def scatterplot_beam(self):
        fig, ax = plt.subplots()

        # Plot the beam
        ax.scatter(self.z, self.x, color="red", s=10, label="Beam Distribution")

        # Perform linear regression to find the best fit line
        slope, intercept, r_value, p_value, std_err = self.linear_fit

        # Calculate the best fit line using the slope and intercept
        z_fit = np.array([self.z.min(), self.z.max()])
        x_fit = intercept + slope * z_fit

        # Plot the line of best fit
        ax.plot(z_fit, x_fit, color="black", linewidth=2, label="Line of Best Fit")
        ax.axis("equal")

        plt.show()

    ### Various properties of the beam ###
    @property
    def mean_x(self):
        return np.mean(self.particle.x)

    @property
    def mean_y(self):
        return np.mean(self.particle.y)

    @property
    def sigma_x(self):
        return np.std(self.particle.x)

    @property
    def sigma_z(self):
        return np.std(self.particle.z)

    @property
    def mean_z(self):
        return np.mean(self.particle.z)
    
    @property
    def tilt_angle(self):
        return self.get_tilt_angle()
        
    @property
    def init_energy(self):
        return self._init_energy

    @property
    def init_gamma(self):
        return self._init_gamma

    @property
    def energy(self):
        return (self.particle.pz+1)*self.particle.p0c

    @property
    def mean_energy(self):
        return np.mean(self.energy)

    @property
    def gamma(self):
        return self.energy/MC2

    @property
    def sigma_energy(self):
        return np.std(self.energy)

    @property
    def x(self):
        return self.particle.x

    @property
    def px(self):
        return self.particle.px

    @property
    def z(self):
        return self.particle.z

    @property
    def pz(self):
        return self.particle.pz

    @property
    def linear_fit(self):
        """
        Computers the line of best fit for (x,z) point distribution.
        """
        linear_fit = linregress(self.z, self.x)
        return linear_fit
        
    @property
    def charge(self):
        return self._charge

    @property
    def twiss(self):
        return twiss_from_bmadx_particles(self.particle)

    @property
    def particle_group(self):
        pg = bmadx_particles_to_openpmd(self.particle, self.charge)
        return pg
