import numpy as np
from distgen import Generator
from .physical_constants import MC2
from scipy.interpolate import RegularGridInterpolator
from bmadx import Particle, M_ELECTRON
#from bmadx.pmd_utils import openpmd_to_bmadx_particles, bmadx_particles_to_openpmd
from .interfaces import  openpmd_to_bmadx_particles, bmadx_particles_to_openpmd
from bmadx import track_element
from pmd_beamphysics import ParticleGroup
#from line_profiler_pycharm import profile
from .twiss import  twiss_from_bmadx_particles
class Beam():
    """
    Beam class to initialize, track and apply wakes
    """
    def __init__(self, input_beam):

        self.check_inputs(input_beam)
        self.input_beam_config = input_beam
        self.style = input_beam['style']

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
            gen = Generator(filename)
            gen.run()
            pg = gen.particles
            self._charge = pg['charge']
            self._init_energy = np.mean(pg['energy'])

            self.particle = openpmd_to_bmadx_particles(pg, self._init_energy, 0.0, MC2)   #Bmad X particle
            #self.particleGroup = pg              # Particle Group

        else:
            ParticleGroup_h5 = input_beam['ParticleGroup_h5']
            pg = ParticleGroup(ParticleGroup_h5)

            self._charge = pg['charge']
            self._init_energy = np.mean(pg['energy'])

            self.particle = openpmd_to_bmadx_particles(pg, self._init_energy, 0.0, MC2)  # Bmad X particle
            #self.particleGroup = pg  # Particle Group

            # unchanged, initial energy and gamma
        self._init_gamma = self._init_energy/MC2
        #self._n_particle = self.particles.shape[0]

        self.position = 0
        self.step = 0

        self.update_status()


    def check_inputs(self, input_beam):
        assert 'style' in input_beam, 'ERROR: input_beam must have keyword <style>'
        if input_beam['style'] == 'from_file':
            self.required_inputs = ['style', 'beamfile', 'charge','energy']
        elif input_beam['style'] == 'distgen':
            self.required_inputs = ['style', 'distgen_input_file']
        elif input_beam['style'] == 'ParticleGroup':
            self.required_inputs = ['style', 'particleGroup_h5']
        else:
            raise Exception("input beam parsing Error: invalid input style")

        allowed_params = self.required_inputs + ['verbose']
        for input_param in input_beam:
            assert input_param in allowed_params, f'Incorrect param given to {self.__class__.__name__}.__init__(**kwargs): {input_param}\nAllowed params: {allowed_params}'

        # Make sure all required parameters are specified
        for req in self.required_inputs:
            assert req in input_beam, f'Required input parameter {req} to {self.__class__.__name__}.__init__(**kwargs) was not found.'
#    @profile
    def update_status(self):
        #self.particleGroup = bmadx_particles_to_openpmd(self.particle)
        self._sigma_x = self.sigma_x
        self._sigma_z = self.sigma_z
        self._slope = self.slope
        #self._sigma_x_transform = self.sigma_x_transform
        self._mean_x = self.mean_x
        self._mean_z = self.mean_z
        #self._twiss = self.twiss
        #self._sigma_energy = self.sigma_energy
        #self._mean_energy = self.mean_energy

 #   @profile
    def track(self, element, step_size, update_step=True):
        self.particle = track_element(self.particle, element)
        self.position += step_size
        if update_step:
            self.step += 1
        self.update_status()
  #  @profile
    def apply_wakes(self, dE_dct, x_kick, xrange, zrange, step_size, transverse_on):
        # Todo: add options for transverse or longitudinal kick only
        dE_E1 = step_size * dE_dct * 1e6 / self.init_energy  # self.energy in eV
        interp = RegularGridInterpolator((xrange, zrange), dE_E1, fill_value=0.0, bounds_error=False)
        dE_Es = interp(np.array([self.x_transform, self.z]).T)
        #self.particle.pz += dE_Es
        pz_new = self.particle.pz + dE_Es

        if transverse_on:
            dxp = step_size * x_kick * 1e6 / self.init_energy
            interp = RegularGridInterpolator((xrange, zrange), dxp, fill_value=0.0, bounds_error=False)
            dxps = interp(np.array([self.x_transform, self.z]).T)
            #self.particle.px += dxps
            px_new = self.particle.px + dxps
            self.particle = Particle(self.particle.x, px_new,
                                 self.particle.y, self.particle.py,
                                 self.particle.z, pz_new,
                                 self.particle.s, self.particle.p0c, self.particle.mc2)

        self.update_status()

    def frog_leap(self):
        # Todo: track half step, apply kicks, track another half step
        pass

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
    def slope(self):
        p = np.polyfit(self.z, self.x, deg=1)
        return p

    @property
    def x_transform(self):
        """
        :return: x coordinates after removing the x-z chirp
        """
        return self.x - np.polyval(self.slope, self.z)

    @property
    def sigma_x_transform(self):
        return np.std(self.x_transform)


    @property
    def charge(self):
        return self._charge

    @property
    def twiss(self):
        return twiss_from_bmadx_particles(self.particle)

    @property
    def particle_group(self):
        pg = bmadx_particles_to_openpmd(self.particle, self.charge)
        #pg.weight = np.abs(pg.weight)
        return pg