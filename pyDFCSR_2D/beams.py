import numpy as np
import sys
from distgen import Generator

class Beam():
    """
    Beam class to initialize and install particle information
    maybe forward track?
    """
    def __init__(self, input_beam):

        self.check_inputs(input_beam)
        self.input_beam_config = input_beam
        self.style = input_beam['style']

        if self.style == 'from_file':
            #Todo: make a safe parser to load ascii
            filename = input_beam['beamfile']
            self.particles = np.loadtxt(filename)
            #Todo：check how to deal with unit
            assert self.particles.shape[1] == 6, f'Error: input beam must have 6 dimension, but get {self.particles.shape[1]} instead'
            self._charge = input_beam['charge']
            self._energy = input_beam['energy']
        else:
            filename = input_beam['distgen_input_file']
            gen = Generator(filename)
            gen.run()
            pg = gen.particles
            gamma = pg.gamma
            delta = (gamma - np.mean(gamma))/np.mean(gamma)
            self.particles = np.vstack((pg.x, pg.xp, pg.y, pg.yp, pg.z, delta)).T
            #Todo: update beam mean energy after each step
            self.mean_gamma = np.mean(gamma)
            self.get_xz_chirp()
            #Todo： check how to get charge and energy from particles group and how to deal with unit
            self._charge = pg['q']
            self._energy = pg['energy']

        self._n_particle = self.particles.shape[0]
        self._particles_keys = ['x', 'px', 'y', 'py', 'z', 'delta']
        self.position = 0
        self.step = 0








    def check_inputs(self, input_beam):
        assert 'style' in input_beam, f'ERROR: input_beam must have keyword <style>'
        if input_beam['style'] == 'from_file':
            self.required_inputs = ['style', 'beamfile', 'charge','energy']
        elif input_beam['style'] == 'distgen':
            self.required_inputs = ['style', 'distgen_input_file']
        else:
            raise Exception("input beam parsing Error: invalid input style")

        allowed_params = self.required_inputs + ['verbose']
        for input_param in input_beam:
            assert input_param in allowed_params, f'Incorrect param given to {self.__class__.__name__}.__init__(**kwargs): {input_param}\nAllowed params: {allowed_params}'

        # Make sure all required parameters are specified
        for req in self.required_inputs:
            assert req in input_beam, f'Required input parameter {req} to {self.__class__.__name__}.__init__(**kwargs) was not found.'

    def get_xz_chirp(self):
        p = np.polyfit(self.particles[:, 0], self.particles[:, 4], deg=1)
        self.xz_chirp = p[0]

    @property
    def x_mean(self):
        #Todo: add xmean
        pass

    @property
    def y_mean(self):
        #Todo: add ymean
        pass

    @property
    def energy_mean(self):
        #Todo: add energy_mean
        pass
    @property
    def slope(self):
        #Todo: calculate linear chirp in x-z plane
        pass
    @property
    def sigma_z(self):
        return np.std(self.particles[:,4])

    @property
    def step(self):
        return self._step


    def emitt(self):
        #Todo: add emittance
        pass

    def twiss(self):
        #Todo: add twiss
        pass
    def track(self, r6, step_size):
        #Todo: forward or back propagate a step, remember to update every thing
        self.particles = np.matmul(r6, self.particles.T)
        self.position += step_size
        self.step += 1
        self.update_status()
    @property
    def x(self):
        return self.particles[:,0]
    @property
    def xp(self):
        return self.particles[:,1]

    @property
    def z(self):
        return self.particles[:,4]



    def update_status(self):
        #Todo: check if there are others to updates
        self.mean_gamma += self.mean_gamma*np.mean(self.particles[:,5])
        self.get_xz_chirp()

    def apply_wakes(self, dE_E, x_kick):
        #Todo: apply longitudinal and transverse CSR wakes
        self.update_status()
        pass

    def frog_leap(self):
        #Todo: track half step, apply kicks, track another half step
        pass
