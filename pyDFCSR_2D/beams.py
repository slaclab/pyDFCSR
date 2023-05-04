import numpy as np
import sys
from distgen import Generator
from physical_constants import *
from scipy.interpolate import RegularGridInterpolator

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
            self.particles = np.loadtxt(filename)
            assert self.particles.shape[1] == 6, f'Error: input beam must have 6 dimension, but get {self.particles.shape[1]} instead'
            self._charge = input_beam['charge']
            self._init_energy = input_beam['energy']


        else:
            filename = input_beam['distgen_input_file']
            gen = Generator(filename)
            gen.run()
            pg = gen.particles
            gamma = pg.gamma
            delta = (gamma - np.mean(gamma))/np.mean(gamma)
            self.particles = np.vstack((pg.x, pg.xp, pg.y, pg.yp, pg.z, delta)).T
            self._charge = pg['charge']
            self._init_energy = np.mean(pg['energy'])
            # unchanged, initial energy and gamma
        self._init_gamma = self._init_energy/MC2
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



    @property
    def mean_x(self):
        return np.mean(self.x)

    @property
    def mean_xp(self):
        return np.mean(self.xp)

    @property
    def mean_y(self):
        return np.mean(self.y)

    @property
    def init_energy(self):
        return self._init_energy

    @property
    def init_gamma(self):
        return self._init_gamma

    @property
    def gamma(self):
        return self.init_gamma + self.init_gamma*np.mean(self.particles[:,5])

    @property
    def energy(self):
        return self.gamma/MC2


    @property
    def slope(self):
        p = np.polyfit(self.z, self.x, deg=1)
        return p

    @property
    def sigma_z(self):
        return np.std(self.particles[:,4])

    @property
    def mean_z(self):
        return np.mean(self.particles[:,4])

    @property
    def sigma_x(self):
        return np.std(self.x)

    @property
    def sigma_xp(self):
        return np.std(self.xp)

    @property
    def sigma_delta(self):
        return np.std(self.delta)

    @property
    def x_xp(self):
        return np.mean(self.x*self.xp)

    @property
    def charge(self):
        return self._charge

    @property
    def emitX(self):
        return np.sqrt(np.mean(self.x**2)*np.mean(self.xp**2) - np.mean(self.x*self.xp)**2)
    @property
    def norm_emitX(self):
        return self.init_gamma*self.emitX
    @property
    def betaX(self):
        return self.sigma_x**2/self.emitX
    @property
    def alphaX(self):
        return -self.x_xp/self.emitX


    def stats_minus_dispersion(self, Rtot = np.eye(6)):
        """
        :param Rtot: reverse transfer matrix
        :return:x-emittance, beta and alpha minus dispersion effect
        """
        iRtot = np.linalg.inv(Rtot)
        SIGj = np.cov(self.particles.T)
        SIGj0 = np.matmul(iRtot, np.matmul(SIGj, iRtot.T))
        SIGj2 = np.matmul(Rtot[0:2, 0:2], np.matmul(SIGj0[0:2, 0:2], Rtot[0:2, 0:2].T))

        emitX =  np.sqrt(np.linalg.det(SIGj0[0:2,0:2]))
        norm_emitX = emitX*self.init_gamma
        beta = SIGj2[0,0]/emitX
        alpha = - SIGj2[0,1]/emitX
        return emitX, norm_emitX, beta, alpha



    def track(self, r6, step_size):
        self.particles = np.matmul(r6, self.particles.T)
        self.particles = self.particles.T
        self.position += step_size
        self.step += 1

    @property
    def x(self):
        return self.particles[:,0]
    @property
    def xp(self):
        return self.particles[:,1]

    @property
    def z(self):
        return self.particles[:,4]

    @property
    def delta(self):
        return self.particles[:,5]

    @property
    def x_transform(self):
        """
        :return: x coordinates after removing the x-z chirp
        """
        return self.x - np.polyval(self.slope, self.z)


    def apply_wakes(self, dE_dct, x_kick, xrange, zrange, step_size):
        # Todo: add options for transverse or longitudinal kick only
        dE_E1 = step_size*dE_dct*1e6/self.init_energy # self.energy in eV
        interp = RegularGridInterpolator((xrange, zrange), dE_E1, fill_value=0.0, bounds_error=False)
        dE_Es = interp(np.array([self.x_transform, self.z]).T)
        self.particles[:,5] += dE_Es


        dxp = step_size*x_kick*1e6/self.init_energy
        interp = RegularGridInterpolator((xrange, zrange), dxp, fill_value=0.0,bounds_error=False)
        dxps = interp(np.array([self.x_transform, self.z]).T)
        self.particles[:,1] += dxps


        #self.update_status()


    def frog_leap(self):
        #Todo: track half step, apply kicks, track another half step
        pass
