import numpy as np
from distgen import Generator
from lattice import *
from beams import *
from scipy.interpolate import RegularGridInterpolator
from yaml_parser import *
from tools import *
from beams import *
from lattice import *
from deposit import *
from interpolation import *
from integration import *
from physical_constants import *
from r_gen6 import *


class CSR2D:
    """
    The main class to calculate 2D CSR
    """

    def __init__(self, input_file=None):

        if input_file:
            self.parse_input(input_file)
            self.input_file = input_file

        self.initialization()  #process the initial beam

    def parse_input(self, input_file):
        input = parse_yaml(input_file)
        self.check_input_consistency(input)
        self.input = input
        self.beam = Beam(input['input_beam'])
        self.lattice = Lattice(input['input_lattice'])

        if 'particle_deposition' in input:
            self.DF_tracker = DF_tracker(input['particle_deposition'])
        else:
            self.DF_tracker = DF_tracker()

        if 'distribution_interpolation' in input:
            self.interpolation = Interpolation(input['distribution_interpolation'])
        else:
            self.interpolation = Interpolation()

        if 'CSR_integration' in input:
            self.integration = Integration(input['CSR_integration'])
        else:
            self.integration = Integration()

    def initialization(self):
        """
        deposit the initial beam
        :return:
        """
        self.DF_tracker.get_DF(x = self.beam.x, z =self.beam.z, xp=self.beam.xp, t = self.beam.position)
        self.DF_tracker.append_DF()
        self.DF_tracker.append_interpolant(formation_length=float('inf'),
                                           n_formation_length=self.integration.n_formation_length,
                                           interpolation=self.interpolation)


    def check_input_consistency(self, input):
        # Todo: need modification if config.yaml format changed
        self.required_inputs = ['input_beam', 'input_lattice']

        allowed_params = self.required_inputs + ['particle_deposition', 'distribution_interpolation', 'CSR_integration',
                                                 'CSR_computation']
        for input_param in input:
            assert input_param in allowed_params, f'Incorrect param given to {self.__class__.__name__}.__init__(**kwargs): {input_param}\nAllowed params: {allowed_params}'

        # Make sure all required parameters are specified
        for req in self.required_inputs:
            assert req in input, f'Required input parameter {req} to {self.__class__.__name__}.__init__(**kwargs) was not found.'

    def get_formation_length(self, R, sigma_z, type = 'inbend'):
        #Todo: add entrance and exit
        if type == 'inbend':
            self.formation_length =  (24*R**2*sigma_z)**(1/3)

    def run(self):
        for ele in range(self.lattice.lattice_config):
            # Todo: add sextupole, maybe Bmad Tracking?
            # -----------------------load current lattice params-----------------#
            steps = self.lattice.lattice_config[ele]['steps']
            L = self.lattice.lattice_config[ele]['L']
            type = self.lattice.lattice_config[ele]['type']
            DL = L / steps
            R = float('inf')
            if type == 'dipole':
                angle = self.lattice.lattice_config[ele]['angle']
                R = L/angle
                E1 = self.lattice.lattice_config[ele]['E1']
                E2 = self.lattice.lattice_config[ele]['E2']
                dang = angle / steps
            if type == 'quad':
                k1 = self.lattice.lattice_config[ele]['strength']
            # -----------------------tracking---------------------------------
            for step in range(steps):
                # get R6
                if type == 'dipole':
                    if step == 0:
                        dR6 = r_gen6(L=DL, angle=angle, E1=E1)
                    elif step == steps:
                        dR6 = r_gen6(L=DL, angle=angle, E1=0, E2=E2)
                    else:
                        dR6 = r_gen6(L=DL, angle=angle, E1=0, E2=0)
                elif type == 'drift':
                    dR6 = r_gen6(L=DL)
                elif type == 'quad':
                    dR6 = r_gen6(L=DL, k1=k1)

                # Propagate beam for one step
                self.beam.track(dR6, DL)
                # get the density functions
                self.DF_tracker.get_DF(x = self.beam.x, z =self.beam.z, xp=self.beam.xp, t = self.beam.position)
                # append the density functions to the log
                self.DF_tracker.append_DF()
                # build interpolant with the new DFs by interpolation
                formation_length = self.get_formation_length(R = R, sigma_z=self.beam.sigma_z)
                self.DF_tracker.append_interpolant(formation_length=formation_length, n_formation_length=self.integration.n_formation_length, interpolation=self.interpolation)




