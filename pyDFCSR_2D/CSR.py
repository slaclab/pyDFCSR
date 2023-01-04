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
from params import *
from physical_constants import *
from r_gen6 import *
import h5py


class CSR2D:
    """
    The main class to calculate 2D CSR
    """

    def __init__(self, input_file=None):


        if input_file:
            self.parse_input(input_file)
            self.input_file = input_file
        self.formation_length = None
        self.initialization()  # process the initial beam

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
            self.interpolation_params = Interpolation_params(input['distribution_interpolation'])
        else:
            self.interpolation_params = Interpolation_params()

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
        deposit the initial beam
        :return:
        """
        self.DF_tracker.get_DF(x=self.beam.x, z=self.beam.z, xp=self.beam.xp, t=self.beam.position)
        self.DF_tracker.append_DF()
        self.DF_tracker.append_interpolant(formation_length=float('inf'),
                                           n_formation_length=self.integration_params.n_formation_length,
                                           interpolation=self.interpolation_params)
        #Todo: add more flexible unit conversion, for both charge and energy
        self.CSR_scaling = 8.98755e-6 * self.beam.charge # charge must in nC (8.98755e-6 MeV/m for 1nC/m^2)

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

    def get_formation_length(self, R, sigma_z, type='inbend'):
        # Todo: add entrance and exit
        if type == 'inbend':
            self.formation_length = (24 * R ** 2 * sigma_z) ** (1 / 3)

    def run(self):
        for ele in self.lattice.lattice_config:
            self.lattice.update(ele)
            # Todo: add sextupole, maybe Bmad Tracking?
            # -----------------------load current lattice params-----------------#
            steps = self.lattice.lattice_config[ele]['steps']
            L = self.lattice.lattice_config[ele]['L']
            type = self.lattice.lattice_config[ele]['type']
            DL = L / steps
            R = float('inf')
            if type == 'dipole':
                angle = self.lattice.lattice_config[ele]['angle']
                R = L / angle
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
                self.DF_tracker.get_DF(x=self.beam.x, z=self.beam.z, xp=self.beam.xp, t=self.beam.position)
                # append the density functions to the log
                self.DF_tracker.append_DF()
                # append 3D matrix for interpolation with the new DFs by interpolation
                self.get_formation_length(R=R, sigma_z=self.beam.sigma_z)
                self.DF_tracker.append_interpolant(formation_length=self.formation_length,
                                                   n_formation_length=self.integration.n_formation_length,
                                                   interpolation=self.interpolation_params)
                # build interpolant based on the 3D matrix
                self.DF_tracker.build_interpolant()

                print('Calculating CSR at s=', str(self.beam.position))

                # calculate CSR mesh given beam shape
                self.get_CSR_mesh()
                # Calculate CSR on the mesh
                self.calculate_2D_CSR()
                # Apply CSR kick to the beam
                if self.CSR_params.apply_CSR:
                    self.beam.apply_wakes(self.dE_dct, self.x_kick,
                                          self.CSR_xrange_transformed, self.CSR_zrange, DL)

                if self.CSR_params.write_beam:
                    self.write_beam()

                if self.CSR_params.write_wakes:
                    self.write_wakes()





    def get_CSR_mesh(self):
        """
        calculating the mesh of observation points by taking linear transformation
        (xmesh, zmesh) TWO 1D arrays representiong (x, z) coordinates on a linear transformed mesh
        :return:
        """

        x_transform = self.beam.x_transform

        sig_x = np.std(x_transform)
        mean_x = np.mean(x_transform)
        sig_z = self.beam.sigma_z
        mean_z = self.beam.mean_z
        xlim = self.CSR_params.xlim
        zlim = self.CSR_params.zlim
        xbins = self.CSR_params.xbins
        zbins = self.CSR_params.zbins

        zrange = np.linspace(mean_z - zlim * sig_z, mean_z + zlim * sig_z, zbins)
        xrange = np.linsapce(mean_x - xlim * sig_x, mean_x + xlim * sig_x, xbins)

        # Todo: check the order
        xmesh_transform, zmesh = np.meshgrid(xrange, zrange, indexing='ij')

        xmesh_transform = xmesh_transform.flatten()
        zmesh = zmesh.flatten()

        xmesh = xmesh_transform +  np.polyval(p, zmesh)

        self.CSR_xmesh = xmesh
        self.CSR_zmesh = zmesh
        self.CSR_zrange = zrange
        self.CSR_xrange_transformed = xrange

    def calculate_2D_CSR(self):
        #Todo: Parallel
        N = self.CSR_params.xbins*self.CSR_params.zbins
        self.dE_dct = np.zeros((N,))
        self.x_kick = np.zeros((N,))

        for i in range(N):
            if i%int(N//10) == 0:
                print('Complete', str(np.round(i/N*00)), '%')

            s = self.beam.position + self.CSR_zmesh[i]
            x = self.CSR_xmesh[i]

            self.dE_dct[i], self.x_kick[i] = self.get_CSR_integrand(s,x)

        self.dE_dct = self.dE_dct.reshape((self.CSR_params.xbins, self.CSR_params.zbins))
        self.x_kick = self.x_kick.reshape((self.CSR_params.xbins, self.CSR_params.zbins))






    def get_CSR_integrand(self,s ,x):
        t = self.beam.position
        #-------------------------------------------------------------------------
        #Todo: kind of hard coding. Change in the future
        start_point = self.DF_tracker.start_time
        end_point = s + self.integration_params.zlim_end*self.beam.sigma_z
        mid_point1 = s - self.integration_params.zlim_mid1*self.beam.sigma_x
        mid_point2 = s - self.integration_params.zlim_mid2*self.beam.sigma_x
        xlim_L = x - self.integration_params.xlim*self.beam.sigma_x
        xlim_R = x + self.integration_params.xlim*self.beam.sigma_x

        zbins_1 = self.integration_params.zbins_1
        zbins_2 = self.integration_params.zbins_2
        zbins_3 = self.integration_params.zbins_3
        s_range_t1 = np.linspace(start_point, mid_point2, zbins_1)
        s_range_t2 = np.linspace(mid_point2, mid_point1, zbins_2)
        s_range_t3 = np.linspace(mid_point1, end_point, zbins_3)
        s_range_t = np.concatenate((s_range_t1, s_range_t2[1:], s_range_t3[1:]))

        x_range_t = np.linspace(xlim_L, xlim_R, self.integration_params.xbins)

        [xp, sp] = np.meshgrid(x_range_t, s_range_t, indexing='ij')
        #----------------------------------------------------------------------------------

        vx = self.DF_tracker.F_vx([t, x, s - t])

        X0_s = self.lattice.F_x_ref([s])
        #Todo: check it
        X0_sp = self.lattice.F_x_ref(sp.ravel())
        Y0_s = self.lattice.F_y_ref([s])
        Y0_sp = self.lattice.F_y_ref(sp.ravel())
        n_vec_s_x = self.lattice.F_n_vec_x([s])
        n_vec_sp_x = self.lattice.F_n_vec_x(sp.ravel())
        n_vec_s_y = self.lattice.F_n_vec_y([s])
        n_vec_sp_y = self.lattice.F_n_vec_y(sp.ravel())
        tau_vec_s_x = self.lattice.F_tau_vec_x([s])
        tau_vec_sp_x = self.lattice.F_tau_vec_x(sp.ravel())
        tau_vec_s_y = self.lattice.F_tau_vec_y([s])
        tau_vec_sp_y = self.lattice.F_tau_vec_y(sp.ravel())

        r_minus_rp_x = X0_s - X0_sp + x * n_vec_s_x - xp * n_vec_sp_x
        r_minus_rp_y = Y0_s - Y0_sp + x * n_vec_s_y - xp * n_vec_sp_y
        r_minus_rp = np.sqrt(r_minus_rp_x**2 + r_minus_rp_y**2)

        #Todo: different from matlab version. May doublecheck
        rho_sp = self.lattice.F_rho(sp.ravel())

        t_ret = t - r_minus_rp

        density_ret = self.DF_tracker.F_density(np.array([t_ret, xp.ravel(), sp.ravel()]).T)
        density_x_ret = self.DF_tracker.F_density_x(np.array([t_ret, xp.ravel(), sp.ravel()]).T)
        density_z_ret = self.DF_tracker.F_density_z(np.array([t_ret, xp.ravel(), sp.ravel()]).T)
        vx_ret = self.DF_tracker.F_vx(np.array([t_ret, xp.ravel(), sp.ravel()]).T)
        vx_x_ret = self.DF_tracker.F_vx_x(np.array([t_ret, xp.ravel(), sp.ravel()]).T)

        ## Todo: More accurate vs
        vs = 1
        vs_ret = 1
        vs_s_ret = 0
        vx_t = 0
        vs_t = 0

        scale_term =  1 + xp.ravel()*rho_sp


        velocity_x = vs * tau_vec_s_x + vx * n_vec_s_x
        velocity_y = vs * tau_vec_s_y + vx * n_vec_s_y

        velocity_ret_x = vs_ret * tau_vec_sp_x + vx_ret * n_vec_sp_x
        velocity_ret_y = vs_ret * tau_vec_sp_y + vx_ret * n_vec_sp_y

        velocity_partial_t_x = vs_t * tau_vec_sp_x + vx_t * n_vec_sp_x
        velocity_partial_t_y = vs_t * tau_vec_sp_y + vx_t * n_vec_sp_y

        nabla_density_ret_x = density_x_ret  * n_vec_sp_x + density_z_ret / scale_term * tau_vec_sp_x
        nabla_density_ret_y = density_x_ret * n_vec_sp_y + density_z_ret / scale_term * tau_vec_sp_y

        div_velocity = vs_s_ret + vx_x_ret  #???

        # Todo: Consider using general form
        ## general form
        #part1 = velocity_x * velocity_ret_x + velocity_y * velocity_ret_y
        #CSR_numerator1 = scale_term * ((velocity_x - part1 * velocity_ret_x) * nabla_density_ret_x  + \
        #                  (velocity_y - part1 * velocity_ret_y)*nabla_density_ret_y)
        #CSR_numerator2 = -scale_term * part1 * density_ret * div_velocity
        #CSR_numerator3 = scale_term * density_ret * (velocity_partial_t_x * velocity_x + velocity_partial_t_y * velocity_y)

        #CSR_denominator = r_minus_rp

        #self.CSR_integrand = CSR_numerator1/CSR_denominator + (CSR_numerator3 + CSR_numerator3)/CSR_denominator


        #Todo: Check the formula. Also check the scale term
        CSR_numerator1 = scale_term * (((n_vec_sp_x * tau_vec_s_x + n_vec_sp_y * tau_vec_s_y) +
                                        (vx - vx_ret) * (tau_vec_sp_x * tau_vec_s_x + tau_vec_sp_y * tau_vec_s_y)) * density_x_ret -
                                       vx_ret * (n_vec_sp_x * tau_vec_s_x + n_vec_sp_y * tau_vec_s_y)/scale_term * density_z_ret)

        CSR_numerator2 = -((tau_vec_sp_x * tau_vec_s_x + tau_vec_sp_y * tau_vec_s_y) +
                           (vx - vx_ret) * (n_vec_s_x * tau_vec_sp_x + n_vec_s_y * tau_vec_sp_y)) * density_ret * vx_x_ret

        CSR_numerator3 = scale_term * density_ret * (velocity_partial_t_x * velocity_x + velocity_partial_t_y * velocity_y)

        CSR_denominator = r_minus_rp

        CSR_integrand_z = CSR_numerator1/CSR_denominator + (CSR_numerator2 + CSR_numerator3)/CSR_denominator

        n_minus_np_x = n_vec_s_x - n_vec_sp_x
        n_minus_np_y = n_vec_s_y - n_vec_sp_y

        #part: (r-r')(n - n')
        part1 = r_minus_rp_x * n_minus_np_x + r_minus_rp_y * n_minus_np_y

        #part2: n tau'
        part2 = n_vec_s_x * tau_vec_sp_x + n_vec_s_y * tau_vec_sp_y

        # part3: partial density/partial t_ret
        partial_density = - (velocity_ret_x * nabla_density_ret_x + velocity_ret_y * nabla_density_ret_y) - \
                          density_ret * div_velocity

        W1 = scale_term * part1 / (r_minus_rp * r_minus_rp * r_minus_rp) * density_ret
        W2 = scale_term * part1 / (r_minus_rp * r_minus_rp) * partial_density
        W3 = -scale_term * part2 / r_minus_rp * partial_density

        CSR_integrand_x = W1 + W2 + W3

        CSR_integrand_x = CSR_integrand_x.reshape(xp.shape)
        CSR_integrand_z = CSR_integrand_z.reshape(xp.shape)

        dE_dct = -self.CSR_scaling * np.trapz(y = np.trapz(y = CSR_integrand_z, x = x_range_t, axis = 0), x = s_range_t)
        x_kick = self.CSR_scaling * np.trapz(y = np.trapz(y = CSR_integrand_x, x = x_range_t, axis = 0), x = s_range_t)

        return dE_dct, x_kick

    def write_beam(self):

        filename = self.CSR_params.workdir + '\\particles.h5'
        hf = h5py.File(filename, 'a')

        step = self.beam.step
        groupname = 'step_' + str(step)
        g = hf.create_group(groupname)
        g.attrs['step'] = step
        g.attrs['position']  = self.beam.position
        g.attrs['mean_gamma'] = self.beam.mean_gamma
        g.attrs['beam_energy'] = self.beam.energy
        g.attrs['element'] = self.lattice.current_element
        g.attrs['charge'] = self.beam.charge
        g2 = g.create_group('particles')
        g2.create_dataset('x', data = self.beam.particles[:, 0])
        g2.create_dataset('xp', data = self.beam.particles[:, 1])
        g2.create_dataset('y', data = self.beam.particles[:, 2])
        g2.create_dataset('yp', data=self.beam.particles[:, 3])
        g2.create_dataset('z', data=self.beam.particles[:,4])
        g2.create_dataset('delta', data=self.beam.particles[:, 5])

    def write_wakes(self):

        filename = self.CSR_params.workdir + '\\wakes.h5'
        hf = h5py.File(filename, 'a')

        step = self.beam.step
        groupname = 'step_' + str(step)
        g = hf.create_group(groupname)
        g.attrs['step'] = step
        g.attrs['position']  = self.beam.position
        g.attrs['mean_gamma'] = self.beam.mean_gamma
        g.attrs['beam_energy'] = self.beam.energy
        g.attrs['element'] = self.lattice.current_element
        g.attrs['charge'] = self.beam.charge
        g1 = g.create_group('longitudinal')
        g1.attrs['unit'] = 'MeV/m'
        g1.create_dataset('x_grids', data = self.CSR_xmesh)
        g1.create_dataset('z_grids', data = self.CSR_zmesh)
        g1.create_dataset('dE_dct', data = self.dE_dct)
        g2  = g.create_group('transverse')
        g2.attrs['unit'] = 'MeV/m'
        g2.create_dataset('x_grids', data = self.CSR_xmesh)
        g2.create_dataset('z_grids', data = self.CSR_zmesh)
        g2.greate_dataset('xkicks', data = self.x_kick)














