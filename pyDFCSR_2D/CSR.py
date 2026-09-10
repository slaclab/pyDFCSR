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
from .deposit_smooth import DF_tracker_smooth, DF_tracker_comoving
from .interp1D import interpolate1D
from .interp3D import (interpolate3D, interpolate3D_transformed,
                       interpolate3D_comoving_fields, get_poly_deriv_blended)
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

    def __init__(self, input_file=None, parallel = False):

        self.timestamp = isotime()
        if input_file:
            self.parse_input(input_file)
            self.input_file = input_file
        self.formation_length = None
        self.initialization()  # process the initial beam

        self.prefix = f'{self.CSR_params.write_name}-{self.timestamp}'

        if parallel:
            self.init_MPI()
        else:
            self.parallel = False

    def parse_input(self, input_file):
        input = parse_yaml(input_file)
        self.check_input_consistency(input)
        self.input = input
        self.beam = Beam(input['input_beam'])
        self.lattice = Lattice(input['input_lattice'])

        if 'particle_deposition' in input:
            deposition_config = input['particle_deposition']
            method = deposition_config.get('method', 'legacy')
            if method == 'bspline_comoving':
                self.DF_tracker = DF_tracker_comoving(deposition_config)
            elif method == 'bspline_fft':
                self.DF_tracker = DF_tracker_smooth(deposition_config)
            else:
                self.DF_tracker = DF_tracker(deposition_config)
        else:
            self.DF_tracker = DF_tracker()
        self.use_comoving = isinstance(self.DF_tracker, DF_tracker_comoving)
        # both B-spline paths deposit in the tilt-removed frame, so both need the
        # xi-following integration bands rather than the sigma_x-wide rectangles
        self.use_smooth_deposit = (isinstance(self.DF_tracker, DF_tracker_smooth)
                                   or self.use_comoving)

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
        self.DF_tracker.get_DF(x=self.beam.x, z=self.beam.z, px=self.beam.px, t=self.beam.position)
        self.DF_tracker.append_DF()
        self.DF_tracker.append_interpolant(formation_length=float('inf'),
                                           n_formation_length=self.integration_params.n_formation_length)
        #Todo: add more flexible unit conversion, for both charge and energy
        self.CSR_scaling = 8.98755e3 * self.beam.charge # charge in C (8.98755e-6 MeV/m for 1nC/m^2)
        self.init_statistics()
    def init_statistics(self):
        Nstep = self.lattice.total_steps
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

        self.update_statistics(step = 0)


        self.inbend = False
        self.afterbend = False
        self.R_rec = None
        self.phi_rec = None

    def init_MPI(self):
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
        # Todo: need modification if dipole_config.yaml format changed
        self.required_inputs = ['input_beam', 'input_lattice']

        allowed_params = self.required_inputs + ['particle_deposition', 'distribution_interpolation', 'CSR_integration',
                                                 'CSR_computation']
        for input_param in input:
            assert input_param in allowed_params, f'Incorrect param given to {self.__class__.__name__}.__init__(**kwargs): {input_param}\nAllowed params: {allowed_params}'

        # Make sure all required parameters are specified
        for req in self.required_inputs:
            assert req in input, f'Required input parameter {req} to {self.__class__.__name__}.__init__(**kwargs) was not found.'

    def get_formation_length(self, R, sigma_z, phi = 0.0, inbend=True):
        if inbend:
            self.formation_length = (24 * (R ** 2) * sigma_z) ** (1 / 3)
        else:
            self.formation_length = (3*R**2*phi**4)/(4*(-6*sigma_z + R*phi**3))

    def _refresh_formation_length(self, R):
        """
        Recompute L_f from the CURRENT bunch length. Returns False only in the
        pre-first-bend drift, where formation_length is an accumulated drift length
        rather than a function of sigma_z and must not be recomputed.

        Must be called every step, not just on element entry. sigma_z can grow by an
        order of magnitude inside a single bend at high tilt (57 um -> 1362 um at
        shear 50 here), and L_f ~ sigma_z^(1/3) sets both the integration reach
        s1 = s2 - n_formation_length*L_f and the history truncation in
        append_interpolant. Pinning it to the entrance bunch length left the retained
        history several times too short and the wake 12.9% off at shear 20.
        """
        if self.inbend:
            self.get_formation_length(R=R, sigma_z=5 * self.beam.sigma_z,
                                      inbend=True)
            return True
        if self.afterbend:
            self.get_formation_length(R=self.R_rec, sigma_z=5 * self.beam.sigma_z,
                                      inbend=True)
            return True
        return False

    def get_bmadx_element(self, ele,  DL, entrance = False, exit = False):
        input_dic = self.lattice.lattice_config[ele].copy()
        input_dic.pop('nsep')
        L = input_dic.pop('L')
        type = input_dic.pop('type')


        if type == 'dipole':
            if 'angle' in input_dic.keys():
                angle = input_dic.pop('angle')
                G = angle / L

            if 'G' in input_dic.keys():
                G = input_dic.pop('G')

            if 'E1' in input_dic.keys():
                E1 = input_dic.pop('E1')
            else:
                E1 = 0

            if 'E2' in input_dic.keys():
                E2 = input_dic.pop('E2')
            else:
                E2 = 0

            if 'FRINGE_AT' in input_dic.keys():
                FRINGE_AT = input_dic.pop('FRINGE_AT')


            if entrance and exit:
                element = SBend(L = DL, P0C = self.beam.init_energy, G = G, E1 = E1, E2 = E2, FRINGE_AT = FRINGE_AT, **input_dic)

            elif entrance:
                element = SBend(L=DL, P0C=self.beam.init_energy, G=G, E1=E1, E2=0.0, FRINGE_AT = "entrance_end", **input_dic)


            elif exit:
                element = SBend(L=DL, P0C=self.beam.init_energy, G=G, E1=0.0, E2=E2, FRINGE_AT = "exit_end", **input_dic)

            else:
                element = SBend(L=DL, P0C=self.beam.init_energy, G=G, E1=0.0, E2=0.0, FRINGE_AT = "no_end", **input_dic)

        elif type == 'drift':
            element = Drift(L = DL)

        elif type == 'quad':
            K1 = input_dic.pop('K1')
            element = Quadrupole(L=DL, K1=K1, **input_dic)

        elif type == 'sextupole':
            K2 = input_dic.pop('K2')
            element = Sextupole(L=DL, K2=K2, **input_dic)
        print(element)
        return element

#    @profile
    def run(self, stop_time = None, debug = False):

        if (not self.parallel) or (self.rank == 0):
            print('Starting the DFCSR run')

        step_count = 1

        DL = self.lattice.step_size
        ele_count = 0
        skip_ele = False
        self.inbend = False
        self.afterbend = False
        self.formation_length = 0.0

        for ele in list(self.lattice.lattice_config.keys())[1:]:

            self.lattice.update(ele)
            # Todo: add sextupole, maybe Bmad Tracking?
            # -----------------------load current lattice params-----------------#
            # Pre-process the lattice params
            L = self.lattice.lattice_config[ele]['L']
            type = self.lattice.lattice_config[ele]['type']
            steps = self.lattice.steps_per_element[ele_count]
            R = float('inf')

            ####### A step over the boundary of the elements, deal with the part of the step in the previous element
            if (not skip_ele) and ele_count > 0:
                DL_1 = self.lattice.distance[ele_count - 1] - self.beam.position   # The remaining distance in last element
                #Todo: Bmadx seems to have some problems when DL is very
                if DL_1 > 1.0e-6:
                # calculate the part in the previous element
                    element = self.get_bmadx_element(ele=ele_prev, DL=DL_1, exit=True)
                    self.beam.track(element, DL_1, update_step=False)
                else:
                    DL_1 = 0.0
            # If no steps inside an element
            if steps == 0:    #If one step over the whole element
                skip_ele = True
                element = self.get_bmadx_element(ele=ele,  DL=L, exit=True, entrance = True)
                self.beam.track(element, L, update_step=False)




            if type == 'dipole':
                angle = self.lattice.lattice_config[ele]['angle']
                R = L / angle

                self.inbend = True

                self.afterbend = True
                self.R_rec = R
                self.phi_rec = angle

                self._refresh_formation_length(R)


            else:  # If not in a bend
                self.inbend = False

                #Todo: Verify the formation length in the drift. The phi-dependent
                # out-of-bend form in get_formation_length is not used; the in-bend
                # expression with the recorded R is used instead.
                if not self._refresh_formation_length(R):
                    # first drift in the lattice, before any bend
                    self.formation_length += L



            distance_in_current_ele = 0.0
            # -----------------------tracking---------------------------------
            for step in range(steps):
                time0  = time.time()

                # Deal with boundary condition. A step over the boundary of two adjacent elements
                if (step == 0) and (ele_count > 0):
                    # If enter a new element, split the step

                    DL_2 = self.lattice._positions_record[step_count] - self.lattice.distance[ele_count - 1]

                    # calculate the part in the new element
                    element = self.get_bmadx_element(ele = ele,  DL = DL_2, entrance = True)
                    self.beam.track(element, DL_2)
                    distance_in_current_ele += DL_2
                    skip_ele = False    # Reset the flag

                else:
                    element = self.get_bmadx_element(ele = ele,  DL = DL)
                    # Propagate beam for one step
                    self.beam.track(element, DL)
                    distance_in_current_ele += DL


                # sigma_z evolves within the element, so L_f has to be refreshed here
                # and not only on element entry -- it truncates the history below and
                # sets the integration reach in get_CSR_wake.
                self._refresh_formation_length(R)

                if debug or self.CSR_params.compute_CSR:
                    # get the density functions
                    self.DF_tracker.get_DF(x=self.beam.x, z=self.beam.z, px=self.beam.px, t=self.beam.position)
                    # append the density functions to the log
                    self.DF_tracker.append_DF()
                    # append 3D matrix for interpolation with the new DFs by interpolation
                    self.DF_tracker.append_interpolant(formation_length=self.formation_length,
                                                       n_formation_length=self.integration_params.n_formation_length)
                    self.DF_tracker.build_interpolant()

                # If beam is in an after-bend drift and away from the previous bend for more than n*formation_length, stop calculating wakes
                #Todo: formation length not correct here
                #if  self.afterbend and (not self.inbend) and distance_in_current_ele > 3*self.formation_length:
                #    CSR_blocker = True
                #    if (not self.parallel) or (self.rank == 0):
                #        print("Far away from a bending magnet, stopping calculating CSR")

                #else:
                #    CSR_blocker = False
                CSR_blocker = False
                
                
                if self.CSR_params.compute_CSR and (not CSR_blocker):
                    if step % self.lattice.nsep[ele_count] == 0:
                        # calculate CSR mesh given beam shape
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
                        if (self.CSR_params.write_beam == 'all' or
                                (isinstance(self.CSR_params.write_beam, list) and (step_count in self.CSR_params.write_beam))):
                            self.dump_beam(label = step_count)
                        if self.CSR_params.write_wakes:
                            self.write_wakes()

                # recording statistics at each step
                self.update_statistics(step = step_count)

                if not self.parallel or self.rank == 0:
                    print("Finish step {}, s = {},  in {} seconds".format(step_count, self.beam.position, time.time() - time0))

                step_count += 1

                if stop_time and self.beam.position > stop_time:
                    return

            ele_prev = ele
            type_prev = type

            ele_count += 1

        self.dump_beam(label='end')
        self.write_statistics()


    def get_CSR_mesh(self):
        """
        calculating the mesh of observation points by taking linear transformation
        (xmesh, zmesh) TWO 1D arrays representiong (x, z) coordinates on a linear transformed mesh
        :return:
        """

        x_transform = self.beam.x_transform
        p = self.beam.slope

        sig_x = np.std(x_transform)
        mean_x = np.mean(x_transform)
        sig_z = self.beam.sigma_z
        mean_z = self.beam.mean_z
        xlim = self.CSR_params.xlim
        zlim = self.CSR_params.zlim
        xbins = self.CSR_params.xbins
        zbins = self.CSR_params.zbins

        zrange = np.linspace(mean_z - zlim * sig_z, mean_z + zlim * sig_z, zbins)
        xrange = np.linspace(mean_x - xlim * sig_x, mean_x + xlim * sig_x, xbins)

        # Todo: check the order
        xmesh_transform, zmesh = np.meshgrid(xrange, zrange, indexing='ij')

        xmesh_transform = xmesh_transform.flatten()
        zmesh = zmesh.flatten()

        xmesh = xmesh_transform +  np.polyval(p, zmesh)

        self.CSR_xmesh = xmesh
        self.CSR_zmesh = zmesh
        self.CSR_zrange = zrange
        self.CSR_xrange_transformed = xrange
    
#    @profile
    def calculate_2D_CSR(self):

        N = self.CSR_params.xbins*self.CSR_params.zbins
        self.dE_dct = np.zeros((N,))
        self.x_kick = np.zeros((N,))

        start_time = time.time()
        for i in range(N):

            #if i == 210:
            #    print(i)

            #if i%int(N//10) == 0:
            #    print('Complete', str(np.round(i/N*100,2)), '%')

            s = self.beam.position + self.CSR_zmesh[i]
            x = self.CSR_xmesh[i]

            self.dE_dct[i], self.x_kick[i] = self.get_CSR_wake(s,x)

        self.dE_dct = self.dE_dct.reshape((self.CSR_params.xbins, self.CSR_params.zbins))
        self.x_kick = self.x_kick.reshape((self.CSR_params.xbins, self.CSR_params.zbins))

    def calculate_2D_CSR_parallel(self):
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

    def _lattice_at(self, sval):
        """(X0, Y0, n_x, n_y) of the reference orbit at arc length(s) sval."""
        lat = self.lattice
        return tuple(interpolate1D(xval=sval, data=d, min_x=lat.min_x, delta_x=lat.delta_x)
                     for d in (lat.coords[:, 0], lat.coords[:, 1],
                               lat.n_vec[:, 0], lat.n_vec[:, 1]))

    def _retarded_xi_band(self, s, x, t, sp, n_iter=3):
        """
        Locate the retarded density support in lab x' for each s' column.

        Returns (lo, hi), both shaped like sp: the lab-x' interval spanned by the
        deposition grid at the retarded time of that column.

        The integrand is proportional to the retarded density and its derivatives,
        which the interpolant returns as exactly 0.0 outside the deposition grid.
        Restricting the transverse domain to that grid is therefore exact, not an
        approximation -- it discards only nodes that were contributing hard zeros.

        t_ret depends on x' through |r - r'|, and the x' where the density lives
        depends on t_ret through the frame polynomial, so this is a fixed point.
        |r - r'| varies slowly with x' (transverse offsets are small compared with
        the retarded distance), so a few passes converge.

        Two snapshots straddle each t_ret and the interpolant blends them in the
        LAB frame, so at large tilt-rate the "support" really is the union of two
        shear-displaced ribbons (the ghosting of interp3D). The band has to cover
        both. Once the time blend is made co-moving the two collapse onto each
        other and these bands tighten by themselves.
        """
        tr = self.DF_tracker
        n_t = tr.data_density_interp.shape[0]

        X0_s, Y0_s, n_s_x, n_s_y = (v[0] for v in self._lattice_at(np.array([s])))
        X0_sp, Y0_sp, n_sp_x, n_sp_y = self._lattice_at(sp)

        xp = np.full(sp.shape, x, dtype=float)
        lo = hi = dead = None
        for _ in range(n_iter):
            dx = X0_s - X0_sp + x * n_s_x - xp * n_sp_x
            dy = Y0_s - Y0_sp + x * n_s_y - xp * n_sp_y
            t_ret = t - np.sqrt(dx * dx + dy * dy)
            z_ret = sp - t_ret

            t_idx = (t_ret - tr.min_x) / tr.delta_x
            k = np.clip(np.floor(t_idx).astype(int), 0, max(n_t - 2, 0))
            k1 = np.minimum(k + 1, n_t - 1)

            if self.use_comoving:
                # The co-moving interpolant places the density in the single
                # BLENDED frame, so the band is one interval. Mirror its blending
                # exactly -- linear in the poly and the means, log-linear in the
                # sigmas -- or the band will not sit where the density is.
                a = np.clip(t_idx - k, 0.0, 1.0)
                b = 1.0 - a
                p_a = (b * self._eval_poly_rows(tr.poly_coeffs_interp[k], z_ret)
                       + a * self._eval_poly_rows(tr.poly_coeffs_interp[k1], z_ret))
                xi_bar = b * tr.xi_bar_arr[k] + a * tr.xi_bar_arr[k1]
                z_bar = b * tr.z_bar_arr[k] + a * tr.z_bar_arr[k1]
                s_xi = np.exp(b * np.log(tr.sigma_xi_arr[k])
                              + a * np.log(tr.sigma_xi_arr[k1]))
                s_z = np.exp(b * np.log(tr.sigma_z_arr[k])
                             + a * np.log(tr.sigma_z_arr[k1]))
                w = (z_ret - z_bar) / s_z
                in_z = (w >= tr.w_start) & (w < tr.w_end)
                centre = p_a + xi_bar
                lo = np.where(in_z, centre + tr.u_start * s_xi, np.inf)
                hi = np.where(in_z, centre + tr.u_end * s_xi, -np.inf)
            else:
                # The lab-frame blend evaluates each snapshot in its OWN frame and
                # adds them, so the support is the union of two shear-displaced
                # ribbons. The band has to cover both.
                lo = np.full(sp.shape, np.inf)
                hi = np.full(sp.shape, -np.inf)
                for kk in (k, k1):
                    xi_lo, xi_hi, z_lo, z_hi = self._grid_extents(kk)
                    # a snapshot whose z grid does not reach z_ret contributes
                    # nothing, so it must not widen the band either
                    in_z = (z_ret >= z_lo) & (z_ret < z_hi)
                    base = self._eval_poly_rows(tr.poly_coeffs_interp[kk], z_ret)
                    lo = np.where(in_z, np.minimum(lo, base + xi_lo), lo)
                    hi = np.where(in_z, np.maximum(hi, base + xi_hi), hi)

            dead = ~np.isfinite(lo)
            lo = np.where(dead, x, lo)
            hi = np.where(dead, x, hi)
            xp = 0.5 * (lo + hi)

        mid = 0.5 * (lo + hi)
        half = 0.5 * (hi - lo) * self.integration_params.xi_band_margin
        # A dead column's z_ret is off the deposition z grid, so the density and
        # all its derivatives are exactly 0 there and the column contributes
        # nothing whatever band we give it. Give it a normal width rather than
        # zero, so no degenerate mesh reaches the 1/|r-r'| kernel.
        if dead.any():
            fallback = half[~dead].max() if (~dead).any() else np.abs(x) + 1e-6
            half = np.where(dead, fallback, half)
        return mid - half, mid + half

    def _grid_extents(self, kk):
        """
        (xi_lo, xi_hi, z_lo, z_hi) of each snapshot in the index array kk, in the
        tilt-removed frame.

        The two B-spline trackers describe the same support with different
        metadata: DF_tracker_smooth stores a physical grid origin and spacing per
        snapshot, DF_tracker_comoving stores one shared normalized grid plus each
        snapshot's (mean, sigma).
        """
        tr = self.DF_tracker
        if self.use_comoving:
            s_xi = tr.sigma_xi_arr[kk]
            s_z = tr.sigma_z_arr[kk]
            return (tr.xi_bar_arr[kk] + tr.u_start * s_xi,
                    tr.xi_bar_arr[kk] + tr.u_end * s_xi,
                    tr.z_bar_arr[kk] + tr.w_start * s_z,
                    tr.z_bar_arr[kk] + tr.w_end * s_z)

        n_xi, n_z = tr.data_density_interp.shape[1:]
        # bilinear_single needs int(idx) <= n - 2, so the usable extent stops one
        # cell short of the last node
        return (tr.min_xi_arr[kk],
                tr.min_xi_arr[kk] + (n_xi - 1) * tr.delta_xi_arr[kk],
                tr.min_z_arr[kk],
                tr.min_z_arr[kk] + (n_z - 1) * tr.delta_z_arr[kk])

    @staticmethod
    def _eval_poly_rows(coeffs, z):
        """Evaluate a different polynomial per element of z. coeffs: (n, deg+1)."""
        out = np.zeros_like(z)
        for j in range(coeffs.shape[1]):
            out = out * z + coeffs[:, j]
        return out

    def _comoving_frame_at(self, t_ret):
        """
        The blended co-moving frame at each element of t_ret.

        Mirrors interpolate3D_comoving_fields exactly -- linear in the polynomial
        and the means, log-linear in the sigmas. A band located with a different
        blend than the interpolant uses will not sit where the density is.

        Returns (poly, xi_bar, z_bar, sigma_xi, sigma_z); poly is (n, deg+1) with
        the highest power first, matching np.polyfit and _eval_poly_rows.
        """
        tr = self.DF_tracker
        n_t = tr.poly_coeffs_interp.shape[0]
        t_idx = (t_ret - tr.min_x) / tr.delta_x
        k = np.clip(np.floor(t_idx).astype(int), 0, max(n_t - 2, 0))
        k1 = np.minimum(k + 1, n_t - 1)
        a = np.clip(t_idx - k, 0.0, 1.0)
        b = 1.0 - a
        poly = (b[:, None] * tr.poly_coeffs_interp[k]
                + a[:, None] * tr.poly_coeffs_interp[k1])
        return (poly,
                b * tr.xi_bar_arr[k] + a * tr.xi_bar_arr[k1],
                b * tr.z_bar_arr[k] + a * tr.z_bar_arr[k1],
                np.exp(b * np.log(tr.sigma_xi_arr[k])
                       + a * np.log(tr.sigma_xi_arr[k1])),
                np.exp(b * np.log(tr.sigma_z_arr[k])
                       + a * np.log(tr.sigma_z_arr[k1])))

    @staticmethod
    def _eq424(T, nq, q2, tau, b, sign):
        """
        Vectorized thesis Eq 4.24, in the form corrected in Step 4f.

            Tb  = T - b/tau
            rad = (tau^2 - 1)(Tb^2 - q2) + (nq*tau + Tb)^2
            x'  = tau [ Tb + nq*tau +- sqrt(rad) ] / (tau^2 - 1)

        where the beam axis is x' = tau*z' + b and l = |r - r'| = Tb + x'/tau.

        Returns (x', rad); x' is nan where the branch does not exist. Every guard
        below is load-bearing:

          |tau| ~ 0   the axis is the horizontal line x' = b, independent of l, so
                      there is a single root and no squaring was involved.
          |tau| ~ 1   (tau^2 - 1) -> 0. This is the alpha = +-pi/4 degeneracy the
                      thesis flags and the origin of the code's |tan theta| <= 1
                      branch switch; the quadratic collapses to a linear equation.
          rad < 0     the light cone and the beam axis do not meet at this s', so
                      the branch is genuinely absent.
          l <= 0      SPURIOUS ROOT. The derivation squares l = Tb + x'/tau, which
                      admits l < 0, i.e. t_ret > t -- a source point in the future.
                      Omitting this guard puts quadrature nodes where the integrand
                      is identically zero (Step 5d found this the hard way).
        """
        flat = np.abs(tau) < 1e-9
        tau_s = np.where(flat, 1.0, tau)
        deg = (~flat) & (np.abs(np.abs(tau_s) - 1.0) < 1e-9)
        Tb = T - b / tau_s

        rad = (tau_s ** 2 - 1.0) * (Tb ** 2 - q2) + (nq * tau_s + Tb) ** 2
        gen = (~flat) & (~deg) & (rad >= 0.0)
        xp = np.where(gen,
                      tau_s * (Tb + nq * tau_s
                               + sign * np.sqrt(np.maximum(rad, 0.0)))
                      / np.where(gen, tau_s ** 2 - 1.0, 1.0),
                      np.nan)

        Bc = -2.0 * T / tau_s - 2.0 * nq
        lin = deg & (Bc != 0.0)
        xp = np.where(lin, -(q2 - T ** 2) / np.where(lin, Bc, 1.0), xp)
        xp = np.where(flat, b, xp)

        # the l > 0 guard does not apply where no squaring took place
        l = np.where(flat, 1.0, Tb + xp / tau_s)
        return np.where(np.isfinite(xp) & (l > 0.0), xp, np.nan), rad

    def _retarded_xi_bands(self, s, x, t, sp, n_iter=12):
        """
        Locate EVERY branch of the retarded density support in lab x': one (lo, hi)
        interval per branch per s' column.

        _retarded_xi_band runs a single fixed point started at x' = x, so it
        converges to one root of "light cone meets beam axis" and never covers the
        other. Step 4e measured the branch it misses at up to 67% of a column's
        contribution. Here BOTH roots of the corrected Eq 4.24 are taken, each
        refined by the same fixed point, each given its own sub-band.

        The roots are deliberately NOT labelled narrow/chirp. Which sign is which
        flips as tau^2 crosses 1, and at low tilt they merge into a single ridge
        (Step 5d), so they are treated symmetrically and any overlap is removed by
        _disjoint_bands.

        Eq 4.24 assumes one straight beam axis, so this applies only to the
        co-moving interpolant at poly_degree 1. Everything else falls back to the
        single-band locator, leaving 'legacy' and 'bspline_fft' untouched.
        """
        tr = self.DF_tracker
        if not (self.use_comoving and getattr(tr, 'poly_degree', None) == 1):
            return [self._retarded_xi_band(s, x, t, sp)]

        X0_s, Y0_s, n_s_x, n_s_y = (v[0] for v in self._lattice_at(np.array([s])))
        X0_sp, Y0_sp, n_sp_x, n_sp_y = self._lattice_at(sp)

        qx = X0_s - X0_sp + x * n_s_x
        qy = Y0_s - Y0_sp + x * n_s_y
        nq = n_sp_x * qx + n_sp_y * qy
        q2 = qx * qx + qy * qy
        T = t - sp
        margin = self.integration_params.xi_band_margin

        bands, diag = [], []
        for sign in (-1.0, +1.0):
            xp = np.full(sp.shape, float(x))
            live = np.ones(sp.shape, dtype=bool)
            rad = np.zeros(sp.shape)
            for _ in range(n_iter):
                dx = qx - xp * n_sp_x
                dy = qy - xp * n_sp_y
                poly, xi_bar, _, _, _ = self._comoving_frame_at(
                    t - np.sqrt(dx * dx + dy * dy))
                new, rad = self._eq424(T, nq, q2, poly[:, 0],
                                       poly[:, -1] + xi_bar, sign)
                ok = np.isfinite(new)
                xp = np.where(ok, new, xp)
                live &= ok

            # the width and the z support come from the frame at the converged root
            dx = qx - xp * n_sp_x
            dy = qy - xp * n_sp_y
            r_phys = np.sqrt(dx * dx + dy * dy)
            z_ret = sp - (t - r_phys)
            poly, xi_bar, z_bar, s_xi, s_z = self._comoving_frame_at(t - r_phys)
            w = (z_ret - z_bar) / s_z
            live &= (w >= tr.w_start) & (w < tr.w_end)

            centre = self._eval_poly_rows(poly, z_ret) + xi_bar
            half = 0.5 * (tr.u_end - tr.u_start) * s_xi * margin
            # A dead branch carries no density whatever band it is given, so it gets
            # zero width and _integrate_xi_region drops its contribution outright.
            # Its position still has to be finite and sane: on a dead column the
            # fixed point had no root to converge to and can wander (45 m has been
            # observed), which contributes nothing but produces absurd node
            # coordinates in debug output and risks landing on the 1/|r-r'| pole.
            # Park it a few band widths off the observation point instead: finite,
            # in a zero-density region, and never exactly at r = 0.
            parked = x + 4.0 * half
            bands.append((np.where(live, centre - half, parked),
                          np.where(live, centre + half, parked)))
            diag.append({'rad': rad, 'live': live, 'xp': xp, 'r': r_phys,
                         'centre': centre, 'half': half})

        self._branch_diag = {'sp': sp, 'branches': diag}
        return bands

    @staticmethod
    def _disjoint_bands(bands):
        """
        Make per-column intervals mutually disjoint by SUBTRACTION.

        The bands all have the same width, so two of them can only overlap
        one-sidedly: whichever starts first covers [lo1, hi1], and the part of the
        other not already covered is [hi1, hi2]. Clipping to that is exact, and a
        band fully inside another collapses to zero width.

        Deliberately not a union. Replacing two overlapping bands by their union is
        also exact, but it spreads the same xbins nodes over up to twice the width,
        doubling the transverse cell size precisely where the integrand peaks --
        which would undo Step 4. Subtraction never widens a band.

        NOTE: the returned bands are ordered by position PER COLUMN, so band j here
        is not branch j of _retarded_xi_bands -- the two swap wherever lo_B < lo_A.
        Aliveness must therefore be tested as (hi > lo) on the returned bands, never
        by pairing them with _branch_diag['branches'][j]['live'], which is recorded
        before this reordering.
        """
        if len(bands) < 2:
            return bands
        if len(bands) > 2:
            raise NotImplementedError('subtraction is exact for two bands only')
        (loA, hiA), (loB, hiB) = bands
        a_first = loA <= loB
        lo1 = np.where(a_first, loA, loB)
        hi1 = np.where(a_first, hiA, hiB)
        lo2 = np.where(a_first, loB, loA)
        hi2 = np.where(a_first, hiB, hiA)
        return [(lo1, hi1), (np.minimum(np.maximum(lo2, hi1), hi2), hi2)]

    def _integrate_xi_region(self, s, x, t, sp, ignore_vx, taper=None):
        """
        Integrate one s' region with the transverse nodes riding the density ribbon.

        Each localization branch gets its own set of x' nodes per s' column, so the
        inner trapezoid rule needs a per-column spacing. For a uniform grid
        trapz(y, dx=h) = h * trapz(y, dx=1), so the unit-spacing result is simply
        scaled by each column's dx.
        """
        bands = self._disjoint_bands(self._retarded_xi_bands(s, x, t, sp))
        nx = self.integration_params.xbins
        frac = np.linspace(0.0, 1.0, nx)

        dE = xk = 0.0
        meshes = []
        for lo, hi in bands:
            width = hi - lo
            alive = width > 0.0
            if not alive.any():
                continue
            xp_mesh = lo[None, :] + frac[:, None] * width[None, :]
            sp_mesh = np.broadcast_to(sp[None, :], xp_mesh.shape).copy()
            dxp = width / (nx - 1)

            gz, gx = self.get_CSR_integrand(s=s, t=t, x=x, xp=xp_mesh, sp=sp_mesh,
                                            ignore_vx=ignore_vx, taper=taper)
            # A zero-width column has dxp = 0 and contributes nothing. Select it out
            # rather than multiplying by zero, so a degenerate node that landed on
            # the 1/|r-r'| pole cannot turn the whole region into a nan.
            iz = np.where(alive, np.trapz(y=gz, axis=0) * dxp, 0.0)
            ix = np.where(alive, np.trapz(y=gx, axis=0) * dxp, 0.0)
            dE += -self.CSR_scaling * np.trapz(y=iz, x=sp)
            xk += self.CSR_scaling * np.trapz(y=ix, x=sp)
            meshes.append((xp_mesh, sp_mesh, gz, gx))

        if not meshes:
            z = np.zeros((nx, len(sp)))
            return (0.0, 0.0, z, np.broadcast_to(sp[None, :], z.shape).copy(), z, z)
        return (dE, xk) + tuple(np.concatenate(a, axis=0) for a in zip(*meshes))

    def _near_patch_radii(self, s, s3, s4):
        """
        (R1, R2) of the polar near-field patch, or None if it is disabled.

        R2 is clipped so the disc stays strictly inside the s' region that owns it,
        which keeps the partition of unity exact: the Cartesian piece carries weight
        w, the disc carries 1 - w, and 1 - w vanishes for |r - r'| > R2.
        """
        n_sig = self.integration_params.near_patch
        if not n_sig:
            return None
        R2 = n_sig * self.beam._sigma_x_transform
        R2 = min(R2, 0.4 * (s4 - s), 0.4 * (s - s3))
        if R2 <= 0:
            return None
        return 0.5 * R2, R2

    def _integrate_near_patch(self, s, x, t, R1, R2, ignore_vx):
        """
        Integrate the near field on a polar mesh centred on x' = x, s' = s.

        The integrand diverges as 1/|r - r'| there. Verified numerically: along rays
        out of that point, integrand * |r - r'| is constant over four decades. The
        polar area element r dr dphi cancels that factor exactly, leaving a bounded
        smooth integrand -- which is what the trapezoid rule needs. A uniform
        Cartesian mesh instead accumulates equal contributions per decade of r and
        never converges.

        Radial nodes sit at cell midpoints so r = 0 is never evaluated. phi is
        periodic, so a plain sum is already the trapezoid rule.
        """
        nr = self.integration_params.near_patch_nr
        nphi = self.integration_params.near_patch_nphi

        dr = R2 / nr
        r = (np.arange(nr) + 0.5) * dr
        dphi = 2.0 * np.pi / nphi
        phi = np.arange(nphi) * dphi

        xp_mesh = x + r[:, None] * np.cos(phi)[None, :]
        sp_mesh = s + r[:, None] * np.sin(phi)[None, :]

        gz, gx = self.get_CSR_integrand(s=s, t=t, x=x, xp=xp_mesh, sp=sp_mesh,
                                        ignore_vx=ignore_vx,
                                        taper=(R1, R2, True))
        jac = r[:, None] * dr * dphi
        dE = -self.CSR_scaling * np.sum(gz * jac)
        xk = self.CSR_scaling * np.sum(gx * jac)
        return dE, xk, xp_mesh, sp_mesh, gz, gx

#    @profile
    def get_CSR_wake(self, s, x, debug = False):

        t = self.beam.position

        #if t >= 0.5:
        #    print('')

        sigma_z = self.beam._sigma_z
        sigma_x = self.beam._sigma_x
        tan_theta = self.beam._slope[0]

        #TODO： why?
        x0 = (s-t)*self.beam._slope[0]
        xmean = self.beam._mean_x

        
        ######### For Debug ########################################################## 
        if np.abs(tan_theta) <= 1:  # if theta <45 degre, the chirp band can be ignored. theta is the angle in z-x plane
            ignore_vx = False
        else:
            ignore_vx = False

        ############################################################################

        chirp_band = False

        if np.abs(tan_theta) <= 1:  # if chirp is small, the chirp band can be ignored. theta is the angle in z-x plane
            s2 = s - 500 * sigma_z
            s3 = s - 20*sigma_z
            s4 = s + 5 * sigma_z
            x1_w = x0 - 20 * sigma_x
            x2_w = x0 + 20 * sigma_x

            x1_n = x0 - 10 * sigma_x
            x2_n = x0 + 10 * sigma_x

        else:
            chirp_band = True
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
        
        s1 = np.max((0, s2 - self.integration_params.n_formation_length * self.formation_length))

        if self.use_smooth_deposit and self.integration_params.xi_bands:
            # The s' decomposition above is kept as-is; only the transverse extents
            # change. The chirp case's regions 3 and 4 tiled x' over the *same* s'
            # range (sp3), so once both are replaced by the same ribbon they must be
            # merged into one region or the ribbon would be counted twice.
            nz = self.integration_params.zbins
            sps = [np.linspace(a, b, nz) for a, b in ((s1, s2), (s2, s3), (s3, s4))]

            # Only the last region straddles s' = s, so only it owns the 1/|r-r'|
            # singularity and gets the polar patch.
            radii = self._near_patch_radii(s, s3, s4)
            tapers = [None, None, None if radii is None else (*radii, False)]

            parts = [self._integrate_xi_region(s, x, t, spk, ignore_vx, tp)
                     for spk, tp in zip(sps, tapers)]
            if radii is not None:
                parts.append(self._integrate_near_patch(s, x, t, *radii, ignore_vx))

            if debug:
                return {'mode': 'xi_bands',
                        'near_patch_radii': radii,
                        'sp': list(sps),
                        'xp_mesh': [p[2] for p in parts],
                        'sp_mesh': [p[3] for p in parts],
                        'integrand_z': [p[4] for p in parts],
                        'integrand_x': [p[5] for p in parts]}
            return sum(p[0] for p in parts), sum(p[1] for p in parts)

        if chirp_band:
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
            sp1 = np.linspace(s1, s2, self.integration_params.zbins)
            sp2 = np.linspace(s2, s3, self.integration_params.zbins)
            sp3 = np.linspace(s3, s4, self.integration_params.zbins)
            xp_w = np.linspace(x1_w, x2_w, 2*self.integration_params.xbins)
            xp_n = np.linspace(x1_n, x2_n, self.integration_params.xbins)

            [xp_mesh1, sp_mesh1] = np.meshgrid(xp_w, sp1, indexing='ij')
            [xp_mesh2, sp_mesh2] = np.meshgrid(xp_n, sp2, indexing='ij')
            [xp_mesh3, sp_mesh3] = np.meshgrid(xp_n, sp3, indexing='ij')

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
          
          
    def _comoving_fields(self, xval, zval, tval):
        """
        (rho, drho/dx, drho/dz|_x, vx, dvx/dx) from the co-moving history, in
        physical units. Thin binding of the tracker's metadata onto the numba
        kernel; see interp3D.interpolate3D_comoving_fields.
        """
        tr = self.DF_tracker
        return interpolate3D_comoving_fields(
            xval, zval, tval,
            tr.data_density_interp, tr.data_density_u_interp,
            tr.data_density_w_interp, tr.data_vx_interp, tr.data_vx_u_interp,
            tr.poly_coeffs_interp, tr.xi_bar_arr, tr.sigma_xi_arr,
            tr.z_bar_arr, tr.sigma_z_arr,
            tr.u_start, tr.delta_u, tr.w_start, tr.delta_w,
            tr.min_x, tr.delta_x)

    def get_CSR_integrand(self,s ,x, t, sp, xp, ignore_vx = False, taper = None):
        """
        taper: (R1, R2, keep_inner) — multiply the integrands by a radial partition
        function of |r - r'|. With keep_inner False the weight rises 0 -> 1 across
        [R1, R2], removing the neighbourhood of the r' -> r singularity; with True
        it is the complement. The two pieces sum to the untapered integral exactly.
        """

        sp_flat = sp.ravel()
        xp_flat = xp.ravel()

        if self.use_comoving:
            vx = self._comoving_fields(np.array([x]), np.array([s - t]),
                                       np.array([t]))[3][0]
        elif self.use_smooth_deposit:
            vx = interpolate3D_transformed(
                xval=np.array([x]), zval=np.array([s - t]), tval=np.array([t]),
                data=self.DF_tracker.data_vx_interp,
                poly_coeffs=self.DF_tracker.poly_coeffs_interp,
                min_xi_arr=self.DF_tracker.min_xi_arr, min_z_arr=self.DF_tracker.min_z_arr, min_t=self.DF_tracker.min_x,
                delta_xi_arr=self.DF_tracker.delta_xi_arr, delta_z_arr=self.DF_tracker.delta_z_arr, delta_t=self.DF_tracker.delta_x)[0]
        else:
            vx = interpolate3D(xval=np.array([t]), yval=np.array([x]), zval=np.array([s-t]),
                                 data=self.DF_tracker.data_vx_interp,
                                 min_x=self.DF_tracker.min_x, min_y=self.DF_tracker.min_y,
                                 min_z=self.DF_tracker.min_z,
                                 delta_x=self.DF_tracker.delta_x, delta_y=self.DF_tracker.delta_y,
                                 delta_z=self.DF_tracker.delta_z)[0]

        X0_s = interpolate1D(xval = np.array([s]), data = self.lattice.coords[:, 0], min_x = self.lattice.min_x,
                             delta_x = self.lattice.delta_x)[0]
        X0_sp = interpolate1D(xval = sp_flat, data = self.lattice.coords[:, 0], min_x = self.lattice.min_x,
                              delta_x = self.lattice.delta_x)
        Y0_s = interpolate1D(xval = np.array([s]), data = self.lattice.coords[:, 1], min_x = self.lattice.min_x,
                             delta_x = self.lattice.delta_x)[0]
        Y0_sp = interpolate1D(xval = sp_flat, data = self.lattice.coords[:, 1], min_x = self.lattice.min_x,
                              delta_x = self.lattice.delta_x)
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


        r_minus_rp_x = X0_s - X0_sp + x * n_vec_s_x - xp_flat * n_vec_sp_x
        r_minus_rp_y = Y0_s - Y0_sp + x * n_vec_s_y - xp_flat * n_vec_sp_y
        r_minus_rp = np.sqrt(r_minus_rp_x**2 + r_minus_rp_y**2)


        rho_sp = np.zeros(sp_flat.shape)
        for count in range(self.lattice.Nelement):
            if count == 0:
                rho_sp[sp_flat < self.lattice.distance[count]] = self.lattice.rho[count]
            else:
                rho_sp[(sp_flat < self.lattice.distance[count]) & (sp_flat >= self.lattice.distance[count - 1])] = self.lattice.rho[count]

        t_ret = t - r_minus_rp

        if self.use_comoving:
            # One pass returns all five fields: they share the frame construction
            # and the stencil indices, so five separate calls would redo that work.
            # The chain rule for d rho/dz is applied inside, with the time-blended
            # p', which the pre-baked per-snapshot version could not do.
            z_ret = sp_flat - t_ret
            (density_ret, density_x_ret, density_z_ret,
             vx_ret, vx_x_ret) = self._comoving_fields(xp_flat, z_ret, t_ret)

        elif self.use_smooth_deposit:
            # Use transformed interpolation — query in physical (x, z) frame,
            # transform to xi internally using per-timestep polynomials
            z_ret = sp_flat - t_ret

            density_ret = interpolate3D_transformed(
                xval=xp_flat, zval=z_ret, tval=t_ret,
                data=self.DF_tracker.data_density_interp,
                poly_coeffs=self.DF_tracker.poly_coeffs_interp,
                min_xi_arr=self.DF_tracker.min_xi_arr, min_z_arr=self.DF_tracker.min_z_arr, min_t=self.DF_tracker.min_x,
                delta_xi_arr=self.DF_tracker.delta_xi_arr, delta_z_arr=self.DF_tracker.delta_z_arr, delta_t=self.DF_tracker.delta_x)

            density_x_ret = interpolate3D_transformed(
                xval=xp_flat, zval=z_ret, tval=t_ret,
                data=self.DF_tracker.data_density_x_interp,
                poly_coeffs=self.DF_tracker.poly_coeffs_interp,
                min_xi_arr=self.DF_tracker.min_xi_arr, min_z_arr=self.DF_tracker.min_z_arr, min_t=self.DF_tracker.min_x,
                delta_xi_arr=self.DF_tracker.delta_xi_arr, delta_z_arr=self.DF_tracker.delta_z_arr, delta_t=self.DF_tracker.delta_x)

            density_z_stored = interpolate3D_transformed(
                xval=xp_flat, zval=z_ret, tval=t_ret,
                data=self.DF_tracker.data_density_z_interp,
                poly_coeffs=self.DF_tracker.poly_coeffs_interp,
                min_xi_arr=self.DF_tracker.min_xi_arr, min_z_arr=self.DF_tracker.min_z_arr, min_t=self.DF_tracker.min_x,
                delta_xi_arr=self.DF_tracker.delta_xi_arr, delta_z_arr=self.DF_tracker.delta_z_arr, delta_t=self.DF_tracker.delta_x)

            # Chain rule already applied in deposit_smooth.py (density_z_stored is lab-frame)
            density_z_ret = density_z_stored

            vx_ret = interpolate3D_transformed(
                xval=xp_flat, zval=z_ret, tval=t_ret,
                data=self.DF_tracker.data_vx_interp,
                poly_coeffs=self.DF_tracker.poly_coeffs_interp,
                min_xi_arr=self.DF_tracker.min_xi_arr, min_z_arr=self.DF_tracker.min_z_arr, min_t=self.DF_tracker.min_x,
                delta_xi_arr=self.DF_tracker.delta_xi_arr, delta_z_arr=self.DF_tracker.delta_z_arr, delta_t=self.DF_tracker.delta_x)

            vx_x_ret = interpolate3D_transformed(
                xval=xp_flat, zval=z_ret, tval=t_ret,
                data=self.DF_tracker.data_vx_x_interp,
                poly_coeffs=self.DF_tracker.poly_coeffs_interp,
                min_xi_arr=self.DF_tracker.min_xi_arr, min_z_arr=self.DF_tracker.min_z_arr, min_t=self.DF_tracker.min_x,
                delta_xi_arr=self.DF_tracker.delta_xi_arr, delta_z_arr=self.DF_tracker.delta_z_arr, delta_t=self.DF_tracker.delta_x)

        else:
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

        ## Todo: More accurate vx, maybe add vs
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

        scale_term =  1 + xp_flat*rho_sp


        velocity_x = vs * tau_vec_s_x + vx * n_vec_s_x
        velocity_y = vs * tau_vec_s_y + vx * n_vec_s_y

        velocity_ret_x = vs_ret * tau_vec_sp_x + vx_ret * n_vec_sp_x
        velocity_ret_y = vs_ret * tau_vec_sp_y + vx_ret * n_vec_sp_y

        #velocity_partial_t_x = vs_t * tau_vec_sp_x + vx_t * n_vec_sp_x
        #velocity_partial_t_y = vs_t * tau_vec_sp_y + vx_t * n_vec_sp_y

        nabla_density_ret_x = density_x_ret  * n_vec_sp_x + density_z_ret / scale_term * tau_vec_sp_x
        nabla_density_ret_y = density_x_ret * n_vec_sp_y + density_z_ret / scale_term * tau_vec_sp_y

        div_velocity = vs_s_ret + vx_x_ret  #???

        # Todo: Consider using general form
        ## general form
        part1 = velocity_x * velocity_ret_x + velocity_y * velocity_ret_y
        CSR_numerator1 = scale_term * ((velocity_x - part1 * velocity_ret_x) * nabla_density_ret_x  + \
                          (velocity_y - part1 * velocity_ret_y)*nabla_density_ret_y)
        CSR_numerator2 = -scale_term * part1 * density_ret * div_velocity
        #CSR_numerator3 = scale_term * density_ret * (velocity_partial_t_x * velocity_x + velocity_partial_t_y * velocity_y)

        #CSR_denominator = r_minus_rp

        #self.CSR_integrand = CSR_numerator1/CSR_denominator + (CSR_numerator2 + CSR_numerator3)/CSR_denominator
        CSR_integrand_z = CSR_numerator1 /r_minus_rp + (CSR_numerator2) / r_minus_rp



        #CSR_numerator1 = scale_term * (((n_vec_sp_x * tau_vec_s_x + n_vec_sp_y * tau_vec_s_y) +
        #                                (vx - vx_ret) * (tau_vec_sp_x * tau_vec_s_x + tau_vec_sp_y * tau_vec_s_y)) * density_x_ret -
        #                               vx_ret * (n_vec_sp_x * tau_vec_s_x + n_vec_sp_y * tau_vec_s_y)/scale_term * density_z_ret)

        #CSR_numerator2 = -((tau_vec_sp_x * tau_vec_s_x + tau_vec_sp_y * tau_vec_s_y) +
        #                   (vx - vx_ret) * (n_vec_s_x * tau_vec_sp_x + n_vec_s_y * tau_vec_sp_y)) * density_ret * vx_x_ret

        #CSR_numerator3 = scale_term * density_ret * (velocity_partial_t_x * velocity_x + velocity_partial_t_y * velocity_y)

        #CSR_denominator = r_minus_rp

        #CSR_integrand_z = CSR_numerator1/CSR_denominator + (CSR_numerator2 + CSR_numerator3)/CSR_denominator

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
        #CSR_integrand_x = W1

        if taper is not None:
            R1, R2, keep_inner = taper
            u = np.clip((r_minus_rp - R1) / (R2 - R1), 0.0, 1.0)
            # C2 smootherstep, so the tapered integrand has no curvature jump
            w = u * u * u * (u * (6.0 * u - 15.0) + 10.0)
            if keep_inner:
                w = 1.0 - w
            CSR_integrand_z = CSR_integrand_z * w
            CSR_integrand_x = CSR_integrand_x * w

        CSR_integrand_x = CSR_integrand_x.reshape(xp.shape)
        CSR_integrand_z = CSR_integrand_z.reshape(xp.shape)



        return CSR_integrand_z, CSR_integrand_x

    def dump_beam(self, label):
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



















