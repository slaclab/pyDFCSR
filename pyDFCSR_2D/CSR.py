import os
import time
from bmadx import  Drift, SBend, Quadrupole
from .tracking import SextupoleKDK
from .tools import dict2hdf5
import h5py
import numpy as np
from mpi4py import MPI

from .beams import Beam
# from .deposit import histogram_cic_1d, histogram_cic_2d
from .deposit import DF_tracker
from .deposit_smooth import DF_tracker_smooth, DF_tracker_comoving
from .interp1D import interpolate1D
from .integrand_kernel import csr_integrand_kernel
from .interp3D import (interpolate3D, interpolate3D_transformed,
                       interpolate3D_comoving_fields, interpolate3D_comoving_ibp)
from .lattice import Lattice  # , get_referece_traj
from .lookup import lookup_vec
from .schedule import SCHEDULE_ELEMENT_KEYS
from .waist import (scan_waists, sigma_from_coords, format_report,
                    retention_schedule)
from .params import Integration_params, CSR_params
# from .physical_constants import c, e, qe, me, MC2
from .r_gen6 import r_gen6
#from line_profiler_pycharm import profile
# from .tools import (find_nearest_ind, full_path, isotime, plot_2D_contour,
#                     plot_surface)
from .tools import full_path, isotime
from .twiss_R import twiss_R
from .yaml_parser import parse_yaml



def degenerate_margin_factor(sin2a, factor, lo=0.8, hi=0.95):
    """
    Ribbon-margin multiplier near the band-branch degeneracy: 1 for |sin 2a| <= lo, `factor` for
    |sin 2a| >= hi, a C1 smoothstep in between (a jump would make the wake discontinuous in s).
    """
    if not factor or factor <= 1.0:
        return 1.0
    u = (abs(float(sin2a)) - lo) / (hi - lo)
    u = min(max(u, 0.0), 1.0)
    return 1.0 + (float(factor) - 1.0) * u * u * (3.0 - 2.0 * u)


class CSR2D:
    """
    The main class to calculate 2D CSR
    """

    def __init__(self, input_file=None, parallel = False):

        self.timestamp = isotime()
        self._parallel_requested = bool(parallel)
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
        # step_control defaults to mode 'auto' with midpoint kicks (since 2026-11-06), the
        # schedule every validated run used. A config that wants the old uniform stepping
        # must now say step_control: {mode: legacy}.
        sc = dict(input.get('step_control') or {})
        sc.setdefault('mode', 'auto')
        if sc['mode'] == 'auto':
            sc.setdefault('kick_interval', 'midpoint')
        sigma0 = None
        if sc.get('mode') == 'auto':
            b = self.beam
            sigma0 = sigma_from_coords(b.x, b.px, b.particle.y, b.particle.py, b.z, b.pz)
        if sc.get('kick_placement') == 'csr1d':
            # kicks where the CSR is: a 1D pre-pass (seconds) gives the wake strength along s,
            # schedule.kicks_from_profile places the 2D kicks on it. Rank 0 runs it, all share it.
            if sc.get('mode') != 'auto':
                raise ValueError("step_control kick_placement 'csr1d' needs mode 'auto'")
            from .csr1d import csr1d_kick_profile
            par = getattr(self, '_parallel_requested', False) and MPI.COMM_WORLD.Get_size() > 1
            prof = None
            if (not par) or MPI.COMM_WORLD.Get_rank() == 0:
                prof = csr1d_kick_profile(input, sc.get('kick_profile_1d'),
                                          n_particles=sc.get('kick_prepass_particles'))
            if par:
                prof = MPI.COMM_WORLD.bcast(prof, root=0)
            sc['kick_profile'] = prof
        elif sc.get('kick_placement', 'kappa') != 'kappa':
            raise ValueError(f"step_control kick_placement must be 'kappa' or 'csr1d', got "
                             f"{sc.get('kick_placement')!r}")
        self.lattice = Lattice(input['input_lattice'], step_control=sc, sigma0=sigma0)

        if 'particle_deposition' in input:
            deposition_config = input['particle_deposition']
            # default 'bspline_comoving' since 2026-11-06; 'legacy' (CIC) must be asked for
            method = deposition_config.get('method', 'bspline_comoving')
            if method == 'bspline_comoving':
                self.DF_tracker = DF_tracker_comoving(deposition_config)
            elif method == 'bspline_fft':
                self.DF_tracker = DF_tracker_smooth(deposition_config)
            else:
                self.DF_tracker = DF_tracker(deposition_config)
        else:
            self.DF_tracker = DF_tracker_comoving({})
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

    def _resolve_auto_frame(self):
        """
        particle_deposition.frame 'auto': decide 'peak' or 'quantile' from the initial beam, once.
        The peak window is previewed on the beam; the 1D wake its dropped charge would radiate at
        the exit of the FIRST bend (csr1d, steady state developed over the whole bend) is compared
        with the full beam's (deposit_smooth.choose_auto_frame). Printed on rank 0; the choice is
        kept for the run and recorded in statistics.h5 (quantile_frame attrs).
        """
        from .csr1d import CSR1DGeometry
        from .deposit_smooth import choose_auto_frame
        tr = self.DF_tracker
        geom1d = CSR1DGeometry(self.lattice.lattice_config, self.lattice.distance)
        first = next((el for el in geom1d.elements if el['bend']), None)
        if first is None:
            geom = dict(case='none', R=np.inf, phi=0.0, phim=0.0, x_R=0.0, bend=None)
        else:
            geom = geom1d.at(first['end'], first['name'])
        choice, f_peak, fl, ff = choose_auto_frame(tr, self.beam.z, geom)
        tr.resolve_frame(choice, (choice, f_peak, fl, ff, geom['bend']))
        if (not getattr(self, 'parallel', False)) or getattr(self, 'rank', 0) == 0:
            where = (f"at the exit of {first['name']} (R = {geom['R']:.2f} m)" if first is not None
                     else 'nowhere: the lattice has no bend')
            print(f"  [frame auto] '{choice}': the peak window (+-{tr.peak_halfwidth:g} FWHM around the "
                  f"highest current peak) holds {100 * f_peak:.1f}% of the charge; the dropped "
                  f"{100 * (1 - f_peak):.1f}% would carry {100 * fl:.1f}% of the 1D energy loss and "
                  f"{100 * ff:.1f}% of the 1D field on the window {where}; threshold "
                  f"auto_pedestal_max {tr.auto_pedestal_max:g}.", flush=True)

    def initialization(self):
        """
        deposit the initial beam
        :return:
        """
        if self.use_comoving and getattr(self.DF_tracker, 'frame', None) == 'auto':
            self._resolve_auto_frame()
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
                                                 'CSR_computation', 'step_control']
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
        # Schedule-only keys must not reach bmad-x: whatever is left in input_dic is forwarded
        # as **kwargs to SBend/Quadrupole/Sextupole, which would raise on an unexpected name.
        for _k in SCHEDULE_ELEMENT_KEYS:
            input_dic.pop(_k, None)
        L = input_dic.pop('L')
        type = input_dic.pop('type')
        # NUM_STEPS in the YAML is for the WHOLE element (as in Bmad); the run loop tracks it
        # in pieces of length DL, so each piece gets its share. Passing the full count to every
        # piece made the integration step depend on the CSR step size.
        if 'NUM_STEPS' in input_dic and L > 0:
            input_dic['NUM_STEPS'] = max(1, int(round(input_dic['NUM_STEPS'] * DL / L)))


        # 'vbend' is a dipole that bends out of the (x, z) plane (TILT = pi/2, e.g. the BC20
        # vertical wiggler). It is tracked as an SBend but is a straight element everywhere
        # else -- reference orbit, formation length, schedule and the CSR integral -- because
        # the 2D code cannot represent a vertical bend.
        if type in ('dipole', 'vbend'):
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

            # was undefined unless given in the YAML, so a bend tracked in one piece
            # (entrance and exit) raised NameError
            FRINGE_AT = input_dic.pop('FRINGE_AT', 'both_ends')


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
            # not bmadx.Sextupole: its drift and kick are wrong for canonical px (tracking.py)
            K2 = input_dic.pop('K2')
            element = SextupoleKDK(L=DL, K2=K2, **input_dic)
        print(element)
        return element

    def _ends_element(self, s_end, ele_count):
        """
        True if a step ending at s_end ends element ele_count, so its exit edge belongs to it.

        The exit edge used to be applied only by the split step that crosses into the next
        element, and that piece is skipped when it is shorter than 1e-6 m. When a node lies ON
        the boundary -- always in modes 'manual' and 'auto', and whenever L is a multiple of
        step_size in 'legacy' -- no step carried the exit edge and E2 was silently dropped.
        The tolerance is that same 1e-6, so exactly one of the two paths applies it.
        """
        return abs(float(self.lattice.distance[ele_count]) - s_end) <= 1.0e-6

#    @profile
    def build_retention_plan(self):
        """
        Precompute how far back the history must reach at every point in the lattice.

        pop_left_DF only ever pops, so truncating on the INSTANTANEOUS L_f is safe only
        while sigma_z decreases monotonically. At a waist L_f collapses and then
        recovers, and the snapshots the later steps need are already gone -- measured as
        a 0.123 m shortfall at shear 20, which interp3D absorbed silently by clamping to
        the oldest surviving snapshot. The plan supplies the suffix-minimum floor
        instead, so nothing needed later is ever discarded.

        Failure is non-fatal and leaves self._retention = None, which restores the old
        instantaneous behaviour exactly.
        """
        self._retention = None
        self._retention_short = 0
        self._retention_worst = 0.0
        if not self.use_comoving:
            return
        try:
            b = self.beam
            sigma0 = sigma_from_coords(b.x, b.px, b.particle.y, b.particle.py,
                                       b.z, b.pz)
            self._retention = retention_schedule(
                self.lattice.lattice_config, sigma0,
                self.integration_params.n_formation_length)
            plan = self._retention
            print(f'  [retention] precomputed floor from linear optics: max depth '
                  f'{plan.max_depth:.4f} m -> '
                  f'{plan.max_snapshots(self.lattice.step_size)} snapshots at '
                  f'step_size {self.lattice.step_size:g} m (safety {plan.safety:g})')
        except Exception as exc:                       # noqa: BLE001
            print(f'  [retention] precomputation skipped '
                  f'({type(exc).__name__}: {exc}); falling back to instantaneous L_f')

    def _retention_floor(self):
        f = (self._retention.floor_at(self.beam.position)
             if self._retention is not None else None)
        # drift_reach: from a bend's entrance until the next bend, never discard the
        # history of that bend -- it is the source of the exit wake downstream
        if self.integration_params.drift_reach and self.afterbend:
            sb = getattr(self, 'bend_start_rec', None)
            if sb is not None:
                f = sb if f is None else min(f, sb)
        pb = self._prev_bend_reach_start()
        if pb is not None:
            f = pb if f is None else min(f, pb)
        return f

    def _check_retention(self):
        """
        Did the retained history actually cover what the integration asked for?

        The plan is linear optics, so it can be wrong. This compares the ACTUAL
        requirement against what survived, every step, so a bad prediction cannot pass
        unnoticed -- the failure mode being guarded is precisely the silent clamp in
        interpolate3D_comoving_fields (k < 0 -> k = 0), which substitutes the oldest
        surviving snapshot with no complaint.
        """
        if self.formation_length is None or not len(self.DF_tracker.time_log):
            return
        need = max(0.0, self.beam.position
                   - self.integration_params.n_formation_length * self.formation_length)
        sb = self._drift_reach_start()
        if sb is not None:
            need = min(need, sb)
        pb = self._prev_bend_reach_start()
        if pb is not None:
            need = min(need, pb)
        have = self.DF_tracker.time_log[0]
        if have > need + 1e-9:
            self._retention_short += 1
            self._retention_worst = max(self._retention_worst, have - need)

    def report_schedule(self):
        """
        Print the realised schedule, and REFUSE up front if it will not fit in memory.

        §6bb measured a chicane demanding 22-32 GB of history under uniform stepping. Failing at
        step 0 with the number and the knob that would fix it beats an OOM kill at step 8000, and
        the total is known before tracking precisely because the schedule is precomputed.
        """
        sch = getattr(self.lattice, 'schedule', None)
        if sch is None:
            return
        print(sch.summary())
        dep = self.DF_tracker
        nx = getattr(dep, 'xbins', 0)
        nz = getattr(dep, 'zbins', 0)
        per = 5 * nx * nz * 8 / 1e9 if (nx and nz) else 0.0
        plan = getattr(self, '_retention', None)
        # worst-case retained snapshots: the deepest retention window at the finest local spacing
        n_hold = sch.n_snap
        if plan is not None:
            fine = float(np.min(sch.dl[1:])) if sch.n_steps else self.lattice.step_size
            n_hold = min(sch.n_snap, int(np.ceil(plan.max_depth / max(fine, 1e-12))) + 8)
        gb = n_hold * per
        budget = (self.lattice.step_control or {}).get('max_memory_gb')
        if per:
            print(f'  [schedule] history at most {n_hold} snapshots x {per*1e3:.2f} MB '
                  f'= {gb:.2f} GB at {nx}x{nz} deposition')
        if budget is not None and gb > float(budget):
            raise MemoryError(
                f'schedule needs up to {gb:.2f} GB of density history but max_memory_gb is '
                f'{float(budget):.2f}. Raise h_max, raise max_memory_gb, or reduce the '
                f'deposition grid ({nx}x{nz} costs {per*1e3:.2f} MB per snapshot).')
        if sch.mode != 'legacy' and not self.use_comoving:
            raise ValueError(
                f"step_control mode '{sch.mode}' produces non-uniform snapshot times, but the "
                f"deposition method is not 'bspline_comoving'. The legacy and bspline_fft "
                f"interpolation paths still assume uniform times and would read the history "
                f"with the wrong index.")

    def report_waists(self):
        """
        Warn, before tracking, about longitudinal waists step_size cannot resolve.

        The co-moving frame blend cannot reconstruct the retarded density across an
        unresolved waist, and it fails SILENTLY: progress 6n measured sigma_z = 50.00 and
        50.09 um at the endpoints of an interval whose true minimum is 2.51 um, so
        nothing in the stored history says a waist was crossed. 6v then ruled out the
        s'-quadrature as the cause, leaving sampling. This is the cheapest useful
        response -- tell the user which results to distrust and what step_size would fix
        it, rather than silently changing their run.

        Only meaningful for the co-moving path, whose frame blend is the thing at risk.
        Never fatal: a diagnostic that can abort a run is worse than the problem, and
        this one is linear optics on the design lattice, so it can be wrong.
        """
        if not self.use_comoving:
            return
        try:
            b = self.beam
            sigma0 = sigma_from_coords(b.x, b.px, b.particle.y, b.particle.py,
                                       b.z, b.pz)
            report = scan_waists(self.lattice.lattice_config, sigma0,
                                 self.lattice.step_size)
            text = format_report(report)
            if text:
                print(text)
        except Exception as exc:                       # noqa: BLE001
            print(f'  [waist scan] skipped ({type(exc).__name__}: {exc})')

    def run(self, stop_time = None, debug = False):

        # `run` is not resumable. It restarts its `for ele in lattice_config` loop at the first
        # element and `step_count` at 1, so a second call re-tracks an already-advanced beam from
        # the lattice entrance -- asked to stop at s = 0.750 after a first call to 0.450, it landed
        # at 0.778 with a density history that was not one forward pass. Refuse instead of
        # returning a plausible-looking wrong answer; construct a fresh CSR2D per stop_time.
        if getattr(self, '_has_run', False):
            raise RuntimeError(
                f'run() has already been called on this CSR2D (beam is at s = '
                f'{self.beam.position:.6f} m). It is not resumable: it would restart tracking '
                f'from the lattice entrance with the current beam. Construct a new CSR2D for '
                f'each stop_time.')
        self._has_run = True

        if (not self.parallel) or (self.rank == 0):
            print('Starting the DFCSR run')
            self.report_waists()
            self.report_schedule()
        self.build_retention_plan()

        step_count = 1
        # arc position of the previous CSR kick; must persist ACROSS elements, since the
        # per-element `step` counter restarting is exactly what broke the old DL*nsep
        self._s_last_kick = self.beam.position

        # Every per-step length and decision now comes from the precomputed schedule.
        # `step_count` starts at 1 and increments once per step, so it IS the schedule's node
        # index -- node i is the end of step i, and node 0 is the s = 0 entrance handled by
        # initialization(). Under mode 'legacy' the schedule reproduces the old uniform node set,
        # kick cadence and steps_per_element exactly.
        sched = self.lattice.schedule
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

                # the bend left behind is still radiating into this one: its exit tail
                # arrives here (Integration_params.prev_bend_reach)
                self.prev_bend_start = (getattr(self, 'bend_start_rec', None)
                                        if self.afterbend else None)
                self.afterbend = True
                self.R_rec = R
                self.phi_rec = angle
                # entrance of the bend just entered: in the drift that follows, the exit
                # transient's source is this arc (Integration_params.drift_reach)
                self.bend_start_rec = (float(self.lattice.distance[ele_count - 1])
                                       if ele_count > 0 else 0.0)

                self._refresh_formation_length(R)


            else:  # If not in a bend
                self.inbend = False
                self.prev_bend_start = None

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
                    element = self.get_bmadx_element(
                        ele = ele,  DL = DL_2, entrance = True,
                        exit = self._ends_element(self.beam.position + DL_2, ele_count))
                    self.beam.track(element, DL_2)
                    distance_in_current_ele += DL_2
                    skip_ele = False    # Reset the flag

                else:
                    DL_i = sched.dl[step_count]
                    # step 0 lands here only in the first element, whose entrance edge was
                    # never applied either
                    element = self.get_bmadx_element(
                        ele = ele,  DL = DL_i, entrance = (step == 0),
                        exit = self._ends_element(self.beam.position + DL_i, ele_count))
                    # Propagate beam for one step
                    self.beam.track(element, DL_i)
                    distance_in_current_ele += DL_i


                # sigma_z evolves within the element, so L_f has to be refreshed here
                # and not only on element entry -- it truncates the history below and
                # sets the integration reach in get_CSR_wake.
                self._refresh_formation_length(R)

                if (debug or self.CSR_params.compute_CSR) and sched.is_snap[step_count]:
                    # get the density functions
                    self.DF_tracker.get_DF(x=self.beam.x, z=self.beam.z, px=self.beam.px, t=self.beam.position)
                    # append the density functions to the log
                    self.DF_tracker.append_DF()
                    self._thin_drift_history()
                    # append 3D matrix for interpolation with the new DFs by interpolation
                    self.DF_tracker.append_interpolant(formation_length=self.formation_length,
                                                       n_formation_length=self.integration_params.n_formation_length,
                                                       min_start_time=self._retention_floor())
                    self.DF_tracker.build_interpolant()
                    self._check_retention()

                # Stop computing wakes once the beam is far enough past a bend that the
                # exit transient has died. The scale is R*phi of the bend just left, NOT
                # a multiple of the formation length -- which is what the old commented
                # form got wrong, and why it was never enabled. R*phi is the geometric
                # scale of the exit decay: Stupakov & Emma Eq. 10 gives
                # W ~ 1/(phi + 2 d/R), halving at d = R*phi/2, and the retarded slippage
                # saturates at the same place (both cross at d/R = 0.5000 phi).
                #
                # Measured on the 1 rad test dipole, on-axis peak relative to the exit
                # face: 0.338 at d = 0.1 R*phi, 0.173 at 0.2, 0.054 at 0.5. The real
                # decay is FASTER than Eq. 10 (0.338 vs 0.833 at d = 0.1) because Eq. 10
                # assumes phi << 1, so R*phi is a conservative cutoff for a strong bend.
                #
                # Counted, and reported once, because silently returning no wake is
                # exactly the failure mode that cost a set of wrong figures in 6f: a
                # skipped kick must be visible in the log.
                # abs() on BOTH: R_rec = L/angle carries the sign of the bend, so a
                # negative-angle dipole gave a NEGATIVE threshold and every step in its
                # downstream drift compared "distance > -0.5 m" -> true. Measured on the
                # chicane, whose B2 and B3 bend the other way: those two lost every kick
                # past the exit face while B1 and B4 kept five each. The arc length
                # |R*phi| is what the decay scale means; it cannot be signed.
                cutoff = self.integration_params.drift_cutoff
                CSR_blocker = bool(
                    cutoff and self._exit_transient() and self.R_rec
                    and distance_in_current_ele > cutoff * abs(self.R_rec)
                    * abs(self.phi_rec))
                if CSR_blocker:
                    # steps, not kicks: this tests every step, and only a subset of steps
                    # are kick nodes. Reporting it as "kicks skipped" overstated it 5x.
                    self._steps_blocked = getattr(self, '_steps_blocked', 0) + 1
                    if sched.is_kick[step_count]:
                        self._kicks_skipped = getattr(self, '_kicks_skipped', 0) + 1
                    if not getattr(self, '_warned_blocker', False):
                        self._warned_blocker = True
                        if (not self.parallel) or (self.rank == 0):
                            print(f'Beyond {cutoff:g} |R*phi| = '
                                  f'{cutoff * abs(self.R_rec) * abs(self.phi_rec):.4f} m '
                                  f'past the last bend; skipping CSR from here in '
                                  f'this drift.')
                
                
                if self.CSR_params.compute_CSR and (not CSR_blocker):
                    # was `step % nsep[ele_count] == 0`, with `step` restarting per element.
                    # The schedule encodes that cadence for mode 'legacy' and lets 'auto' and
                    # 'manual' choose kicks independently of the snapshot spacing -- which is the
                    # point, since a kick costs ~5000x what a snapshot costs.
                    if sched.is_kick[step_count]:
                        # calculate CSR mesh given beam shape
                        self.get_CSR_mesh()
                        # Calculate CSR on the mesh
                        if self.parallel:
                            self.calculate_2D_CSR_parallel()
                        else:
                            self.calculate_2D_CSR()
                        if self._quantile_frame() and ((not self.parallel) or self.rank == 0):
                            self._quantile_diagnostics()
                        # Apply CSR kick to the beam
                        if self.CSR_params.apply_CSR:
                            # The kick is a rectangle-rule quadrature of int(W ds), so the
                            # length must be the arc ACTUALLY covered since the previous
                            # kick. DL*nsep is wrong in two cases that both occur in
                            # ordinary lattices:
                            #   * the first kick of the run, applied after a single step
                            #     but weighted by nsep steps  (+200% at nsep = 3)
                            #   * element boundaries, where `step` restarts at 0 so the
                            #     gap since the last kick is shorter than nsep steps
                            #     (+50% measured), and where the boundary step is split
                            #     into DL_1 + DL_2 across two elements
                            # On a 4-element test lattice with nsep = 3 this over-weighted
                            # the integrated CSR kick by 14.3%.
                            if sched.kick_interval == 'midpoint':
                                # Centred on the sample point, so the quadrature is a midpoint
                                # rule (O(h^2)) rather than a trailing rectangle (O(h)). Needs
                                # the NEXT kick position, hence only possible with a precomputed
                                # schedule -- which is why §6dd deferred it.
                                L_kick = (sched.kick_hi[step_count]
                                          - sched.kick_lo[step_count])
                            else:
                                # Trailing: the arc actually covered since the previous kick.
                                # Must NOT be taken from the schedule -- beam.position
                                # accumulates by `position += step_size`, so it differs from the
                                # schedule node i*h in the last bits for 31 of 38 nodes, and
                                # substituting would break the bit-identity guarantee.
                                L_kick = self.beam.position - self._s_last_kick
                            self.beam.apply_wakes(self.dE_dct, self.x_kick,
                                              self.CSR_xrange_transformed, self.CSR_zrange, L_kick,
                                                  self.CSR_params.transverse_on,
                                                  x_shift=self._mesh_ridge(self.beam.z),
                                                  clamp_x=self.CSR_params.mesh_x_quantile is not None)
                            self._s_last_kick = self.beam.position
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

                # `>=` within a tolerance, not `>`: with step_control's force_nodes a node can land
                # EXACTLY on stop_time, and a strict `>` would then walk a full step past the one
                # position the caller asked for.
                if stop_time and self.beam.position >= stop_time - 1e-12 * max(abs(stop_time), 1.0):
                    # The node grid is fixed, so a requested stop_time between nodes cannot be
                    # hit exactly; the loop lands on the first node past it. That used to be
                    # SILENT and overshot by up to a full step -- at step_size 0.1,
                    # run(stop_time=0.60) landed at 0.700, 17% past the request with a 27%
                    # different beam slope, which confounded a whole convergence study in §6i.
                    over = self.beam.position - stop_time
                    if over > 1e-9 * max(abs(stop_time), 1.0):
                        print(f'  [run] stopped at s = {self.beam.position:.6f}, '
                              f'{over:.6f} m past the requested stop_time = {stop_time:.6f}. '
                              f'No node lands exactly there; assert on beam.position when '
                              f'comparing runs at different step sizes.')
                    return

            ele_prev = ele
            type_prev = type

            ele_count += 1

        self.dump_beam(label='end')
        self.write_statistics()


    def _mesh_ridge(self, z):
        """
        Lab-x offset of the beam ridge from the straight tilt fit, sigma_xi * g(w), from
        the newest snapshot's curvature shear (particle_deposition.curvature). Zero when
        the shear is off, so the mesh and the kick lookup are then exactly as before.
        """
        tr = self.DF_tracker
        if not (self.use_comoving and getattr(tr, 'curvature', False) and tr.sigma_z):
            return np.zeros_like(np.asarray(z, dtype=float))
        from .deposit_smooth import _curv_g_vec
        w = (np.asarray(z, dtype=float) - tr.z_bar) / tr.sigma_z
        if getattr(tr, 'frame', 'rms') in ('quantile', 'peak'):
            # tail particles lie outside the window; continuing g linearly there would
            # throw mesh rows (and the kick lookup) far off the beam. Hold it at the edge.
            w = np.clip(w, tr.w_start, tr.w_end)
        return tr.sigma_xi * _curv_g_vec(tr.curv_c2, tr.curv_c3, w, tr.curv_zlim)

    def get_CSR_mesh(self):
        """
        calculating the mesh of observation points by taking linear transformation
        (xmesh, zmesh) TWO 1D arrays representiong (x, z) coordinates on a linear transformed mesh
        :return:
        """

        x_transform = self.beam.x_transform
        p = self.beam.slope
        # With the curvature shear the beam is a banana around the straight fit; the
        # mesh rows must follow the ridge or the edge rows land outside the sheared
        # deposition grid (D1 5.01 m: row 11 at u' = 5.06, v_x = 0 there, wake 100x off).
        ridge = self._mesh_ridge(self.beam.z)
        x_transform = x_transform - ridge

        sig_x = np.std(x_transform)
        mean_x = np.mean(x_transform)
        sig_z = self.beam.sigma_z
        mean_z = self.beam.mean_z
        xlim = self.CSR_params.xlim
        zlim = self.CSR_params.zlim
        xbins = self.CSR_params.xbins
        zbins = self.CSR_params.zbins

        if (self.use_comoving and self.CSR_params.mesh_mode == 'uniform'
                and not getattr(self, '_taper_mesh_warned', False)
                and xlim > self.DF_tracker.u_flat() + 1e-9):
            # The outer mesh rows sit where the deposit's u edge taper has removed part of the
            # source charge (DF_tracker_comoving.u_taper), so their near field is weakened.
            # Few particles are there, but say so once.
            self._taper_mesh_warned = True
            print(f'  [CSR2D] note: the wake mesh spans +-{xlim} sigma_x, beyond the deposit\'s '
                  f'full-weight region |u| <= {self.DF_tracker.u_flat():.2f} (u_taper); the '
                  f'outermost rows see a tapered source. Lower CSR_computation.xlim, or raise '
                  f'particle_deposition xlim/xbins.')

        if self.CSR_params.mesh_mode == 'charge':
            # node density follows the projected charge (peakfind.charge_nodes), smoothed
            # over mesh_smooth deposition resolution lengths so the field just outside a
            # spike is covered too; the axes span every particle
            from .peakfind import charge_nodes
            cp = self.CSR_params
            res_z, res_x = self._deposition_resolution()
            zrange = charge_nodes(self.beam.z, zbins, cp.mesh_uniform_frac,
                                  smooth=cp.mesh_smooth * res_z, pad=cp.mesh_pad,
                                  max_ratio=cp.mesh_max_ratio)
            xq = x_transform
            if cp.mesh_x_quantile is not None:
                xq = np.clip(x_transform, *np.quantile(x_transform, [cp.mesh_x_quantile,
                                                                     1.0 - cp.mesh_x_quantile]))
            xrange = charge_nodes(xq, xbins, cp.mesh_uniform_frac,
                                  smooth=cp.mesh_smooth * res_x, pad=cp.mesh_pad,
                                  max_ratio=cp.mesh_max_ratio)
        else:
            zrange = np.linspace(mean_z - zlim * sig_z, mean_z + zlim * sig_z, zbins)
            xrange = np.linspace(mean_x - xlim * sig_x, mean_x + xlim * sig_x, xbins)

        # Todo: check the order
        xmesh_transform, zmesh = np.meshgrid(xrange, zrange, indexing='ij')

        xmesh_transform = xmesh_transform.flatten()
        zmesh = zmesh.flatten()

        xmesh = xmesh_transform +  np.polyval(p, zmesh) + self._mesh_ridge(zmesh)

        self.CSR_xmesh = xmesh
        self.CSR_zmesh = zmesh
        self.CSR_zrange = zrange
        self.CSR_xrange_transformed = xrange
        if self._quantile_frame():
            self._fit_beam_vx()

    def _quantile_frame(self):
        # frames 'quantile' and 'peak' share the windowed machinery (taper, mesh, observer vx, log)
        return self.use_comoving and getattr(self.DF_tracker, 'frame', 'rms') in ('quantile', 'peak')

    def _deposition_resolution(self):
        """
        (z, x) effective resolution of the newest density snapshot [m]: cell times
        sqrt(smoothing^2 + 2/3), the B-spline deposit and evaluation adding 1/3 cell^2 each.
        """
        tr = self.DF_tracker
        if not self.use_comoving or tr.sigma_z is None:
            return 0.0, 0.0
        sx = tr.smoothing_sigma
        sz = sx if getattr(tr, 'smoothing_sigma_z', None) is None else tr.smoothing_sigma_z
        sz = getattr(tr, 'sz_used', sz)      # after the smoothing_res_fwhm cap, if any
        return (tr.delta_w * tr.sigma_z * np.sqrt(sz * sz + 2.0 / 3.0),
                tr.delta_u * tr.sigma_xi * np.sqrt(sx * sx + 2.0 / 3.0))

    def _fit_beam_vx(self):
        """
        Beam-wide fit px ~ c0 + b x_t + cubic in (z - mean)/sigma_z, over ALL particles, for
        the observer velocity outside the quantile window (get_CSR_integrand), where the
        windowed source density -- and so its vx -- is zero.
        """
        b = self.beam
        w = (b.z - b.mean_z) / b.sigma_z
        wq = np.clip(w, -3.0, 3.0)
        A = np.column_stack([np.ones(w.size), b.x_transform, wq, wq ** 2, wq ** 3])
        self._beam_vx = (np.linalg.lstsq(A, b.px, rcond=None)[0], b.mean_z, b.sigma_z,
                         np.array(b.slope))

    def _quantile_diagnostics(self):
        """
        Frame 'quantile', once per kick, rank 0. Logs to self.quantile_log and warns once each:

        * dropped tails: the 1D wake (csr1d kernels, current geometry) of the charge outside
          the window, (1 - T) * lambda, against that of the full beam. Both are binned on the
          same grid and smoothed linearly, so W_full = W_window + W_dropped exactly.
          Reported: <W_drop> / <W_full> over all particles (the mean energy loss), and
          max|W_drop| / max|W_full| inside the window (the field on the source).
        * resolution: FWHM of the highest current peak against the deposition's effective
          z resolution (warn below 5) and against the local wake-mesh spacing (below 3).
          A WARNING only; nothing here moves the frame.
        """
        from .csr1d import CSR1DGeometry, dropped_wake_fraction
        from .peakfind import peak_fwhm
        tr = self.DF_tracker
        z = np.asarray(self.beam.z, dtype=float)
        T = tr.window_weight(z)
        if not hasattr(self, '_geom1d'):
            self._geom1d = CSR1DGeometry(self.lattice.lattice_config, self.lattice.distance)
            self.quantile_log = []
        geom = self._geom1d.at(self.beam.position, self.lattice.current_element)
        fwhm = peak_fwhm(z)
        frac_loss, frac_field = dropped_wake_fraction(z, T, geom)
        res_z, _ = self._deposition_resolution()
        zr = np.asarray(self.CSR_zrange)
        h_mesh = np.diff(zr)
        # mesh spacing where the charge is densest (the finest local spacing it reaches)
        h_at_peak = float(h_mesh.min()) if h_mesh.size else np.inf
        self.quantile_log.append((self.beam.position, tr.window[0], tr.window[1], tr.window[2], tr.window[3],
                                  tr.f_window, res_z, fwhm, h_at_peak, frac_loss, frac_field))
        cp = self.CSR_params
        if max(abs(frac_loss), frac_field) > cp.pedestal_warn and not getattr(self, '_warned_ped', False):
            self._warned_ped = True
            print(f'  [quantile frame] WARNING at s = {self.beam.position:.4f} m: the dropped '
                  f'{100 * (1 - tr.f_window):.2f}% of the charge carries {100 * frac_loss:.1f}% of '
                  f'the 1D energy loss and {100 * frac_field:.1f}% of the 1D field on the window '
                  f'(pedestal_warn {cp.pedestal_warn:g}). '
                  + ("frame 'auto' chose 'peak' on the initial beam; the pedestal matters now: use frame 'quantile'."
                     if getattr(tr, 'frame_requested', None) == 'auto' and tr.frame == 'peak'
                     else 'Lower particle_deposition.frame_quantile.'))
        if res_z > 0 and fwhm / res_z < 5 and not getattr(self, '_warned_res', False):
            self._warned_res = True
            print(f'  [quantile frame] WARNING at s = {self.beam.position:.4f} m: the highest '
                  f'current peak (FWHM {fwhm*1e6:.3g} um) is only {fwhm/res_z:.1f}x the '
                  f'deposition z resolution ({res_z*1e6:.3g} um). Raise zbins or lower '
                  f'smoothing_sigma_z.')
        if fwhm / h_at_peak < 3 and not getattr(self, '_warned_mesh', False):
            self._warned_mesh = True
            print(f'  [quantile frame] WARNING at s = {self.beam.position:.4f} m: the finest '
                  f'wake-mesh z spacing ({h_at_peak*1e6:.3g} um) is coarser than FWHM/3 of the '
                  f'highest current peak ({fwhm*1e6:.3g} um). Raise CSR_computation.zbins.')

    def _observer_vx(self, x, zobs, vx_src):
        """vx at the observer: T * (windowed source vx) + (1 - T) * (beam-wide fit)."""
        tr = self.DF_tracker
        T = float(tr.window_weight(np.array([zobs]))[0])
        if T >= 1.0:
            return vx_src
        c, mz, sz, p = self._beam_vx
        wq = min(max((zobs - mz) / sz, -3.0), 3.0)
        xt = x - np.polyval(p, zobs)
        vb = c[0] + c[1] * xt + c[2] * wq + c[3] * wq ** 2 + c[4] * wq ** 3
        return T * vx_src + (1.0 - T) * vb
    
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
        self._ibp_consistency_check()

    def calculate_2D_CSR_parallel(self):
        """
        The wake mesh split over the ranks INTERLEAVED (rank r takes points r, r + size, ...),
        not in contiguous blocks. A point's cost varies 15x across the mesh (24-340 ms on the
        BC20 loss beam at B2RE: the near-adapt and x' refinement work where the horns are, the
        core rows cost 1.7x the edge rows), and contiguous blocks are whole rows, so the slowest
        rank held 1.16x the mean at 3 ranks and 1.27-1.30x at 6-12; interleaving brings it to
        1.01-1.02x (measured on the pickled 37.74 m state). Every point is computed once, by one
        rank, with the same arithmetic as before, so the result is bit-identical to the serial
        and to the block split; the gather is an Allreduce of zero-padded full arrays.
        """
        work_size = self.CSR_params.xbins * self.CSR_params.zbins
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        dE_local = np.zeros((work_size,))
        xk_local = np.zeros((work_size,))
        for k in range(rank, work_size, size):
            s = self.beam.position + self.CSR_zmesh[k]
            x = self.CSR_xmesh[k]
            dE_local[k], xk_local[k] = self.get_CSR_wake(s, x)

        self.dE_dct = np.zeros((work_size,))
        self.x_kick = np.zeros((work_size,))
        comm.Allreduce(dE_local, self.dE_dct, op=MPI.SUM)
        comm.Allreduce(xk_local, self.x_kick, op=MPI.SUM)

        self.dE_dct = self.dE_dct.reshape((self.CSR_params.xbins, self.CSR_params.zbins))
        self.x_kick = self.x_kick.reshape((self.CSR_params.xbins, self.CSR_params.zbins))
        if rank == 0:
            self._ibp_consistency_check()

    def _lattice_vectors(self, sval):
        """
        (X0, Y0, n_x, n_y, tau_x, tau_y) of the reference orbit at arc length(s) sval.

        Past the lattice end the table itself is padded with a straight line (lattice.py). Before
        the start it is not: interpolate1D extrapolates with the first interval for one cell
        below the table and returns ZERO further down -- position at the origin, normal (0, 0) --
        so an observer more than one table cell before s = 0 (the bunch tail at the first kick,
        when the table is fine: a short lattice, or many samples) got a collapsed geometry and a
        transverse integrand of 1e16. Here the orbit is continued backwards along the initial
        tangent, exactly as the physical straight section before the lattice. Queries inside the
        table, and within one cell below it, are untouched (bit-identical).
        """
        lat = self.lattice
        sval = np.asarray(sval, dtype=float)
        out = [interpolate1D(xval=sval, data=d, min_x=lat.min_x, delta_x=lat.delta_x)
               for d in (lat.coords[:, 0], lat.coords[:, 1], lat.n_vec[:, 0], lat.n_vec[:, 1],
                         lat.tau_vec[:, 0], lat.tau_vec[:, 1])]
        before = sval <= lat.min_x - lat.delta_x        # where interpolate1D returns 0
        if before.any():
            ds = sval[before] - lat.min_x
            tx, ty = float(lat.tau_vec[0, 0]), float(lat.tau_vec[0, 1])
            vals = (lat.coords[0, 0] + ds * tx, lat.coords[0, 1] + ds * ty,
                    lat.n_vec[0, 0], lat.n_vec[0, 1], tx, ty)
            for k, v in enumerate(vals):
                out[k] = out[k].copy()
                out[k][before] = v
        return tuple(out)

    def _lattice_at(self, sval):
        """(X0, Y0, n_x, n_y) of the reference orbit at arc length(s) sval."""
        return self._lattice_vectors(sval)[:4]

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

            # shares its arithmetic with the interpolant via lookup_vec: a band located with a
            # different blend than the interpolant uses will not sit where the density is
            k, a_w = lookup_vec(t_ret, tr.t_arr, tr.is_uniform, tr.min_x, tr.delta_x, n_t)
            k1 = np.minimum(k + 1, n_t - 1)

            if self.use_comoving:
                # The co-moving interpolant places the density in the single
                # BLENDED frame, so the band is one interval. Mirror its blending
                # exactly -- linear in the poly and the means, log-linear in the
                # sigmas -- or the band will not sit where the density is.
                a = a_w
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

        Mirrors interpolate3D_comoving_fields exactly, in whichever frame_blend mode
        is active. A band located with a different blend than the interpolant uses
        will not sit where the density is.

        Returns (poly, xi_bar, z_bar, sigma_xi, sigma_z); poly is (n, deg+1) with
        the highest power first, matching np.polyfit and _eval_poly_rows. In
        'moment' mode the centroid pivot is folded back into the coefficients, so
        every consumer -- _eval_poly_rows, poly[:, 0] as the slope, and
        poly[:, -1] + xi_bar as the intercept at z = 0 for _eq424 -- keeps working
        unchanged.
        """
        tr = self.DF_tracker
        n_t = tr.poly_coeffs_interp.shape[0]
        k, a = lookup_vec(t_ret, tr.t_arr, tr.is_uniform, tr.min_x, tr.delta_x, n_t)
        k1 = np.minimum(k + 1, n_t - 1)
        b = 1.0 - a

        if tr.frame_blend_code != tr.BLEND_COEFF:
            var_z = b * tr.var_z_arr[k] + a * tr.var_z_arr[k1]
            cov = b * tr.cov_arr[k] + a * tr.cov_arr[k1]
            z_bar = b * tr.z_bar_arr[k] + a * tr.z_bar_arr[k1]
            x_bar = b * tr.x_bar_arr[k] + a * tr.x_bar_arr[k1]

            tau = cov / var_z
            poly = np.stack([tau, x_bar - tau * z_bar], axis=1)
            if tr.frame_blend_code == tr.BLEND_MOMENT:
                var_x = b * tr.var_x_arr[k] + a * tr.var_x_arr[k1]
                s_xi = np.sqrt(np.maximum(var_x - cov * cov / var_z, 0.0))
                s_z = np.sqrt(var_z)
            else:
                s_xi = np.exp(b * np.log(tr.sigma_xi_arr[k])
                              + a * np.log(tr.sigma_xi_arr[k1]))
                s_z = np.exp(b * np.log(tr.sigma_z_arr[k])
                             + a * np.log(tr.sigma_z_arr[k1]))
            return poly, np.zeros_like(tau), z_bar, s_xi, s_z

        poly = (b[:, None] * tr.poly_coeffs_interp[k]
                + a[:, None] * tr.poly_coeffs_interp[k1])
        return (poly,
                b * tr.xi_bar_arr[k] + a * tr.xi_bar_arr[k1],
                b * tr.z_bar_arr[k] + a * tr.z_bar_arr[k1],
                np.exp(b * np.log(tr.sigma_xi_arr[k])
                       + a * np.log(tr.sigma_xi_arr[k1])),
                np.exp(b * np.log(tr.sigma_z_arr[k])
                       + a * np.log(tr.sigma_z_arr[k1])))

    def _curv_shift_at(self, t_ret, z_ret):
        """
        sigma_xi * g(w) at the retarded frame: the lab-x' offset of the beam ridge from
        the straight localization axis (DF_tracker_comoving.curvature). 0 when off.
        Blends the curvature coefficients exactly as interpolate3D_comoving_fields does.
        """
        tr = self.DF_tracker
        if not getattr(tr, 'curvature', False):
            return 0.0
        from .deposit_smooth import _curv_g_vec
        n_t = tr.poly_coeffs_interp.shape[0]
        k, a = lookup_vec(t_ret, tr.t_arr, tr.is_uniform, tr.min_x, tr.delta_x, n_t)
        k1 = np.minimum(k + 1, n_t - 1)
        b = 1.0 - a
        c2 = b * tr.curv_c2_arr[k] + a * tr.curv_c2_arr[k1]
        c3 = b * tr.curv_c3_arr[k] + a * tr.curv_c3_arr[k1]
        _, _, z_bar, s_xi, s_z = self._comoving_frame_at(t_ret)
        w = (z_ret - z_bar) / s_z
        wc = np.clip(w, -tr.curv_zlim, tr.curv_zlim)
        g = c2 * (wc * wc - 1.0) + c3 * wc ** 3 + (2.0 * c2 * wc + 3.0 * c3 * wc * wc) * (w - wc)
        return s_xi * g

    @staticmethod
    def _eq424(T, nq, q2, tau, b, sign, tangent=False):
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
          tangent     with tangent=True, rad < 0 returns the column's closest
                      approach to the axis instead of nan. The retarded offset from
                      the axis is xi_ret(x') = -tau (l(x') - Tb - x'/tau), extremal
                      where dl/dx' = 1/tau, i.e. at
                          x' = nq + sign(tau) sqrt((q2 - nq^2) / (tau^2 - 1)),
                      which exists for |tau| > 1. (The vertex of the squared quadratic
                      agrees only to leading order in 1/tau.) See
                      Integration_params.tangent_bands for why the beam is still
                      there although its axis is not.
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

        if tangent:
            tan = (~flat) & (~deg) & (rad < 0.0) & (np.abs(tau_s) > 1.0)
            p2 = np.maximum(q2 - nq * nq, 0.0)
            xp = np.where(tan, nq + np.sign(tau_s) * np.sqrt(
                p2 / np.where(tan, tau_s ** 2 - 1.0, 1.0)), xp)

        Bc = -2.0 * T / tau_s - 2.0 * nq
        lin = deg & (Bc != 0.0)
        xp = np.where(lin, -(q2 - T ** 2) / np.where(lin, Bc, 1.0), xp)
        xp = np.where(flat, b, xp)

        # the l > 0 guard does not apply where no squaring took place
        l = np.where(flat, 1.0, Tb + xp / tau_s)
        return np.where(np.isfinite(xp) & (l > 0.0), xp, np.nan), rad

    def _retarded_xi_bands(self, s, x, t, sp, n_iter=12, scale_bands=True):
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
        tau0 = self._frame_tilt()
        margin = margin * degenerate_margin_factor(
            2.0 * tau0 / (1.0 + tau0 * tau0),
            getattr(self.integration_params, 'xi_band_margin_degenerate', 1.0))

        # BOTH roots are always taken. The branch degeneracy of thesis 4.4.2 is a
        # property of the tilt at the RETARDED time a(s'), which varies column to
        # column -- the retarded tilt sweeps +19.4 -> -0.22 -> -19.8 across one near
        # region -- so it cannot be decided from a(s) at the observation point. Doing
        # so discarded 28-31% of the wake wherever it fired.
        # The per-column decision needs no new logic: on a degenerate column the
        # second root either fails the _eq424 guards (rad < 0, l <= 0) and is parked
        # dead, or lands on top of the first and is collapsed to zero width by
        # _disjoint_bands.
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
                                       poly[:, -1] + xi_bar, sign,
                                       tangent=self.integration_params.tangent_bands)
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
            # with the curvature shear the beam ridge sits sigma_xi*g(w) off the straight
            # axis Eq 4.24 solved on; centre the band on the ridge (the root error from
            # solving on the axis is inside the band, as for the frame blend)
            # du/dx' along the column (see band_jacobian_min below); the ridge offset is
            # sigma_xi*g in u, i.e. sigma_xi*g/J in x'
            drdx = -(dx * n_sp_x + dy * n_sp_y) / np.maximum(r_phys, 1e-300)
            J = 1.0 - poly[:, 0] * drdx
            Jc = np.where(J < 0, -1.0, 1.0) * np.maximum(np.abs(J), 0.2)
            centre = centre + self._curv_shift_at(t - r_phys, z_ret) / Jc
            half = 0.5 * (tr.u_end - tr.u_start) * s_xi * margin
            jmin = self.integration_params.band_jacobian_min
            if jmin and scale_bands:
                # The band is sized in lab x', but the beam is sized in u at the
                # retarded time, and z_ret moves with x' through |r - r'|:
                #     du/dx' = (1 - tau * d|r-r'|/dx') / sigma_xi
                # (the frame's own change across one band is second order). Where the
                # line of sight is oblique to a tilted beam -- columns in a bend seen
                # from the drift after it -- |J| falls to ~0.25, so +-10 sigma_xi of x'
                # covers only +-2.5 sigma_xi of the beam (18% of the D3 far integral
                # missed). Scale the band by 1/|J| so it covers the same +-xlim*margin
                # in u everywhere, and so the same nodes per beam width.
                half = half / np.maximum(np.abs(J), jmin)
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

    def _integrate_xi_region(self, s, x, t, sp, ignore_vx, taper=None, scale_bands=True, adapt_x=False):
        """
        Integrate one s' region with the transverse nodes riding the density ribbon.

        Each localization branch gets its own set of x' nodes per s' column, so the
        inner trapezoid rule needs a per-column spacing. For a uniform grid
        trapz(y, dx=h) = h * trapz(y, dx=1), so the unit-spacing result is simply
        scaled by each column's dx.
        """
        bands = self._disjoint_bands(self._retarded_xi_bands(s, x, t, sp, scale_bands=scale_bands))
        nx = self.integration_params.xbins
        frac = np.linspace(0.0, 1.0, nx)
        ibp = self.integration_params.ibp and self.use_comoving

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

            ip = self.integration_params
            do_adapt_x = bool(adapt_x and ip.near_adapt and ip.near_adapt_x and not ibp)
            ex = {} if (ibp or do_adapt_x) else None
            gz, gx = self.get_CSR_integrand(s=s, t=t, x=x, xp=xp_mesh, sp=sp_mesh,
                                            ignore_vx=ignore_vx, taper=taper, extras=ex)
            if do_adapt_x:
                gz, gx, dxp_col = self._adapt_x_columns(s, x, t, sp, xp_mesh, gz, gx, ex, ignore_vx, taper)
            else:
                dxp_col = None
            # A zero-width column has dxp = 0 and contributes nothing. Select it out
            # rather than multiplying by zero, so a degenerate node that landed on
            # the 1/|r-r'| pole cannot turn the whole region into a nan.
            if dxp_col is not None:
                # refined columns: (iz, ix) already integrated over their own x' nodes
                iz = np.where(alive, dxp_col[0], 0.0)
                ix = np.where(alive, dxp_col[1], 0.0)
                dE += -self.CSR_scaling * np.trapz(y=iz, x=sp)
                xk += self.CSR_scaling * np.trapz(y=ix, x=sp)
                meshes.append((xp_mesh, sp_mesh, gz, gx))
                continue
            iz = np.where(alive, np.trapz(y=gz, axis=0) * dxp, 0.0)
            if ibp:
                # the shipped total is kept for the consistency check (get_CSR_wake)
                self._ibp_ship_acc += -self.CSR_scaling * np.trapz(y=iz, x=sp)
                gz, bnd, am, zone = self._ibp_integrand(gz, ex, t, taper, xp_mesh.shape)
                iz = np.where(alive, np.trapz(y=gz, axis=0) * dxp + bnd, 0.0)
                wcol = np.where(alive, dxp, 0.0) * np.gradient(sp)
                self._ibp_gross_acc += float(np.sum(am.sum(axis=0) * wcol))
                self._ibp_zone_acc += float(np.sum((am * zone).sum(axis=0) * wcol))
            ix = np.where(alive, np.trapz(y=gx, axis=0) * dxp, 0.0)
            dE += -self.CSR_scaling * np.trapz(y=iz, x=sp)
            xk += self.CSR_scaling * np.trapz(y=ix, x=sp)
            meshes.append((xp_mesh, sp_mesh, gz, gx))

        if not meshes:
            z = np.zeros((nx, len(sp)))
            return (0.0, 0.0, z, np.broadcast_to(sp[None, :], z.shape).copy(), z, z)
        return (dE, xk) + tuple(np.concatenate(a, axis=0) for a in zip(*meshes))

    def _adapt_x_columns(self, s, x, t, sp, xp_mesh, gz, gx, ex, ignore_vx, taper):
        """
        near_adapt_x: re-evaluate the columns whose retarded density jumps by more than
        near_adapt_eps * rho_max between neighbouring x' nodes, on x' nodes equidistributing
        1/h_uniform + |d(rho/rho_max)/dx'| / near_adapt_eps (every flagged column gets the same
        count, so one rectangular evaluation covers them all). Returns the uniform-mesh integrands
        (unchanged, for diagnostics) and the per-column integrals (iz, ix) over x', with the
        flagged columns replaced by their refined values.
        """
        ip = self.integration_params
        nx, ncol = xp_mesh.shape
        width = xp_mesh[-1] - xp_mesh[0]
        dxp = width / (nx - 1)
        iz = np.trapz(y=gz, axis=0) * dxp
        ix = np.trapz(y=gx, axis=0) * dxp
        rho = ex['rho'].reshape(xp_mesh.shape)
        rho_max = rho.max()
        if not rho_max > 0:
            return gz, gx, (iz, ix)
        eps = ip.near_adapt_eps
        xp_cur, rho_cur = xp_mesh, rho
        cols = np.flatnonzero((np.abs(np.diff(rho_cur, axis=0)).max(axis=0) > eps * rho_max) & (width > 0))
        n_ref = 0
        for _ in range(ip.near_adapt_x_rounds):
            if cols.size == 0:
                break
            xc, rc = xp_cur[:, cols], rho_cur[:, cols] / rho_max
            h_uni = (xc[-1] - xc[0]) / (xc.shape[0] - 1)
            with np.errstate(divide='ignore', invalid='ignore'):
                grad = np.abs(np.gradient(rc, axis=0) / np.gradient(xc, axis=0))
            # coincident nodes (a flat stretch of the previous round's monitor) give 0/0 here
            m = 1.0 / h_uni[None, :] + np.where(np.isfinite(grad), grad, 0.0) / eps
            F = np.concatenate([np.zeros((1, cols.size)),
                                np.cumsum(0.5 * (m[1:] + m[:-1]) * np.diff(xc, axis=0), axis=0)])
            ok = np.isfinite(F[-1]) & (F[-1] > 0)
            if not ok.all():
                cols, xc, F = cols[ok], xc[:, ok], F[:, ok]
                if cols.size == 0:
                    break
            n_new = int(min(max(np.ceil(F[-1]).max() + 1, nx), ip.near_adapt_x_max * nx))
            xn = np.empty((n_new, cols.size))
            for j in range(cols.size):
                xn[:, j] = np.interp(np.linspace(0.0, F[-1, j], n_new), F[:, j], xc[:, j])
            # strictly increasing nodes: trapz needs it and the next round's gradient divides by dx
            xn = np.maximum.accumulate(xn, axis=0)
            tiny = 1e-9 * (xn[-1] - xn[0])
            for k in range(1, n_new):
                xn[k] = np.maximum(xn[k], xn[k - 1] + tiny)
            sn = np.broadcast_to(sp[cols][None, :], xn.shape).copy()
            exn = {}
            gzn, gxn = self.get_CSR_integrand(s=s, t=t, x=x, xp=xn, sp=sn, ignore_vx=ignore_vx,
                                              taper=taper, extras=exn)
            iz[cols] = np.trapz(y=gzn, x=xn, axis=0)
            ix[cols] = np.trapz(y=gxn, x=xn, axis=0)
            n_ref += cols.size * n_new
            rn = exn['rho'].reshape(xn.shape)
            # next round: only the columns that still jump on the refined nodes
            still = np.abs(np.diff(rn, axis=0)).max(axis=0) > eps * rho_max
            xp_cur = np.zeros((n_new, ncol)); rho_cur = np.zeros((n_new, ncol))
            xp_cur[:, cols], rho_cur[:, cols] = xn, rn
            cols = cols[still]
        self._last_adapt_x = dict(cols=int(n_ref // max(n_new, 1)) if n_ref else 0, evals=n_ref)
        return gz, gx, (iz, ix)

    def _band_alive(self, s, x, t, sp):
        """Whether the retarded band has nonzero width, for each s' in the array."""
        sp = np.atleast_1d(np.asarray(sp, float))
        bands = self._disjoint_bands(self._retarded_xi_bands(s, x, t, sp))
        w = np.zeros(sp.size)
        for lo, hi in bands:
            w += np.maximum(hi - lo, 0.0)
        return w > 0.0

    def _causal_edge(self, s, x, t, sp_far, sp_near, n_scan=64):
        """
        Lowest s' in [sp_far, sp_near] whose retarded band still has nonzero width.

        Below this the retarded source has slipped outside the bunch and every column is
        identically zero, so integrating there is pure cost. ONE vectorised scan, and
        the LAST DEAD node below the first alive one is returned -- deliberately
        conservative, since truncating a live domain is a one-sided error that cannot
        average out, while starting one cell early costs one cell.

        Three things rule out anything cleverer, all measured:

        - NOT a root-find. The alive mask is *not* monotone: at shear 20 it has a dead
          gap inside the live region at 2 of 40 mesh points, so bisection can land in
          the gap and return an edge far above the true one. "First alive from below"
          on a scan has no such failure mode.
        - NOT a closed form. The edge is where the retarded z leaves the deposition
          window, and the slippage that sets it is exact only on the ray the band
          actually sits on. Inverting it analytically on the x' = x ray is free
          (0.06 ms) and matches this to 4 far-cells at zero tilt, but at shear 20 the
          band sits at x' = +2.9..3.6 mm against an observer at x = -71 um, and the
          wrong chord put the edge 37 mm too high -- rel L2 1.0 instead of 1.3e-03.
        - NOT refined further. A 44-step scalar bisection located the edge to 1e-14 m
          and cost 75% of a whole wake point, for a number that is then discretised
          onto a far_cell*sigma_z = 673 um grid. Padding the target by 2 full zlim
          moves the answer by 0.3%.

        Returns sp_far if the first sample is already alive, or None if the whole
        interval is dead.
        """
        grid = np.linspace(sp_far, sp_near, n_scan)
        alive = self._band_alive(s, x, t, grid)
        if not alive.any():
            return None
        if alive[0]:
            return sp_far
        return grid[max(int(np.argmax(alive)) - 1, 0)]

    def _far_region_nodes(self, s, x, t, sp_far, sp_near, transient=None):
        """
        s' nodes for the merged far region, spanning sp_far (= s1, the causal reach
        s - n_formation_length*L_f) up to sp_near (= s3, where the near region starts).

        TWO constructions, because the far integrand has two quite different shapes:

        IN A BEND (or before any bend) -- UNIFORM at 2*far_zbins nodes. This is the
        cost-neutral replacement for the old (s1,s2)+(s2,s3) pair, which gave each
        sub-region far_zbins nodes. It is deliberately NOT log-graded: grading assumes
        the contribution decays as 1/u away from the observer, and in a bend at high
        tilt it does not -- measured at shear 20, 0.30 m in, 10-99% of the far integral
        sits at u/L_f = 0.12-0.56, i.e. near the FAR end, where distance grading is
        coarsest. Pure log grading there gave rel L2 1.79 against 5.3e-02 for the old
        layout, and refining near_grade 0.05 -> 0.005 only reached 1.2e-02 at 456 nodes.

        PAST A BEND EXIT -- causal clip, then a uniform fine window, then log grading.
        Here the source is the arc still inside the dipole, so the contribution is a
        narrow band far upstream (u/L_f ~ 2.7-3.1, 10-99% of it inside ~0.3 L_f) and
        everything below the causal edge is identically zero. Log grading alone put a
        single 30 mm cell across the entire contribution and was 400x worse than the old
        layout; with the window it is 28x better at the same node count. See
        _exit_transient, _causal_edge and Integration_params.
        """
        ip = self.integration_params
        if sp_near <= sp_far:
            return np.array([sp_far, sp_near])

        if transient is None:
            transient = self._exit_transient()
        if not (ip.causal_clip and transient):
            # Not an exit transient: uniform, and no band solve. The causal edge would
            # be redundant here anyway -- measured in a bend it sits at u ~ 1.0 L_f
            # (0.994 at shear 0, 0.783 at shear 20), which is what n_formation_length
            # already provides.
            n = max(2 * ip.far_zbins, 3)
            return self._record_far_nodes(np.linspace(sp_far, sp_near, n))

        lo = sp_far
        e = self._causal_edge(s, x, t, sp_far, sp_near)
        if e is not None:
            lo = min(max(e, sp_far), sp_near)
            if e <= sp_far:
                # The edge should be the tighter limit; if the reach binds instead, the
                # integral is truncated by the history budget rather than by causality.
                # Honest, but a different limitation -- count it rather than hide it.
                self._far_clip_at_reach = getattr(self, '_far_clip_at_reach', 0) + 1

        parts = []
        top = lo
        if ip.far_window and ip.far_cell:
            top = min(lo + ip.far_window * self.formation_length, sp_near)
            cell = ip.far_cell * self.beam._sigma_z
            if ip.far_fine_max:
                # cap the fine-window node count: far_cell*sigma_z is set by the bunch,
                # but the band it resolves scales with L_f, so a short bunch (sigma_z
                # 19 um in D3) put 3300 nodes in a window that ~300 resolve to 3e-4
                cell = max(cell, (top - lo) / ip.far_fine_max)
            n_fine = int(np.clip(round((top - lo) / cell) + 1, 3, 100 * ip.far_zbins))
            parts.append(np.linspace(lo, top, n_fine))
        if top < sp_near:
            g = ip.near_grade
            if g:
                u_hi, u_lo = s - top, max(s - sp_near, 1e-12)
                if u_hi > u_lo:
                    k = int(np.ceil(np.log(u_hi / u_lo) / np.log1p(g)))
                    u = u_lo * (1.0 + g) ** np.arange(k + 1)
                    parts.append(s - np.minimum(u, u_hi))
            else:
                parts.append(np.linspace(top, sp_near, max(ip.far_zbins, 3)))
        if not parts:
            parts.append(np.linspace(lo, sp_near, max(ip.far_zbins, 3)))
        return self._record_far_nodes(np.unique(np.concatenate(parts)))

    def _record_far_nodes(self, sp):
        """Log the realised far node count, which is now derived rather than fixed."""
        self._last_region_nodes = list(getattr(self, '_last_region_nodes',
                                               [self.integration_params.far_zbins] * 2))
        self._last_region_nodes[0] = len(sp)
        return sp

    def _exit_transient(self):
        """
        Is the beam in a drift after a bend, i.e. computing an EXIT transient?

        This is the only regime where the causal edge is worth locating: it is the only
        one where it differs from the n_formation_length*L_f reach (u ~ 3 L_f past the
        exit against ~1 L_f inside the bend), and the only one where the contributing
        source is a narrow band far upstream rather than a 1/u tail.
        """
        return bool(self.afterbend and not self.inbend)

    def _layout_bounds(self, s, x):
        """
        The two s' integration regions for the xi_bands path.

        Only the region EXTENT is decided here. Whether the two Eq 4.24 branches are
        resolvable is a per-column property of the retarded tilt a(s'), so it is left
        to _retarded_xi_bands; the extent cannot be, since it defines s3 itself.

        Two regions, not three. The old middle seam s2 = s3 - 200 sigma_z dated from the
        legacy decomposition, where it marked where the +-20 sigma_x WIDE transverse
        window stopped being needed. With per-column ribbons there is no wide window, so
        s2 no longer separates anything physical -- but because every region got the same
        NODE COUNT while their lengths differed ~45x, it was still acting as a crude
        two-level grading (cells 45.7 / 1.0 / 0.1 sigma_z). Merging the two far regions
        naively therefore LOST accuracy (3-10x at matched cost); the replacement is
        _far_region_nodes, which grades the merged region explicitly and is 28x better
        than the old split at a bend-exit transient for the same node count.

        Everything is keyed on sin 2a and cos 2a, never on tan 2a and never on |tau|:

          sin 2a = 2 tau/(1 + tau^2)      cos 2a = (1 - tau^2)/(1 + tau^2)

        Both are bounded and pole-free, whereas tan 2a = 2 tau/(1 - tau^2) has a pole at
        |tau| = 1. Thesis 4.4.2 notes that x2 = x - (s-s')tan(2a) degenerates to x2 = x
        at a = 0 AND at a = +-pi/2, so the chirp branch merges into the narrow one in
        BOTH limits -- and |sin 2a| is small in both, so a single test catches them.

        The legacy (xi_bands = False) decomposition in get_CSR_wake is deliberately left
        alone; it keys on |tan_theta| <= 1, which fires where the branches are maximally
        SEPARATED and which sent the near region swinging
        d = 99.9 mm -> 807.7 mm -> (branch flip) s3 = s - 50.9 um -> 99.9 mm
        within ~1 mm of path length across a longitudinal waist.
        """
        ip = self.integration_params
        b = self.beam
        sigma_z, sigma_x = b._sigma_z, b._sigma_x
        # Use the SAME tilt the density frame uses (core fit) rather than beams.slope
        # (all particles). They disagree most exactly where the correlation vanishes.
        tau = self._frame_tilt()

        sin2a = 2.0 * tau / (1.0 + tau * tau)
        cos2a = (1.0 - tau * tau) / (1.0 + tau * tau)

        # Near-region length as ONE continuous expression, with no branch on it at all.
        # An earlier version switched the extent formula at branch_sin_min and
        # reintroduced a discontinuity (125 mm -> 0.347 mm across the threshold).
        #
        #   d = (10 sigma_x + x - xmean) |cos 2a| / max(|sin 2a|, branch_sin_min)
        #
        # This is the legacy d = (10 sigma_x + x - xmean)/|tan 2a| wherever the branches
        # are resolvable, and it saturates at branch_sin_min as sin 2a -> 0, where the
        # branches merge and d stops meaning anything.
        #
        # It is then floored at near_floor sigma_z, which is NOT cosmetic. As |tau| -> 1,
        # cos 2a -> 0 and the chirp band becomes vertical, so it does exit the beam at
        # once -- but d is the length of the NEAR region, which must also contain the
        # NARROW band and the neighbourhood of the 1/|r-r'| pole. Sizing it from the
        # chirp exit alone starves both. Measured at shear 50, 0.80 m into the dipole
        # (tau = -0.999, cos 2a = 8.8e-4): the upstream reach collapsed to 15 um =
        # 0.009 sigma_z against sigma_z = 1744 um, so 1-2 of the 88 graded nodes landed
        # on the causal side and the pole sat 15 um from the region edge, leaving
        # region 2's uniform 1.74 mm cells to carry it. The wake went visibly blocky.
        # The floor is the s3 = s - 20 sigma_z that the deleted |tan theta| <= 1 branch
        # used to provide; removing that branch in 6o dropped it silently, and 6f's
        # coarser s grid (tau = -1.225 then -0.814) stepped over the region where it
        # matters -- the defect only bites within ~1% of |tau| = 1.
        #
        # The (x - xmean) term is kept deliberately: the chirp branch starts at the
        # observation point, so one starting near the beam edge exits sooner than one
        # starting at the centre, and d legitimately varies ~1.9x across a wake mesh.
        # Dropping it makes the wake look ~4x smoother at fixed resolution, but the two
        # variants converge to the same answer (rel L2 0.045 -> 0.0017 as zbins/xbins
        # refine), so that smoothness is per-column node-layout jitter being removed
        # from the roughness METRIC, not accuracy being gained. Optimising the metric
        # rather than the answer is not worth losing the correct exit condition.
        d = ((10.0 * sigma_x + x - b._mean_x) * abs(cos2a)
             / max(abs(sin2a), ip.branch_sin_min))
        d = max(d, ip.near_floor * sigma_z)
        # and never reach back further than the retained history supports
        d = min(max(d, 0.0), ip.n_formation_length * self.formation_length)

        s4 = s + 3.0 * sigma_z
        s3 = max(0.0, s - d)
        # The far edge is the causal reach measured from s3, and it is clamped at 0
        # because s' < 0 is before the lattice exists -- interpolate1D returns 0 there
        # silently, so nodes below 0 are not merely wasted but invisible. The old
        # s1 = s2 - n_fl*L_f with s2 = s3 - 200 sigma_z reached back 200 sigma_z FURTHER
        # than the retained history supports, and inverted (s1 > s2) whenever s2 went
        # negative while s1 clamped.
        s1 = max(0.0, s3 - ip.n_formation_length * self.formation_length)
        # In a drift after a bend the source of the exit wake is the bend's own arc,
        # which recedes behind the observer; a fixed n_fl*L_f reach loses it ~1.7 R*phi
        # past the exit and the wake then changes sign (progress.md, 2026-11-03).
        # drift_reach extends the far edge back to the bend's entrance; the causal-edge
        # scan in _far_region_nodes then trims it to where the source still contributes.
        s_bend = self._drift_reach_start()
        if s_bend is not None:
            s1 = min(s1, s_bend)
        return ((s1, s3), (s3, s4))

    def _thin_drift_history(self):
        """
        drift_snap_thin: in a drift after a bend, drop the previous snapshot if linear
        interpolation between its neighbours reproduces it within the tolerance (see
        DF_tracker_comoving.thin_previous). Online, O(1) per step, newest always kept.
        """
        tol = self.integration_params.drift_snap_thin
        if not (tol and self.use_comoving and self._exit_transient()):
            return
        # never thin the bend itself: P (the candidate) must lie past the bend exit
        tr = self.DF_tracker
        if len(tr.time_log) < 3 or tr.time_log[-2] <= self._bend_exit_rec():
            return
        gap = self.integration_params.n_formation_length * self.formation_length
        dropped, err = tr.thin_previous(tol, max_gap=gap)
        self._thin_dropped = getattr(self, '_thin_dropped', 0) + int(dropped)
        self._thin_kept = getattr(self, '_thin_kept', 0) + int(not dropped)

    def _bend_exit_rec(self):
        sb = getattr(self, 'bend_start_rec', None)
        if sb is None or self.R_rec is None:
            return -np.inf
        return sb + abs(self.R_rec * self.phi_rec)

    def _prev_region_nodes(self, s, pb, s1):
        """
        s' nodes of the prev_bend_reach region [pb, s1]: log-graded in u = s - s' at
        prev_bend_grade. Measured at the chicane's B4 entrance against a 4001-node
        reference, the region is a smooth 1/u-like tail of the drift history (53% within
        0.5 m of s1, 90% within 2 m), not a narrow band: grading 0.02 gives 113 nodes and
        <= 0.4% error, where the exit-transient layout used 544 nodes for 0.1%.
        """
        g = self.integration_params.prev_bend_grade
        u_lo, u_hi = max(s - s1, 1e-12), s - pb
        k = int(np.ceil(np.log(u_hi / u_lo) / np.log1p(g)))
        return np.unique(s - np.minimum(u_lo * (1.0 + g) ** np.arange(k + 1), u_hi))

    def _prev_bend_reach_start(self):
        """Entrance of the previous bend while inside the next one, or None."""
        ip = self.integration_params
        if not (ip.drift_reach and ip.prev_bend_reach and self.inbend):
            return None
        pb = getattr(self, 'prev_bend_start', None)
        return None if pb is None else max(0.0, float(pb))

    def _drift_reach_start(self):
        """Far edge for the exit transient (the last bend's entrance), or None."""
        if not (self.integration_params.drift_reach and self._exit_transient()):
            return None
        sb = getattr(self, 'bend_start_rec', None)
        return None if sb is None else max(0.0, float(sb))

    def _frame_tilt(self):
        """
        The tilt the DENSITY frame uses, i.e. the core fit stored by the tracker.

        beams.slope fits ALL particles (beams.py); DF_tracker_comoving fits the core
        only (fit_zlim). They disagree maximally where the x-z correlation vanishes,
        and the integration layout must agree with the frame the density actually
        lives in, or the bands sit beside the ribbon.
        """
        tr = self.DF_tracker
        pc = getattr(tr, 'poly_coeffs', None)
        if pc is None or not self.use_comoving:
            return float(self.beam._slope[0])
        return float(pc[0])

    def _region_node_counts(self, bounds):
        """
        Longitudinal node count for each s' region, near region last.

        A flat count per region is wrong once the regions differ in length by ~100x.
        The near region sets the requirement: its cell must be ~sigma_xi (measured to
        collapse across tilt amplification 66x and 167x), while its length grows with
        tilt, so its node count scales as sigma_x/sigma_xi. The far regions need far
        fewer nodes -- the far edge is invariant to machine precision there.

        near_cell = 0 restores the historical behaviour of `zbins` everywhere.
        """
        ip = self.integration_params
        if not ip.near_cell:
            self._last_region_nodes = [ip.zbins] * len(bounds)
            return self._last_region_nodes
        target = ip.near_cell * self.beam._sigma_x_transform
        counts = [ip.far_zbins] * len(bounds)
        a, b = bounds[-1]
        want = int(round(abs(b - a) / target))
        cap = 100 * ip.zbins
        # Warn once per run when the cap binds. It binds where |tan 2 alpha| collapses
        # -- near a bend entrance the chirp band can stretch to d = 10 sigma_x /
        # |tan 2 alpha| ~ 150 mm while the cell stays at sigma_xi, asking for ~25000
        # columns. Silently clipping that returns an under-resolved wake that looks
        # perfectly plausible, which is exactly how it went unnoticed once already.
        # ... but only when this count is what the near region will actually use.
        # With near_grade set, _near_region_nodes discards `want` entirely and builds a
        # graded grid whose finest cell IS near_cell*sigma_xi, so the cap never binds
        # and warning about it is not merely noise: it reports a resolution loss that
        # is not happening, and advises "use graded nodes" when they are already on.
        graded = bool(ip.near_grade)
        if want > cap and not graded and not getattr(self, '_warned_near_cap', False):
            self._warned_near_cap = True
            if (not self.parallel) or self.rank == 0:
                print(f'WARNING: near-region nodes capped at {cap} '
                      f'(wanted {want}); near-region cell is '
                      f'{want/cap:.1f}x coarser than near_cell requests. '
                      f'Raise zbins, raise near_cell, or use graded nodes.')
        counts[-1] = int(np.clip(want, 3, cap))
        self._last_region_nodes = counts
        return counts

    def _near_region_nodes(self, s, a, b, x=None, t=None):
        """
        s' nodes for the near region, graded so the cell grows with |s - s'|.

        The cell is  du = max(near_cell*sigma_xi, near_grade*u),  u = |s - s'|:
        an absolute floor set by the transverse support width close to the
        observation point, geometric growth outside. Built independently on each
        side, because the near region straddles s' = s.

        Rationale: for u larger than the transverse offsets, |r - r'| ~ u, so the
        per-column contribution falls as 1/u and equal contributions come from equal
        logarithmic intervals. Uniform nodes are therefore under-resolved where the
        accuracy is set and over-resolved where the cost is, forcing
        N ~ u_max/sigma_xi -- which is why the count had to scale with tilt. Graded,
        N ~ 1/near_grade + ln(u_max/u*)/ln(1+near_grade), growing only
        logarithmically with tilt.

        The outer integral uses np.trapz(..., x=sp), which handles the non-uniform
        spacing correctly.
        """
        ip = self.integration_params
        floor = ip.near_cell * self.beam._sigma_x_transform
        g = ip.near_grade
        if not ip.near_cell or not g or floor <= 0.0:
            base = np.linspace(a, b, self._region_node_counts(((a, b),))[-1])
            if ip.near_adapt and x is not None:
                return self._near_adapt_nodes(s, x, t, a, b, base)
            return base

        ustar = floor / g                      # where relative spacing takes over

        def side(umax):
            if umax <= 0.0:
                return np.zeros(1)
            u = np.arange(0.0, min(ustar, umax) + floor, floor)
            if umax > ustar:
                n = int(np.ceil(np.log(umax / ustar) / np.log1p(g)))
                u = np.concatenate((u, ustar * (1.0 + g) ** np.arange(1, n + 1)))
            return np.unique(np.minimum(u, umax))

        uL = side(s - a)
        uR = side(b - s)
        sp = np.concatenate((s - uL[::-1], s + uR))
        self._last_region_nodes = list(getattr(self, '_last_region_nodes',
                                               [ip.far_zbins] * 2))
        self._last_region_nodes[-1] = len(sp)
        sp = np.unique(sp)
        if ip.near_adapt and x is not None:
            return self._near_adapt_nodes(s, x, t, a, b, sp)
        return sp

    def _column_scan(self, s, x, t, sp):
        """
        The retarded source seen by the column at each s': (z_ret, rho) at the located band's
        axis point, the larger of the two Eq 4.24 branches. rho is 0 where no branch is live.
        """
        self._retarded_xi_bands(s, x, t, sp, n_iter=getattr(self.integration_params, 'near_adapt_scan_iter', 4))
        diag = getattr(self, '_branch_diag', None)
        zr = np.full(sp.shape, np.nan)
        rho = np.zeros(sp.shape)
        if diag is None or diag['sp'] is not sp:
            return zr, rho
        for br in diag['branches']:
            live = br['live']
            if not live.any():
                continue
            t_ret = t - br['r'][live]
            z_ret = sp[live] - t_ret
            r_b = self._comoving_fields(br['xp'][live], z_ret, t_ret)[0]
            take = np.zeros(sp.shape, dtype=bool)
            take[live] = r_b >= rho[live]
            rho[take] = r_b[take[live]]
            zr[take] = z_ret[take[live]]
        return zr, rho

    def _near_adapt_nodes(self, s, x, t, a, b, base):
        """
        Near-region s' nodes adapted to the retarded density along the column (Integration_params
        near_adapt). Starts from the fixed layout `base`, scans z_ret and rho along the column and
        bisects every interval whose z_ret still moves by more than near_adapt_scan_dz * res_z
        inside the history's z support, then equidistributes the monitor
            m(s') = 1 / h_fixed(s') + |d(rho / rho_max) / ds'| / near_adapt_eps,
        graded so neighbouring cells differ by at most near_adapt_max_ratio. The first term keeps
        every cell at least as fine as the fixed layout; the second puts ~2/eps nodes across each
        full-height rise and fall of the source, wherever it is and however many there are.
        Returns `base` unchanged when the column sees no charge.
        """
        ip = self.integration_params
        tr = self.DF_tracker
        if not self.use_comoving or getattr(tr, 'poly_degree', None) != 1 or base.size < 3:
            return base
        res_z = self._deposition_resolution()[0]
        if not res_z > 0:
            return base
        delta = ip.near_adapt_scan_dz * res_z
        n_hist = len(tr.time_log)
        zb, sz = np.asarray(tr.z_bar_arr)[:n_hist], np.asarray(tr.sigma_z_arr)[:n_hist]
        z_lo, z_hi = float(np.min(zb + tr.w_start * sz)), float(np.max(zb + tr.w_end * sz))

        sp = np.asarray(base, dtype=float)
        zr, rho = self._column_scan(s, x, t, sp)
        # Each round splits every interval that still needs it into ceil(|dz_ret| / delta) pieces
        # at once (z_ret is close to linear within an interval), so 2-3 rounds suffice; a round is
        # one band solve, ~4 ms of Python overhead whatever its size.
        for _ in range(4):
            ds = np.diff(sp)
            dz = np.abs(np.diff(zr))
            lo = np.fmin(zr[:-1], zr[1:])
            hi = np.fmax(zr[:-1], zr[1:])
            # a dead end (nan) next to a live one: the support edge lies inside; locate it to delta
            edge = (np.isnan(zr[:-1]) != np.isnan(zr[1:])) & (ds > delta)
            need = ((dz > delta) & (hi > z_lo) & (lo < z_hi)) | edge
            if not need.any() or sp.size >= ip.near_adapt_max_scan:
                break
            k = np.where(edge, 2, np.minimum(np.ceil(np.nan_to_num(dz, nan=0.0) / delta), 64)).astype(int)[need]
            room = max(ip.near_adapt_max_scan - sp.size, 0)
            if (k - 1).sum() > room:
                k = np.maximum(1, np.floor(k * room / max((k - 1).sum(), 1))).astype(int)
            a0, a1 = sp[:-1][need], sp[1:][need]
            new = np.concatenate([a0[i] + (a1[i] - a0[i]) * np.arange(1, k[i]) / k[i]
                                  for i in range(need.sum())]) if need.any() else np.zeros(0)
            if new.size == 0:
                break
            zm, rm = self._column_scan(s, x, t, new)
            # unique: a split point can coincide with an existing one to round-off, and a zero
            # spacing would put inf/nan into the gradient monitor below
            sp, order = np.unique(np.concatenate([sp, new]), return_index=True)
            zr = np.concatenate([zr, zm])[order]
            rho = np.concatenate([rho, rm])[order]
        if sp.size >= ip.near_adapt_max_scan and not getattr(self, '_warned_adapt_scan', False):
            self._warned_adapt_scan = True
            if (not self.parallel) or self.rank == 0:
                print(f'  [near_adapt] WARNING at s = {t:.4f} m: the column scan hit '
                      f'near_adapt_max_scan = {ip.near_adapt_max_scan} points before resolving '
                      f'{delta*1e6:.3g} um of retarded z per interval.')
        rho_max = rho.max()
        if not rho_max > 0:
            return base

        floor = ip.near_cell * self.beam._sigma_x_transform
        g = ip.near_grade
        if ip.near_cell and g and floor > 0.0:
            h_fixed = np.maximum(floor, g * np.abs(sp - s))
        else:
            h_fixed = np.full(sp.shape, (b - a) / max(base.size - 1, 1))
        grad = np.abs(np.gradient(rho / rho_max, sp))
        m = 1.0 / h_fixed + np.where(np.isfinite(grad), grad, 0.0) / ip.near_adapt_eps
        # grading on the non-uniform scan grid: with h = 1/m, |d ln m| per cell <= ln(max_ratio)
        lr = np.log(ip.near_adapt_max_ratio)
        dg = np.diff(sp)
        lm = np.log(m)
        for _ in range(2):
            for i in range(1, lm.size):
                lm[i] = max(lm[i], lm[i - 1] - lr * dg[i - 1] * np.exp(lm[i - 1]))
            for i in range(lm.size - 2, -1, -1):
                lm[i] = max(lm[i], lm[i + 1] - lr * dg[i] * np.exp(lm[i + 1]))
        m = np.exp(lm)
        F = np.concatenate([[0.0], np.cumsum(0.5 * (m[1:] + m[:-1]) * dg)])
        if not np.isfinite(F[-1]):
            return base
        n = int(np.ceil(F[-1])) + 1
        if n > ip.near_adapt_max_nodes:
            if not getattr(self, '_warned_adapt_nodes', False):
                self._warned_adapt_nodes = True
                if (not self.parallel) or self.rank == 0:
                    print(f'  [near_adapt] WARNING at s = {t:.4f} m: {n} near-region nodes wanted, '
                          f'capped at near_adapt_max_nodes = {ip.near_adapt_max_nodes}.')
            n = ip.near_adapt_max_nodes
        nodes = np.interp(np.linspace(0.0, F[-1], max(n, base.size)), F, sp)
        nodes = np.unique(np.concatenate([nodes, [a, s, b]]))
        nodes = nodes[(nodes >= a) & (nodes <= b)]
        self._last_region_nodes = list(getattr(self, '_last_region_nodes', [ip.far_zbins] * 2))
        self._last_region_nodes[-1] = len(nodes)
        self._last_adapt = dict(scan=sp.size, nodes=len(nodes), base=base.size)
        return nodes

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

        self._ibp_ship_acc = 0.0
        self._ibp_gross_acc = 0.0
        self._ibp_zone_acc = 0.0
        if self.use_smooth_deposit and self.integration_params.xi_bands:
            # The transverse extents are replaced by the located ribbons, so the chirp
            # case's regions 3 and 4 (which tiled x' over the SAME s' range) merge into
            # one region or the ribbon would be counted twice.
            #
            # The s' decomposition is also recomputed here, pole-free and keyed on
            # sin 2a instead of |tan_theta| <= 1. The legacy bounds above are left
            # untouched so the xi_bands = False path stays byte-identical.
            bnds = self._layout_bounds(s, x)
            (s1, s3), (_, s4) = bnds
            self._region_node_counts(bnds)      # sets _last_region_nodes for diagnostics
            # far region: causal-clipped, fine over the contributing band, graded above
            # near region: graded about s' = s
            sps = [self._far_region_nodes(s, x, t, s1, s3),
                   self._near_region_nodes(s, s3, s4, x=x, t=t)]

            # Only the last region straddles s' = s, so only it owns the 1/|r-r'|
            # singularity and gets the polar patch.
            radii = self._near_patch_radii(s, s3, s4)
            tapers = [None, None if radii is None else (*radii, False)]

            # band_jacobian_min scales the FAR region only. Its columns are the oblique
            # ones (a bend seen from far downstream, |J| down to 0.25); the near region
            # has |J| ~ 1 except within a few band widths of the observer, where the
            # 1/|r-r'| pole and the polar patch already set the node layout, and
            # widening there lost accuracy (D3: up to 16% at a 0.1 floor).
            parts = [self._integrate_xi_region(s, x, t, spk, ignore_vx, tp, scale_bands=(k == 0),
                                               adapt_x=(k == 1) or (k == 0 and self.integration_params.far_adapt_x))
                     for k, (spk, tp) in enumerate(zip(sps, tapers))]
            # prev_bend_reach: inside a bend, the previous bend's exit tail still arrives
            # (1D: -0.05 MeV/m at the chicane's B4 entrance). Its source lies below this
            # bend's own reach s1, as a narrow band far upstream -- the exit-transient
            # shape -- so it gets that layout (causal edge, fine window, grading).
            pb = self._prev_bend_reach_start()
            if pb is not None and pb < s1:
                sp_prev = self._prev_region_nodes(s, pb, s1)
                parts.append(self._integrate_xi_region(s, x, t, sp_prev, ignore_vx, None,
                                                       scale_bands=True))
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
            if self.integration_params.ibp and self.use_comoving:
                # the shipped form's total: the regions' shipped sums plus the patch,
                # which both forms share
                self._last_wake_shipped = (self._ibp_ship_acc
                                           + (parts[-1][0] if radii is not None else 0.0))
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
            tr.min_x, tr.delta_x,
            tr.frame_blend_code, tr.var_z_arr, tr.cov_arr, tr.var_x_arr,
            tr.x_bar_arr,
            tr.t_arr, tr.bucket, tr.bucket_inv_h, tr.bucket_t0, tr.bucket_M,
            tr.is_uniform, tr.curv_c2_arr, tr.curv_c3_arr, float(tr.curv_zlim))

    def _comoving_fields_ibp(self, xval, zval, tval, want_fields=True):
        """
        _comoving_fields plus the by-parts coefficients (A, B); see
        interp3D.interpolate3D_comoving_ibp. want_fields False returns only A (the rest 0).
        """
        tr = self.DF_tracker
        return interpolate3D_comoving_ibp(
            xval, zval, tval,
            tr.data_density_interp, tr.data_density_u_interp,
            tr.data_density_w_interp, tr.data_vx_interp, tr.data_vx_u_interp,
            tr.poly_coeffs_interp, tr.xi_bar_arr, tr.sigma_xi_arr,
            tr.z_bar_arr, tr.sigma_z_arr,
            tr.u_start, tr.delta_u, tr.w_start, tr.delta_w,
            tr.min_x, tr.delta_x,
            tr.frame_blend_code, tr.var_z_arr, tr.cov_arr, tr.var_x_arr,
            tr.x_bar_arr,
            tr.t_arr, tr.bucket, tr.bucket_inv_h, tr.bucket_t0, tr.bucket_M,
            tr.is_uniform, tr.curv_c2_arr, tr.curv_c3_arr, float(tr.curv_zlim),
            bool(want_fields))

    @staticmethod
    def _radial_taper(R, taper):
        """The near-patch partition weight of get_CSR_integrand (1 if taper is None)."""
        if taper is None:
            return np.ones_like(R)
        R1, R2, keep_inner = taper
        u = np.clip((R - R1) / (R2 - R1), 0.0, 1.0)
        w = u * u * u * (u * (6.0 * u - 15.0) + 10.0)
        return 1.0 - w if keep_inner else w

    @staticmethod
    def _ibp_chi(J, J0):
        """Partition weight of the shipped form: 1 for |J| <= J0, 0 for |J| >= 2 J0, C2 between."""
        if not J0:
            return np.zeros_like(J)
        u = np.clip((np.abs(J) - J0) / J0, 0.0, 1.0)
        return 1.0 - u * u * u * (u * (6.0 * u - 15.0) + 10.0)

    def _ibp_Q(self, ex, t, delta, taper, J0, A=None):
        """
        Q = (1 - chi(J)) G0 / J at the nodes shifted by delta in x', G0 = h' (v.n') T(R) / R.
        Pure geometry and frame (A from the field-free kernel unless given), no density.
        Returns (Q, J, chi, G0, R_x).
        """
        rx = ex['rx'] - delta * ex['nx']
        ry = ex['ry'] - delta * ex['ny']
        R = np.sqrt(rx * rx + ry * ry)
        Rx = -(rx * ex['nx'] + ry * ex['ny']) / R
        xp = ex['xp'] + delta
        if A is None:
            A = self._comoving_fields_ibp(xp, ex['sp'] - (t - R), t - R, want_fields=False)[5]
        J = 1.0 - A * Rx
        chi = self._ibp_chi(J, J0)
        G0 = (1.0 + xp * ex['kappa']) * ex['vn'] * self._radial_taper(R, taper) / R
        safe = np.where(chi < 1.0, J, 1.0)
        Q = np.where(chi < 1.0, (1.0 - chi) * G0 / safe, 0.0)
        return Q, J, chi, G0, Rx

    def _ibp_integrand(self, gz, ex, t, taper, shape):
        """
        The longitudinal integrand with its main term integrated by parts along each column.

        Node integrand  gz - (1 - chi) G0 rho_x - Q' rho - Q B R_x, and per column the boundary
        term Q rho |last node - Q rho |first node. Their column integral equals that of gz up to
        quadrature error (progress_cancellation.md §2). Q' is a central difference with step
        1e-3 of the smallest sigma_xi in the history, on geometry and frame only. Columns where
        |J| <= 2 J0 on a node with charge keep gz (see below).
        """
        J0 = self.integration_params.ibp_j0
        h = 1e-3 * float(np.min(self.DF_tracker.sigma_xi_arr))
        Q0, J, chi, G0, Rx = self._ibp_Q(ex, t, 0.0, taper, J0, A=ex['A'])
        Qp = self._ibp_Q(ex, t, +h, taper, J0)[0]
        Qm = self._ibp_Q(ex, t, -h, taper, J0)[0]
        dQ = (Qp - Qm) / (2.0 * h)
        rho = ex['rho']
        main = G0 * ex['rho_x']
        g = (gz.ravel() - (1.0 - chi) * main - dQ * rho - Q0 * ex['B'] * Rx).reshape(shape)
        Qr = (Q0 * rho).reshape(shape)
        bnd = Qr[-1] - Qr[0]
        # Column-wise choice of form. The identity holds column by column, so a column can keep
        # the shipped form without any boundary term. Where J gets small on a node that carries
        # charge, the chi partition inside the column becomes a steep ramp in x' (J changes at
        # A (1 - R_x^2)/R near the observer, A ~ the beam tilt) whose chi and 1 - chi pieces
        # cancel each other, so the by-parts column is far less converged than the shipped one.
        # Measured on the chicane, frozen at B3 (tau = -30) and D3 (tau = -49): with the
        # partition alone the by-parts wake was off by up to 30x at 3200 nodes. So a column uses
        # the by-parts form only if |J| > 2 J0 at every node with charge; chi then vanishes there
        # except on empty nodes, where it only guards the 1/J.
        #
        # J must also keep ONE SIGN over the whole column. Along a column R_x is monotonic and A
        # varies slowly, so J crosses zero at most once, but at high tilt it can do so between
        # two nodes: chicane D3 (tau = -49), a column 40 um from the observer has J from -20 to
        # +20, its 200 nodes miss the crossing (smallest sampled |J| 3.0), and the 1/J pole
        # between them made that one column 100x too large. A sign test sees it at any node count.
        live = (rho > 1e-6 * max(float(rho.max()), 1e-300)).reshape(shape)
        Jr = J.reshape(shape)
        Jc = np.where(live, np.abs(Jr), np.inf).min(axis=0)
        one_sign = np.all(Jr > 0.0, axis=0) | np.all(Jr < 0.0, axis=0)
        col = (Jc > 2.0 * J0) & one_sign
        g = np.where(col[None, :], g, gz)
        bnd = np.where(col, bnd, 0.0)
        # |main term| per node, and the part of it in columns kept in the shipped form, for
        # the consistency check (the share where the cancellation is still being summed)
        am = np.abs(main).reshape(shape)
        zone = np.broadcast_to(~col[None, :], shape)
        return g, bnd, am, zone

    def _ibp_consistency_check(self):
        """
        Both forms at a few points of the mesh's centre row, at ibp_check_xbins x' nodes.
        See Integration_params.ibp_check. Appends (s, relative L2 difference, points used,
        points skipped, largest zone share) to ibp_check_log.

        Also recorded per kick: the largest share of the main term's sum|node| in columns that
        keep the shipped form (|J| <= 2 J0 on a node with charge). There the two forms coincide,
        so a share near 1 means the check sees little. Points above ibp_check_zone_max are
        skipped (default 1: none).
        """
        ip = self.integration_params
        n = ip.ibp_check
        if n <= 0 or not self.use_comoving:
            return
        nz = self.CSR_params.zbins
        ix = self.CSR_params.xbins // 2
        js = np.unique(np.round(np.linspace(0, nz - 1, n + 2)[1:-1]).astype(int))
        saved = (ip.ibp, ip.xbins)
        ip.ibp, ip.xbins = True, ip.ibp_check_xbins
        a, b, fz = [], [], []
        skipped = 0
        try:
            for j in js:
                k = ix * nz + j
                dE, _ = self.get_CSR_wake(self.beam.position + self.CSR_zmesh[k], self.CSR_xmesh[k])
                f = self._ibp_zone_acc / max(self._ibp_gross_acc, 1e-300)
                fz.append(f)
                if f > ip.ibp_check_zone_max:
                    skipped += 1
                    continue
                a.append(self._last_wake_shipped)
                b.append(dE)
        finally:
            ip.ibp, ip.xbins = saved
        a, b = np.array(a), np.array(b)
        d = (float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300)) if len(a)
             else float('nan'))
        if not hasattr(self, 'ibp_check_log'):
            self.ibp_check_log = []
        self.ibp_check_log.append((float(self.beam.position), d, len(a), skipped, max(fz)))
        if len(a) and d > ip.ibp_check_tol and not getattr(self, '_ibp_check_warned', False):
            self._ibp_check_warned = True
            print(f'  [CSR2D] warning: at s = {self.beam.position:.4f} the shipped and the '
                  f'integrated-by-parts longitudinal wakes differ by {d:.1e} (relative L2, '
                  f'{len(a)} points). They are the same integral: rho and d rho/dx are '
                  f'inconsistent somewhere (grid edge, blending). '
                  f'Further kicks are logged in ibp_check_log, not printed.')

    def _assert_history(self):
        """
        Refuse to evaluate a wake without a history to interpolate.

        `compute_CSR: 0` gates get_DF/append_DF, so the run produces a beam with NO density
        history. Anything calling get_CSR_integrand afterwards then got near-zeros with no error
        raised -- it cost a full set of wrong figures in §6f, and bit again while benchmarking
        the B-spline hoist in Step 10, where debug=False gave a one-snapshot history and
        data_rho[k+1] read out of bounds. The name suggests a pure output switch and it is not.
        """
        n = len(getattr(self.DF_tracker, 'time_log', ()))
        if n < 2:
            raise RuntimeError(
                f'the density history holds {n} snapshot(s); the wake needs at least 2 to '
                f'interpolate between. Set CSR_computation.compute_CSR = 1, or pass '
                f'debug=True to run(), so that snapshots are recorded.')

    def get_CSR_integrand(self,s ,x, t, sp, xp, ignore_vx = False, taper = None, extras = None):
        """
        taper: (R1, R2, keep_inner) — multiply the integrands by a radial partition
        function of |r - r'|. With keep_inner False the weight rises 0 -> 1 across
        [R1, R2], removing the neighbourhood of the r' -> r singularity; with True
        it is the complement. The two pieces sum to the untapered integral exactly.

        extras: a dict to fill with what the integrated-by-parts form needs (CSR_integration.ibp;
        bspline_comoving only). The fields then come from interpolate3D_comoving_ibp, which is
        bit-identical for them, so the returned integrands do not change.
        """

        self._assert_history()
        sp_flat = sp.ravel()
        xp_flat = xp.ravel()

        if self.use_comoving:
            vx = self._comoving_fields(np.array([x]), np.array([s - t]),
                                       np.array([t]))[3][0]
            if self._quantile_frame():
                vx = self._observer_vx(x, s - t, vx)
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

        (X0_s, Y0_s, n_vec_s_x, n_vec_s_y,
         tau_vec_s_x, tau_vec_s_y) = (v[0] for v in self._lattice_vectors(np.array([s])))
        (X0_sp, Y0_sp, n_vec_sp_x, n_vec_sp_y,
         tau_vec_sp_x, tau_vec_sp_y) = self._lattice_vectors(sp_flat)

        r_minus_rp_x = X0_s - X0_sp + x * n_vec_s_x - xp_flat * n_vec_sp_x
        r_minus_rp_y = Y0_s - Y0_sp + x * n_vec_s_y - xp_flat * n_vec_sp_y
        r_minus_rp = np.sqrt(r_minus_rp_x**2 + r_minus_rp_y**2)


        # curvature of the element each s' falls in (0 past the lattice end). One searchsorted
        # instead of the per-element mask loop: identical values, 3.8 -> 0.5 ms on 80k points
        # with 65 elements (BC20), i.e. ~15% of a wake point.
        dist = np.asarray(self.lattice.distance)
        rho_el = np.asarray(self.lattice.rho)
        k_el = np.searchsorted(dist, sp_flat, side='right')
        rho_sp = np.where(k_el < dist.size, rho_el[np.minimum(k_el, dist.size - 1)], 0.0)

        t_ret = t - r_minus_rp

        if self.use_comoving:
            # One pass returns all five fields: they share the frame construction
            # and the stencil indices, so five separate calls would redo that work.
            # The chain rule for d rho/dz is applied inside, with the time-blended
            # p', which the pre-baked per-snapshot version could not do.
            z_ret = sp_flat - t_ret
            if extras is None or not (self.integration_params.ibp and self.use_comoving):
                # the x' refinement (near_adapt_x / far_adapt_x) asks for extras only to read the
                # retarded density, which the plain pass returns; the ibp pass computes A and B on
                # top (22.3 vs 17.4 ms per 80k points, bit-identical first five fields)
                (density_ret, density_x_ret, density_z_ret,
                 vx_ret, vx_x_ret) = self._comoving_fields(xp_flat, z_ret, t_ret)
                A_ibp = B_ibp = None
            else:
                (density_ret, density_x_ret, density_z_ret,
                 vx_ret, vx_x_ret, A_ibp, B_ibp) = self._comoving_fields_ibp(xp_flat, z_ret, t_ret)

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
        if ignore_vx:
            vx = 0
            vx_x_ret = 0
            vx_ret = 0
        if getattr(self.integration_params, 'integrand_kernel', 'numpy') == 'numba':
            n = xp_flat.size
            CSR_integrand_z = np.empty(n); CSR_integrand_x = np.empty(n); vn = np.empty(n)
            as_arr = lambda v: np.broadcast_to(np.asarray(v, dtype=float), (n,)) if np.ndim(v) == 0 else np.ascontiguousarray(v, dtype=float)
            R1, R2, keep_inner = taper if taper is not None else (0.0, 1.0, False)
            csr_integrand_kernel(np.ascontiguousarray(xp_flat, dtype=float), as_arr(rho_sp), as_arr(r_minus_rp_x), as_arr(r_minus_rp_y),
                                 as_arr(r_minus_rp), as_arr(n_vec_sp_x), as_arr(n_vec_sp_y), as_arr(tau_vec_sp_x), as_arr(tau_vec_sp_y),
                                 as_arr(density_ret), as_arr(density_x_ret), as_arr(density_z_ret), as_arr(vx_ret), as_arr(vx_x_ret),
                                 float(vx), float(n_vec_s_x), float(n_vec_s_y), float(tau_vec_s_x), float(tau_vec_s_y),
                                 getattr(self.integration_params, 'transverse_terms', 'all') == 'W1', taper is not None,
                                 float(R1), float(R2), bool(keep_inner), CSR_integrand_z, CSR_integrand_x, vn)
            if extras is not None:
                extras.update(rx=r_minus_rp_x, ry=r_minus_rp_y, nx=n_vec_sp_x, ny=n_vec_sp_y,
                              xp=xp_flat, sp=sp_flat, kappa=rho_sp, vn=vn,
                              rho=density_ret, rho_x=density_x_ret, A=A_ibp, B=B_ibp)
            CSR_integrand_x = CSR_integrand_x.reshape(xp.shape)
            CSR_integrand_z = CSR_integrand_z.reshape(xp.shape)
            return CSR_integrand_z, CSR_integrand_x

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

        if getattr(self.integration_params, 'transverse_terms', 'all') == 'W1':
            CSR_integrand_x = W1
        else:
            CSR_integrand_x = W1 + W2 + W3

        if extras is not None:
            extras.update(rx=r_minus_rp_x, ry=r_minus_rp_y, nx=n_vec_sp_x, ny=n_vec_sp_y,
                          xp=xp_flat, sp=sp_flat, kappa=rho_sp,
                          vn=velocity_x * n_vec_sp_x + velocity_y * n_vec_sp_y,
                          rho=density_ret, rho_x=density_x_ret, A=A_ibp, B=B_ibp)

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
            if self.CSR_params.mesh_mode != 'uniform':
                g.attrs['mesh_mode'] = self.CSR_params.mesh_mode
                g.create_dataset('x_axis_transformed', data=np.asarray(self.CSR_xrange_transformed))
                g.create_dataset('z_axis', data=np.asarray(self.CSR_zrange))
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
    def _write_schedule(self, hf):
        """
        Record the REALISED step schedule in the output.

        Without this an adaptive run is not reproducible: the node set depends on the beam's own
        linear optics and on tolerances whose defaults may change, so "mode: auto" in the input
        YAML does not pin down what was actually done. Writing the nodes, the snapshot and kick
        masks, and the tolerances that produced them makes a run replayable and comparable.
        """
        sch = getattr(self.lattice, 'schedule', None)
        if sch is None:
            return
        g = hf.create_group('step_schedule')
        g.attrs['mode'] = str(sch.mode)
        g.attrs['kick_interval'] = str(sch.kick_interval)
        g.attrs['n_nodes'] = int(sch.n_nodes)
        g.attrs['n_snapshots'] = int(sch.n_snap)
        g.attrs['n_kicks'] = int(sch.n_kick)
        g.create_dataset('s_nodes', data=sch.s_nodes)
        g.create_dataset('dl', data=sch.dl)
        g.create_dataset('ele_of', data=sch.ele_of)
        g.create_dataset('is_snap', data=sch.is_snap.astype('u1'))
        g.create_dataset('is_kick', data=sch.is_kick.astype('u1'))
        g.create_dataset('kick_lo', data=sch.kick_lo)
        g.create_dataset('kick_hi', data=sch.kick_hi)
        for k, v in (getattr(sch, 'auto_opts', None) or {}).items():
            if isinstance(v, (int, float, bool)) or v is None:
                g.attrs[f'auto_{k}'] = (np.nan if v is None else v)
        if hasattr(sch, 'auto_h_eff'):
            g.create_dataset('auto_s_scan', data=sch.auto_s_scan)
            g.create_dataset('auto_h_eff', data=sch.auto_h_eff)

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
            self._write_schedule(hf)
            hf.create_dataset(name = 'step_positions', data = self.lattice.steps_record, shape = self.lattice.steps_record.shape)
            hf.create_dataset(name='coords', data=self.lattice.coords)
            hf.create_dataset(name='n_vec', data=self.lattice.n_vec)
            hf.create_dataset(name='tau_vec', data=self.lattice.tau_vec)
            dict2hdf5(hf, self.statistics)
            if getattr(self, 'quantile_log', None):
                # one row per kick: s, window lo, hi, margin, f_window, res_z, peak FWHM,
                # finest mesh z spacing, dropped-tail fraction of the 1D loss and field
                q = hf.create_group('quantile_frame')
                q.attrs['frame'] = str(self.DF_tracker.frame)
                q.attrs['frame_requested'] = str(getattr(self.DF_tracker, 'frame_requested', self.DF_tracker.frame))
                if getattr(self.DF_tracker, 'frame_decision', None):
                    ch, fp, fl, ff, bend = self.DF_tracker.frame_decision
                    q.attrs['auto_decision'] = f'{ch}: peak window keeps {fp:.4f} of the charge; dropped part carries {fl:.4f} of the 1D loss, {ff:.4f} of the field at {bend}'
                cols = ('s', 'window_lo', 'window_hi', 'window_margin', 'window_margin_hi', 'f_window', 'res_z',
                        'peak_fwhm', 'mesh_dz_min', 'dropped_loss_frac', 'dropped_field_frac')
                arr = np.array(self.quantile_log)
                for k, c in enumerate(cols):
                    q.create_dataset(c, data=arr[:, k])



















