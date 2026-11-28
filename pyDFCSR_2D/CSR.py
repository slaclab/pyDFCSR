import os
import sys
import time

import h5py
import numpy as np
from bmadx import Drift, SBend, Quadrupole

# MPICH's libfabric 'sockets' provider spins two kernel threads per process on macOS (2.75 cores
# per rank); 'tcp' does not. It must be chosen before mpi4py initialises MPI (progress_facet_bc20.md 11h).
if sys.platform == 'darwin':
    os.environ.setdefault('FI_PROVIDER', 'tcp')
from mpi4py import MPI

from .bands import BandsMixin, degenerate_margin_factor  # noqa: F401 (re-exported)
from .beams import Beam
from .deposit_smooth import DF_tracker_comoving
from .ibp import IBPMixin
from .lattice import Lattice
from .layout import LayoutMixin
from .params import Integration_params, CSR_params
from .schedule import SCHEDULE_ELEMENT_KEYS
from .tools import dict2hdf5, full_path, isotime
from .tracking import SextupoleKDK
from .waist import scan_waists, sigma_from_coords, format_report, retention_schedule
from .wake import WakeMixin
from .yaml_parser import parse_yaml


class CSR2D(WakeMixin, LayoutMixin, BandsMixin, IBPMixin):
    """
    The 2D CSR driver: tracks the beam through the lattice (bmad-x), deposits the density history,
    computes the longitudinal and transverse CSR wake on a mesh at the kick nodes, and applies it.

    The wake integral lives in the mixins: wake.py (integral and integrand), layout.py (where the
    s' columns go), bands.py (where the retarded beam is in x'), ibp.py (integration by parts).
    See docs/DFCSR_2D_methods.md.

        sim = CSR2D(input_file='config.yaml')
        sim.run()
    """

    # The bend the beam is in or has last left, as the run loop records it on element entry.
    BEND_STATE = dict(inbend=False, afterbend=False,   # in a dipole / past at least one dipole
                      R_rec=None, phi_rec=None,       # radius and angle of the current or last bend
                      bend_start_rec=None,            # its entrance
                      bend_len_rec=0.0,               # its arc length
                      prev_bend_start=None,           # inside a bend: the previous bend's entrance
                      prev_bend_end=None,             # inside a bend: the previous bend's exit
                      drift_prev_start=None)          # the previous bend's entrance, kept through the drift

    def __init__(self, input_file=None, parallel = False):

        self.timestamp = isotime()
        self._parallel_requested = bool(parallel)
        self.parallel = False
        self.rank = MPI.COMM_WORLD.Get_rank() if parallel else 0
        self.__dict__.update(self.BEND_STATE)
        self._last_region_nodes = [0, 0]
        self.counters = {}          # per-run diagnostics, reported at the end of run() (_report_counters)
        self._warned = set()        # keys of the once-only warnings already printed on this rank
        if input_file:
            self.parse_input(input_file)
            self.input_file = input_file
        self.formation_length = None
        self.initialization()  # process the initial beam

        self.prefix = f'{self.CSR_params.write_name}-{self.timestamp}'

        if parallel:
            self.init_MPI()
            sc = self._step_control_cfg
            # the two replicated-work splits (progress_facet_bc20.md 11h): tracking (bit-identical)
            # and deposition (1e-16 differences) over contiguous particle slices per rank
            if sc.get('particle_split', False):
                self.beam.set_particle_split(MPI.COMM_WORLD)
            if sc.get('deposit_split', False):
                self.DF_tracker.set_deposit_split(MPI.COMM_WORLD)
        else:
            self.parallel = False

    def __getstate__(self):
        # the torch evaluator holds device tensors and a back reference; it is rebuilt on demand
        return {k: v for k, v in self.__dict__.items() if k != '_torch_wake'}

    def __setstate__(self, state):
        self.__dict__.update(self.BEND_STATE)
        self._last_region_nodes = [0, 0]
        self.counters, self._warned = {}, set()
        self.__dict__.update(state)

    def _is_root(self):
        # by rank, not by self.parallel: that is set only in init_MPI, after parsing and the first deposit
        return getattr(self, 'rank', 0) == 0

    def _note(self, key, msg, every_rank=False):
        """
        Print msg the first time `key` fires. every_rank False: on rank 0 only, for conditions every
        rank sees alike (lattice, schedule, beam). every_rank True: on whichever rank hits it, tagged
        with the rank, for conditions met at individual wake-mesh points (the mesh is split over ranks).
        """
        if key in self._warned or not (every_rank or self._is_root()):
            return
        self._warned.add(key)
        print((f'[rank {self.rank}] ' if (self.parallel and every_rank) else '') + msg, flush=True)

    def _count(self, key, n=1):
        self.counters[key] = self.counters.get(key, 0) + n

    def _report_counters(self):
        """
        Sum the per-rank counters (MPI), print them once on rank 0, keep them for statistics.h5.
        retention_short: steps whose retained history started later than the integration asked for;
        steps_blocked / kicks_skipped: drift_cutoff; far_clip_at_reach / drift_causal_clipped: far
        regions ended by the reach or by the retained history rather than by causality;
        thin_dropped / thin_kept: drift_snap_thin.
        """
        c = dict(self.counters)
        if self.parallel:
            keys = sorted(set().union(*MPI.COMM_WORLD.allgather(sorted(c))))
            c = {k: MPI.COMM_WORLD.allreduce(c.get(k, 0), op=MPI.MAX if k.endswith('_worst') else MPI.SUM)
                 for k in keys}
        self.diagnostics = c
        if c and self._is_root():
            print('  [diagnostics] ' + ', '.join(f'{k} = {v:.4g}' if isinstance(v, float) else f'{k} = {v}'
                                               for k, v in sorted(c.items())), flush=True)

    def parse_input(self, input_file):
        input = parse_yaml(input_file)
        self.check_input_consistency(input)
        self.input = input
        self.beam = Beam(input['input_beam'])
        # step_control defaults to mode 'auto' with midpoint kicks (since 2026-11-06), the
        # schedule every validated run used. A config that wants the old uniform stepping
        # must now say step_control: {mode: legacy}.
        sc = dict(input.get('step_control') or {})
        self._step_control_cfg = sc
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

        deposition_config = input.get('particle_deposition') or {}
        method = deposition_config.get('method', 'bspline_comoving')
        if method != 'bspline_comoving':
            raise ValueError(f"particle_deposition.method {method!r} is not supported: only 'bspline_comoving' "
                             f"(the default) remains; the legacy CIC and bspline_fft deposits were removed in the "
                             f"2026-11 cleanup")
        self.DF_tracker = DF_tracker_comoving(deposition_config)
        self.DF_tracker.verbose = self._is_root()
        if self.DF_tracker.poly_degree != 1:
            raise ValueError('particle_deposition.poly_degree must be 1: the two-branch band solver (Eq 4.24) '
                             'assumes a straight beam axis')
        self.integration_params = Integration_params(input.get('CSR_integration'))
        ip = self.integration_params
        if ip.ibp and input.get('CSR_computation', {}).get('backend', 'numba') == 'torch':
            raise NotImplementedError('CSR_integration.ibp is not implemented in the torch backend')
        if ip.ibp and ip.near_adapt and ip.near_adapt_x:
            self._note('ibp_refined', '  [CSR2D] note: CSR_integration.ibp acts only on columns without x\' '
                       'refinement, and near_adapt_x refines every column of its regions (far_adapt_x: all '
                       'regions), so the by-parts form is used only where both are off. ibp_check still '
                       'compares the two forms on uniform nodes.')

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
        if self._is_root():
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
        if self.DF_tracker.frame == 'auto':
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
        self.__dict__.update(CSR2D.BEND_STATE)      # also used by CSR1D, which borrows this method

    def init_MPI(self):
        self.parallel = True
        comm = MPI.COMM_WORLD
        self.rank = comm.Get_rank()

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

    def get_formation_length(self, R, sigma_z):
        """Steady-state formation length (24 R^2 sigma_z)^(1/3), stored in self.formation_length."""
        self.formation_length = (24 * (R ** 2) * sigma_z) ** (1 / 3)

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
            self.get_formation_length(R=R, sigma_z=5 * self.beam.sigma_z)
            return True
        if self.afterbend:
            self.get_formation_length(R=self.R_rec, sigma_z=5 * self.beam.sigma_z)
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
        try:
            b = self.beam
            sigma0 = sigma_from_coords(b.x, b.px, b.particle.y, b.particle.py,
                                       b.z, b.pz)
            self._retention = retention_schedule(
                self.lattice.lattice_config, sigma0,
                self.integration_params.n_formation_length)
            plan = self._retention
            if self._is_root():
                print(f'  [retention] precomputed floor from linear optics: max depth '
                      f'{plan.max_depth:.4f} m -> '
                      f'{plan.max_snapshots(self.lattice.step_size)} snapshots at '
                      f'step_size {self.lattice.step_size:g} m (safety {plan.safety:g})')
        except Exception as exc:                       # noqa: BLE001
            self._note('retention', f'  [retention] precomputation skipped '
                                    f'({type(exc).__name__}: {exc}); falling back to instantaneous L_f')

    def _retention_floor(self):
        f = (self._retention.floor_at(self.beam.position)
             if self._retention is not None else None)
        # drift_reach: from a bend's entrance until the next bend, never discard the
        # history of that bend -- it is the source of the exit wake downstream
        if self.integration_params.drift_reach and self.afterbend:
            sb = self.bend_start_rec
            if sb is not None:
                f = sb if f is None else min(f, sb)
        pb = self._prev_bend_reach_start()
        if pb is not None:
            f = pb if f is None else min(f, pb)
        dp = self.drift_prev_start
        if (self.integration_params.drift_causal_reach and self._exit_transient()
                and dp is not None):
            # drift_causal_reach: the previous bend (and n_fl*L_f before it) is still a source here
            fd = max(0.0, float(dp) - self.integration_params.n_formation_length * self.formation_length)
            f = fd if f is None else min(f, fd)
        g = self._geo_reach(self.beam.position + 3.0 * self.beam._sigma_z)
        if g is not None:
            # straight-section reach (Integration_params.reach_straight) for the bunch head, not
            # below the previous bend's exit (the history there is drift_reach's)
            gs = self.beam.position + 3.0 * self.beam._sigma_z - g
            pe = self.prev_bend_end
            if pe is not None:
                gs = max(gs, float(pe))
            f = gs if f is None else min(f, gs)
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
        g = self._geo_reach(self.beam.position)
        if g is not None:
            gs = self.beam.position - g
            pe = self.prev_bend_end
            if pe is not None:
                gs = max(gs, float(pe))
            need = min(need, max(0.0, gs))
        if (self.integration_params.drift_causal_reach and self._exit_transient()
                and self.drift_prev_start is not None):
            need = min(need, max(0.0, float(self.drift_prev_start)
                                 - self.integration_params.n_formation_length * self.formation_length))
        have = self.DF_tracker.time_log[0]
        if have > need + 1e-9:
            self._count('retention_short')
            self.counters['retention_worst'] = max(self.counters.get('retention_worst', 0.0), have - need)

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

        Never fatal: a diagnostic that can abort a run is worse than the problem, and
        this one is linear optics on the design lattice, so it can be wrong.
        """
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
        """
        Track through the lattice on the schedule's nodes: record density snapshots, compute and
        apply the CSR kicks, write statistics and output. Not resumable (docs/code_notes.md).
        debug: record snapshots even with compute_CSR = 0.
        """
        if getattr(self, '_has_run', False):
            raise RuntimeError(
                f'run() has already been called on this CSR2D (beam is at s = '
                f'{self.beam.position:.6f} m). It is not resumable: it would restart tracking '
                f'from the lattice entrance with the current beam. Construct a new CSR2D for '
                f'each stop_time.')
        self._has_run = True

        if self._is_root():
            print('Starting the DFCSR run')
            self.report_waists()
            self.report_schedule()
        self.build_retention_plan()

        sched = self.lattice.schedule
        step_count = 1                              # the schedule's node index (node 0: the entrance)
        self._s_last_kick = self.beam.position      # for trailing kicks; persists across elements
        ele_count = 0
        skip_ele = False                            # the previous element was tracked in one piece
        ele_prev = None
        self.inbend = False
        self.afterbend = False
        self.formation_length = 0.0

        for ele in list(self.lattice.lattice_config.keys())[1:]:
            self.lattice.update(ele)
            L = self.lattice.lattice_config[ele]['L']
            etype = self.lattice.lattice_config[ele]['type']
            steps = self.lattice.steps_per_element[ele_count]

            if (not skip_ele) and ele_count > 0:
                self._finish_element(ele_prev, ele_count)
            if steps == 0:                          # no node inside: one piece over the whole element
                skip_ele = True
                element = self.get_bmadx_element(ele=ele, DL=L, exit=True, entrance=True)
                self.beam.track(element, L, update_step=False)
            R = self._enter_element(ele_count, etype, L)

            distance_in_current_ele = 0.0
            for step in range(steps):
                time0 = time.time()
                if (step == 0) and (ele_count > 0):
                    # the first step of an element starts at its entrance (the rest of the step that
                    # crossed the boundary was tracked in the previous element)
                    dl = self.lattice._positions_record[step_count] - self.lattice.distance[ele_count - 1]
                    skip_ele = False
                else:
                    dl = sched.dl[step_count]
                element = self.get_bmadx_element(
                    ele=ele, DL=dl, entrance=(step == 0),
                    exit=self._ends_element(self.beam.position + dl, ele_count))
                self.beam.track(element, dl)
                distance_in_current_ele += dl

                self._refresh_formation_length(R)
                if (debug or self.CSR_params.compute_CSR) and sched.is_snap[step_count]:
                    self._record_snapshot()
                blocked = self._drift_cutoff_reached(distance_in_current_ele, sched.is_kick[step_count])
                if self.CSR_params.compute_CSR and (not blocked) and sched.is_kick[step_count]:
                    self._kick(step_count)

                self.update_statistics(step = step_count)
                if self._is_root():
                    print("Finish step {}, s = {},  in {} seconds".format(step_count, self.beam.position, time.time() - time0))
                step_count += 1
                if self._stop_reached(stop_time):
                    self._report_counters()
                    return

            ele_prev = ele
            ele_count += 1

        self._report_counters()
        self.dump_beam(label='end')
        self.write_statistics()

    def _finish_element(self, ele_prev, ele_count):
        """Track the rest of the step that crossed out of the previous element (pieces < 1 um skipped)."""
        DL_1 = self.lattice.distance[ele_count - 1] - self.beam.position
        if DL_1 > 1.0e-6:
            element = self.get_bmadx_element(ele=ele_prev, DL=DL_1, exit=True)
            self.beam.track(element, DL_1, update_step=False)

    def _enter_element(self, ele_count, etype, L):
        """
        Update the bend bookkeeping (BEND_STATE) and the formation length on entering an element.
        Returns the bending radius (inf outside a dipole).
        """
        R = float('inf')
        if etype == 'dipole':
            angle = self.lattice.lattice_config[self.lattice.current_element]['angle']
            R = L / angle
            self.inbend = True
            # the bend left behind still radiates into this one (prev_bend_reach); the section before
            # this bend is straight only from that bend's exit (reach_straight); its entrance is kept
            # through the next drift (drift_causal_reach retention)
            self.prev_bend_start = self.bend_start_rec if self.afterbend else None
            self.prev_bend_end = (self.bend_start_rec + self.bend_len_rec) if self.afterbend else None
            self.drift_prev_start = self.prev_bend_start
            self.bend_len_rec = L
            self.afterbend = True
            self.R_rec = R
            self.phi_rec = angle
            # in the drift that follows, the exit transient's source is this arc (drift_reach)
            self.bend_start_rec = (float(self.lattice.distance[ele_count - 1])
                                   if ele_count > 0 else 0.0)
            self._refresh_formation_length(R)
        else:
            self.inbend = False
            self.prev_bend_start = None
            # in a drift L_f keeps the in-bend expression with the last bend's R; before the first
            # bend it is the accumulated drift length
            if not self._refresh_formation_length(R):
                self.formation_length += L
        return R

    def _record_snapshot(self):
        """Deposit the beam, append it to the history, truncate the history and rebuild the interpolant."""
        self.DF_tracker.get_DF(x=self.beam.x, z=self.beam.z, px=self.beam.px, t=self.beam.position)
        self.DF_tracker.append_DF()
        self._thin_drift_history()
        self.DF_tracker.append_interpolant(formation_length=self.formation_length,
                                           n_formation_length=self.integration_params.n_formation_length,
                                           min_start_time=self._retention_floor())
        self.DF_tracker.build_interpolant()
        self._check_retention()

    def _drift_cutoff_reached(self, distance_in_element, is_kick):
        """
        CSR_integration.drift_cutoff: past cutoff * |R phi| of the last bend in the drift after it, the
        exit transient has died and no wake is computed (docs/code_notes.md). Counted and noted once.
        """
        cutoff = self.integration_params.drift_cutoff
        blocked = bool(cutoff and self._exit_transient() and self.R_rec
                       and distance_in_element > cutoff * abs(self.R_rec) * abs(self.phi_rec))
        if blocked:
            self._count('steps_blocked')
            if is_kick:
                self._count('kicks_skipped')
            self._note('blocker', f'Beyond {cutoff:g} |R*phi| = '
                                  f'{cutoff * abs(self.R_rec) * abs(self.phi_rec):.4f} m '
                                  f'past the last bend; skipping CSR from here in this drift.')
        return blocked

    def _kick(self, step_count):
        """The wake on the mesh at this node, applied over the kick length, then the per-kick output."""
        sched = self.lattice.schedule
        self.get_CSR_mesh()
        if self.parallel:
            self.calculate_2D_CSR_parallel()
        else:
            self.calculate_2D_CSR()
        if self._windowed_frame() and self._is_root():
            self._quantile_diagnostics()
        if self.CSR_params.apply_CSR:
            # the arc this kick stands for (docs/code_notes.md): centred on the node for midpoint
            # kicks, else the arc actually covered since the previous kick
            if sched.kick_interval == 'midpoint':
                L_kick = sched.kick_hi[step_count] - sched.kick_lo[step_count]
            else:
                L_kick = self.beam.position - self._s_last_kick
            self.beam.apply_wakes(self.dE_dct, self.x_kick,
                                  self.CSR_xrange_transformed, self.CSR_zrange, L_kick,
                                  self.CSR_params.transverse_on,
                                  x_shift=self._mesh_ridge(self.beam.z),
                                  clamp_x=self.CSR_params.mesh_x_quantile is not None,
                                  clamp_z=self.CSR_params.mesh_z_quantile is not None)
            self._s_last_kick = self.beam.position
        wb = self.CSR_params.write_beam
        if wb == 'all' or (isinstance(wb, list) and step_count in wb):
            self.dump_beam(label = step_count)
        if self.CSR_params.write_wakes:
            self.write_wakes()

    def _stop_reached(self, stop_time):
        """
        True once the beam has reached stop_time (within round-off). The node grid is fixed, so a
        stop_time between nodes stops at the first node past it, which is printed.
        """
        if not (stop_time and self.beam.position >= stop_time - 1e-12 * max(abs(stop_time), 1.0)):
            return False
        over = self.beam.position - stop_time
        if over > 1e-9 * max(abs(stop_time), 1.0):
            print(f'  [run] stopped at s = {self.beam.position:.6f}, '
                  f'{over:.6f} m past the requested stop_time = {stop_time:.6f}. '
                  f'No node lands exactly there; assert on beam.position when '
                  f'comparing runs at different step sizes.')
        return True


    def _mesh_ridge(self, z):
        """
        Lab-x offset of the beam ridge from the straight tilt fit, sigma_xi * g(w), from
        the newest snapshot's curvature shear (particle_deposition.curvature). Zero when
        the shear is off, so the mesh and the kick lookup are then exactly as before.
        """
        tr = self.DF_tracker
        if not (tr.curvature and tr.sigma_z):
            return np.zeros_like(np.asarray(z, dtype=float))
        from .deposit_smooth import _curv_g_vec
        w = (np.asarray(z, dtype=float) - tr.z_bar) / tr.sigma_z
        if tr.frame in ('quantile', 'peak'):
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

        if (self.CSR_params.mesh_mode == 'uniform'
                and xlim > self.DF_tracker.u_flat() + 1e-9):
            # The outer mesh rows sit where the deposit's u edge taper has removed part of the
            # source charge (DF_tracker_comoving.u_taper), so their near field is weakened.
            # Few particles are there, but say so once.
            self._note('taper_mesh', f'  [CSR2D] note: the wake mesh spans +-{xlim} sigma_x, beyond the deposit\'s '
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
            zq = self.beam.z
            if cp.mesh_z_quantile is not None:
                # the axis on the [q, 1-q] span, the particles outside DROPPED (not clipped onto
                # the edge: a clipped pile is a spike in the monitor); they get the edge kick
                zlo, zhi = np.quantile(zq, [cp.mesh_z_quantile, 1.0 - cp.mesh_z_quantile])
                zq = zq[(zq >= zlo) & (zq <= zhi)]
            zrange = charge_nodes(zq, zbins, cp.mesh_uniform_frac,
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

        xmesh_transform, zmesh = np.meshgrid(xrange, zrange, indexing='ij')

        xmesh_transform = xmesh_transform.flatten()
        zmesh = zmesh.flatten()

        xmesh = xmesh_transform +  np.polyval(p, zmesh) + self._mesh_ridge(zmesh)

        self.CSR_xmesh = xmesh
        self.CSR_zmesh = zmesh
        self.CSR_zrange = zrange
        self.CSR_xrange_transformed = xrange
        if self._windowed_frame():
            self._fit_beam_vx()

    def _windowed_frame(self):
        # frames 'quantile' and 'peak' share the windowed machinery (taper, mesh, observer vx, log)
        return self.DF_tracker.frame in ('quantile', 'peak')

    def _deposition_resolution(self):
        """
        (z, x) effective resolution of the newest density snapshot [m]: cell times
        sqrt(smoothing^2 + 2/3), the B-spline deposit and evaluation adding 1/3 cell^2 each.
        """
        tr = self.DF_tracker
        if tr.sigma_z is None:
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
        if max(abs(frac_loss), frac_field) > cp.pedestal_warn:
            self._note('pedestal', f'  [quantile frame] WARNING at s = {self.beam.position:.4f} m: the dropped '
                  f'{100 * (1 - tr.f_window):.2f}% of the charge carries {100 * frac_loss:.1f}% of '
                  f'the 1D energy loss and {100 * frac_field:.1f}% of the 1D field on the window '
                  f'(pedestal_warn {cp.pedestal_warn:g}). '
                  + ("frame 'auto' chose 'peak' on the initial beam; the pedestal matters now: use frame 'quantile'."
                     if getattr(tr, 'frame_requested', None) == 'auto' and tr.frame == 'peak'
                     else 'Lower particle_deposition.frame_quantile.'))
        if res_z > 0 and fwhm / res_z < 5:
            self._note('resolution', f'  [quantile frame] WARNING at s = {self.beam.position:.4f} m: the highest '
                  f'current peak (FWHM {fwhm*1e6:.3g} um) is only {fwhm/res_z:.1f}x the '
                  f'deposition z resolution ({res_z*1e6:.3g} um). Raise zbins or lower '
                  f'smoothing_sigma_z.')
        if fwhm / h_at_peak < 3:
            self._note('mesh_resolution', f'  [quantile frame] WARNING at s = {self.beam.position:.4f} m: the finest '
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
    
    def _numba_wakes(self, ks, batch=256):
        """
        (dE/dct, x kick) at the mesh points ks: layouts built `batch` points at a time
        (wake_layouts, one band solve per batch instead of per region), evaluated point by point.
        """
        s_all = self.beam.position + np.asarray(self.CSR_zmesh)[ks]
        x_all = np.asarray(self.CSR_xmesh)[ks]
        out = np.zeros((2, len(ks)))
        for a in range(0, len(ks), batch):
            lays = self.wake_layouts(s_all[a:a + batch], x_all[a:a + batch])
            for j, lay in enumerate(lays):
                out[:, a + j] = self.get_CSR_wake(s_all[a + j], x_all[a + j], layout=lay)
        return out[0], out[1]

    def _torch_wakes(self, ks):
        """(dE/dct, x kick) at the mesh points ks with the torch backend (torch_backend.evaluate)."""
        if getattr(self, '_torch_wake', None) is None:
            from .torch_backend.evaluate import TorchWake
            self._torch_wake = TorchWake(self, device=self.CSR_params.device, chunk=self.CSR_params.torch_chunk,
                                         compile=self.CSR_params.torch_compile, bands=self.CSR_params.torch_bands)
        z = np.asarray(self.CSR_zmesh)[ks]
        return self._torch_wake.compute(self.beam.position + z, np.asarray(self.CSR_xmesh)[ks])

    def calculate_2D_CSR(self):

        N = self.CSR_params.xbins*self.CSR_params.zbins
        self.dE_dct = np.zeros((N,))
        self.x_kick = np.zeros((N,))

        if self.CSR_params.backend == 'torch':
            self.dE_dct[:], self.x_kick[:] = self._torch_wakes(np.arange(N))
        else:
            self.dE_dct[:], self.x_kick[:] = self._numba_wakes(np.arange(N))

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
        if self.CSR_params.backend == 'torch':
            ks = np.arange(rank, work_size, size)
            dE_local[ks], xk_local[ks] = self._torch_wakes(ks)
        else:
            ks = np.arange(rank, work_size, size)
            dE_local[ks], xk_local[ks] = self._numba_wakes(ks)

        self.dE_dct = np.zeros((work_size,))
        self.x_kick = np.zeros((work_size,))
        comm.Allreduce(dE_local, self.dE_dct, op=MPI.SUM)
        comm.Allreduce(xk_local, self.x_kick, op=MPI.SUM)

        self.dE_dct = self.dE_dct.reshape((self.CSR_params.xbins, self.CSR_params.zbins))
        self.x_kick = self.x_kick.reshape((self.CSR_params.xbins, self.CSR_params.zbins))
        if rank == 0:
            self._ibp_consistency_check()


    def _thin_drift_history(self):
        """
        drift_snap_thin: in a drift after a bend, drop the previous snapshot if linear
        interpolation between its neighbours reproduces it within the tolerance (see
        DF_tracker_comoving.thin_previous). Online, O(1) per step, newest always kept.
        """
        tol = self.integration_params.drift_snap_thin
        if not (tol and self._exit_transient()):
            return
        # never thin the bend itself: P (the candidate) must lie past the bend exit
        tr = self.DF_tracker
        if len(tr.time_log) < 3 or tr.time_log[-2] <= self._bend_exit_rec():
            return
        gap = self.integration_params.n_formation_length * self.formation_length
        dropped, err = tr.thin_previous(tol, max_gap=gap)
        self._count('thin_dropped', int(dropped))
        self._count('thin_kept', int(not dropped))


          
          


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


        if not getattr(self, '_wakes_started', False):
            self._wakes_started = True
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
            diag = getattr(self, 'diagnostics', None)      # CSR1D borrows this method and has none
            if diag:
                g = hf.create_group('diagnostics')
                for k, v in diag.items():
                    g.attrs[k] = v
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


