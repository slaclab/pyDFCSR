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

from .beams import Beam
from .deposit_smooth import DF_tracker_comoving
from .integrand_kernel import csr_integrand_kernel
from .interp1D import interpolate1D
from .interp3D import interpolate3D_comoving_fields, interpolate3D_comoving_ibp
from .lattice import Lattice
from .lookup import lookup_vec
from .params import Integration_params, CSR_params
from .schedule import SCHEDULE_ELEMENT_KEYS
from .tools import dict2hdf5, full_path, isotime
from .tracking import SextupoleKDK
from .waist import scan_waists, sigma_from_coords, format_report, retention_schedule
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

    def __setstate__(self, state):
        self.__dict__.update(self.BEND_STATE)
        self._last_region_nodes = [0, 0]
        self.counters, self._warned = {}, set()
        self.__dict__.update(state)

    def _is_root(self):
        return (not self.parallel) or self.rank == 0

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
        ele_count = 0
        skip_ele = False
        self.inbend = False
        self.afterbend = False
        self.formation_length = 0.0

        for ele in list(self.lattice.lattice_config.keys())[1:]:

            self.lattice.update(ele)
            L = self.lattice.lattice_config[ele]['L']
            type = self.lattice.lattice_config[ele]['type']
            steps = self.lattice.steps_per_element[ele_count]
            R = float('inf')

            ####### A step over the boundary of the elements, deal with the part of the step in the previous element
            if (not skip_ele) and ele_count > 0:
                DL_1 = self.lattice.distance[ele_count - 1] - self.beam.position   # The remaining distance in last element
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
                self.prev_bend_start = self.bend_start_rec if self.afterbend else None
                # ... and its exit: the section before THIS bend is straight only from there
                # (Integration_params.reach_straight)
                self.prev_bend_end = (self.bend_start_rec + self.bend_len_rec) if self.afterbend else None
                # kept through the drift after this bend (drift_causal_reach retention)
                self.drift_prev_start = self.prev_bend_start
                self.bend_len_rec = L
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

                # in a drift L_f keeps the in-bend expression with the last bend's R
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
                    self._count('steps_blocked')
                    if sched.is_kick[step_count]:
                        self._count('kicks_skipped')
                    self._note('blocker', f'Beyond {cutoff:g} |R*phi| = '
                                          f'{cutoff * abs(self.R_rec) * abs(self.phi_rec):.4f} m '
                                          f'past the last bend; skipping CSR from here in this drift.')
                
                
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
                                                  clamp_x=self.CSR_params.mesh_x_quantile is not None,
                                                  clamp_z=self.CSR_params.mesh_z_quantile is not None)
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
                    self._report_counters()
                    return

            ele_prev = ele

            ele_count += 1

        self._report_counters()
        self.dump_beam(label='end')
        self.write_statistics()


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
        if self._quantile_frame():
            self._fit_beam_vx()

    def _quantile_frame(self):
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
    
    def calculate_2D_CSR(self):

        N = self.CSR_params.xbins*self.CSR_params.zbins
        self.dE_dct = np.zeros((N,))
        self.x_kick = np.zeros((N,))

        for i in range(N):
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
        if not tr.curvature:
            return 0.0
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

        BOTH roots of the corrected Eq 4.24 are taken, each refined by a fixed point (the
        retarded frame depends on x' through t_ret), each given its own sub-band; a single
        fixed point from x' = x finds one root and missed up to 67% of a column (Step 4e).

        The roots are deliberately NOT labelled narrow/chirp. Which sign is which
        flips as tau^2 crosses 1, and at low tilt they merge into a single ridge
        (Step 5d), so they are treated symmetrically and any overlap is removed by
        _disjoint_bands.

        Eq 4.24 assumes one straight beam axis: poly_degree 1 (checked in parse_input).
        """
        tr = self.DF_tracker

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
            self.integration_params.xi_band_margin_degenerate)

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

    def _integrate_xi_region(self, s, x, t, sp, taper=None, scale_bands=True, adapt_x=False):
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
        ibp = self.integration_params.ibp

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
            # the x' refinement (near_adapt_x / far_adapt_x) and the by-parts form now coexist: a
            # refined column -- one where the retarded density jumps between x' nodes, i.e. a horn
            # crossed obliquely -- keeps the shipped form on its own refined nodes (the by-parts
            # form's gain is where the density is smooth across the band), every other column
            # takes the by-parts integrand on the uniform nodes. Until 2026-11-23 ibp switched
            # the refinement off, and on the loss beam that brought the aliased cells back
            # (max |dW| 34% of the horn's W at 200 nodes, ibp_nodes.py).
            do_adapt_x = bool(adapt_x and ip.near_adapt and ip.near_adapt_x)
            ex = {} if ibp else None
            gz, gx = self.get_CSR_integrand(s=s, t=t, x=x, xp=xp_mesh, sp=sp_mesh, taper=taper, extras=ex)
            refined = None
            if do_adapt_x:
                dxp_col, refined = self._adapt_x_error(s, x, t, sp, lo, width, gz, gx, taper)
            # A zero-width column has dxp = 0 and contributes nothing. Select it out
            # rather than multiplying by zero, so a degenerate node that landed on
            # the 1/|r-r'| pole cannot turn the whole region into a nan.
            iz = np.where(alive, np.trapz(y=gz, axis=0) * dxp, 0.0)
            ix = np.where(alive, np.trapz(y=gx, axis=0) * dxp, 0.0)
            iz_ship = iz
            if ibp:
                g_ibp, bnd, am, zone = self._ibp_integrand(gz, ex, t, taper, xp_mesh.shape)
                iz = np.where(alive, np.trapz(y=g_ibp, axis=0) * dxp + bnd, 0.0)
                wcol = np.where(alive, dxp, 0.0) * np.gradient(sp)
                self._ibp_gross_acc += float(np.sum(am.sum(axis=0) * wcol))
                self._ibp_zone_acc += float(np.sum((am * zone).sum(axis=0) * wcol))
            if refined is not None and refined.any():
                # refined columns: (iz, ix) already integrated over their own x' nodes, in BOTH
                # forms (the shipped total of the consistency check must see the same refinement)
                iz = np.where(alive & refined, dxp_col[0], iz)
                iz_ship = np.where(alive & refined, dxp_col[0], iz_ship)
                ix = np.where(alive & refined, dxp_col[1], ix)
            if ibp:
                # the shipped total is kept for the consistency check (get_CSR_wake)
                self._ibp_ship_acc += -self.CSR_scaling * np.trapz(y=iz_ship, x=sp)
            dE += -self.CSR_scaling * np.trapz(y=iz, x=sp)
            xk += self.CSR_scaling * np.trapz(y=ix, x=sp)
            meshes.append((xp_mesh, sp_mesh, gz, gx))

        if not meshes:
            z = np.zeros((nx, len(sp)))
            return (0.0, 0.0, z, np.broadcast_to(sp[None, :], z.shape).copy(), z, z)
        return (dE, xk) + tuple(np.concatenate(a, axis=0) for a in zip(*meshes))

    def _adapt_x_error(self, s, x, t, sp, lo, width, gz, gx, taper):
        """
        Per-column x' quadrature from the integrand's own error estimate (methods §5.4).

        gz, gx are the integrands on the xbins uniform nodes of every column (lo + k h). Each
        round adds the midpoints of the current intervals of the columns still flagged, so the
        trapezoid sum is refined in place, T_2m = T_m / 2 + (h / 2) sum f(mid), and the error of the
        finer sum is estimated as |T_2m - T_m| / 3. The first round covers every live column; after
        it, the columns are ranked by their s'-weighted error w_j e_j and the smallest set whose
        removal leaves at most x_adapt_tol * sum_j w_j |T_j| (W and F separately) is refined again,
        up to x_adapt_rounds rounds in all. Returns ((iz, ix), refined) with refined = every live
        column, since all of them now carry at least the 2n-node value.
        """
        ip = self.integration_params
        nx, ncol = gz.shape
        alive = width > 0.0
        h = np.where(alive, width / (nx - 1), 0.0)
        Tz = np.trapz(gz, axis=0) * h
        Tx = np.trapz(gx, axis=0) * h
        m = nx - 1                                          # intervals per column (same for all live ones)
        wcol = np.zeros(ncol)
        if ncol > 1:
            d = np.diff(sp); wcol[:-1] += 0.5 * d; wcol[1:] += 0.5 * d
        else:
            wcol[:] = 1.0
        wcol = np.abs(wcol)
        cols = np.flatnonzero(alive)
        h_c = h.copy()
        m_c = np.full(ncol, m)
        n_eval = 0
        for rnd in range(max(ip.x_adapt_rounds, 1)):
            if cols.size == 0:
                break
            mm = int(m_c[cols].max())                       # columns refined together have the same count
            k = np.arange(mm) + 0.5
            xn = lo[None, cols] + k[:, None] * h_c[None, cols]
            sn = np.broadcast_to(sp[cols][None, :], xn.shape).copy()
            fz, fx = self.get_CSR_integrand(s=s, t=t, x=x, xp=xn, sp=sn, taper=taper)
            n_eval += xn.size
            Tz_new = 0.5 * Tz[cols] + 0.5 * h_c[cols] * fz.sum(axis=0)
            Tx_new = 0.5 * Tx[cols] + 0.5 * h_c[cols] * fx.sum(axis=0)
            ez = np.abs(Tz_new - Tz[cols]) / 3.0
            ex_ = np.abs(Tx_new - Tx[cols]) / 3.0
            Tz[cols], Tx[cols] = Tz_new, Tx_new
            h_c[cols] *= 0.5
            m_c[cols] *= 2
            # which columns still carry too much of the error: per component, the largest weighted errors
            # until what is left unrefined is within tolerance of the region's gross sum
            keep = np.zeros(cols.size, dtype=bool)
            for e, T in ((ez, Tz), (ex_, Tx)):
                we = wcol[cols] * e
                budget = ip.x_adapt_tol * float(np.sum(wcol * np.abs(T)))
                if not we.sum() > budget:
                    continue
                order = np.argsort(-we)
                rest = we.sum() - np.cumsum(we[order])
                nk = int(np.searchsorted(-rest, -budget)) + 1
                keep[order[:nk]] = True
            cols = cols[keep]
        self._last_adapt_x = dict(cols=int(alive.sum()), evals=n_eval, unconverged=int(cols.size))
        return (Tz, Tx), alive

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
                self._count('far_clip_at_reach')

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

    def _geo_reach(self, s):
        """
        Entrance-transient reach for an observer at s, measured back from s, or None.

        The straight section the beam came along before the last bend entrance is a source
        retarded by ~ R phi^3/6 (phi = arc angle traversed) whose contribution decays only as
        1/u with the distance u before the entrance, so it is needed out to
        reach_straight * R phi, i.e. reach = (s - entrance) + reach_straight * R phi. It is
        applied while that retardation is within reach_slip_max sigma_z of the bunch
        (Integration_params.reach_straight): inside the bend slip = a^3/(6R^2) with a the arc
        length past the entrance; in the drift x past the exit the straight sources slip a
        further x phi^2/2 and fall out of the bunch within a fraction of R phi.
        Inside the bend only (see below). The straight section is integrated as its own graded
        region below the regular layout (_straight_region_nodes); the retention floor and check use
        the same reach.
        """
        ip = self.integration_params
        rs = ip.reach_straight
        sb = self.bend_start_rec
        if not rs or sb is None:
            return None
        R = abs(float(self.R_rec or 0.0))
        if R <= 0.0:
            return None
        a = float(s) - float(sb)
        if a <= 0.0:
            return None
        if not self.inbend:
            # NOT in the drift after the bend: there the frozen straight history is seen from a
            # different straight line, each column contributes ~ phi^2 lambda'/u and the sum grows
            # as ln(reach) -- measured on Indranil's gain beam 0.19 m past B1LE, -0.18 MeV/m per
            # doubling of the reach out to 170 m (55x the 1D wake at 40 R phi), with or without the
            # velocity terms. Its value is set by the real length of the upstream straight, not by
            # local physics, so the regular layout (n_formation_length * L_f) is kept there. Inside
            # the bend the same contribution converges as 1/U (talman_cancellation.md 7.7).
            return None
        phi = a / R
        slip = R * phi ** 3 / 6.0
        if slip > ip.reach_slip_max * self.beam._sigma_z:
            return None
        return a + rs * R * phi

    def _drift_causal_nodes(self, s, x, t, s1):
        """
        s' nodes of the causal drift region [edge, s1) below the regular drift layout, or None
        (Integration_params.drift_causal_reach): only in the drift after a bend, from the lowest s'
        whose retarded band still meets the bunch, bounded by s' = 0 and the retained history.
        """
        ip = self.integration_params
        if not (ip.drift_causal_reach and self._drift_reach_start() is not None):
            return None
        tl = self.DF_tracker.time_log
        h0 = max(0.0, float(tl[0]) if len(tl) else 0.0)
        if not h0 < s1 - 1e-9:
            return None
        lo = self._causal_edge(s, x, t, h0, s1, n_scan=ip.drift_causal_scan)
        if lo is None or not lo < s1 - 1e-9:
            return None
        if lo <= h0 + 1e-12 and h0 > 0.0:
            # still alive at the oldest retained snapshot: the history, not causality, ends the region
            self._count('drift_causal_clipped')
        g = ip.drift_causal_grade
        u_lo, u_hi = max(s - s1, 1e-12), s - lo
        k = max(int(np.ceil(np.log(u_hi / u_lo) / np.log1p(g))), 2)
        return np.unique(s - np.minimum(u_lo * (1.0 + g) ** np.arange(k + 1), u_hi))

    def _straight_region_lo(self, s):
        """
        Lower s' bound of the straight-section extension for an observer at s, or None:
        s - _geo_reach(s), but never below the previous bend's exit (the section before the
        entrance is straight only from there; the previous bend's own tail is prev_bend_reach's
        business), never below the retained history unless that is the synthetic pre-history,
        and never below -prehistory.
        """
        g = self._geo_reach(s)
        if g is None:
            return None
        lo = s - g
        lo = max(lo, -self.integration_params.prehistory)
        pe = self.prev_bend_end
        if pe is not None:
            lo = max(lo, float(pe))
        tl = self.DF_tracker.time_log
        if len(tl) and float(tl[0]) > 1e-12:
            lo = max(lo, float(tl[0]))
        return lo

    def _straight_region_nodes(self, s, lo, hi):
        """
        s' nodes for the straight-section extension [lo, hi) below the regular layout
        (reach_straight). Its sources are far from the observer and their retarded z is nearly
        constant, so the column integral is a smooth ~1/u^2 in u = s - s': geometric grading in u
        with ratio 1 + near_grade, ~ln(u_lo/u_hi)/near_grade nodes (41 for 10 m behind a 0.5 m
        bend at 5%), where the fixed far cell would need thousands.
        """
        g = self.integration_params.near_grade or 0.05
        u_hi, u_lo = max(s - hi, 1e-9), s - lo
        k = int(np.ceil(np.log(u_lo / u_hi) / np.log1p(g)))
        u = u_hi * (1.0 + g) ** np.arange(k + 1)
        u = np.minimum(u, u_lo)
        if u[-1] < u_lo:
            u = np.append(u, u_lo)
        return np.unique(s - u)

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
        reach = ip.n_formation_length * self.formation_length
        if (ip.seam_ratio and ip.near_grade
                and not (ip.causal_clip and self._exit_transient())):
            # Seam continuity: keep grading the near region until its cell near_grade*u has
            # reached 1/seam_ratio of the far region's flat cell, so the far columns never
            # inherit structure finer than they can carry (Integration_params.seam_ratio).
            # Only where the far layout IS flat (in a bend); past a bend exit the far region
            # is itself graded at near_grade from the causal edge, so the seam is continuous
            # already and extending the near region there only moved the 0.4% node-phase
            # jitter of the golden drift kicks.
            far_cell = ip.n_formation_length * self.formation_length / max(2 * ip.far_zbins, 3)
            d = max(d, far_cell / (ip.near_grade * ip.seam_ratio))
        # and never reach back further than the retained history supports
        d = min(max(d, 0.0), reach)

        s4 = s + 3.0 * sigma_z
        # prehistory: the lattice start is not a wall. Sources on the synthetic straight section
        # s' in [-prehistory, 0) are the first snapshot translated along the initial tangent
        # (Integration_params.prehistory); 0 keeps the old clamp at s' = 0.
        lo = -ip.prehistory
        s3 = max(lo, s - d)
        # The far edge is the causal reach measured from s3, and it is clamped at 0
        # because s' < 0 is before the lattice exists -- interpolate1D returns 0 there
        # silently, so nodes below 0 are not merely wasted but invisible. The old
        # s1 = s2 - n_fl*L_f with s2 = s3 - 200 sigma_z reached back 200 sigma_z FURTHER
        # than the retained history supports, and inverted (s1 > s2) whenever s2 went
        # negative while s1 clamped.
        s1 = max(lo, s3 - reach)
        # In a drift after a bend the source of the exit wake is the bend's own arc,
        # which recedes behind the observer; a fixed n_fl*L_f reach loses it ~1.7 R*phi
        # past the exit and the wake then changes sign (progress.md, 2026-11-03).
        # drift_reach extends the far edge back to the bend's entrance; the causal-edge
        # scan in _far_region_nodes then trims it to where the source still contributes.
        s_bend = self._drift_reach_start()
        if s_bend is not None:
            if ip.drift_causal_reach:
                # the regular exit-transient layout always has the n_formation_length = 1 reach in a drift,
                # so its fine window stays on the last bend's arc whatever n_fl is (at n_fl = 6 it slid
                # into D31 and B1RE's exit wake came out 65% wrong); earlier sources go to the causal
                # drift region (Integration_params.drift_causal_reach)
                s1 = max(lo, s3 - self.formation_length)
            s1 = min(s1, s_bend)
        return ((s1, s3), (s3, s4))

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

    def _bend_exit_rec(self):
        sb = self.bend_start_rec
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
        pb = self.prev_bend_start
        return None if pb is None else max(0.0, float(pb))

    def _drift_reach_start(self):
        """Far edge for the exit transient (the last bend's entrance), or None."""
        if not (self.integration_params.drift_reach and self._exit_transient()):
            return None
        sb = self.bend_start_rec
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
        if pc is None:
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
        if want > cap and not graded:
            self._note('near_cap', f'WARNING: near-region nodes capped at {cap} (wanted {want}); near-region '
                                   f'cell is {want/cap:.1f}x coarser than near_cell requests. Raise zbins, '
                                   f'raise near_cell, or use graded nodes.', every_rank=True)
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
        self._retarded_xi_bands(s, x, t, sp, n_iter=self.integration_params.near_adapt_scan_iter)
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
        if base.size < 3:
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
        if sp.size >= ip.near_adapt_max_scan:
            self._note('adapt_scan', f'  [near_adapt] WARNING at s = {t:.4f} m: the column scan hit '
                                     f'near_adapt_max_scan = {ip.near_adapt_max_scan} points before resolving '
                                     f'{delta*1e6:.3g} um of retarded z per interval.', every_rank=True)
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
            self._note('adapt_nodes', f'  [near_adapt] WARNING at s = {t:.4f} m: {n} near-region nodes wanted, '
                                      f'capped at near_adapt_max_nodes = {ip.near_adapt_max_nodes}.', every_rank=True)
            n = ip.near_adapt_max_nodes
        nodes = np.interp(np.linspace(0.0, F[-1], max(n, base.size)), F, sp)
        nodes = np.unique(np.concatenate([nodes, [a, s, b]]))
        nodes = nodes[(nodes >= a) & (nodes <= b)]
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

    def _integrate_near_patch(self, s, x, t, R1, R2):
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

        gz, gx = self.get_CSR_integrand(s=s, t=t, x=x, xp=xp_mesh, sp=sp_mesh, taper=(R1, R2, True))
        jac = r[:, None] * dr * dphi
        dE = -self.CSR_scaling * np.sum(gz * jac)
        xk = self.CSR_scaling * np.sum(gx * jac)
        return dE, xk, xp_mesh, sp_mesh, gz, gx

    def get_CSR_wake(self, s, x, debug=False):
        """
        (dE/dct, x kick) at the observer (s, x): the band-localised integral over every s'
        region, plus the polar patch around the 1/|r - r'| pole (methods §5.1). debug=True
        returns the regions' node meshes and integrands instead.
        """
        t = self.beam.position
        ip = self.integration_params
        self._ibp_ship_acc = 0.0
        self._ibp_gross_acc = 0.0
        self._ibp_zone_acc = 0.0
        bnds = self._layout_bounds(s, x)
        (s1, s3), (_, s4) = bnds
        self._region_node_counts(bnds)      # sets _last_region_nodes for diagnostics
        sps = [self._far_region_nodes(s, x, t, s1, s3),
               self._near_region_nodes(s, s3, s4, x=x, t=t)]
        # only the near region straddles s' = s, so only it owns the pole and gets the patch
        radii = self._near_patch_radii(s, s3, s4)
        tapers = [None, None if radii is None else (*radii, False)]
        # band_jacobian_min widens the far region's oblique columns only (methods §5.3)
        parts = [self._integrate_xi_region(s, x, t, spk, tp, scale_bands=(k == 0),
                                           adapt_x=(k == 1) or (k == 0 and ip.far_adapt_x))
                 for k, (spk, tp) in enumerate(zip(sps, tapers))]
        for sp_extra in self._extra_region_nodes(s, x, t, s1):
            parts.append(self._integrate_xi_region(s, x, t, sp_extra, None, scale_bands=True,
                                                   adapt_x=ip.far_adapt_x))
        if radii is not None:
            parts.append(self._integrate_near_patch(s, x, t, *radii))

        if debug:
            return {'mode': 'xi_bands',
                    'near_patch_radii': radii,
                    'sp': list(sps),
                    'xp_mesh': [p[2] for p in parts],
                    'sp_mesh': [p[3] for p in parts],
                    'integrand_z': [p[4] for p in parts],
                    'integrand_x': [p[5] for p in parts]}
        if ip.ibp:
            # the shipped form's total for the consistency check: the regions' shipped sums plus
            # the patch, which both forms share
            self._last_wake_shipped = self._ibp_ship_acc + (parts[-1][0] if radii is not None else 0.0)
        return sum(p[0] for p in parts), sum(p[1] for p in parts)

    def _extra_region_nodes(self, s, x, t, s1):
        """
        s' nodes of the regions below the regular far edge s1, in integration order:
          prev_bend_reach     inside a bend, the previous bend's exit tail, from its entrance
          reach_straight      inside a bend, the straight section before this bend's entrance
          drift_causal_reach  in the drift after a bend, the earlier sources from the causal edge
        """
        out = []
        pb = self._prev_bend_reach_start()
        if pb is not None and pb < s1:
            out.append(self._prev_region_nodes(s, pb, s1))
        lo_pre = self._straight_region_lo(s)
        if lo_pre is not None:
            lo_cur = min(s1, pb) if pb is not None else s1
            lo_cur = min(lo_cur, float(self.bend_start_rec))      # never the arc itself
            if lo_pre < lo_cur - 1e-12:
                out.append(self._straight_region_nodes(s, lo_pre, lo_cur))
        sp_drift = self._drift_causal_nodes(s, x, t, s1)
        if sp_drift is not None:
            out.append(sp_drift)
        return out
          
          
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
        if n <= 0:
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
        # The warning is judged against the mesh's largest wake, not against the checked points
        # themselves: the centre-row points can sit where W is ~0 (the gain beam's first bend,
        # W 0.002 MeV/m against a mesh max of 0.02), and there a 4% relative difference was
        # 1e-4 MeV/m, which the beam never sees. d (relative to the points) stays in the log.
        wmax = float(np.abs(self.dE_dct).max()) if getattr(self, 'dE_dct', None) is not None else 0.0
        d_mesh = float(np.abs(a - b).max() / max(wmax, 1e-300)) if len(a) else float('nan')
        if not hasattr(self, 'ibp_check_log'):
            self.ibp_check_log = []
        self.ibp_check_log.append((float(self.beam.position), d, len(a), skipped, max(fz), d_mesh))
        if len(a) and d_mesh > ip.ibp_check_tol:
            self._note('ibp_check', f'  [CSR2D] warning: at s = {self.beam.position:.4f} the shipped and the '
                  f'integrated-by-parts longitudinal wakes differ by {d_mesh:.1e} of the mesh\'s '
                  f'largest |W| ({d:.1e} relative to the {len(a)} checked points). They are the '
                  f'same integral: rho and d rho/dx are inconsistent somewhere (grid edge, '
                  f'blending). Further kicks are logged in ibp_check_log, not printed.')

    def _assert_history(self):
        """
        Refuse to evaluate a wake without a history to interpolate.

        `compute_CSR: 0` gates get_DF/append_DF, so the run produces a beam with NO density
        history. Anything calling get_CSR_integrand afterwards then got near-zeros with no error
        raised -- it cost a full set of wrong figures in §6f, and bit again while benchmarking
        the B-spline hoist in Step 10, where debug=False gave a one-snapshot history and
        data_rho[k+1] read out of bounds. The name suggests a pure output switch and it is not.
        """
        n = len(self.DF_tracker.time_log)
        if n < 2:
            raise RuntimeError(
                f'the density history holds {n} snapshot(s); the wake needs at least 2 to '
                f'interpolate between. Set CSR_computation.compute_CSR = 1, or pass '
                f'debug=True to run(), so that snapshots are recorded.')

    def get_CSR_integrand(self, s, x, t, sp, xp, taper=None, extras=None):
        """
        Longitudinal and transverse integrands at the source points (xp, sp) for the observer
        (s, x) at time t (thesis Eqs 4.4 and 4.8; the arithmetic is integrand_kernel).

        taper: (R1, R2, keep_inner) -- multiply by a radial partition of |r - r'|: with keep_inner
        False the weight rises 0 -> 1 across [R1, R2] (removing the pole's neighbourhood), with True
        it is the complement; the two pieces sum to the untapered integral exactly.
        extras: a dict to fill with what the integrated-by-parts form needs (CSR_integration.ibp);
        the fields then come from interpolate3D_comoving_ibp, bit-identical for the five fields.
        """
        self._assert_history()
        ip = self.integration_params
        sp_flat = sp.ravel()
        xp_flat = xp.ravel()

        vx = self._comoving_fields(np.array([x]), np.array([s - t]), np.array([t]))[3][0]
        if self._quantile_frame():
            vx = self._observer_vx(x, s - t, vx)

        (X0_s, Y0_s, n_vec_s_x, n_vec_s_y,
         tau_vec_s_x, tau_vec_s_y) = (v[0] for v in self._lattice_vectors(np.array([s])))
        (X0_sp, Y0_sp, n_vec_sp_x, n_vec_sp_y,
         tau_vec_sp_x, tau_vec_sp_y) = self._lattice_vectors(sp_flat)
        r_minus_rp_x = X0_s - X0_sp + x * n_vec_s_x - xp_flat * n_vec_sp_x
        r_minus_rp_y = Y0_s - Y0_sp + x * n_vec_s_y - xp_flat * n_vec_sp_y
        r_minus_rp = np.sqrt(r_minus_rp_x**2 + r_minus_rp_y**2)

        # curvature of the element each s' falls in (0 past the lattice end, and on the synthetic
        # straight pre-history before s' = 0 even when the lattice starts with a dipole)
        dist = np.asarray(self.lattice.distance)
        rho_el = np.asarray(self.lattice.rho)
        k_el = np.searchsorted(dist, sp_flat, side='right')
        rho_sp = np.where(k_el < dist.size, rho_el[np.minimum(k_el, dist.size - 1)], 0.0)
        if sp_flat.min() < 0.0:
            rho_sp = np.where(sp_flat < 0.0, 0.0, rho_sp)

        t_ret = t - r_minus_rp
        z_ret = sp_flat - t_ret
        if extras is None or not ip.ibp:
            # the five fields share the frame construction and stencil, so one pass returns all
            (density_ret, density_x_ret, density_z_ret,
             vx_ret, vx_x_ret) = self._comoving_fields(xp_flat, z_ret, t_ret)
            A_ibp = B_ibp = None
        else:
            (density_ret, density_x_ret, density_z_ret,
             vx_ret, vx_x_ret, A_ibp, B_ibp) = self._comoving_fields_ibp(xp_flat, z_ret, t_ret)

        n = xp_flat.size
        CSR_integrand_z = np.empty(n)
        CSR_integrand_x = np.empty(n)
        vn = np.empty(n)
        as_arr = lambda v: (np.broadcast_to(np.asarray(v, dtype=float), (n,)) if np.ndim(v) == 0
                            else np.ascontiguousarray(v, dtype=float))
        R1, R2, keep_inner = taper if taper is not None else (0.0, 1.0, False)
        csr_integrand_kernel(np.ascontiguousarray(xp_flat, dtype=float), as_arr(rho_sp), as_arr(r_minus_rp_x),
                             as_arr(r_minus_rp_y), as_arr(r_minus_rp), as_arr(n_vec_sp_x), as_arr(n_vec_sp_y),
                             as_arr(tau_vec_sp_x), as_arr(tau_vec_sp_y),
                             as_arr(density_ret), as_arr(density_x_ret), as_arr(density_z_ret),
                             as_arr(vx_ret), as_arr(vx_x_ret),
                             float(vx), float(n_vec_s_x), float(n_vec_s_y), float(tau_vec_s_x), float(tau_vec_s_y),
                             ip.transverse_terms == 'W1', taper is not None,
                             float(R1), float(R2), bool(keep_inner), CSR_integrand_z, CSR_integrand_x, vn)
        if extras is not None:
            extras.update(rx=r_minus_rp_x, ry=r_minus_rp_y, nx=n_vec_sp_x, ny=n_vec_sp_y,
                          xp=xp_flat, sp=sp_flat, kappa=rho_sp, vn=vn,
                          rho=density_ret, rho_x=density_x_ret, A=A_ibp, B=B_ibp)
        return CSR_integrand_z.reshape(xp.shape), CSR_integrand_x.reshape(xp.shape)

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



















