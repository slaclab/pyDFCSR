"""
Parameters of the 2D CSR computation: CSR_integration (Integration_params) and CSR_computation
(CSR_params).

One line per option here. What each option does and why, with the measurements behind the
defaults, is in docs/DFCSR_2D_methods.md (§5, §11) and docs/parameter_notes.md.
"""
from .tools import full_path


def _identity(v):
    return v


def _transverse_terms(v):
    if v not in ('all', 'W1'):
        raise ValueError(f"CSR_integration.transverse_terms must be 'all' or 'W1', got {v!r}")
    return v


# name: (default, conversion). Conversions are applied exactly as before the 2026-11 cleanup.
INTEGRATION_DEFAULTS = {
    # reach and region layout (methods §5.2)
    'n_formation_length': (1, _identity),          # reach inside a bend, in formation lengths
    'near_cell': (0.25, _identity),                # near-region s' cell at the observer [sigma_xi]
    'near_grade': (0.05, _identity),               # geometric growth of the near-region s' cell
    'near_floor': (20.0, _identity),               # near-region length floor [sigma_z]
    'branch_sin_min': (0.05, _identity),           # |sin 2a| saturation of the near-region length
    'seam_ratio': (2.0, float),                    # near/far s' cell continuity at the seam (0: off)
    'far_zbins': (200, _identity),                 # far region: 2*far_zbins uniform columns in a bend
    'zbins': (200, _identity),                     # near-region node count when near_cell = 0
    'causal_clip': (True, _identity),              # exit-transient layout: start at the causal edge
    'far_window': (1.0, _identity),                # ... with a fine window of far_window*L_f
    'far_cell': (10.0, _identity),                 # ... at far_cell*sigma_z
    'far_fine_max': (500, _identity),              # ... capped at this many nodes
    'drift_reach': (True, _identity),              # in a drift, reach back to the last bend's entrance
    'drift_snap_thin': (0.01, _identity),          # drop drift snapshots that interpolate within this
    'drift_cutoff': (0.0, _identity),              # stop computing beyond this many |R*phi| past a bend (0: never)
    'prev_bend_reach': (True, _identity),          # inside a bend, the previous bend's exit tail
    'prev_bend_grade': (0.02, _identity),          # ... graded in u at this ratio
    'drift_causal_reach': (True, bool),            # in a drift, sources before the last bend from the causal edge
    'drift_causal_grade': (0.005, float),          # ... graded in u at this ratio
    'drift_causal_scan': (128, int),               # ... causal-edge scan samples
    'prehistory': (0.0, float),                    # synthetic straight history before the lattice start [m]
    'reach_straight': (40.0, float),               # straight section before a bend, in units of R*phi (0: off)
    'reach_slip_max': (5.0, float),                # ... while its retardation is below this [sigma_z]
    # transverse bands (methods §5.3)
    'xi_band_margin': (2.0, _identity),            # band half-width factor
    'xi_band_margin_degenerate': (3.0, float),     # ... where the two branches merge
    'tangent_bands': (True, _identity),            # band at the closest approach when the cone misses the axis
    'band_jacobian_min': (0.2, _identity),         # widen oblique far bands by 1/max(|J|, this)
    # x' and near-region s' nodes (methods §5.4)
    'xbins': (200, _identity),                     # x' nodes per band before refinement
    'near_adapt': (False, bool),                   # adaptive near-region s' nodes
    'near_adapt_eps': (0.02, float),               # ... density monitor weight
    'near_adapt_scan_dz': (0.25, float),           # ... scan resolution [res_z]
    'near_adapt_max_scan': (4000, int),
    'near_adapt_max_nodes': (2000, int),
    'near_adapt_max_ratio': (1.5, float),
    'near_adapt_scan_iter': (4, int),              # band-solver passes of the scan
    'near_adapt_x': (False, bool),                 # x' refinement in the near region
    'far_adapt_x': (True, bool),                   # x' refinement in every other region (needs the two above)
    'x_adapt_tol': (1e-3, float),                  # x' refinement: error left below tol of the region's gross sum
    'x_adapt_rounds': (4, int),                    # ... at most this many halvings
    # near-field polar patch (methods §5.5)
    'near_patch': (5.0, _identity),                # patch radius [sigma_xi]
    'near_patch_nr': (100, _identity),
    'near_patch_nphi': (180, _identity),
    # integration by parts (methods §5.7)
    'ibp': (False, bool),                          # by-parts longitudinal main term (robustness option)
    'ibp_j0': (0.25, float),
    'ibp_check': (0, int),                         # compare both forms at this many mesh points per kick
    'ibp_check_xbins': (400, int),
    'ibp_check_tol': (1e-3, float),
    'ibp_check_zone_max': (1.0, float),
    # physics switches
    'transverse_terms': ('all', _transverse_terms),  # 'W1': Coulomb term only (diagnostic)
}

# Options removed in the 2026-11 cleanup. An old config may still name them with the value that is
# now built in (None: any value, the option tuned a removed code path); anything else is an error.
REMOVED_INTEGRATION = {
    'xi_bands': True,              # the pre-band layout is gone
    'integrand_kernel': 'numba',   # the numpy integrand is a test reference now
    'x_adapt': 'error',            # the density-jump x' refinement is gone
    'threads': 1,                  # the thread-parallel kernel twins are gone
    'threads_min_points': None,
    'far_adapt': False,            # density-adaptive far columns are gone
    'far_adapt_base': None,
    'near_adapt_x_rounds': None,   # parameters of the density-jump refinement
    'near_adapt_x_max': None,
}


def _check_removed(section, removed, given):
    for k, allowed in removed.items():
        if k in given and allowed is not None and given[k] != allowed:
            raise ValueError(f'{section}.{k} = {given[k]!r} is no longer supported (removed in the '
                             f'2026-11 cleanup; only {allowed!r} is built in). See docs/DFCSR_2D_methods.md.')


class _Params:
    """Attribute container filled from DEFAULTS; unknown keys raise like a bad keyword argument."""
    DEFAULTS = {}
    REMOVED = {}
    SECTION = ''

    def __init__(self, input_dic=None):
        given = dict(input_dic or {})
        _check_removed(self.SECTION, self.REMOVED, given)
        unknown = [k for k in given if k not in self.DEFAULTS and k not in self.REMOVED]
        if unknown:
            raise TypeError(f'{self.SECTION}: unknown parameter(s) {unknown}')
        for k, (default, conv) in self.DEFAULTS.items():
            setattr(self, k, conv(given[k]) if k in given else default)
        self._validate()

    def _validate(self):
        pass

    def __setstate__(self, state):
        # pickles written before a parameter existed get its current default
        for k, (default, _) in self.DEFAULTS.items():
            setattr(self, k, default)
        self.__dict__.update({k: v for k, v in state.items() if k not in self.REMOVED})


class Integration_params(_Params):
    DEFAULTS = INTEGRATION_DEFAULTS
    REMOVED = REMOVED_INTEGRATION
    SECTION = 'CSR_integration'


def _opt_float(v):
    return None if v is None else float(v)


CSR_DEFAULTS = {
    'compute_CSR': (1, _identity),
    'apply_CSR': (1, _identity),
    'transverse_on': (1, _identity),
    'xbins': (20, _identity),                      # wake-mesh rows
    'zbins': (30, _identity),                      # wake-mesh columns
    'xlim': (5, _identity),                        # uniform mesh half-width [sigma]
    'zlim': (5, _identity),
    'mesh_mode': ('uniform', _identity),           # 'uniform' or 'charge' (nodes follow the projected charge)
    'mesh_uniform_frac': (0.3, float),             # charge mesh: share of nodes spread uniformly
    'mesh_max_ratio': (1.5, float),                # charge mesh: neighbouring spacing ratio cap
    'mesh_smooth': (3.0, float),                   # charge mesh: monitor smoothing [deposition resolutions]
    'mesh_pad': (0.02, float),                     # charge mesh: axis padding [span]
    'mesh_x_quantile': (None, _opt_float),         # charge mesh x axis on the [q, 1-q] span
    'mesh_z_quantile': (None, _opt_float),         # charge mesh z axis on the [q, 1-q] span
    'pedestal_warn': (0.05, float),                # windowed frames: warn above this dropped-charge share
    'write_beam': (None, _identity),               # None, 'all' or a list of step numbers
    'write_wakes': (True, _identity),
    'write_name': ('', _identity),
    'workdir': ('.', full_path),
    'backend': ('numba', str),                     # wake evaluation: 'numba' or 'torch' (torch_backend)
    'device': ('cpu', str),                        # torch device, e.g. 'cuda'
    'torch_chunk': (1 << 21, int),                 # integrand nodes per torch batch
}



class CSR_params(_Params):
    DEFAULTS = CSR_DEFAULTS
    SECTION = 'CSR_computation'

    def __init__(self, input_dic=None):
        given = dict(input_dic or {})
        super().__init__(given)
        if 'workdir' not in given:
            self.workdir = full_path('.')

    def _validate(self):
        if self.mesh_mode not in ('uniform', 'charge'):
            raise ValueError(f"CSR_computation.mesh_mode must be 'uniform' or 'charge', got {self.mesh_mode!r}")
        if not 0.0 <= self.mesh_uniform_frac <= 1.0:
            raise ValueError(f'CSR_computation.mesh_uniform_frac must be in [0, 1], got {self.mesh_uniform_frac}')
        if not self.mesh_max_ratio > 1.0:
            raise ValueError(f'CSR_computation.mesh_max_ratio must be > 1, got {self.mesh_max_ratio}')
        if self.backend not in ('numba', 'torch'):
            raise ValueError(f"CSR_computation.backend must be 'numba' or 'torch', got {self.backend!r}")
        for k in ('mesh_x_quantile', 'mesh_z_quantile'):
            q = getattr(self, k)
            if q is not None and not 0.0 < q < 0.5:
                raise ValueError(f'CSR_computation.{k} must be in (0, 0.5), got {q}')
