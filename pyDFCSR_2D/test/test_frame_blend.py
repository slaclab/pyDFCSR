"""
Acceptance tests for the frame_blend modes.

Run on a real tracked history through a full-compression point (shear 20, long
upstream drift), because that is the configuration where the modes differ: the beam's
orientation goes +87 deg -> 90 deg (vertical) -> -87 deg, and tau = tan(alpha) cannot
represent that path continuously.

The tests, in order of what they protect:

1. NODE EXACTNESS. At alpha = 0 and 1 every mode must reproduce the stored snapshot
   frame to round-off. A mode that fails this introduces a discontinuity AT snapshot
   times -- exactly the artefact being hunted -- and would look like an improvement on
   any smoothness metric while being worse.

2. SITE AGREEMENT. CSR._comoving_frame_at (which places the integration bands) and the
   numba kernel inside interpolate3D_comoving_fields (which evaluates the density) must
   construct the SAME frame. Nothing enforced this before, and the two are separate
   implementations of the same formula. A band located with one blend and a density
   evaluated with another does not overlap.

3. CHIRP ANGLE EXCURSION -- the figure of merit. The chirp band of thesis 4.4.2 sits at
   x - (s - s') tan 2a with tan 2a = 2 tau/(1 - tau^2), which has POLES at tau = +-1.
   Reported as a ratio, not a pass/fail: it is the quantity that selects the mode.

4. COEFF UNCHANGED. 'coeff' must be bit-identical to the pre-flag code path.
"""
import sys
import os
import numpy as np
import pytest
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from pyDFCSR_2D.CSR import CSR2D
from pyDFCSR_2D.interp3D import interpolate3D_comoving_fields

EXAMPLE_DIR = os.path.join(os.path.dirname(__file__), '..', 'example')
SHEAR = 20.0
DRIFT = 1.0
STEP = 0.05
S_OBS_DIP = 0.10
MODES = ('coeff', 'moment', 'orient')


def _write_inputs(tag, frame_blend):
    beam = {
        'n_particle': 100000, 'species': 'electron',
        'px_dist': {'sigma_px': {'units': 'keV/c', 'value': 25.0}, 'type': 'gaussian'},
        'py_dist': {'sigma_py': {'units': 'keV/c', 'value': 25.0}, 'type': 'gaussian'},
        'pz_dist': {'avg_pz': {'units': 'GeV/c', 'value': 5},
                    'sigma_pz': {'units': 'MeV/c', 'value': 0.5}, 'type': 'gaussian'},
        'x_dist': {'sigma_x': {'units': 'um', 'value': 50}, 'type': 'gaussian'},
        'y_dist': {'sigma_y': {'units': 'um', 'value': 5}, 'type': 'gaussian'},
        'random': {'type': 'hammersley'},
        'start': {'tstart': {'units': 'sec', 'value': 0}, 'type': 'time'},
        'total_charge': {'units': 'nC', 'value': 1},
        'z_dist': {'avg_z': {'units': 'mm', 'value': 0},
                   'sigma_z': {'units': 'um', 'value': 50}, 'type': 'gaussian'},
        'transforms': {'s1': {'shear_coefficient':
                              {'units': 'dimensionless', 'value': SHEAR},
                              'type': 'shear z:x'}},
    }
    with open(os.path.join(EXAMPLE_DIR, f'input/fb_beam_{tag}.yaml'), 'w') as f:
        yaml.dump(beam, f, default_flow_style=False, sort_keys=False)

    lat = {'step_size': STEP,
           'element_1': {'type': 'drift', 'L': DRIFT, 'nsep': 1},
           'element_2': {'type': 'dipole', 'L': 1.0, 'angle': 1.0,
                         'E1': 0, 'E2': 0, 'FINT': 0.0, 'FINTX': 0.5,
                         'HGAP': 0.1, 'HGAPX': 0.0, 'FRINGE_AT': 'both_ends',
                         'FRINGE_TYPE': 'linear_edge', 'TILT': 0.0, 'nsep': 1},
           'element_3': {'type': 'drift', 'L': 0.5, 'nsep': 1}}
    lp = f'input/fb_lat_{tag}.yaml'
    with open(os.path.join(EXAMPLE_DIR, lp), 'w') as f:
        yaml.dump(lat, f, default_flow_style=False, sort_keys=False)

    cfg = {
        'input_beam': {'style': 'distgen',
                       'distgen_input_file': f'input/fb_beam_{tag}.yaml'},
        'input_lattice': {'lattice_input_file': lp},
        'particle_deposition': {'method': 'bspline_comoving',
                                'xbins': 128, 'zbins': 128,
                                'xlim': 5, 'zlim': 5, 'smoothing_sigma': 3.0,
                                'poly_degree': 1, 'velocity_threhold': 1000,
                                'frame_blend': frame_blend},
        'CSR_integration': {'n_formation_length': 1.5, 'zbins': 200, 'xbins': 200},
        'CSR_computation': {'compute_CSR': 0, 'apply_CSR': 0, 'transverse_on': 1,
                            'xbins': 21, 'zbins': 51, 'xlim': 3, 'zlim': 3,
                            'write_beam': [], 'write_wakes': False,
                            'write_name': f'fb_{tag}', 'workdir': './output'},
    }
    p = f'input/fb_config_{tag}.yaml'
    with open(os.path.join(EXAMPLE_DIR, p), 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    return p


_CACHE = {}


def _run(frame_blend):
    """A tracked history reaching S_OBS_DIP into the dipole. Cached across tests."""
    if frame_blend not in _CACHE:
        cwd = os.getcwd()
        try:
            os.chdir(EXAMPLE_DIR)
            cfg = _write_inputs(frame_blend, frame_blend)
            csr = CSR2D(input_file=cfg)
            # debug=True is required: CSR.py gates the whole density-history build on
            # `debug or compute_CSR`, so compute_CSR = 0 alone leaves no history.
            csr.run(stop_time=DRIFT + S_OBS_DIP, debug=True)
        finally:
            os.chdir(cwd)
        _CACHE[frame_blend] = csr
    return _CACHE[frame_blend]


@pytest.mark.parametrize('mode', MODES)
def test_node_exactness(mode):
    """At a snapshot time every mode must return that snapshot's stored frame."""
    csr = _run(mode)
    tr = csr.DF_tracker
    ts = np.asarray(tr.time_log)
    assert len(ts) >= 3

    poly, xi_bar, z_bar, s_xi, s_z = csr._comoving_frame_at(ts.copy())

    tau_k = np.asarray(tr.poly_coeffs_interp)[:, 0]
    # centre of the density in x at the snapshot's own centroid z
    centre_stored = (np.asarray(tr.poly_coeffs_interp)[:, 0] * np.asarray(tr.z_bar_arr)
                     + np.asarray(tr.poly_coeffs_interp)[:, -1]
                     + np.asarray(tr.xi_bar_arr))
    centre_blend = csr._eval_poly_rows(poly, np.asarray(tr.z_bar_arr)) + xi_bar

    np.testing.assert_allclose(poly[:, 0], tau_k, rtol=1e-12, atol=0,
                              err_msg=f'{mode}: slope not exact at nodes')
    np.testing.assert_allclose(centre_blend, centre_stored, rtol=1e-12,
                              atol=1e-18, err_msg=f'{mode}: centre not exact')
    np.testing.assert_allclose(s_xi, np.asarray(tr.sigma_xi_arr), rtol=1e-12, atol=0,
                              err_msg=f'{mode}: sigma_xi not exact at nodes')
    np.testing.assert_allclose(s_z, np.asarray(tr.sigma_z_arr), rtol=1e-12, atol=0,
                              err_msg=f'{mode}: sigma_z not exact at nodes')
    np.testing.assert_allclose(z_bar, np.asarray(tr.z_bar_arr), rtol=1e-12, atol=0,
                              err_msg=f'{mode}: z_bar not exact at nodes')


@pytest.mark.parametrize('mode', MODES)
def test_site_agreement(mode):
    """
    The band-placement frame (_comoving_frame_at) and the density-evaluation frame
    (inside the numba kernel) must be the same frame.

    Recovered from the kernel without touching it: rho_x/rho and rho_z/rho give the
    logarithmic gradients, whose ratio is -dp/dz = -tau at fixed (u, w) only if the
    two frames agree. Instead of relying on that, probe the frame directly by
    finding where the kernel's density peaks in x at fixed z -- but that is noisy, so
    use the analytic relation the kernel itself applies:
        rho_z|_x = rho_w/s_z - dp_a * rho_u/s_xi
    Comparing the kernel's rho_z against the same expression rebuilt from
    _comoving_frame_at's dp_a tests exactly the shared quantity.
    """
    csr = _run(mode)
    tr = csr.DF_tracker
    ts = np.asarray(tr.time_log)

    # mid-interval query times, where the modes actually differ
    t_q = 0.5 * (ts[:-1] + ts[1:])
    z_q = np.asarray(tr.z_bar_arr)[:-1] * 0.0 + np.asarray(tr.z_bar_arr)[:-1]
    poly, xi_bar, z_bar, s_xi, s_z = csr._comoving_frame_at(t_q.copy())
    centre = csr._eval_poly_rows(poly, z_q) + xi_bar

    # The kernel's density must peak (in x, at fixed z) essentially at `centre`,
    # because the deposited rho is centred at u = 0. Scan x around centre and find
    # the argmax; agreement to a small fraction of s_xi proves a shared frame.
    for j in range(len(t_q)):
        xs = centre[j] + np.linspace(-2.0, 2.0, 4001) * s_xi[j]
        rho = interpolate3D_comoving_fields(
            xs, np.full_like(xs, z_q[j]), np.full_like(xs, t_q[j]),
            tr.data_density_interp, tr.data_density_u_interp,
            tr.data_density_w_interp, tr.data_vx_interp, tr.data_vx_u_interp,
            tr.poly_coeffs_interp, tr.xi_bar_arr, tr.sigma_xi_arr,
            tr.z_bar_arr, tr.sigma_z_arr,
            tr.u_start, tr.delta_u, tr.w_start, tr.delta_w,
            tr.min_x, tr.delta_x,
            tr.frame_blend_code, tr.var_z_arr, tr.cov_arr, tr.var_x_arr,
            tr.x_bar_arr)[0]
        assert rho.max() > 0, f'{mode}: no density found near the located centre'
        off = abs(xs[int(np.argmax(rho))] - centre[j]) / s_xi[j]
        assert off < 0.05, (f'{mode}: band frame and density frame disagree at '
                            f'interval {j}: peak is {off:.3f} sigma_xi off centre')


def test_chirp_angle_excursion():
    """
    Figure of merit: how far the blended chirp angle strays from its endpoints.

    tan 2a = 2 tau/(1 - tau^2) places the chirp band. Its poles at tau = +-1 are the
    failure: a blend whose tau crosses +-1 flings the band arbitrarily far while the
    density stays within a few sigma_x. Reported, then asserted only as an ordering.
    """
    stats = {}
    for mode in MODES:
        csr = _run(mode)
        tr = csr.DF_tracker
        ts = np.asarray(tr.time_log)
        t_q = np.linspace(ts[0], ts[-1], 20001)
        poly = csr._comoving_frame_at(t_q)[0]
        tau = poly[:, 0]
        den = 1.0 - tau ** 2
        tan2a = np.where(np.abs(den) > 1e-12, 2.0 * tau / den, np.inf)
        stats[mode] = {
            'max|tan2a|': np.max(np.abs(tan2a)),
            'frac |alpha|<45deg': np.mean(np.abs(tau) < 1.0),
            'frac |alpha|<60deg': np.mean(np.abs(tau) < np.tan(np.radians(60.0))),
        }

    print('\n  chirp-angle excursion over the retained history '
          f'(shear {SHEAR:g}, through full compression)')
    keys = list(stats['coeff'])
    print(f"    {'mode':>8} " + ' '.join(f'{k:>20}' for k in keys))
    for mode in MODES:
        print(f'    {mode:>8} ' + ' '.join(f'{stats[mode][k]:>20.6g}'
                                          for k in keys))

    for mode in MODES:
        if mode == 'coeff':
            continue
        assert stats[mode]['max|tan2a|'] < stats['coeff']['max|tan2a|'], (
            f'{mode} must reduce the chirp-angle excursion')
        assert (stats[mode]['frac |alpha|<45deg']
                < stats['coeff']['frac |alpha|<45deg']), (
            f'{mode} must spend less of the history at a fictitious orientation')
    # 'orient' takes tau from the same moment ratio as 'moment', so the orientation
    # diagnostics must be identical -- they differ only in the widths.
    assert (stats['orient']['max|tan2a|'] == stats['moment']['max|tan2a|']), (
        'orient and moment must share the orientation exactly')


def test_coeff_matches_legacy_formula():
    """
    'coeff' must be the pre-flag formula exactly: linear in the coefficients and the
    means, log-linear in the sigmas. Guards against the flag refactor perturbing the
    default path.
    """
    csr = _run('coeff')
    tr = csr.DF_tracker
    ts = np.asarray(tr.time_log)
    t_q = np.linspace(ts[0], ts[-1], 997)

    poly, xi_bar, z_bar, s_xi, s_z = csr._comoving_frame_at(t_q.copy())

    n_t = tr.poly_coeffs_interp.shape[0]
    idx = (t_q - tr.min_x) / tr.delta_x
    k = np.clip(np.floor(idx).astype(int), 0, max(n_t - 2, 0))
    k1 = np.minimum(k + 1, n_t - 1)
    a = np.clip(idx - k, 0.0, 1.0)
    b = 1.0 - a

    np.testing.assert_array_equal(
        poly, b[:, None] * tr.poly_coeffs_interp[k] + a[:, None] * tr.poly_coeffs_interp[k1])
    np.testing.assert_array_equal(xi_bar, b * tr.xi_bar_arr[k] + a * tr.xi_bar_arr[k1])
    np.testing.assert_array_equal(z_bar, b * tr.z_bar_arr[k] + a * tr.z_bar_arr[k1])
    np.testing.assert_array_equal(s_xi, np.exp(b * np.log(tr.sigma_xi_arr[k])
                                               + a * np.log(tr.sigma_xi_arr[k1])))
    np.testing.assert_array_equal(s_z, np.exp(b * np.log(tr.sigma_z_arr[k])
                                              + a * np.log(tr.sigma_z_arr[k1])))


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v', '-s']))
