"""
Check what fraction of retarded-time query points fall out of bounds
at each bracketing timestep in the interpolation.
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..', 'example'))

from pyDFCSR_2D.CSR import CSR2D
from pyDFCSR_2D.deposit_smooth import DF_tracker_smooth
from pyDFCSR_2D.interp1D import interpolate1D

DEP_CONFIG = {
    'method': 'bspline_fft',
    'xbins': 200, 'zbins': 200,
    'xlim': 5, 'zlim': 5,
    'smoothing_sigma': 3.0,
    'poly_degree': 3,
    'velocity_threhold': 1000,
}

# Run new method to target step, then inspect query points
csr = CSR2D(input_file='input/dipole_chirp500_config_fine.yaml')
csr.CSR_params.apply_CSR = 0
csr.DF_tracker = DF_tracker_smooth(DEP_CONFIG)
csr.use_smooth_deposit = True
csr.DF_tracker.get_DF(x=csr.beam.x, z=csr.beam.z, px=csr.beam.px, t=csr.beam.position)
csr.DF_tracker.append_DF()
csr.DF_tracker.append_interpolant(formation_length=float('inf'),
                                   n_formation_length=csr.integration_params.n_formation_length)

# Monkey-patch to capture OOB stats at selected steps
TARGET_STEPS = [6, 10, 14, 18, 22]
oob_stats = {}

original_get_integrand = csr.get_CSR_integrand

def capture_integrand(s, x, t, sp, xp, ignore_vx=False):
    step = csr.beam.step
    if step in TARGET_STEPS:
        sp_flat = sp.ravel()
        xp_flat = xp.ravel()
        tracker = csr.DF_tracker

        # Compute retarded time (same as in get_CSR_integrand)
        X0_s = interpolate1D(xval=np.array([s]), data=csr.lattice.coords[:, 0],
                             min_x=csr.lattice.min_x, delta_x=csr.lattice.delta_x)[0]
        X0_sp = interpolate1D(xval=sp_flat, data=csr.lattice.coords[:, 0],
                              min_x=csr.lattice.min_x, delta_x=csr.lattice.delta_x)
        Y0_s = interpolate1D(xval=np.array([s]), data=csr.lattice.coords[:, 1],
                             min_x=csr.lattice.min_x, delta_x=csr.lattice.delta_x)[0]
        Y0_sp = interpolate1D(xval=sp_flat, data=csr.lattice.coords[:, 1],
                              min_x=csr.lattice.min_x, delta_x=csr.lattice.delta_x)
        n_s_x = interpolate1D(xval=np.array([s]), data=csr.lattice.n_vec[:, 0],
                              min_x=csr.lattice.min_x, delta_x=csr.lattice.delta_x)[0]
        n_sp_x = interpolate1D(xval=sp_flat, data=csr.lattice.n_vec[:, 0],
                               min_x=csr.lattice.min_x, delta_x=csr.lattice.delta_x)
        n_s_y = interpolate1D(xval=np.array([s]), data=csr.lattice.n_vec[:, 1],
                              min_x=csr.lattice.min_x, delta_x=csr.lattice.delta_x)[0]
        n_sp_y = interpolate1D(xval=sp_flat, data=csr.lattice.n_vec[:, 1],
                               min_x=csr.lattice.min_x, delta_x=csr.lattice.delta_x)

        r_x = X0_s - X0_sp + x * n_s_x - xp_flat * n_sp_x
        r_y = Y0_s - Y0_sp + x * n_s_y - xp_flat * n_sp_y
        r_minus_rp = np.sqrt(r_x**2 + r_y**2)
        t_ret = t - r_minus_rp
        z_ret = sp_flat - t_ret

        # Check OOB for each query point against each timestep
        n_t = tracker.data_density_interp.shape[0]
        n_pts = len(xp_flat)

        step_stats = {
            's': s, 'x': x, 't': t, 'n_pts': n_pts,
            'n_t': n_t, 'per_timestep': []
        }

        # Time range
        t_ret_min, t_ret_max = t_ret.min(), t_ret.max()
        step_stats['t_ret_range'] = (t_ret_min, t_ret_max)
        step_stats['z_ret_range'] = (z_ret.min(), z_ret.max())

        for k in range(n_t):
            t_k = tracker.min_x + k * tracker.delta_x
            min_xi = tracker.min_xi_arr[k]
            max_xi = min_xi + tracker.delta_xi_arr[k] * 199
            min_z = tracker.min_z_arr[k]
            max_z = min_z + tracker.delta_z_arr[k] * 199
            poly_k = tracker.poly_coeffs_interp[k]

            # Transform query x -> xi at this timestep
            xi_at_k = xp_flat - np.polyval(poly_k, z_ret)

            # Check bounds
            oob_xi = (xi_at_k < min_xi) | (xi_at_k > max_xi)
            oob_z = (z_ret < min_z) | (z_ret > max_z)
            oob_either = oob_xi | oob_z

            step_stats['per_timestep'].append({
                'k': k, 't_k': t_k,
                'xi_range': (min_xi * 1e6, max_xi * 1e6),
                'z_range': (min_z * 1e6, max_z * 1e6),
                'oob_xi_frac': oob_xi.sum() / n_pts,
                'oob_z_frac': oob_z.sum() / n_pts,
                'oob_total_frac': oob_either.sum() / n_pts,
                'slope_at_0': np.polyval(np.polyder(poly_k), 0),
            })

        oob_stats[step] = step_stats

    return original_get_integrand(s, x, t, sp, xp, ignore_vx)

csr.get_CSR_integrand = capture_integrand
csr.run()

# Print results
print("=" * 80)
print("Out-of-Bounds Analysis for Retarded-Time Queries")
print("=" * 80)

for step in TARGET_STEPS:
    if step not in oob_stats:
        continue
    stats = oob_stats[step]
    print(f"\nStep {step} (s={stats['s']:.3f}m, x={stats['x']*1e6:.1f}um):")
    print(f"  t_ret range: [{stats['t_ret_range'][0]:.4f}, {stats['t_ret_range'][1]:.4f}]")
    print(f"  z_ret range: [{stats['z_ret_range'][0]*1e6:.1f}, {stats['z_ret_range'][1]*1e6:.1f}]um")
    print(f"  n_pts={stats['n_pts']}, n_t={stats['n_t']}")
    print(f"  {'k':<3} | {'t_k':<6} | {'slope':<7} | {'xi range (um)':<20} | {'z range (um)':<20} | {'OOB_xi':<8} | {'OOB_z':<8} | {'OOB_tot':<8}")
    print(f"  {'-'*95}")
    for ts in stats['per_timestep']:
        print(f"  {ts['k']:<3} | {ts['t_k']:<6.3f} | {ts['slope_at_0']:<+7.2f} | [{ts['xi_range'][0]:>8.0f},{ts['xi_range'][1]:>8.0f}] | [{ts['z_range'][0]:>8.0f},{ts['z_range'][1]:>8.0f}] | {ts['oob_xi_frac']:<8.1%} | {ts['oob_z_frac']:<8.1%} | {ts['oob_total_frac']:<8.1%}")
