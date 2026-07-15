"""Check if density is actually zero or nonzero at the Region 1 query point."""
import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..', 'example'))

from pyDFCSR_2D.CSR import CSR2D
from pyDFCSR_2D.deposit_smooth import DF_tracker_smooth

dep_config = {'method': 'bspline_fft', 'xbins': 100, 'zbins': 100,
              'xlim': 5, 'zlim': 5, 'smoothing_sigma': 3.0,
              'poly_degree': 3, 'velocity_threhold': 1000}

csr = CSR2D(input_file='input/single_dipole_tilted_config.yaml')
csr.CSR_params.apply_CSR = 0
csr.DF_tracker = DF_tracker_smooth(dep_config)
csr.use_smooth_deposit = True
csr.DF_tracker.get_DF(x=csr.beam.x, z=csr.beam.z, px=csr.beam.px, t=csr.beam.position)
csr.DF_tracker.append_DF()
csr.DF_tracker.append_interpolant(formation_length=float('inf'),
                                   n_formation_length=csr.integration_params.n_formation_length)
csr.run(stop_time=0.41)

print(f'Beam at t={csr.beam.position:.4f}:')
print(f'  mean_x = {np.mean(csr.beam.x)*1e6:.1f} um')
print(f'  mean_z = {np.mean(csr.beam.z)*1e6:.1f} um')
print(f'  sigma_x = {np.std(csr.beam.x)*1e6:.1f} um')
print(f'  sigma_z = {np.std(csr.beam.z)*1e6:.1f} um')
print(f'  slope = {np.polyfit(csr.beam.z, csr.beam.x, 1)[0]:.4f}')
print(f'  x range: [{csr.beam.x.min()*1e6:.0f}, {csr.beam.x.max()*1e6:.0f}] um')
print(f'  z range: [{csr.beam.z.min()*1e6:.0f}, {csr.beam.z.max()*1e6:.0f}] um')

# The query point: xp=714um, z_ret=150um, t_ret=0.3999
xp_query = 714e-6
z_query = 150e-6
t_query = 0.3999

tr = csr.DF_tracker
t_idx = (t_query - tr.min_x) / tr.delta_x
k = int(t_idx)
n_t = tr.data_density_interp.shape[0]
print(f'\nQuery: xp={xp_query*1e6:.0f}um, z={z_query*1e6:.0f}um, t_ret={t_query:.4f}')
print(f'  t_idx={t_idx:.3f}, k={k}, n_t={n_t}')
print(f'  Time grid: min={tr.min_x:.4f}, max={tr.max_x:.4f}, delta={tr.delta_x:.4f}')

# Check bounds at bracketing timesteps
for kk in range(max(0, k), min(k+2, n_t)):
    poly_k = tr.poly_coeffs_interp[kk]
    xi_k = xp_query - np.polyval(poly_k, z_query)
    min_xi = tr.min_xi_arr[kk]
    max_xi = min_xi + tr.delta_xi_arr[kk] * (tr.data_density_interp.shape[1] - 1)
    min_z = tr.min_z_arr[kk]
    max_z = min_z + tr.delta_z_arr[kk] * (tr.data_density_interp.shape[2] - 1)
    slope_k = np.polyval(np.polyder(poly_k), 0)

    print(f'\n  Timestep kk={kk} (t={tr.min_x + kk*tr.delta_x:.4f}, slope={slope_k:.2f}):')
    print(f'    poly(z=150um) = {np.polyval(poly_k, z_query)*1e6:.1f} um')
    print(f'    xi = {xp_query*1e6:.0f} - {np.polyval(poly_k, z_query)*1e6:.1f} = {xi_k*1e6:.1f} um')
    print(f'    xi grid: [{min_xi*1e6:.1f}, {max_xi*1e6:.1f}] um')
    print(f'    z grid: [{min_z*1e6:.1f}, {max_z*1e6:.1f}] um')
    print(f'    xi IN BOUNDS: {min_xi <= xi_k <= max_xi}')
    print(f'    z IN BOUNDS: {min_z <= z_query <= max_z}')

# Is xp=714um physically within the beam at this time?
x_p = csr.beam.x
z_p = csr.beam.z
near = np.sum((np.abs(x_p - xp_query) < 300e-6) & (np.abs(z_p - z_query) < 200e-6))
print(f'\n  Particles near (xp=714, z=150)um (|dx|<300, |dz|<200): {near}/{len(x_p)}')

# What is the actual density from the deposited grid at this (xi, z)?
if k < n_t:
    poly_k = tr.poly_coeffs_interp[k]
    xi_k = xp_query - np.polyval(poly_k, z_query)
    xi_idx = (xi_k - tr.min_xi_arr[k]) / tr.delta_xi_arr[k]
    z_idx = (z_query - tr.min_z_arr[k]) / tr.delta_z_arr[k]
    print(f'\n  Grid indices at k={k}: xi_idx={xi_idx:.2f}, z_idx={z_idx:.2f} (valid: 0 to 99)')
