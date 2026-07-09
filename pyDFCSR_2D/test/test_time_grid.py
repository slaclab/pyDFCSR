"""Check time grid and formation length for new method."""
import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..', 'example'))

from pyDFCSR_2D.CSR import CSR2D
from pyDFCSR_2D.deposit_smooth import DF_tracker_smooth

dep_config = {
    'method': 'bspline_fft',
    'xbins': 200, 'zbins': 200,
    'xlim': 5, 'zlim': 5,
    'smoothing_sigma': 3.0,
    'poly_degree': 3,
    'velocity_threhold': 1000,
}

csr = CSR2D(input_file='input/dipole_chirp500_config.yaml')
csr.DF_tracker = DF_tracker_smooth(dep_config)
csr.use_smooth_deposit = True
csr.DF_tracker.get_DF(x=csr.beam.x, z=csr.beam.z, px=csr.beam.px, t=csr.beam.position)
csr.DF_tracker.append_DF()
csr.DF_tracker.append_interpolant(formation_length=float('inf'),
                                   n_formation_length=csr.integration_params.n_formation_length)
csr.run(stop_time=0.75)

tracker = csr.DF_tracker
print(f'Beam position = {csr.beam.position:.4f}')
print(f'Formation length = {csr.formation_length:.6f}')
print(f'n_formation_length = {csr.integration_params.n_formation_length}')
print(f'\nNew method time grid:')
print(f'  min_t = {tracker.min_x:.6f}')
print(f'  max_t = {tracker.max_x:.6f}')
print(f'  delta_t = {tracker.delta_x:.6f}')
n_t = tracker.data_density_interp.shape[0]
print(f'  n_t = {n_t}')
times = [tracker.min_x + i*tracker.delta_x for i in range(n_t)]
print(f'  times: {[round(t,4) for t in times]}')
print(f'\nPer-timestep grids:')
for k in range(n_t):
    min_xi = tracker.min_xi_arr[k]
    max_xi = min_xi + tracker.delta_xi_arr[k] * 199
    min_z = tracker.min_z_arr[k]
    max_z = min_z + tracker.delta_z_arr[k] * 199
    pc = tracker.poly_coeffs_interp[k]
    slope = np.polyval(np.polyder(pc), 0)
    print(f'  k={k}: t={times[k]:.4f}, xi=[{min_xi*1e6:.0f},{max_xi*1e6:.0f}]um, z=[{min_z*1e6:.0f},{max_z*1e6:.0f}]um, slope(0)={slope:.2f}')

# Now check: at the retarded-time query point from earlier diagnostic
# x'=-85.77um, z_ret=22.5um, t_ret=0.799
t_ret_test = 0.799
x_test = -85.77e-6
z_test = 22.5e-6
print(f'\nTest query: x={x_test*1e6:.2f}um, z={z_test*1e6:.2f}um, t_ret={t_ret_test:.4f}')
t_idx = (t_ret_test - tracker.min_x) / tracker.delta_x
k = int(t_idx)
alpha = t_idx - k
print(f'  t_idx = {t_idx:.4f}, k={k}, alpha={alpha:.4f}')
print(f'  Blending between t={times[k]:.4f} (k={k}) and t={times[min(k+1,n_t-1)]:.4f} (k+1={k+1})')

# What xi does this map to in each bracketing timestep?
pc_k = tracker.poly_coeffs_interp[k]
pc_k1 = tracker.poly_coeffs_interp[min(k+1, n_t-1)]
xi_k = x_test - np.polyval(pc_k, z_test)
xi_k1 = x_test - np.polyval(pc_k1, z_test)
print(f'  At timestep k={k}: poly(z)={np.polyval(pc_k, z_test)*1e6:.2f}um, xi={xi_k*1e6:.2f}um')
print(f'  At timestep k+1={k+1}: poly(z)={np.polyval(pc_k1, z_test)*1e6:.2f}um, xi={xi_k1*1e6:.2f}um')
print(f'  Grid k: xi range [{tracker.min_xi_arr[k]*1e6:.0f}, {(tracker.min_xi_arr[k]+tracker.delta_xi_arr[k]*199)*1e6:.0f}]um')
if k+1 < n_t:
    print(f'  Grid k+1: xi range [{tracker.min_xi_arr[k+1]*1e6:.0f}, {(tracker.min_xi_arr[k+1]+tracker.delta_xi_arr[k+1]*199)*1e6:.0f}]um')
