"""Manual lookup: trace exactly what interpolate3D_transformed does at the query point."""
import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..', 'example'))

from pyDFCSR_2D.CSR import CSR2D
from pyDFCSR_2D.deposit_smooth import DF_tracker_smooth
from scipy.interpolate import RegularGridInterpolator

dep_config = {'method': 'bspline_fft', 'xbins': 200, 'zbins': 200, 'xlim': 5, 'zlim': 5,
              'smoothing_sigma': 3.0, 'poly_degree': 3, 'velocity_threhold': 1000}

csr = CSR2D(input_file='input/dipole_chirp500_config.yaml')
csr.CSR_params.apply_CSR = 0
csr.DF_tracker = DF_tracker_smooth(dep_config)
csr.use_smooth_deposit = True
csr.DF_tracker.get_DF(x=csr.beam.x, z=csr.beam.z, px=csr.beam.px, t=csr.beam.position)
csr.DF_tracker.append_DF()
csr.DF_tracker.append_interpolant(formation_length=float('inf'),
                                   n_formation_length=csr.integration_params.n_formation_length)
csr.run(stop_time=0.75)

tracker = csr.DF_tracker
x_q = -84.1e-6
z_q = 23.85e-6
t_q = 0.799

t_idx = (t_q - tracker.min_x) / tracker.delta_x
k = int(t_idx)
alpha = t_idx - k
print(f'Query: x={x_q*1e6:.2f}um, z={z_q*1e6:.2f}um, t_ret={t_q}')
print(f'Time: t_idx={t_idx:.4f}, k={k}, alpha={alpha:.4f}')
print(f'Blending: {1-alpha:.4f} * k={k} + {alpha:.4f} * k={k+1}')
print()

for kk in [k, min(k+1, tracker.data_density_interp.shape[0]-1)]:
    pc = tracker.poly_coeffs_interp[kk]
    poly_val = np.polyval(pc, z_q)
    xi = x_q - poly_val
    print(f'Timestep k={kk}:')
    print(f'  poly_coeffs = {pc}')
    print(f'  poly({z_q*1e6:.2f}um) = {poly_val*1e6:.4f}um')
    print(f'  xi = x - poly(z) = {x_q*1e6:.2f} - ({poly_val*1e6:.4f}) = {xi*1e6:.4f}um')

    min_xi = tracker.min_xi_arr[kk]
    delta_xi = tracker.delta_xi_arr[kk]
    min_z = tracker.min_z_arr[kk]
    delta_z = tracker.delta_z_arr[kk]
    xi_idx = (xi - min_xi) / delta_xi
    z_idx = (z_q - min_z) / delta_z
    print(f'  xi_idx={xi_idx:.4f}, z_idx={z_idx:.4f} (valid: 0-199)')

    # Manual scipy lookup
    xi_grids = np.linspace(min_xi, min_xi + delta_xi*199, 200)
    z_grids = np.linspace(min_z, min_z + delta_z*199, 200)

    rho_x_slice = tracker.data_density_x_interp[kk]
    interp_rx = RegularGridInterpolator((xi_grids, z_grids), rho_x_slice, bounds_error=False, fill_value=0)
    val_rx = interp_rx(np.array([[xi, z_q]]))[0]
    print(f'  density_x(xi, z) via scipy = {val_rx:.4e}')

    rho_slice = tracker.data_density_interp[kk]
    interp_r = RegularGridInterpolator((xi_grids, z_grids), rho_slice, bounds_error=False, fill_value=0)
    val_r = interp_r(np.array([[xi, z_q]]))[0]
    print(f'  density(xi, z) via scipy = {val_r:.4e}')

    # Check sign: is xi on positive or negative side of the distribution center?
    rho_col = rho_slice[:, int(z_idx)]  # density along xi at this z
    center_idx = np.argmax(rho_col)
    print(f'  Distribution center in xi: idx={center_idx}, xi_center={xi_grids[center_idx]*1e6:.2f}um')
    print(f'  Query xi={xi*1e6:.2f}um -> {"RIGHT of center" if xi > xi_grids[center_idx] else "LEFT of center"}')
    print(f'  Expected sign of d_rho/d_xi: {"NEGATIVE (decreasing)" if xi > xi_grids[center_idx] else "POSITIVE (increasing)"}')
    print(f'  Actual stored density_x sign at query: {"NEGATIVE" if val_rx < 0 else "POSITIVE"} ({val_rx:.4e})')
    print()

# Now check what interpolate3D_transformed returns
from pyDFCSR_2D.interp3D import interpolate3D_transformed
result = interpolate3D_transformed(
    xval=np.array([x_q]), zval=np.array([z_q]), tval=np.array([t_q]),
    data=tracker.data_density_x_interp, poly_coeffs=tracker.poly_coeffs_interp,
    min_xi_arr=tracker.min_xi_arr, min_z_arr=tracker.min_z_arr, min_t=tracker.min_x,
    delta_xi_arr=tracker.delta_xi_arr, delta_z_arr=tracker.delta_z_arr, delta_t=tracker.delta_x)
print(f'interpolate3D_transformed result for density_x: {result[0]:.4e}')
