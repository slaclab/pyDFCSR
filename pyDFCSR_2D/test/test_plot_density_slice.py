"""Plot density slice at z=98um from both methods, and verify the value at x=-387um."""
import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..', 'example'))

from pyDFCSR_2D.CSR import CSR2D
from pyDFCSR_2D.deposit_smooth import DF_tracker_smooth
from pyDFCSR_2D.interp3D import interpolate3D, interpolate3D_transformed
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

dep_config = {'method': 'bspline_fft', 'xbins': 200, 'zbins': 100,
              'xlim': 10, 'zlim': 5, 'smoothing_sigma': 3.0,
              'poly_degree': 3, 'velocity_threhold': 1000}

# Run both
csr_leg = CSR2D(input_file='input/single_dipole_tilted_config.yaml')
csr_leg.CSR_params.apply_CSR = 0
csr_leg.run(stop_time=0.86)

csr_new = CSR2D(input_file='input/single_dipole_tilted_config.yaml')
csr_new.CSR_params.apply_CSR = 0
csr_new.DF_tracker = DF_tracker_smooth(dep_config)
csr_new.use_smooth_deposit = True
csr_new.DF_tracker.get_DF(x=csr_new.beam.x, z=csr_new.beam.z, px=csr_new.beam.px, t=csr_new.beam.position)
csr_new.DF_tracker.append_DF()
csr_new.DF_tracker.append_interpolant(formation_length=float('inf'),
                                       n_formation_length=csr_new.integration_params.n_formation_length)
csr_new.run(stop_time=0.86)

# Query density along x at z=98um, t=0.499
t_query = 0.499
z_query = 98e-6
x_scan = np.linspace(-3000e-6, 3000e-6, 500)
t_arr = np.full(500, t_query)
z_arr = np.full(500, z_query)

tr_leg = csr_leg.DF_tracker
rho_leg = interpolate3D(t_arr, x_scan, z_arr,
                         tr_leg.data_density_interp, tr_leg.min_x, tr_leg.min_y, tr_leg.min_z,
                         tr_leg.delta_x, tr_leg.delta_y, tr_leg.delta_z)

tr_new = csr_new.DF_tracker
rho_new = interpolate3D_transformed(x_scan, z_arr, t_arr,
                                     tr_new.data_density_interp, tr_new.poly_coeffs_interp,
                                     tr_new.min_xi_arr, tr_new.min_z_arr, tr_new.min_x,
                                     tr_new.delta_xi_arr, tr_new.delta_z_arr, tr_new.delta_x)

# Print key values
print(f"Query: z={z_query*1e6:.0f}um, t={t_query:.4f}")
print(f"Legacy peak: {rho_leg.max():.4e} at x={x_scan[np.argmax(rho_leg)]*1e6:.0f}um")
print(f"New peak:    {rho_new.max():.4e} at x={x_scan[np.argmax(rho_new)]*1e6:.0f}um")
idx_query = np.argmin(np.abs(x_scan - (-387e-6)))
print(f"At x=-387um: legacy={rho_leg[idx_query]:.4e}, new={rho_new[idx_query]:.4e}")

# Plot
fig, axes = plt.subplots(2, 1, figsize=(12, 10))

axes[0].plot(x_scan*1e6, rho_leg, 'b-', lw=2, label='Legacy')
axes[0].plot(x_scan*1e6, rho_new, 'r--', lw=2, label='New (xlim=10, xbins=200)')
axes[0].axvline(-387, color='green', ls=':', lw=1.5, label='query x=-387um')
axes[0].axvline(-784, color='gray', ls='--', lw=1, label='beam center at z=98um')
axes[0].set_xlabel('x (μm)')
axes[0].set_ylabel('ρ')
axes[0].set_title(f'Density slice at z={z_query*1e6:.0f}μm, t={t_query:.4f}')
axes[0].legend()
axes[0].set_xlim([-3000, 3000])

# Log scale
axes[1].semilogy(x_scan*1e6, np.maximum(rho_leg, 1e-10), 'b-', lw=2, label='Legacy')
axes[1].semilogy(x_scan*1e6, np.maximum(rho_new, 1e-10), 'r--', lw=2, label='New')
axes[1].axvline(-387, color='green', ls=':', lw=1.5, label='query x=-387um')
axes[1].axvline(-784, color='gray', ls='--', lw=1, label='beam center')
axes[1].set_xlabel('x (μm)')
axes[1].set_ylabel('ρ (log scale)')
axes[1].set_title('Same, log scale')
axes[1].legend()
axes[1].set_xlim([-3000, 3000])
axes[1].set_ylim([1e-2, 1e7])

plt.tight_layout()
out_dir = '../test/benchmark_results/single_dipole_tilted'
plt.savefig(os.path.join(out_dir, 'density_slice_z98_t0499.png'), dpi=150)
plt.close()
print(f"\nPlot: {out_dir}/density_slice_z98_t0499.png")
