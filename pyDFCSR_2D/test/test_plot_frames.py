"""Plot beam distribution at t~0.4 in both lab frame and xi-frame."""
import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..', 'example'))

from pyDFCSR_2D.CSR import CSR2D
from pyDFCSR_2D.deposit_smooth import DF_tracker_smooth
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

dep_config = {'method': 'bspline_fft', 'xbins': 100, 'zbins': 100,
              'xlim': 5, 'zlim': 5, 'smoothing_sigma': 3.0,
              'poly_degree': 3, 'velocity_threhold': 1000}

# Run legacy
csr_leg = CSR2D(input_file='input/single_dipole_tilted_config.yaml')
csr_leg.CSR_params.apply_CSR = 0
csr_leg.run(stop_time=0.41)

# Run new
csr_new = CSR2D(input_file='input/single_dipole_tilted_config.yaml')
csr_new.CSR_params.apply_CSR = 0
csr_new.DF_tracker = DF_tracker_smooth(dep_config)
csr_new.use_smooth_deposit = True
csr_new.DF_tracker.get_DF(x=csr_new.beam.x, z=csr_new.beam.z, px=csr_new.beam.px, t=csr_new.beam.position)
csr_new.DF_tracker.append_DF()
csr_new.DF_tracker.append_interpolant(formation_length=float('inf'),
                                       n_formation_length=csr_new.integration_params.n_formation_length)
csr_new.run(stop_time=0.41)

# Get density at the timestep closest to t=0.4
# Legacy: stored on shared (x, z) grid
tr_leg = csr_leg.DF_tracker
# Find timestep for t=0.4
t_target = 0.4
k_leg = round((t_target - tr_leg.min_x) / tr_leg.delta_x)
k_leg = min(k_leg, tr_leg.data_density_interp.shape[0] - 1)
print(f'Legacy: k={k_leg}, t={tr_leg.min_x + k_leg*tr_leg.delta_x:.4f}')
print(f'  Grid: x=[{tr_leg.min_y*1e6:.0f}, {(tr_leg.min_y + tr_leg.delta_y*(tr_leg.data_density_interp.shape[1]-1))*1e6:.0f}]um')
print(f'         z=[{tr_leg.min_z*1e6:.0f}, {(tr_leg.min_z + tr_leg.delta_z*(tr_leg.data_density_interp.shape[2]-1))*1e6:.0f}]um')

# New: stored on per-timestep (xi, z) grid
tr_new = csr_new.DF_tracker
k_new = round((t_target - tr_new.min_x) / tr_new.delta_x)
k_new = min(k_new, tr_new.data_density_interp.shape[0] - 1)
poly_k = tr_new.poly_coeffs_interp[k_new]
slope_k = np.polyval(np.polyder(poly_k), 0)
print(f'\nNew: k={k_new}, t={tr_new.min_x + k_new*tr_new.delta_x:.4f}')
print(f'  xi grid: [{tr_new.min_xi_arr[k_new]*1e6:.0f}, {(tr_new.min_xi_arr[k_new] + tr_new.delta_xi_arr[k_new]*99)*1e6:.0f}]um')
print(f'  z grid: [{tr_new.min_z_arr[k_new]*1e6:.0f}, {(tr_new.min_z_arr[k_new] + tr_new.delta_z_arr[k_new]*99)*1e6:.0f}]um')
print(f'  poly slope at z=0: {slope_k:.2f}')

# Plot
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle(f'Density at t≈0.4 (step {k_leg})', fontsize=14)

# Legacy density in (x, z) lab frame
rho_leg = tr_leg.data_density_interp[k_leg]
x_leg = np.linspace(tr_leg.min_y, tr_leg.min_y + tr_leg.delta_y*(rho_leg.shape[0]-1), rho_leg.shape[0]) * 1e6
z_leg = np.linspace(tr_leg.min_z, tr_leg.min_z + tr_leg.delta_z*(rho_leg.shape[1]-1), rho_leg.shape[1]) * 1e6

im = axes[0].imshow(rho_leg.T, origin='lower', extent=[x_leg[0], x_leg[-1], z_leg[0], z_leg[-1]],
                     aspect='auto', cmap='viridis')
axes[0].set_title(f'Legacy ρ(x, z)\nx=[{x_leg[0]:.0f},{x_leg[-1]:.0f}]μm, z=[{z_leg[0]:.0f},{z_leg[-1]:.0f}]μm')
axes[0].set_xlabel('x (μm)')
axes[0].set_ylabel('z (μm)')
axes[0].axhline(150, color='r', ls='--', lw=1, label='z_ret=150μm')
axes[0].axvline(714, color='r', ls=':', lw=1, label='xp=714μm')
axes[0].legend(fontsize=8)
plt.colorbar(im, ax=axes[0])

# New density in (xi, z) frame
rho_new = tr_new.data_density_interp[k_new]
xi_new = np.linspace(tr_new.min_xi_arr[k_new], tr_new.min_xi_arr[k_new] + tr_new.delta_xi_arr[k_new]*99, 100) * 1e6
z_new = np.linspace(tr_new.min_z_arr[k_new], tr_new.min_z_arr[k_new] + tr_new.delta_z_arr[k_new]*99, 100) * 1e6

im = axes[1].imshow(rho_new.T, origin='lower', extent=[xi_new[0], xi_new[-1], z_new[0], z_new[-1]],
                     aspect='auto', cmap='viridis')
axes[1].set_title(f'New ρ(ξ, z)\nξ=[{xi_new[0]:.0f},{xi_new[-1]:.0f}]μm, z=[{z_new[0]:.0f},{z_new[-1]:.0f}]μm')
axes[1].set_xlabel('ξ (μm)')
axes[1].set_ylabel('z (μm)')
# Mark where the query point maps in xi-frame
xi_query = (714e-6 - np.polyval(poly_k, 150e-6)) * 1e6
axes[1].axhline(150, color='r', ls='--', lw=1, label=f'z_ret=150μm')
axes[1].axvline(xi_query, color='r', ls=':', lw=1, label=f'ξ={xi_query:.0f}μm (OOB!)')
axes[1].legend(fontsize=8)
plt.colorbar(im, ax=axes[1])

# Particle scatter plot showing the query point
x_p = csr_new.beam.x * 1e6
z_p = csr_new.beam.z * 1e6
subsample = np.random.choice(len(x_p), min(10000, len(x_p)), replace=False)
axes[2].scatter(x_p[subsample], z_p[subsample], s=0.1, alpha=0.3, c='blue')
axes[2].axhline(150, color='r', ls='--', lw=1)
axes[2].axvline(714, color='r', ls=':', lw=1)
axes[2].plot(714, 150, 'rx', ms=15, mew=3, label='query (714,150)μm')
axes[2].set_xlabel('x (μm)')
axes[2].set_ylabel('z (μm)')
axes[2].set_title('Particle positions at t≈0.4\n(query point marked)')
axes[2].legend()
axes[2].set_xlim([-8000, 8000])
axes[2].set_ylim([-1000, 1000])

plt.tight_layout()
out_dir = '../test/benchmark_results/single_dipole_tilted'
plt.savefig(os.path.join(out_dir, 'density_frames_t04.png'), dpi=150)
plt.close()
print(f'\nPlot: {out_dir}/density_frames_t04.png')
