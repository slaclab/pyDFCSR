"""2D density plot at t=0.499 where legacy returns 1.1e6 at query point."""
import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..', 'example'))

from pyDFCSR_2D.CSR import CSR2D
from pyDFCSR_2D.deposit_smooth import DF_tracker_smooth
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

dep_config = {'method': 'bspline_fft', 'xbins': 200, 'zbins': 100,
              'xlim': 10, 'zlim': 5, 'smoothing_sigma': 3.0,
              'poly_degree': 3, 'velocity_threhold': 1000}

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

# Get density slices at the timestep closest to t=0.499
tr_leg = csr_leg.DF_tracker
t_target = 0.499
k_leg = round((t_target - tr_leg.min_x) / tr_leg.delta_x)
k_leg = min(k_leg, tr_leg.data_density_interp.shape[0] - 1)
t_actual = tr_leg.min_x + k_leg * tr_leg.delta_x

tr_new = csr_new.DF_tracker
k_new = round((t_target - tr_new.min_x) / tr_new.delta_x)
k_new = min(k_new, tr_new.data_density_interp.shape[0] - 1)

print(f"Legacy: k={k_leg}, t={t_actual:.4f}")
print(f"New: k={k_new}, t={tr_new.min_x + k_new*tr_new.delta_x:.4f}")

# Legacy 2D density
rho_leg = tr_leg.data_density_interp[k_leg]
x_leg = np.linspace(tr_leg.min_y, tr_leg.min_y + tr_leg.delta_y*(rho_leg.shape[0]-1), rho_leg.shape[0]) * 1e6
z_leg = np.linspace(tr_leg.min_z, tr_leg.min_z + tr_leg.delta_z*(rho_leg.shape[1]-1), rho_leg.shape[1]) * 1e6

# New 2D density (in xi frame)
rho_new = tr_new.data_density_interp[k_new]
xi_new = np.linspace(tr_new.min_xi_arr[k_new], tr_new.min_xi_arr[k_new] + tr_new.delta_xi_arr[k_new]*(rho_new.shape[0]-1), rho_new.shape[0]) * 1e6
z_new = np.linspace(tr_new.min_z_arr[k_new], tr_new.min_z_arr[k_new] + tr_new.delta_z_arr[k_new]*(rho_new.shape[1]-1), rho_new.shape[1]) * 1e6
poly_k = tr_new.poly_coeffs_interp[k_new]

print(f"Legacy grid: x=[{x_leg[0]:.0f},{x_leg[-1]:.0f}]um, z=[{z_leg[0]:.0f},{z_leg[-1]:.0f}]um, shape={rho_leg.shape}")
print(f"New grid: xi=[{xi_new[0]:.0f},{xi_new[-1]:.0f}]um, z=[{z_new[0]:.0f},{z_new[-1]:.0f}]um, shape={rho_new.shape}")
print(f"New poly slope: {np.polyval(np.polyder(poly_k), 0):.2f}")

# Query point
xp_query = -387  # um
z_query = 98  # um

fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle(f'Density at t≈0.499 (step {k_leg}). Query: x={xp_query}μm, z={z_query}μm', fontsize=13)

# Legacy
im = axes[0].imshow(rho_leg.T, origin='lower', extent=[x_leg[0], x_leg[-1], z_leg[0], z_leg[-1]],
                     aspect='auto', cmap='viridis')
axes[0].plot(xp_query, z_query, 'rx', ms=15, mew=3)
axes[0].set_title(f'Legacy ρ(x, z)\nx=[{x_leg[0]:.0f},{x_leg[-1]:.0f}]μm')
axes[0].set_xlabel('x (μm)')
axes[0].set_ylabel('z (μm)')
plt.colorbar(im, ax=axes[0])

# Legacy zoomed around query point
zoom_x = [-1500, 500]
zoom_z = [-300, 300]
axes[1].imshow(rho_leg.T, origin='lower', extent=[x_leg[0], x_leg[-1], z_leg[0], z_leg[-1]],
               aspect='auto', cmap='viridis')
axes[1].plot(xp_query, z_query, 'rx', ms=15, mew=3, label=f'query ({xp_query},{z_query})μm')
axes[1].set_xlim(zoom_x)
axes[1].set_ylim(zoom_z)
axes[1].set_title('Legacy ρ(x, z) — ZOOMED')
axes[1].set_xlabel('x (μm)')
axes[1].set_ylabel('z (μm)')
axes[1].legend()

# New in xi frame
im = axes[2].imshow(rho_new.T, origin='lower', extent=[xi_new[0], xi_new[-1], z_new[0], z_new[-1]],
                     aspect='auto', cmap='viridis')
xi_query = xp_query - np.polyval(poly_k, z_query*1e-6)*1e6
axes[2].plot(xi_query, z_query, 'rx', ms=15, mew=3, label=f'query ξ={xi_query:.0f}μm')
axes[2].set_title(f'New ρ(ξ, z)\nξ=[{xi_new[0]:.0f},{xi_new[-1]:.0f}]μm')
axes[2].set_xlabel('ξ (μm)')
axes[2].set_ylabel('z (μm)')
axes[2].legend()
plt.colorbar(im, ax=axes[2])

plt.tight_layout()
out_dir = '../test/benchmark_results/single_dipole_tilted'
plt.savefig(os.path.join(out_dir, 'density_2d_t0499.png'), dpi=150)
plt.close()
print(f"\nPlot: {out_dir}/density_2d_t0499.png")
