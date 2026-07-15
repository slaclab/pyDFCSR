"""Find where the Region 1 integrand is actually nonzero and compare."""
import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..', 'example'))

from pyDFCSR_2D.CSR import CSR2D
from pyDFCSR_2D.deposit_smooth import DF_tracker_smooth
from pyDFCSR_2D.interp1D import interpolate1D
from pyDFCSR_2D.interp3D import interpolate3D, interpolate3D_transformed

dep_config = {'method': 'bspline_fft', 'xbins': 100, 'zbins': 100,
              'xlim': 5, 'zlim': 5, 'smoothing_sigma': 3.0,
              'poly_degree': 3, 'velocity_threhold': 1000}

# Run both to step 17 (s=0.85)
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

s = csr_leg.beam.position
x = 4930e-6
t = s
print(f"Observation: s={s:.4f}, x={x*1e6:.0f}um")

# Get the Region 1 integrand from debug mode
r_leg = csr_leg.get_CSR_wake(s, x, debug=True)
# chirp_band=True: returns xp1, xp2, xp3, xp4, sp1, sp2, sp3, iz1, ix1, ...
xp4 = r_leg[3]  # Region 1 uses xp4
sp1 = r_leg[4]  # Region 1 uses sp1
iz1_leg = r_leg[7]  # Region 1 integrand_z

r_new = csr_new.get_CSR_wake(s, x, debug=True)
iz1_new = r_new[7]

print(f"Region 1: xp4=[{xp4[0]*1e6:.0f},{xp4[-1]*1e6:.0f}]um ({len(xp4)} pts)")
print(f"          sp1=[{sp1[0]:.5f},{sp1[-1]:.5f}] ({len(sp1)} pts)")
print(f"          iz1 shape: {iz1_leg.shape}")
print(f"          iz1 legacy: max={abs(iz1_leg).max():.4e}, sum={iz1_leg.sum():.4e}")
print(f"          iz1 new:    max={abs(iz1_new).max():.4e}, sum={iz1_new.sum():.4e}")

# Find where the integrand is largest
idx_max = np.unravel_index(np.argmax(np.abs(iz1_leg)), iz1_leg.shape)
print(f"\n  Legacy max at index {idx_max}: xp={xp4[idx_max[0]]*1e6:.0f}um, sp={sp1[idx_max[1]]:.5f}")
print(f"    iz1_leg = {iz1_leg[idx_max]:.4e}")
print(f"    iz1_new = {iz1_new[idx_max]:.4e}")

# Find all points where |iz1_leg| > 10% of max
threshold = 0.1 * abs(iz1_leg).max()
nonzero_mask = np.abs(iz1_leg) > threshold
n_nonzero = nonzero_mask.sum()
print(f"\n  Points with |integrand| > 10% of max: {n_nonzero}/{iz1_leg.size}")

if n_nonzero > 0:
    # Get the (xp, sp) indices where integrand is significant
    ix_nz, is_nz = np.where(nonzero_mask)
    xp_nz = xp4[ix_nz]
    sp_nz = sp1[is_nz]
    print(f"  xp range of nonzero: [{xp_nz.min()*1e6:.0f}, {xp_nz.max()*1e6:.0f}]um")
    print(f"  sp range of nonzero: [{sp_nz.min():.5f}, {sp_nz.max():.5f}]")

    # Compute retarded time and z_ret at these points
    X0_s = interpolate1D(np.array([s]), csr_leg.lattice.coords[:, 0], csr_leg.lattice.min_x, csr_leg.lattice.delta_x)[0]
    Y0_s = interpolate1D(np.array([s]), csr_leg.lattice.coords[:, 1], csr_leg.lattice.min_x, csr_leg.lattice.delta_x)[0]
    n_s_x = interpolate1D(np.array([s]), csr_leg.lattice.n_vec[:, 0], csr_leg.lattice.min_x, csr_leg.lattice.delta_x)[0]
    n_s_y = interpolate1D(np.array([s]), csr_leg.lattice.n_vec[:, 1], csr_leg.lattice.min_x, csr_leg.lattice.delta_x)[0]

    # At the max point
    sp_max = sp1[idx_max[1]]
    xp_max = xp4[idx_max[0]]
    X0_sp = interpolate1D(np.array([sp_max]), csr_leg.lattice.coords[:, 0], csr_leg.lattice.min_x, csr_leg.lattice.delta_x)[0]
    Y0_sp = interpolate1D(np.array([sp_max]), csr_leg.lattice.coords[:, 1], csr_leg.lattice.min_x, csr_leg.lattice.delta_x)[0]
    n_sp_x = interpolate1D(np.array([sp_max]), csr_leg.lattice.n_vec[:, 0], csr_leg.lattice.min_x, csr_leg.lattice.delta_x)[0]
    n_sp_y = interpolate1D(np.array([sp_max]), csr_leg.lattice.n_vec[:, 1], csr_leg.lattice.min_x, csr_leg.lattice.delta_x)[0]

    r_x = X0_s - X0_sp + x * n_s_x - xp_max * n_sp_x
    r_y = Y0_s - Y0_sp + x * n_s_y - xp_max * n_sp_y
    r_mag = np.sqrt(r_x**2 + r_y**2)
    t_ret_max = t - r_mag
    z_ret_max = sp_max - t_ret_max

    print(f"\n  At integrand max:")
    print(f"    xp = {xp_max*1e6:.1f}um, sp = {sp_max:.5f}")
    print(f"    |r-r'| = {r_mag*1e6:.1f}um")
    print(f"    t_ret = {t_ret_max:.5f}")
    print(f"    z_ret = {z_ret_max*1e6:.1f}um")

    # Now query the density at this point from both methods
    tr_leg = csr_leg.DF_tracker
    rho_leg = interpolate3D(np.array([t_ret_max]), np.array([xp_max]), np.array([z_ret_max]),
                            tr_leg.data_density_interp, tr_leg.min_x, tr_leg.min_y, tr_leg.min_z,
                            tr_leg.delta_x, tr_leg.delta_y, tr_leg.delta_z)[0]

    tr_new = csr_new.DF_tracker
    rho_new = interpolate3D_transformed(np.array([xp_max]), np.array([z_ret_max]), np.array([t_ret_max]),
                                         tr_new.data_density_interp, tr_new.poly_coeffs_interp,
                                         tr_new.min_xi_arr, tr_new.min_z_arr, tr_new.min_x,
                                         tr_new.delta_xi_arr, tr_new.delta_z_arr, tr_new.delta_x)[0]

    print(f"    rho (legacy) = {rho_leg:.4e}")
    print(f"    rho (new)    = {rho_new:.4e}")

    # Check xi bounds for new method at this point
    t_idx = (t_ret_max - tr_new.min_x) / tr_new.delta_x
    k = int(t_idx)
    k = min(k, tr_new.data_density_interp.shape[0] - 2)
    poly_k = tr_new.poly_coeffs_interp[k]
    xi_at_k = xp_max - np.polyval(poly_k, z_ret_max)
    min_xi = tr_new.min_xi_arr[k]
    max_xi = min_xi + tr_new.delta_xi_arr[k] * (tr_new.data_density_interp.shape[1] - 1)
    slope_k = np.polyval(np.polyder(poly_k), 0)

    print(f"\n    New method at k={k} (t={tr_new.min_x + k*tr_new.delta_x:.4f}, slope={slope_k:.2f}):")
    print(f"      poly(z_ret={z_ret_max*1e6:.1f}um) = {np.polyval(poly_k, z_ret_max)*1e6:.1f}um")
    print(f"      xi = xp - poly(z_ret) = {xp_max*1e6:.1f} - {np.polyval(poly_k, z_ret_max)*1e6:.1f} = {xi_at_k*1e6:.1f}um")
    print(f"      xi grid: [{min_xi*1e6:.1f}, {max_xi*1e6:.1f}]um")
    print(f"      IN BOUNDS: {min_xi <= xi_at_k <= max_xi}")
