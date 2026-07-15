"""Check if the legacy density at (714, 150)um is real or an artifact."""
import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..', 'example'))

from pyDFCSR_2D.CSR import CSR2D

# Run legacy
csr = CSR2D(input_file='input/single_dipole_tilted_config.yaml')
csr.CSR_params.apply_CSR = 0
csr.run(stop_time=0.41)

# Compute true density by direct particle binning at (x=714, z=150)
x_p = csr.beam.x
z_p = csr.beam.z
N = len(x_p)

# At z=150um, beam center in x is at slope*z = -8*150um = -1200um
# xi = x - (-8*z) = x + 8*z
xi_p = x_p - (-8) * z_p  # tilt-removed

z_query = 150e-6
x_query = 714e-6
xi_query = x_query - (-8) * z_query  # = 714 + 1200 = 1914 um

print(f"Query point: x={x_query*1e6:.0f}um, z={z_query*1e6:.0f}um")
print(f"  xi = x - slope*z = {xi_query*1e6:.0f}um")
print(f"  sigma_xi = {np.std(xi_p)*1e6:.1f}um")
print(f"  Distance from center: {xi_query/np.std(xi_p):.1f} sigma_xi")
print()

# Direct particle count in different box sizes
for dx in [50e-6, 100e-6, 200e-6, 500e-6]:
    for dz in [50e-6, 100e-6]:
        count = np.sum((np.abs(x_p - x_query) < dx) & (np.abs(z_p - z_query) < dz))
        area = (2*dx) * (2*dz)
        density_est = count / (N * area) if count > 0 else 0
        print(f"  Box ±{dx*1e6:.0f}x±{dz*1e6:.0f}um: {count} particles, est density = {density_est:.2e}")

# What does the legacy stored density look like at this point?
tr = csr.DF_tracker
from pyDFCSR_2D.interp3D import interpolate3D
rho_at_point = interpolate3D(
    np.array([0.4]), np.array([x_query]), np.array([z_query]),
    tr.data_density_interp, tr.min_x, tr.min_y, tr.min_z,
    tr.delta_x, tr.delta_y, tr.delta_z)
print(f"\nLegacy interpolated density at (714, 150, t=0.4): {rho_at_point[0]:.4e}")

# Also check the legacy density at the beam center at this z
x_center = -8 * z_query
rho_at_center = interpolate3D(
    np.array([0.4]), np.array([x_center]), np.array([z_query]),
    tr.data_density_interp, tr.min_x, tr.min_y, tr.min_z,
    tr.delta_x, tr.delta_y, tr.delta_z)
print(f"Legacy interpolated density at center ({x_center*1e6:.0f}, 150, t=0.4): {rho_at_center[0]:.4e}")
print(f"Ratio (query/center): {rho_at_point[0]/rho_at_center[0]:.6f}")

# Check: what's the max density on the legacy grid at this timestep?
k = round((0.4 - tr.min_x) / tr.delta_x)
print(f"\nLegacy grid at k={k}: max density = {tr.data_density_interp[k].max():.4e}")
print(f"Query density / max = {rho_at_point[0] / tr.data_density_interp[k].max():.6e}")
