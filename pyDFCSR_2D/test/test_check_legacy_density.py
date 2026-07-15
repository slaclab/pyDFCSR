"""Check legacy density at the query point and compare to peak."""
import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..', 'example'))

from pyDFCSR_2D.CSR import CSR2D
from pyDFCSR_2D.interp3D import interpolate3D

csr = CSR2D(input_file='input/single_dipole_tilted_config.yaml')
csr.CSR_params.apply_CSR = 0
csr.run(stop_time=0.86)

tr = csr.DF_tracker
t_query = 0.499
k = round((t_query - tr.min_x) / tr.delta_x)
print(f"Time grid: min={tr.min_x:.4f}, delta={tr.delta_x:.4f}, k={k}")
print(f"t at k={k}: {tr.min_x + k*tr.delta_x:.4f}")

# Peak density at this timestep
rho_slice = tr.data_density_interp[k]
print(f"\nDensity slice at k={k}:")
print(f"  shape: {rho_slice.shape}")
print(f"  max: {rho_slice.max():.4e}")
print(f"  grid x: [{tr.min_y*1e6:.0f}, {(tr.min_y + tr.delta_y*(rho_slice.shape[0]-1))*1e6:.0f}]um, dy={tr.delta_y*1e6:.1f}um")
print(f"  grid z: [{tr.min_z*1e6:.0f}, {(tr.min_z + tr.delta_z*(rho_slice.shape[1]-1))*1e6:.0f}]um, dz={tr.delta_z*1e6:.1f}um")

# Query at the point of interest
xp_query = -387e-6
z_query = 98e-6

rho_at_point = interpolate3D(
    np.array([t_query]), np.array([xp_query]), np.array([z_query]),
    tr.data_density_interp, tr.min_x, tr.min_y, tr.min_z,
    tr.delta_x, tr.delta_y, tr.delta_z)[0]

print(f"\nQuery: xp={xp_query*1e6:.0f}um, z={z_query*1e6:.0f}um, t={t_query:.4f}")
print(f"  rho = {rho_at_point:.4e}")
print(f"  rho/peak = {rho_at_point/rho_slice.max():.6f}")

# Where is the beam center at z=98um?
# slope=-8, so x_center = -8 * 98um = -784um
x_center = -8 * z_query
xi_offset = xp_query - x_center
sigma_xi = 64e-6
print(f"\n  Beam center at z={z_query*1e6:.0f}um: x_center={x_center*1e6:.0f}um")
print(f"  xi = xp - x_center = {xp_query*1e6:.0f} - ({x_center*1e6:.0f}) = {xi_offset*1e6:.0f}um")
print(f"  sigma_xi = {sigma_xi*1e6:.0f}um")
print(f"  Distance: {xi_offset/sigma_xi:.1f} sigma_xi")
print(f"  Expected Gaussian: exp(-{(xi_offset/sigma_xi)**2/2:.1f}) = {np.exp(-(xi_offset/sigma_xi)**2/2):.4e}")
print(f"  Expected density: peak * exp(...) = {rho_slice.max() * np.exp(-(xi_offset/sigma_xi)**2/2):.4e}")

# Also check density along x at fixed z=98um
x_scan = np.linspace(-2000e-6, 2000e-6, 100)
rho_scan = interpolate3D(
    np.full(100, t_query), x_scan, np.full(100, z_query),
    tr.data_density_interp, tr.min_x, tr.min_y, tr.min_z,
    tr.delta_x, tr.delta_y, tr.delta_z)
print(f"\n  Density scan along x at z={z_query*1e6:.0f}um, t={t_query:.4f}:")
print(f"  Peak at x={x_scan[np.argmax(rho_scan)]*1e6:.0f}um, value={rho_scan.max():.4e}")
print(f"  At x=-387um: {rho_scan[np.argmin(np.abs(x_scan - xp_query))]:.4e}")
