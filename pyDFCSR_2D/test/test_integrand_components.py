"""
Compare CSR integrand components (CSR_numerator1, CSR_numerator2, r_minus_rp)
at Region 1 (far history) query points.
"""
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

# Run both to step 17
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

print(f"s={s:.4f}, x={x*1e6:.1f}um, slope={np.polyfit(csr_leg.beam.z, csr_leg.beam.x, 1)[0]:.4f}")

# Pick query points in Region 1 matching where the integrand is nonzero.
# The beam center at early times was at x ~ slope*z ~ -8 * 200um = -1600um initially,
# then grows due to dispersion. Let's pick sp in the mid-bend region and scan xp.
sp_val = s - 0.5  # 500mm behind (in the drift before the bend)
sp_line = np.full(50, sp_val)
# Scan xp across where beam might be
xp_line = np.linspace(-5000e-6, 10000e-6, 50)

# Compute geometry (same for both methods)
sp_flat = sp_line
X0_s = interpolate1D(np.array([s]), csr_leg.lattice.coords[:, 0], csr_leg.lattice.min_x, csr_leg.lattice.delta_x)[0]
Y0_s = interpolate1D(np.array([s]), csr_leg.lattice.coords[:, 1], csr_leg.lattice.min_x, csr_leg.lattice.delta_x)[0]
X0_sp = interpolate1D(sp_flat, csr_leg.lattice.coords[:, 0], csr_leg.lattice.min_x, csr_leg.lattice.delta_x)
Y0_sp = interpolate1D(sp_flat, csr_leg.lattice.coords[:, 1], csr_leg.lattice.min_x, csr_leg.lattice.delta_x)
n_s_x = interpolate1D(np.array([s]), csr_leg.lattice.n_vec[:, 0], csr_leg.lattice.min_x, csr_leg.lattice.delta_x)[0]
n_s_y = interpolate1D(np.array([s]), csr_leg.lattice.n_vec[:, 1], csr_leg.lattice.min_x, csr_leg.lattice.delta_x)[0]
n_sp_x = interpolate1D(sp_flat, csr_leg.lattice.n_vec[:, 0], csr_leg.lattice.min_x, csr_leg.lattice.delta_x)
n_sp_y = interpolate1D(sp_flat, csr_leg.lattice.n_vec[:, 1], csr_leg.lattice.min_x, csr_leg.lattice.delta_x)
tau_s_x = interpolate1D(np.array([s]), csr_leg.lattice.tau_vec[:, 0], csr_leg.lattice.min_x, csr_leg.lattice.delta_x)[0]
tau_s_y = interpolate1D(np.array([s]), csr_leg.lattice.tau_vec[:, 1], csr_leg.lattice.min_x, csr_leg.lattice.delta_x)[0]
tau_sp_x = interpolate1D(sp_flat, csr_leg.lattice.tau_vec[:, 0], csr_leg.lattice.min_x, csr_leg.lattice.delta_x)
tau_sp_y = interpolate1D(sp_flat, csr_leg.lattice.tau_vec[:, 1], csr_leg.lattice.min_x, csr_leg.lattice.delta_x)

r_x = X0_s - X0_sp + x * n_s_x - xp_line * n_sp_x
r_y = Y0_s - Y0_sp + x * n_s_y - xp_line * n_sp_y
r_minus_rp = np.sqrt(r_x**2 + r_y**2)
t_ret = t - r_minus_rp
z_ret = sp_flat - t_ret

print(f"\nQuery line: sp from {sp_flat[0]:.4f} to {sp_flat[-1]:.4f}")
print(f"  t_ret range: [{t_ret.min():.4f}, {t_ret.max():.4f}]")
print(f"  z_ret range: [{z_ret.min()*1e6:.0f}, {z_ret.max()*1e6:.0f}] um")
print(f"  r_minus_rp range: [{r_minus_rp.min()*1e6:.0f}, {r_minus_rp.max()*1e6:.0f}] um")

# Query density and derivatives from both methods
def query_fields(csr, xp, z_ret, t_ret):
    tr = csr.DF_tracker
    if csr.use_smooth_deposit:
        from pyDFCSR_2D.interp3D import interpolate3D_transformed_with_derivs
        rho, rho_x, rho_z = interpolate3D_transformed_with_derivs(
            xp, z_ret, t_ret, tr.data_density_interp, tr.poly_coeffs_interp,
            tr.min_xi_arr, tr.min_z_arr, tr.min_x, tr.delta_xi_arr, tr.delta_z_arr, tr.delta_x)
    else:
        rho = interpolate3D(t_ret, xp, z_ret, tr.data_density_interp,
                            tr.min_x, tr.min_y, tr.min_z, tr.delta_x, tr.delta_y, tr.delta_z)
        rho_x = interpolate3D(t_ret, xp, z_ret, tr.data_density_x_interp,
                              tr.min_x, tr.min_y, tr.min_z, tr.delta_x, tr.delta_y, tr.delta_z)
        rho_z = interpolate3D(t_ret, xp, z_ret, tr.data_density_z_interp,
                              tr.min_x, tr.min_y, tr.min_z, tr.delta_x, tr.delta_y, tr.delta_z)
    return rho, rho_x, rho_z

rho_leg, rho_x_leg, rho_z_leg = query_fields(csr_leg, xp_line, z_ret, t_ret)
rho_new, rho_x_new, rho_z_new = query_fields(csr_new, xp_line, z_ret, t_ret)

# Print comparison
print(f"\n{'sp-s (mm)':<10} | {'r (mm)':<8} | {'t_ret':<6} | {'rho_leg':<10} | {'rho_new':<10} | {'rho_x_leg':<12} | {'rho_x_new':<12} | {'rho_z_leg':<12} | {'rho_z_new':<12}")
print("-" * 120)
for i in range(0, len(sp_line), 5):
    print(f"{(sp_flat[i]-s)*1e3:>8.3f}  | {r_minus_rp[i]*1e3:>6.3f} | {t_ret[i]:.4f} | {rho_leg[i]:>10.3e} | {rho_new[i]:>10.3e} | {rho_x_leg[i]:>12.3e} | {rho_x_new[i]:>12.3e} | {rho_z_leg[i]:>12.3e} | {rho_z_new[i]:>12.3e}")

# Summary
mask = rho_leg > 0.01 * rho_leg.max()
print(f"\nSignificant points: {mask.sum()}/{len(rho_leg)}")
if mask.sum() > 0:
    def rel(a, b, m):
        return np.linalg.norm(a[m] - b[m]) / max(np.linalg.norm(b[m]), 1e-30)
    print(f"  rho:   rel_diff = {rel(rho_new, rho_leg, mask):.4f}")
    print(f"  rho_x: rel_diff = {rel(rho_x_new, rho_x_leg, mask):.4f}")
    print(f"  rho_z: rel_diff = {rel(rho_z_new, rho_z_leg, mask):.4f}")
