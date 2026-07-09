"""
Diagnostic: compare the density and derivative values returned by interpolation
at the same retarded-time query points between legacy and new methods.

This isolates whether the STORED DATA differs (not the interpolation function).
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..', 'example'))

from pyDFCSR_2D.CSR import CSR2D
from pyDFCSR_2D.deposit_smooth import DF_tracker_smooth
from pyDFCSR_2D.interp3D import interpolate3D, interpolate3D_transformed
from pyDFCSR_2D.interp1D import interpolate1D

dep_config = {
    'method': 'bspline_fft',
    'xbins': 200, 'zbins': 200,
    'xlim': 5, 'zlim': 5,
    'smoothing_sigma': 3.0,
    'poly_degree': 3,
    'velocity_threhold': 1000,
}

TARGET_STEP = 7


def run_to_step(method):
    csr = CSR2D(input_file='input/dipole_chirp500_config.yaml')
    csr.CSR_params.apply_CSR = 0  # No kicks — identical beam
    if method == 'bspline_fft':
        csr.DF_tracker = DF_tracker_smooth(dep_config)
        csr.use_smooth_deposit = True
        csr.DF_tracker.get_DF(x=csr.beam.x, z=csr.beam.z, px=csr.beam.px, t=csr.beam.position)
        csr.DF_tracker.append_DF()
        csr.DF_tracker.append_interpolant(formation_length=float('inf'),
                                           n_formation_length=csr.integration_params.n_formation_length)
    csr.run(stop_time=(TARGET_STEP + 0.5) * 0.1)
    return csr


def query_density_fields(csr, xp_arr, sp_arr, s, x, t):
    """Query all density fields at given (xp, sp) integration points."""
    sp_flat = sp_arr.ravel()
    xp_flat = xp_arr.ravel()

    # Compute retarded time (same geometry for both methods)
    X0_s = interpolate1D(xval=np.array([s]), data=csr.lattice.coords[:, 0],
                         min_x=csr.lattice.min_x, delta_x=csr.lattice.delta_x)[0]
    X0_sp = interpolate1D(xval=sp_flat, data=csr.lattice.coords[:, 0],
                          min_x=csr.lattice.min_x, delta_x=csr.lattice.delta_x)
    Y0_s = interpolate1D(xval=np.array([s]), data=csr.lattice.coords[:, 1],
                         min_x=csr.lattice.min_x, delta_x=csr.lattice.delta_x)[0]
    Y0_sp = interpolate1D(xval=sp_flat, data=csr.lattice.coords[:, 1],
                          min_x=csr.lattice.min_x, delta_x=csr.lattice.delta_x)
    n_vec_s_x = interpolate1D(xval=np.array([s]), data=csr.lattice.n_vec[:, 0],
                              min_x=csr.lattice.min_x, delta_x=csr.lattice.delta_x)[0]
    n_vec_sp_x = interpolate1D(xval=sp_flat, data=csr.lattice.n_vec[:, 0],
                               min_x=csr.lattice.min_x, delta_x=csr.lattice.delta_x)
    n_vec_s_y = interpolate1D(xval=np.array([s]), data=csr.lattice.n_vec[:, 1],
                              min_x=csr.lattice.min_x, delta_x=csr.lattice.delta_x)[0]
    n_vec_sp_y = interpolate1D(xval=sp_flat, data=csr.lattice.n_vec[:, 1],
                               min_x=csr.lattice.min_x, delta_x=csr.lattice.delta_x)

    r_minus_rp_x = X0_s - X0_sp + x * n_vec_s_x - xp_flat * n_vec_sp_x
    r_minus_rp_y = Y0_s - Y0_sp + x * n_vec_s_y - xp_flat * n_vec_sp_y
    r_minus_rp = np.sqrt(r_minus_rp_x**2 + r_minus_rp_y**2)
    t_ret = t - r_minus_rp
    z_ret = sp_flat - t_ret

    # Query density fields
    if csr.use_smooth_deposit:
        tracker = csr.DF_tracker
        rho = interpolate3D_transformed(xp_flat, z_ret, t_ret,
                                         tracker.data_density_interp, tracker.poly_coeffs_interp,
                                         tracker.min_xi_arr, tracker.min_z_arr, tracker.min_x,
                                         tracker.delta_xi_arr, tracker.delta_z_arr, tracker.delta_x)
        rho_x = interpolate3D_transformed(xp_flat, z_ret, t_ret,
                                           tracker.data_density_x_interp, tracker.poly_coeffs_interp,
                                           tracker.min_xi_arr, tracker.min_z_arr, tracker.min_x,
                                           tracker.delta_xi_arr, tracker.delta_z_arr, tracker.delta_x)
        rho_z = interpolate3D_transformed(xp_flat, z_ret, t_ret,
                                           tracker.data_density_z_interp, tracker.poly_coeffs_interp,
                                           tracker.min_xi_arr, tracker.min_z_arr, tracker.min_x,
                                           tracker.delta_xi_arr, tracker.delta_z_arr, tracker.delta_x)
    else:
        tracker = csr.DF_tracker
        rho = interpolate3D(t_ret, xp_flat, z_ret,
                            tracker.data_density_interp,
                            tracker.min_x, tracker.min_y, tracker.min_z,
                            tracker.delta_x, tracker.delta_y, tracker.delta_z)
        rho_x = interpolate3D(t_ret, xp_flat, z_ret,
                              tracker.data_density_x_interp,
                              tracker.min_x, tracker.min_y, tracker.min_z,
                              tracker.delta_x, tracker.delta_y, tracker.delta_z)
        rho_z = interpolate3D(t_ret, xp_flat, z_ret,
                              tracker.data_density_z_interp,
                              tracker.min_x, tracker.min_y, tracker.min_z,
                              tracker.delta_x, tracker.delta_y, tracker.delta_z)

    return rho, rho_x, rho_z, t_ret, z_ret, r_minus_rp


def main():
    print("="*70)
    print(f"Density Query Diagnostic: chirp=500, step={TARGET_STEP}")
    print("="*70)

    print("\nRunning legacy...")
    csr_leg = run_to_step('legacy')
    print("\nRunning new...")
    csr_new = run_to_step('bspline_fft')

    # Pick a CSR mesh point
    zbins = csr_leg.CSR_params.zbins
    ix, iz = 5, 15  # center
    k = ix * zbins + iz
    s = csr_leg.beam.position + csr_leg.CSR_zmesh[k]
    x = csr_leg.CSR_xmesh[k]
    t = csr_leg.beam.position

    print(f"\nObservation: s={s:.6f}, x={x*1e6:.2f}um, t={t:.4f}")
    print(f"Beam slope: {csr_leg.beam._slope[0]:.4f}")

    # Create a line of query points along s' (the retarded position axis)
    # Use the region that matters most (sp close to s, where density is significant)
    sigma_z = csr_leg.beam._sigma_z
    sp_line = np.linspace(s - 5*sigma_z, s - 0.5*sigma_z, 50)
    xp_val = x  # query at same x as observation (narrow band)
    xp_line = np.full_like(sp_line, xp_val)

    # Make them into the right shape for the query function
    sp_arr = sp_line.reshape(1, -1)
    xp_arr = xp_line.reshape(1, -1)

    # Query both methods
    rho_leg, rho_x_leg, rho_z_leg, t_ret_leg, z_ret_leg, r_leg = query_density_fields(
        csr_leg, xp_arr, sp_arr, s, x, t)
    rho_new, rho_x_new, rho_z_new, t_ret_new, z_ret_new, r_new = query_density_fields(
        csr_new, xp_arr, sp_arr, s, x, t)

    # Print comparison
    print(f"\n{'sp':<8} | {'z_ret':<10} | {'t_ret':<8} | {'rho_leg':<12} | {'rho_new':<12} | {'ratio':<8} | {'rho_x_leg':<12} | {'rho_x_new':<12} | {'rho_z_leg':<12} | {'rho_z_new':<12}")
    print("-" * 140)

    for i in range(0, len(sp_line), 5):
        ratio_rho = rho_new[i] / rho_leg[i] if abs(rho_leg[i]) > 1e-30 else 0
        print(f"{sp_line[i]:.4f} | {z_ret_leg[i]*1e6:8.2f}um | {t_ret_leg[i]:.5f} | {rho_leg[i]:12.4e} | {rho_new[i]:12.4e} | {ratio_rho:8.4f} | {rho_x_leg[i]:12.4e} | {rho_x_new[i]:12.4e} | {rho_z_leg[i]:12.4e} | {rho_z_new[i]:12.4e}")

    # Summary statistics
    mask = np.abs(rho_leg) > 0.01 * np.abs(rho_leg).max()
    n_sig = mask.sum()
    print(f"\nSignificant points (rho > 1% max): {n_sig}/{len(rho_leg)}")

    if n_sig > 0:
        def rel(a, b, m):
            return np.linalg.norm(a[m] - b[m]) / max(np.linalg.norm(b[m]), 1e-30)

        print(f"  rho:   rel_diff = {rel(rho_new, rho_leg, mask):.4f}")
        print(f"  rho_x: rel_diff = {rel(rho_x_new, rho_x_leg, mask):.4f}")
        print(f"  rho_z: rel_diff = {rel(rho_z_new, rho_z_leg, mask):.4f}")

        # Check if the issue is normalization
        print(f"\n  Peak values:")
        print(f"    rho:   legacy={rho_leg[mask].max():.4e}, new={rho_new[mask].max():.4e}, ratio={rho_new[mask].max()/rho_leg[mask].max():.4f}")
        print(f"    rho_x: legacy max={np.abs(rho_x_leg[mask]).max():.4e}, new max={np.abs(rho_x_new[mask]).max():.4e}")
        print(f"    rho_z: legacy max={np.abs(rho_z_leg[mask]).max():.4e}, new max={np.abs(rho_z_new[mask]).max():.4e}")

    # Also query along x' at fixed s' (transverse cut)
    print(f"\n\n--- Transverse cut at sp = s - 2*sigma_z ---")
    sp_fixed = s - 2 * sigma_z
    sigma_x = csr_leg.beam._sigma_x
    xp_trans = np.linspace(x - 3*sigma_x, x + 3*sigma_x, 50)
    sp_trans = np.full_like(xp_trans, sp_fixed)

    rho_leg_t, rho_x_leg_t, rho_z_leg_t, _, _, _ = query_density_fields(
        csr_leg, xp_trans.reshape(1,-1), sp_trans.reshape(1,-1), s, x, t)
    rho_new_t, rho_x_new_t, rho_z_new_t, _, _, _ = query_density_fields(
        csr_new, xp_trans.reshape(1,-1), sp_trans.reshape(1,-1), s, x, t)

    mask_t = np.abs(rho_leg_t) > 0.01 * np.abs(rho_leg_t).max()
    if mask_t.sum() > 0:
        print(f"  rho:   rel_diff = {rel(rho_new_t, rho_leg_t, mask_t):.4f}")
        print(f"  rho_x: rel_diff = {rel(rho_x_new_t, rho_x_leg_t, mask_t):.4f}")
        print(f"  rho_z: rel_diff = {rel(rho_z_new_t, rho_z_leg_t, mask_t):.4f}")
        print(f"  Peak rho: legacy={rho_leg_t[mask_t].max():.4e}, new={rho_new_t[mask_t].max():.4e}")


if __name__ == '__main__':
    main()
