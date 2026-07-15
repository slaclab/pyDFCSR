"""
Realistic interpolation test matching the actual simulation conditions:
1. Large tilt (slope up to ±12)
2. Fast change in tilt between timesteps (slope flip +7 → -12)
3. Query points that are mostly OOB (like the CSR retarded-time queries)
4. Different grid extents per timestep (sigma_xi changes)
5. Include chain rule correction in stored density_z

Compare interpolate3D_transformed vs analytical at the SAME query points
the real CSR code would use.
"""
import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from pyDFCSR_2D.interp3D import interpolate3D, interpolate3D_transformed

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RESULT_DIR = os.path.join(os.path.dirname(__file__), 'benchmark_results', 'interp_test')
os.makedirs(RESULT_DIR, exist_ok=True)


def run_test():
    sigma_xi = 50e-6
    sigma_z = 50e-6

    # Realistic tilt sequence matching chirp=500 (from benchmark):
    # slope evolves rapidly, with a sign flip
    n_t = 6
    slopes = [+1.0, +3.4, +7.1, -5.3, -9.9, -6.6]
    times = np.array([0.25, 0.35, 0.45, 0.50, 0.60, 0.70])
    delta_t = 0.05  # non-uniform but we use the average

    # Each timestep has DIFFERENT sigma_z (beam compresses/expands)
    sigma_z_per_step = np.array([50e-6, 45e-6, 35e-6, 30e-6, 60e-6, 130e-6])

    print("=" * 70)
    print("Realistic Interpolation Test")
    print("  Large tilts, fast tilt change, varying grid extents, OOB queries")
    print("=" * 70)
    print(f"  Slopes: {slopes}")
    print(f"  sigma_z: {[f'{s*1e6:.0f}' for s in sigma_z_per_step]} um")

    # Build per-timestep data
    n_bins = 200
    xlim = 5

    data_density = np.zeros((n_t, n_bins, n_bins))
    data_density_x = np.zeros((n_t, n_bins, n_bins))
    data_density_z = np.zeros((n_t, n_bins, n_bins))
    min_xi_arr = np.zeros(n_t)
    delta_xi_arr = np.zeros(n_t)
    min_z_arr = np.zeros(n_t)
    delta_z_arr = np.zeros(n_t)
    poly_coeffs_arr = np.zeros((n_t, 2))  # linear: [slope, 0]

    for k in range(n_t):
        slope = slopes[k]
        sz = sigma_z_per_step[k]

        xi_grid = np.linspace(-xlim * sigma_xi, xlim * sigma_xi, n_bins)
        z_grid = np.linspace(-xlim * sz, xlim * sz, n_bins)

        XI, Z = np.meshgrid(xi_grid, z_grid, indexing='ij')
        rho = np.exp(-XI**2 / (2 * sigma_xi**2) - Z**2 / (2 * sz**2))
        rho_dxi = -XI / sigma_xi**2 * rho
        rho_dz_xi = -Z / sz**2 * rho
        # Chain rule: ∂ρ/∂z|_x = ∂ρ/∂z|_ξ - slope * ∂ρ/∂ξ
        rho_dz_lab = rho_dz_xi - slope * rho_dxi

        data_density[k] = rho
        data_density_x[k] = rho_dxi
        data_density_z[k] = rho_dz_lab

        min_xi_arr[k] = xi_grid[0]
        delta_xi_arr[k] = xi_grid[1] - xi_grid[0]
        min_z_arr[k] = z_grid[0]
        delta_z_arr[k] = z_grid[1] - z_grid[0]
        poly_coeffs_arr[k] = [slope, 0.0]

    # Use uniform delta_t for the interpolation
    min_t = times[0]
    dt = (times[-1] - times[0]) / (n_t - 1)

    # ======================================================================
    # Query points mimicking CSR retarded-time queries:
    # Large x range (following the tilt), large z range (formation length),
    # many points OOB relative to any single timestep's grid
    # ======================================================================

    # Query at t_ret = 0.55 (between step 3 and 4, where slope flips from -5.3 to -9.9)
    t_query_val = 0.55
    # CSR queries have z_ret spanning a wide range (much wider than beam sigma_z)
    z_query_1d = np.linspace(-500e-6, 500e-6, 50)  # 10x sigma_z — many OOB
    # x queries follow the beam tilt
    x_query_1d = np.linspace(-3000e-6, 3000e-6, 50)  # wide range

    X_q, Z_q = np.meshgrid(x_query_1d, z_query_1d, indexing='ij')
    x_query = X_q.ravel()
    z_query = Z_q.ravel()
    n_pts = len(x_query)
    t_query = np.full(n_pts, t_query_val)

    # Compute analytical answer: blend between bracketing timesteps
    k_low = int((t_query_val - min_t) / dt)
    alpha = (t_query_val - (min_t + k_low * dt)) / dt
    slope_k = slopes[k_low]
    slope_k1 = slopes[k_low + 1]
    sz_k = sigma_z_per_step[k_low]
    sz_k1 = sigma_z_per_step[k_low + 1]

    def analytical_rho(x, z, slope, sz):
        xi = x - slope * z
        return np.exp(-xi**2 / (2 * sigma_xi**2) - z**2 / (2 * sz**2))

    def analytical_rho_x(x, z, slope, sz):
        xi = x - slope * z
        rho = np.exp(-xi**2 / (2 * sigma_xi**2) - z**2 / (2 * sz**2))
        return -xi / sigma_xi**2 * rho

    def analytical_rho_z(x, z, slope, sz):
        xi = x - slope * z
        rho = np.exp(-xi**2 / (2 * sigma_xi**2) - z**2 / (2 * sz**2))
        drho_dxi = -xi / sigma_xi**2 * rho
        drho_dz_xi = -z / sz**2 * rho
        return drho_dz_xi - slope * drho_dxi

    rho_exact = (1 - alpha) * analytical_rho(x_query, z_query, slope_k, sz_k) + \
                alpha * analytical_rho(x_query, z_query, slope_k1, sz_k1)
    rho_x_exact = (1 - alpha) * analytical_rho_x(x_query, z_query, slope_k, sz_k) + \
                  alpha * analytical_rho_x(x_query, z_query, slope_k1, sz_k1)
    rho_z_exact = (1 - alpha) * analytical_rho_z(x_query, z_query, slope_k, sz_k) + \
                  alpha * analytical_rho_z(x_query, z_query, slope_k1, sz_k1)

    # Count how many are OOB at each timestep
    print(f"\n  Query: t={t_query_val}, blending k={k_low} (slope={slope_k}) and k={k_low+1} (slope={slope_k1})")
    print(f"  alpha={alpha:.3f}")
    print(f"  x range: [{x_query.min()*1e6:.0f}, {x_query.max()*1e6:.0f}] um")
    print(f"  z range: [{z_query.min()*1e6:.0f}, {z_query.max()*1e6:.0f}] um")
    print(f"  Total query points: {n_pts}")

    for k in [k_low, k_low + 1]:
        xi_at_k = x_query - slopes[k] * z_query
        oob_xi = (xi_at_k < min_xi_arr[k]) | (xi_at_k > min_xi_arr[k] + delta_xi_arr[k] * 199)
        oob_z = (z_query < min_z_arr[k]) | (z_query > min_z_arr[k] + delta_z_arr[k] * 199)
        oob = oob_xi | oob_z
        print(f"  Timestep k={k}: OOB_xi={oob_xi.sum()/n_pts:.1%}, OOB_z={oob_z.sum()/n_pts:.1%}, OOB_total={oob.sum()/n_pts:.1%}")

    # How many query points have significant density?
    sig_mask = rho_exact > 0.01 * rho_exact.max()
    print(f"  Points with significant density: {sig_mask.sum()} / {n_pts} ({sig_mask.sum()/n_pts:.1%})")

    # ======================================================================
    # Interpolate with new method
    # ======================================================================
    rho_new = interpolate3D_transformed(x_query, z_query, t_query,
                                         data_density, poly_coeffs_arr,
                                         min_xi_arr, min_z_arr, min_t,
                                         delta_xi_arr, delta_z_arr, dt)
    rho_x_new = interpolate3D_transformed(x_query, z_query, t_query,
                                           data_density_x, poly_coeffs_arr,
                                           min_xi_arr, min_z_arr, min_t,
                                           delta_xi_arr, delta_z_arr, dt)
    rho_z_new = interpolate3D_transformed(x_query, z_query, t_query,
                                           data_density_z, poly_coeffs_arr,
                                           min_xi_arr, min_z_arr, min_t,
                                           delta_xi_arr, delta_z_arr, dt)

    # ======================================================================
    # Compare
    # ======================================================================
    def rel_err(approx, exact, m):
        norm = np.linalg.norm(exact[m])
        if norm == 0:
            return 0
        return np.linalg.norm(approx[m] - exact[m]) / norm

    print(f"\n  Results (only at points with significant density):")
    print(f"  {'Field':<12} | {'New vs Exact':<14} | {'Max |diff|/|exact|':<20}")
    print(f"  {'-'*50}")

    for name, new_arr, exact_arr in [('ρ', rho_new, rho_exact),
                                      ('∂ρ/∂x', rho_x_new, rho_x_exact),
                                      ('∂ρ/∂z', rho_z_new, rho_z_exact)]:
        rel = rel_err(new_arr, exact_arr, sig_mask)
        # Max pointwise relative error
        denom = np.abs(exact_arr[sig_mask])
        denom[denom < 1e-30] = 1e-30
        max_rel = np.max(np.abs(new_arr[sig_mask] - exact_arr[sig_mask]) / denom)
        print(f"  {name:<12} | {rel:<14.6f} | {max_rel:<20.6f}")

    # Also test at multiple query times spanning the full range
    print(f"\n  Sweep across query times:")
    print(f"  {'t_query':<8} | {'k':<3} | {'alpha':<6} | {'ρ rel err':<12} | {'∂ρ/∂x rel':<12} | {'∂ρ/∂z rel':<12}")
    print(f"  {'-'*65}")

    for t_q in np.linspace(times[0] + 0.01, times[-1] - 0.01, 10):
        t_arr = np.full(n_pts, t_q)
        k_l = int((t_q - min_t) / dt)
        k_l = min(k_l, n_t - 2)
        a = (t_q - (min_t + k_l * dt)) / dt

        rho_ex = (1 - a) * analytical_rho(x_query, z_query, slopes[k_l], sigma_z_per_step[k_l]) + \
                 a * analytical_rho(x_query, z_query, slopes[k_l+1], sigma_z_per_step[k_l+1])
        rho_x_ex = (1 - a) * analytical_rho_x(x_query, z_query, slopes[k_l], sigma_z_per_step[k_l]) + \
                   a * analytical_rho_x(x_query, z_query, slopes[k_l+1], sigma_z_per_step[k_l+1])
        rho_z_ex = (1 - a) * analytical_rho_z(x_query, z_query, slopes[k_l], sigma_z_per_step[k_l]) + \
                   a * analytical_rho_z(x_query, z_query, slopes[k_l+1], sigma_z_per_step[k_l+1])

        mask = rho_ex > 0.01 * rho_ex.max()
        if mask.sum() == 0:
            continue

        r_new = interpolate3D_transformed(x_query, z_query, t_arr, data_density, poly_coeffs_arr,
                                           min_xi_arr, min_z_arr, min_t, delta_xi_arr, delta_z_arr, dt)
        rx_new = interpolate3D_transformed(x_query, z_query, t_arr, data_density_x, poly_coeffs_arr,
                                            min_xi_arr, min_z_arr, min_t, delta_xi_arr, delta_z_arr, dt)
        rz_new = interpolate3D_transformed(x_query, z_query, t_arr, data_density_z, poly_coeffs_arr,
                                            min_xi_arr, min_z_arr, min_t, delta_xi_arr, delta_z_arr, dt)

        print(f"  {t_q:<8.4f} | {k_l:<3} | {a:<6.3f} | {rel_err(r_new, rho_ex, mask):<12.6f} | {rel_err(rx_new, rho_x_ex, mask):<12.6f} | {rel_err(rz_new, rho_z_ex, mask):<12.6f}")


if __name__ == '__main__':
    run_test()
