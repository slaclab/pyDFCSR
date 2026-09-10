"""
Test interpolation accuracy with tilted 2D Gaussians.

Creates a time series of 2D Gaussians with varying x-z tilt (matching high-chirp regime).
Compares legacy (shared grid) and new (per-timestep polynomial transform) interpolation
against the known analytical values.
"""
import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from pyDFCSR_2D.interp3D import interpolate3D, interpolate3D_transformed
from pyDFCSR_2D.deposit_smooth import smooth_and_differentiate

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RESULT_DIR = os.path.join(os.path.dirname(__file__), 'benchmark_results', 'interp_test')
os.makedirs(RESULT_DIR, exist_ok=True)


def make_tilted_gaussian(x_grid, z_grid, slope, sigma_xi=50e-6, sigma_z=50e-6):
    """
    Create a 2D Gaussian tilted in x-z space.
    rho(x, z) = exp(-xi^2/(2*sigma_xi^2) - z^2/(2*sigma_z^2))
    where xi = x - slope * z
    """
    X, Z = np.meshgrid(x_grid, z_grid, indexing='ij')
    XI = X - slope * Z
    rho = np.exp(-XI**2 / (2 * sigma_xi**2) - Z**2 / (2 * sigma_z**2))
    # Normalize
    rho /= (rho.sum() * (x_grid[1] - x_grid[0]) * (z_grid[1] - z_grid[0]))
    return rho


def analytical_density(x, z, slope, sigma_xi=50e-6, sigma_z=50e-6):
    """Evaluate tilted Gaussian analytically at arbitrary (x, z)."""
    xi = x - slope * z
    return np.exp(-xi**2 / (2 * sigma_xi**2) - z**2 / (2 * sigma_z**2))


def analytical_density_x(x, z, slope, sigma_xi=50e-6, sigma_z=50e-6):
    """∂ρ/∂x = ∂ρ/∂ξ * ∂ξ/∂x = ∂ρ/∂ξ (since ∂ξ/∂x = 1)"""
    xi = x - slope * z
    rho = np.exp(-xi**2 / (2 * sigma_xi**2) - z**2 / (2 * sigma_z**2))
    return -xi / sigma_xi**2 * rho


def analytical_density_z(x, z, slope, sigma_xi=50e-6, sigma_z=50e-6):
    """∂ρ/∂z|_x = ∂ρ/∂z|_ξ + ∂ρ/∂ξ * (-slope)"""
    xi = x - slope * z
    rho = np.exp(-xi**2 / (2 * sigma_xi**2) - z**2 / (2 * sigma_z**2))
    drho_dxi = -xi / sigma_xi**2 * rho
    drho_dz_at_xi = -z / sigma_z**2 * rho
    return drho_dz_at_xi + drho_dxi * (-slope)


def run_test():
    sigma_xi = 50e-6
    sigma_z = 50e-6

    # Tilt sequence matching chirp=500 case behavior:
    # slope evolves: starts at 0, grows to +5, swings to -10, settles to -5
    n_t = 5
    slopes = [0.0, 2.0, 5.5, -10.0, -5.0]
    times = np.array([0.0, 0.1, 0.2, 0.3, 0.4])  # uniform spacing
    delta_t = 0.1

    print("="*70)
    print("Test: Interpolation of tilted 2D Gaussians")
    print("="*70)
    print(f"\nSlopes per timestep: {slopes}")
    print(f"sigma_xi = {sigma_xi*1e6:.0f} um, sigma_z = {sigma_z*1e6:.0f} um")

    # ======================================================================
    # Setup NEW method: per-timestep ξ-frames
    # ======================================================================
    n_bins = 200
    xlim = 5  # grid covers ±xlim*sigma

    # Per-timestep data in ξ-frames
    data_density_new = np.zeros((n_t, n_bins, n_bins))
    data_density_x_new = np.zeros((n_t, n_bins, n_bins))
    data_density_z_new = np.zeros((n_t, n_bins, n_bins))
    min_xi_arr = np.zeros(n_t)
    delta_xi_arr = np.zeros(n_t)
    min_z_arr = np.zeros(n_t)
    delta_z_arr = np.zeros(n_t)
    poly_coeffs_arr = np.zeros((n_t, 2))  # linear polynomial [slope, intercept=0]

    for k in range(n_t):
        slope = slopes[k]
        # In the ξ-frame (after tilt removal), the Gaussian is untilted
        xi_grid = np.linspace(-xlim * sigma_xi, xlim * sigma_xi, n_bins)
        z_grid = np.linspace(-xlim * sigma_z, xlim * sigma_z, n_bins)
        dx = xi_grid[1] - xi_grid[0]
        dz = z_grid[1] - z_grid[0]

        # Density in ξ-frame: Gaussian centered at origin, no tilt
        XI, Z = np.meshgrid(xi_grid, z_grid, indexing='ij')
        rho = np.exp(-XI**2 / (2 * sigma_xi**2) - Z**2 / (2 * sigma_z**2))

        # Spectral derivatives in ξ-frame
        rho_dxi = -XI / sigma_xi**2 * rho  # ∂ρ/∂ξ
        rho_dz_xi = -Z / sigma_z**2 * rho  # ∂ρ/∂z at fixed ξ

        # Chain rule correction (as done in deposit_smooth.py):
        # ∂ρ/∂z|_x = ∂ρ/∂z|_ξ - slope * ∂ρ/∂ξ
        rho_dz_lab = rho_dz_xi - slope * rho_dxi

        data_density_new[k] = rho
        data_density_x_new[k] = rho_dxi
        data_density_z_new[k] = rho_dz_lab

        min_xi_arr[k] = xi_grid[0]
        delta_xi_arr[k] = dx
        min_z_arr[k] = z_grid[0]
        delta_z_arr[k] = dz
        poly_coeffs_arr[k] = [slope, 0.0]  # x = slope * z + 0 (linear, no intercept)

    # ======================================================================
    # Setup LEGACY method: shared (x, z) grid covering all tilts
    # ======================================================================
    # The shared grid must cover the full x extent across all tilts
    # At z = ±xlim*sigma_z with slope=10, x ranges ±(xlim*sigma_z*10 + xlim*sigma_xi)
    max_slope = max(abs(s) for s in slopes)
    x_extent = xlim * sigma_z * max_slope + xlim * sigma_xi
    n_bins_legacy = 500  # use more bins for shared grid (like the legacy code does)

    x_grid_shared = np.linspace(-x_extent, x_extent, n_bins_legacy)
    z_grid_shared = np.linspace(-xlim * sigma_z, xlim * sigma_z, n_bins_legacy)
    dx_shared = x_grid_shared[1] - x_grid_shared[0]
    dz_shared = z_grid_shared[1] - z_grid_shared[0]

    data_density_leg = np.zeros((n_t, n_bins_legacy, n_bins_legacy))
    data_density_x_leg = np.zeros((n_t, n_bins_legacy, n_bins_legacy))
    data_density_z_leg = np.zeros((n_t, n_bins_legacy, n_bins_legacy))

    for k in range(n_t):
        slope = slopes[k]
        X, Z = np.meshgrid(x_grid_shared, z_grid_shared, indexing='ij')
        XI = X - slope * Z
        rho = np.exp(-XI**2 / (2 * sigma_xi**2) - Z**2 / (2 * sigma_z**2))

        # Lab-frame derivatives (direct, no coordinate transform needed)
        drho_dx = -XI / sigma_xi**2 * rho  # = ∂ρ/∂ξ since ∂ξ/∂x = 1
        drho_dz = -Z / sigma_z**2 * rho + (-slope) * (-XI / sigma_xi**2) * rho
        # = ∂ρ/∂z|_ξ - slope * ∂ρ/∂ξ

        data_density_leg[k] = rho
        data_density_x_leg[k] = drho_dx
        data_density_z_leg[k] = drho_dz

    # ======================================================================
    # Query points: grid of (x, z) at intermediate times
    # ======================================================================
    n_query = 50
    x_query_1d = np.linspace(-3 * sigma_xi, 3 * sigma_xi, n_query)
    z_query_1d = np.linspace(-3 * sigma_z, 3 * sigma_z, n_query)
    X_q, Z_q = np.meshgrid(x_query_1d, z_query_1d, indexing='ij')
    x_query = X_q.ravel()
    z_query = Z_q.ravel()
    n_pts = len(x_query)

    # Test at t = 0.25 (between timestep 2 and 3, where slope goes from +5.5 to -10)
    t_test = 0.25
    t_query = np.full(n_pts, t_test)

    # Blending weight
    k_low = int((t_test - times[0]) / delta_t)
    alpha = (t_test - times[k_low]) / delta_t
    slope_blended = (1 - alpha) * slopes[k_low] + alpha * slopes[k_low + 1]

    print(f"\nQuery time: t={t_test} (between step {k_low} and {k_low+1})")
    print(f"  slope[{k_low}] = {slopes[k_low]}, slope[{k_low+1}] = {slopes[k_low+1]}")
    print(f"  alpha = {alpha:.2f}, blended slope = {slope_blended:.2f}")

    # Reference: the TRUE density at the query time, i.e. a Gaussian sheared by
    # the time-interpolated slope. slope(t) is piecewise linear in t here, so on
    # each interval this is the exact intermediate-time distribution.
    rho_exact = analytical_density(x_query, z_query, slope_blended, sigma_xi, sigma_z)
    drho_dx_exact = analytical_density_x(x_query, z_query, slope_blended, sigma_xi, sigma_z)
    drho_dz_exact = analytical_density_z(x_query, z_query, slope_blended, sigma_xi, sigma_z)

    # The reference this test USED to assert against: the lab-frame blend of the
    # two snapshots. That is the approximation under test, not the truth, so it
    # is reported for contrast only. See test_ghosting.py.
    rho_blend = ((1 - alpha) * analytical_density(x_query, z_query, slopes[k_low], sigma_xi, sigma_z)
                 + alpha * analytical_density(x_query, z_query, slopes[k_low + 1], sigma_xi, sigma_z))

    # ======================================================================
    # Interpolate with NEW method
    # ======================================================================
    rho_new = interpolate3D_transformed(
        xval=x_query, zval=z_query, tval=t_query,
        data=data_density_new, poly_coeffs=poly_coeffs_arr,
        min_xi_arr=min_xi_arr, min_z_arr=min_z_arr, min_t=times[0],
        delta_xi_arr=delta_xi_arr, delta_z_arr=delta_z_arr, delta_t=delta_t)

    drho_dx_new = interpolate3D_transformed(
        xval=x_query, zval=z_query, tval=t_query,
        data=data_density_x_new, poly_coeffs=poly_coeffs_arr,
        min_xi_arr=min_xi_arr, min_z_arr=min_z_arr, min_t=times[0],
        delta_xi_arr=delta_xi_arr, delta_z_arr=delta_z_arr, delta_t=delta_t)

    drho_dz_new = interpolate3D_transformed(
        xval=x_query, zval=z_query, tval=t_query,
        data=data_density_z_new, poly_coeffs=poly_coeffs_arr,
        min_xi_arr=min_xi_arr, min_z_arr=min_z_arr, min_t=times[0],
        delta_xi_arr=delta_xi_arr, delta_z_arr=delta_z_arr, delta_t=delta_t)

    # ======================================================================
    # Interpolate with LEGACY method
    # ======================================================================
    rho_leg = interpolate3D(
        xval=t_query, yval=x_query, zval=z_query,
        data=data_density_leg,
        min_x=times[0], min_y=x_grid_shared[0], min_z=z_grid_shared[0],
        delta_x=delta_t, delta_y=dx_shared, delta_z=dz_shared)

    drho_dx_leg = interpolate3D(
        xval=t_query, yval=x_query, zval=z_query,
        data=data_density_x_leg,
        min_x=times[0], min_y=x_grid_shared[0], min_z=z_grid_shared[0],
        delta_x=delta_t, delta_y=dx_shared, delta_z=dz_shared)

    drho_dz_leg = interpolate3D(
        xval=t_query, yval=x_query, zval=z_query,
        data=data_density_z_leg,
        min_x=times[0], min_y=x_grid_shared[0], min_z=z_grid_shared[0],
        delta_x=delta_t, delta_y=dx_shared, delta_z=dz_shared)

    # ======================================================================
    # Compare
    # ======================================================================
    # Mask: only compare where signal is significant
    mask = rho_exact > 0.01 * rho_exact.max()
    n_sig = mask.sum()

    def rel_err(approx, exact, m):
        norm = np.linalg.norm(exact[m])
        if norm == 0:
            return 0
        return np.linalg.norm(approx[m] - exact[m]) / norm

    print(f"\n{'Field':<12} | {'New vs Exact':<14} | {'Legacy vs Exact':<16} | {'New vs Legacy':<14}")
    print("-" * 65)
    print(f"{'rho':<12} | {rel_err(rho_new, rho_exact, mask):<14.6f} | {rel_err(rho_leg, rho_exact, mask):<16.6f} | {rel_err(rho_new, rho_leg, mask):<14.6f}")
    print(f"{'drho/dx':<12} | {rel_err(drho_dx_new, drho_dx_exact, mask):<14.6f} | {rel_err(drho_dx_leg, drho_dx_exact, mask):<16.6f} | {rel_err(drho_dx_new, drho_dx_leg, mask):<14.6f}")
    print(f"{'drho/dz':<12} | {rel_err(drho_dz_new, drho_dz_exact, mask):<14.6f} | {rel_err(drho_dz_leg, drho_dz_exact, mask):<16.6f} | {rel_err(drho_dz_new, drho_dz_leg, mask):<14.6f}")

    G = abs(slopes[k_low + 1] - slopes[k_low]) * sigma_z / sigma_xi
    print(f"\nGhosting parameter G = |d slope| sigma_z/sigma_xi = {G:.1f}")
    print(f"Old reference (lab-frame blend) vs truth: {rel_err(rho_blend, rho_exact, mask):.6f}")
    print("  -- that is how wrong this test's former 'exact' answer was. Because the")
    print("     interpolator computes the same blend, the old test reported ~0 error.")

    # Also test at t = 0.15 (between step 1 and 2, slope 2 → 5.5, moderate)
    t_test2 = 0.15
    t_query2 = np.full(n_pts, t_test2)
    k2 = int((t_test2 - times[0]) / delta_t)
    alpha2 = (t_test2 - times[k2]) / delta_t

    slope_blended2 = (1 - alpha2) * slopes[k2] + alpha2 * slopes[k2 + 1]
    rho_exact2 = analytical_density(x_query, z_query, slope_blended2)
    drho_dx_exact2 = analytical_density_x(x_query, z_query, slope_blended2)
    drho_dz_exact2 = analytical_density_z(x_query, z_query, slope_blended2)

    rho_new2 = interpolate3D_transformed(x_query, z_query, t_query2, data_density_new, poly_coeffs_arr,
                                          min_xi_arr, min_z_arr, times[0], delta_xi_arr, delta_z_arr, delta_t)
    drho_dx_new2 = interpolate3D_transformed(x_query, z_query, t_query2, data_density_x_new, poly_coeffs_arr,
                                              min_xi_arr, min_z_arr, times[0], delta_xi_arr, delta_z_arr, delta_t)
    drho_dz_new2 = interpolate3D_transformed(x_query, z_query, t_query2, data_density_z_new, poly_coeffs_arr,
                                              min_xi_arr, min_z_arr, times[0], delta_xi_arr, delta_z_arr, delta_t)

    rho_leg2 = interpolate3D(t_query2, x_query, z_query, data_density_leg,
                              times[0], x_grid_shared[0], z_grid_shared[0], delta_t, dx_shared, dz_shared)
    drho_dx_leg2 = interpolate3D(t_query2, x_query, z_query, data_density_x_leg,
                                  times[0], x_grid_shared[0], z_grid_shared[0], delta_t, dx_shared, dz_shared)
    drho_dz_leg2 = interpolate3D(t_query2, x_query, z_query, data_density_z_leg,
                                  times[0], x_grid_shared[0], z_grid_shared[0], delta_t, dx_shared, dz_shared)

    mask2 = rho_exact2 > 0.01 * rho_exact2.max()

    print(f"\nAt t={t_test2} (slope {slopes[k2]} -> {slopes[k2+1]}, moderate change):")
    print(f"{'Field':<12} | {'New vs Exact':<14} | {'Legacy vs Exact':<16} | {'New vs Legacy':<14}")
    print("-" * 65)
    print(f"{'rho':<12} | {rel_err(rho_new2, rho_exact2, mask2):<14.6f} | {rel_err(rho_leg2, rho_exact2, mask2):<16.6f} | {rel_err(rho_new2, rho_leg2, mask2):<14.6f}")
    print(f"{'drho/dx':<12} | {rel_err(drho_dx_new2, drho_dx_exact2, mask2):<14.6f} | {rel_err(drho_dx_leg2, drho_dx_exact2, mask2):<16.6f} | {rel_err(drho_dx_new2, drho_dx_leg2, mask2):<14.6f}")
    print(f"{'drho/dz':<12} | {rel_err(drho_dz_new2, drho_dz_exact2, mask2):<14.6f} | {rel_err(drho_dz_leg2, drho_dz_exact2, mask2):<16.6f} | {rel_err(drho_dz_new2, drho_dz_leg2, mask2):<14.6f}")

    # Plot comparison for the high-tilt case
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle(f't={t_test}, slope {slopes[k_low]}→{slopes[k_low+1]} (blended={slope_blended:.1f})', fontsize=14)

    fields = [
        (rho_exact, rho_new, rho_leg, 'ρ'),
        (drho_dx_exact, drho_dx_new, drho_dx_leg, '∂ρ/∂x'),
        (drho_dz_exact, drho_dz_new, drho_dz_leg, '∂ρ/∂z'),
    ]

    for row, (exact, new, leg, label) in enumerate(fields):
        exact_2d = exact.reshape(n_query, n_query)
        new_2d = new.reshape(n_query, n_query)
        leg_2d = leg.reshape(n_query, n_query)

        ext = [z_query_1d[0]*1e6, z_query_1d[-1]*1e6, x_query_1d[0]*1e6, x_query_1d[-1]*1e6]

        vmax = max(abs(exact_2d).max(), 1e-30)
        im = axes[row, 0].imshow(exact_2d, origin='lower', extent=ext, aspect='auto',
                                  cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        axes[row, 0].set_title(f'{label} (exact)')
        plt.colorbar(im, ax=axes[row, 0])

        diff_new = new_2d - exact_2d
        vmax_d = max(abs(diff_new).max(), 1e-30)
        im = axes[row, 1].imshow(diff_new, origin='lower', extent=ext, aspect='auto',
                                  cmap='RdBu_r', vmin=-vmax_d, vmax=vmax_d)
        axes[row, 1].set_title(f'New - Exact ({label})')
        plt.colorbar(im, ax=axes[row, 1])

        diff_leg = leg_2d - exact_2d
        vmax_d2 = max(abs(diff_leg).max(), 1e-30)
        im = axes[row, 2].imshow(diff_leg, origin='lower', extent=ext, aspect='auto',
                                  cmap='RdBu_r', vmin=-vmax_d2, vmax=vmax_d2)
        axes[row, 2].set_title(f'Legacy - Exact ({label})')
        plt.colorbar(im, ax=axes[row, 2])

    for ax in axes[:, 0]:
        ax.set_ylabel('x (μm)')
    for ax in axes[2, :]:
        ax.set_xlabel('z (μm)')

    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, 'interp_tilted_gaussian_t025.png'), dpi=120)
    plt.close()
    print(f"\nPlot saved: {RESULT_DIR}/interp_tilted_gaussian_t025.png")


if __name__ == '__main__':
    run_test()
