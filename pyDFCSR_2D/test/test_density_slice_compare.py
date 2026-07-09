"""
Direct comparison of stored density slices between legacy and new methods.

For each timestep:
1. Legacy: ρ(x, z) on shared grid
2. New: ρ(ξ, z) on per-timestep grid, with ξ = x - poly(z)

Transform new method's ρ(ξ, z) → ρ(x, z) by inverting the poly transform,
interpolate to legacy's grid, and compare.
"""
import sys, os
import numpy as np
from scipy.interpolate import RegularGridInterpolator

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..', 'example'))

from pyDFCSR_2D.CSR import CSR2D
from pyDFCSR_2D.deposit_smooth import DF_tracker_smooth

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RESULT_DIR = os.path.join(os.path.dirname(__file__), 'benchmark_results', 'density_compare')
os.makedirs(RESULT_DIR, exist_ok=True)

dep_config = {
    'method': 'bspline_fft',
    'xbins': 200, 'zbins': 200,
    'xlim': 5, 'zlim': 5,
    'smoothing_sigma': 3.0,
    'poly_degree': 3,
    'velocity_threhold': 1000,
}

TARGET_STEP = 6  # where slope is ~ -10


def run_and_capture_slices(method):
    """Run to target step, capture density slices at each deposition step.
    CSR is computed but NOT applied, so beam evolution is identical."""
    csr = CSR2D(input_file='input/dipole_chirp500_config.yaml')
    csr.CSR_params.apply_CSR = 0  # Don't apply kicks — beam stays identical

    if method == 'bspline_fft':
        csr.DF_tracker = DF_tracker_smooth(dep_config)
        csr.use_smooth_deposit = True
        csr.DF_tracker.get_DF(x=csr.beam.x, z=csr.beam.z, px=csr.beam.px, t=csr.beam.position)
        csr.DF_tracker.append_DF()
        csr.DF_tracker.append_interpolant(formation_length=float('inf'),
                                           n_formation_length=csr.integration_params.n_formation_length)

    # Capture density at each step
    slices = []
    original_get_DF = csr.DF_tracker.get_DF

    def capture_get_DF(x, z, px, t):
        original_get_DF(x, z, px, t)
        tracker = csr.DF_tracker
        if method == 'bspline_fft':
            slices.append({
                'step': len(slices) + 1,
                't': t,
                'xi_grids': tracker.xi_grids.copy(),
                'z_grids': tracker.z_grids.copy(),
                'density': tracker.density.copy(),
                'density_x': tracker.density_x.copy(),
                'density_z': tracker.density_z.copy(),
                'poly_coeffs': tracker.poly_coeffs.copy(),
            })
        else:
            slices.append({
                'step': len(slices) + 1,
                't': t,
                'x_grids': tracker.x_grids.copy(),
                'z_grids': tracker.z_grids.copy(),
                'density': tracker.density.copy(),
                'density_x': tracker.density_x.copy(),
                'density_z': tracker.density_z.copy(),
            })

    csr.DF_tracker.get_DF = capture_get_DF
    csr.run(stop_time=(TARGET_STEP + 0.5) * 0.1)
    return slices, csr


def compare_slice(leg_slice, new_slice, step_idx):
    """Compare one density slice between legacy and new."""
    print(f"\n{'='*60}")
    print(f"Step {step_idx}: t={leg_slice['t']:.4f}")
    print(f"{'='*60}")

    # Legacy grid
    x_leg = leg_slice['x_grids']
    z_leg = leg_slice['z_grids']
    rho_leg = leg_slice['density']
    rho_x_leg = leg_slice['density_x']
    rho_z_leg = leg_slice['density_z']

    # New method grid (in ξ-frame)
    xi_new = new_slice['xi_grids']
    z_new = new_slice['z_grids']
    rho_new_xi = new_slice['density']
    rho_x_new_xi = new_slice['density_x']  # ∂ρ/∂ξ
    rho_z_new_xi = new_slice['density_z']  # ∂ρ/∂z_lab (chain rule already applied)
    poly = new_slice['poly_coeffs']

    print(f"  Legacy grid: x=[{x_leg[0]*1e6:.1f}, {x_leg[-1]*1e6:.1f}]um ({len(x_leg)} pts), z=[{z_leg[0]*1e6:.1f}, {z_leg[-1]*1e6:.1f}]um ({len(z_leg)} pts)")
    print(f"  New grid:    ξ=[{xi_new[0]*1e6:.1f}, {xi_new[-1]*1e6:.1f}]um ({len(xi_new)} pts), z=[{z_new[0]*1e6:.1f}, {z_new[-1]*1e6:.1f}]um ({len(z_new)} pts)")
    print(f"  Poly coeffs: {poly}")
    poly_deriv = np.polyder(poly)
    print(f"  Poly'(0) = {np.polyval(poly_deriv, 0):.4f} (slope at z=0)")

    # Transform new method to lab frame (x, z):
    # For each point (ξ_i, z_j) in new grid, the lab-frame x = ξ_i + poly(z_j)
    # We need to evaluate new method's ρ at legacy's (x, z) grid points.
    # At legacy grid point (x_k, z_l): ξ = x_k - poly(z_l), then look up ρ(ξ, z_l) on new grid.

    # Build interpolator for new method in (ξ, z) space
    rho_interp = RegularGridInterpolator((xi_new, z_new), rho_new_xi,
                                          bounds_error=False, fill_value=0.0)
    rho_x_interp = RegularGridInterpolator((xi_new, z_new), rho_x_new_xi,
                                            bounds_error=False, fill_value=0.0)
    rho_z_interp = RegularGridInterpolator((xi_new, z_new), rho_z_new_xi,
                                            bounds_error=False, fill_value=0.0)

    # Evaluate at legacy grid points
    X_leg, Z_leg = np.meshgrid(x_leg, z_leg, indexing='ij')
    XI_at_leg = X_leg - np.polyval(poly, Z_leg)  # transform to ξ-frame

    # Query new method's interpolator
    query_pts = np.column_stack([XI_at_leg.ravel(), Z_leg.ravel()])
    rho_new_at_leg = rho_interp(query_pts).reshape(X_leg.shape)
    rho_x_new_at_leg = rho_x_interp(query_pts).reshape(X_leg.shape)
    rho_z_new_at_leg = rho_z_interp(query_pts).reshape(X_leg.shape)

    # Compare where density is significant
    mask = rho_leg > 0.01 * rho_leg.max()
    n_sig = mask.sum()

    def rel_err(a, b, m):
        norm = np.linalg.norm(b[m])
        if norm == 0:
            return 0
        return np.linalg.norm(a[m] - b[m]) / norm

    print(f"\n  Significant points: {n_sig} / {rho_leg.size}")
    print(f"  {'Field':<12} | {'Rel L2 diff':<14} | {'Legacy peak':<14} | {'New peak':<14} | {'Sign match?'}")
    print(f"  {'-'*75}")

    r_rho = rel_err(rho_new_at_leg, rho_leg, mask)
    r_rx = rel_err(rho_x_new_at_leg, rho_x_leg, mask)
    r_rz = rel_err(rho_z_new_at_leg, rho_z_leg, mask)

    # Check sign agreement at peak
    peak_idx = np.unravel_index(np.argmax(np.abs(rho_x_leg)), rho_x_leg.shape)
    sign_x = "YES" if np.sign(rho_x_new_at_leg[peak_idx]) == np.sign(rho_x_leg[peak_idx]) else "NO"
    peak_idx_z = np.unravel_index(np.argmax(np.abs(rho_z_leg)), rho_z_leg.shape)
    sign_z = "YES" if np.sign(rho_z_new_at_leg[peak_idx_z]) == np.sign(rho_z_leg[peak_idx_z]) else "NO"

    print(f"  {'ρ':<12} | {r_rho:<14.4f} | {rho_leg.max():<14.4e} | {rho_new_at_leg[mask].max():<14.4e} | ---")
    print(f"  {'∂ρ/∂x':<12} | {r_rx:<14.4f} | {rho_x_leg[peak_idx]:<14.4e} | {rho_x_new_at_leg[peak_idx]:<14.4e} | {sign_x}")
    print(f"  {'∂ρ/∂z':<12} | {r_rz:<14.4f} | {rho_z_leg[peak_idx_z]:<14.4e} | {rho_z_new_at_leg[peak_idx_z]:<14.4e} | {sign_z}")

    # Plot comparison for this step
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle(f'Step {step_idx}, t={leg_slice["t"]:.4f}, poly\'(0)={np.polyval(poly_deriv, 0):.2f}', fontsize=13)

    extent = [z_leg[0]*1e6, z_leg[-1]*1e6, x_leg[0]*1e6, x_leg[-1]*1e6]

    fields = [
        (rho_leg, rho_new_at_leg, 'ρ'),
        (rho_x_leg, rho_x_new_at_leg, '∂ρ/∂x'),
        (rho_z_leg, rho_z_new_at_leg, '∂ρ/∂z'),
    ]

    for row, (leg_f, new_f, label) in enumerate(fields):
        vmax = max(abs(leg_f).max(), 1e-30)
        im = axes[row, 0].imshow(leg_f, origin='lower', extent=extent, aspect='auto',
                                  cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        axes[row, 0].set_title(f'{label} (legacy)')
        plt.colorbar(im, ax=axes[row, 0])

        im = axes[row, 1].imshow(new_f, origin='lower', extent=extent, aspect='auto',
                                  cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        axes[row, 1].set_title(f'{label} (new→legacy grid)')
        plt.colorbar(im, ax=axes[row, 1])

        diff = new_f - leg_f
        vmax_d = max(abs(diff).max(), 1e-30)
        im = axes[row, 2].imshow(diff, origin='lower', extent=extent, aspect='auto',
                                  cmap='RdBu_r', vmin=-vmax_d, vmax=vmax_d)
        axes[row, 2].set_title(f'Δ{label} (new - legacy)')
        plt.colorbar(im, ax=axes[row, 2])

    for ax in axes[:, 0]:
        ax.set_ylabel('x (μm)')
    for ax in axes[2, :]:
        ax.set_xlabel('z (μm)')

    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, f'density_compare_step{step_idx:02d}.png'), dpi=120)
    plt.close()

    return r_rho, r_rx, r_rz


def main():
    print("="*70)
    print("Direct Density Slice Comparison: Legacy vs New (chirp=500)")
    print("="*70)

    print("\nRunning legacy...")
    leg_slices, csr_leg = run_and_capture_slices('legacy')
    print(f"  Captured {len(leg_slices)} slices")

    print("\nRunning new...")
    new_slices, csr_new = run_and_capture_slices('bspline_fft')
    print(f"  Captured {len(new_slices)} slices")

    # Compare each step
    n = min(len(leg_slices), len(new_slices))
    print(f"\nComparing {n} slices...")

    for i in range(n):
        compare_slice(leg_slices[i], new_slices[i], i + 1)


if __name__ == '__main__':
    main()
