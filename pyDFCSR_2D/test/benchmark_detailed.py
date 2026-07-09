"""
Detailed per-step benchmark: dump density, derivatives, and wakes at every step.
Compare legacy vs new method field-by-field.

Usage:
    conda run -n pydfcsr python pyDFCSR_2D/test/benchmark_detailed.py
"""
import sys
import os
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from pyDFCSR_2D.CSR import CSR2D
from pyDFCSR_2D.deposit_smooth import DF_tracker_smooth

BENCHMARK_DIR = os.path.join(os.path.dirname(__file__), 'benchmark_results')
EXAMPLE_DIR = os.path.join(os.path.dirname(__file__), '..', 'example')


def run_and_capture(config_file, method='legacy', deposition_config=None):
    """
    Run CSR simulation and capture density/derivative/wake data at every step.
    Returns a list of per-step dictionaries.
    """
    os.chdir(EXAMPLE_DIR)
    csr = CSR2D(input_file=config_file)

    if method == 'bspline_fft' and deposition_config:
        csr.DF_tracker = DF_tracker_smooth(deposition_config)
        csr.use_smooth_deposit = True
        csr.DF_tracker.get_DF(x=csr.beam.x, z=csr.beam.z, px=csr.beam.px, t=csr.beam.position)
        csr.DF_tracker.append_DF()
        csr.DF_tracker.append_interpolant(formation_length=float('inf'),
                                           n_formation_length=csr.integration_params.n_formation_length)

    # Monkey-patch to capture data at each step
    step_data = []
    original_get_DF = csr.DF_tracker.get_DF

    def capture_get_DF(x, z, px, t):
        original_get_DF(x, z, px, t)
        tracker = csr.DF_tracker
        step_data.append({
            'step': len(step_data),
            't': t,
            'x_grids': tracker.xi_grids.copy() if hasattr(tracker, 'xi_grids') else tracker.x_grids.copy(),
            'z_grids': tracker.z_grids.copy(),
            'density': tracker.density.copy(),
            'density_x': tracker.density_x.copy(),
            'density_z': tracker.density_z.copy(),
            'vx': tracker.vx.copy(),
            'vx_x': tracker.vx_x.copy(),
            'poly_coeffs': tracker.poly_coeffs.copy() if hasattr(tracker, 'poly_coeffs') and tracker.poly_coeffs is not None else None,
        })

    csr.DF_tracker.get_DF = capture_get_DF

    # Also capture wakes
    wake_data = []
    original_calculate = csr.calculate_2D_CSR

    def capture_calculate():
        original_calculate()
        wake_data.append({
            'step': csr.beam.step,
            'dE_dct': csr.dE_dct.copy(),
            'x_kick': csr.x_kick.copy(),
            'xmesh': csr.CSR_xmesh.copy(),
            'zmesh': csr.CSR_zmesh.copy(),
            'xrange': csr.CSR_xrange_transformed.copy(),
            'zrange': csr.CSR_zrange.copy(),
        })

    csr.calculate_2D_CSR = capture_calculate

    csr.run()
    return step_data, wake_data, csr


def plot_step_comparison(legacy_data, new_data, step_idx, case_dir):
    """Plot side-by-side comparison of all fields at a given step."""
    leg = legacy_data[step_idx]
    new = new_data[step_idx]

    fig, axes = plt.subplots(3, 5, figsize=(25, 15))
    fig.suptitle(f'Step {step_idx} (t={leg["t"]:.4f} m) — Legacy (top) vs New (middle) vs Difference (bottom)', fontsize=14)

    fields = ['density', 'density_x', 'density_z', 'vx', 'vx_x']
    titles = ['ρ', '∂ρ/∂x', '∂ρ/∂z', 'vx', '∂vx/∂x']

    for col, (field_name, title) in enumerate(zip(fields, titles)):
        leg_field = leg[field_name]
        new_field = new[field_name]
        leg_x = leg['x_grids'] * 1e6
        leg_z = leg['z_grids'] * 1e6
        new_x = new['x_grids'] * 1e6
        new_z = new['z_grids'] * 1e6

        # Legacy
        vmax = max(abs(leg_field.max()), abs(leg_field.min())) if leg_field.max() != leg_field.min() else 1
        cmap = 'RdBu_r' if col > 0 else 'viridis'
        im = axes[0, col].imshow(leg_field.T, origin='lower',
                                  extent=[leg_x[0], leg_x[-1], leg_z[0], leg_z[-1]],
                                  aspect='auto', cmap=cmap)
        axes[0, col].set_title(f'{title} (legacy)')
        plt.colorbar(im, ax=axes[0, col])

        # New
        im = axes[1, col].imshow(new_field.T, origin='lower',
                                  extent=[new_x[0], new_x[-1], new_z[0], new_z[-1]],
                                  aspect='auto', cmap=cmap)
        axes[1, col].set_title(f'{title} (new)')
        plt.colorbar(im, ax=axes[1, col])

        # Difference — only if grids are the same size
        if leg_field.shape == new_field.shape:
            diff = new_field - leg_field
            vmax_d = max(abs(diff.max()), abs(diff.min())) if diff.max() != diff.min() else 1
            im = axes[2, col].imshow(diff.T, origin='lower',
                                      extent=[leg_x[0], leg_x[-1], leg_z[0], leg_z[-1]],
                                      aspect='auto', cmap='RdBu_r', vmin=-vmax_d, vmax=vmax_d)
            axes[2, col].set_title(f'Δ{title} (new - legacy)')
            plt.colorbar(im, ax=axes[2, col])
        else:
            axes[2, col].text(0.5, 0.5, f'Different grid sizes\nleg: {leg_field.shape}\nnew: {new_field.shape}',
                             ha='center', va='center', transform=axes[2, col].transAxes)
            axes[2, col].set_title(f'Δ{title}')

    for ax in axes[:, 0]:
        ax.set_ylabel('z (μm)')
    for ax in axes[2, :]:
        ax.set_xlabel('x/ξ (μm)')

    plt.tight_layout()
    plt.savefig(os.path.join(case_dir, f'fields_step_{step_idx:03d}.png'), dpi=120)
    plt.close()


def plot_wake_comparison(legacy_wakes, new_wakes, wake_idx, case_dir):
    """Plot side-by-side comparison of wakes at a given step."""
    if wake_idx >= len(legacy_wakes) or wake_idx >= len(new_wakes):
        return

    leg = legacy_wakes[wake_idx]
    new = new_wakes[wake_idx]

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'CSR Wakes at step {leg["step"]} — Legacy vs New', fontsize=14)

    # Longitudinal wake
    xr_leg = leg['xrange'] * 1e6
    zr_leg = leg['zrange'] * 1e6
    xr_new = new['xrange'] * 1e6
    zr_new = new['zrange'] * 1e6

    vmax = max(abs(leg['dE_dct']).max(), abs(new['dE_dct']).max())
    im = axes[0, 0].imshow(leg['dE_dct'].T, origin='lower',
                            extent=[xr_leg[0], xr_leg[-1], zr_leg[0], zr_leg[-1]],
                            aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    axes[0, 0].set_title('dE/dct (legacy) [MeV/m]')
    plt.colorbar(im, ax=axes[0, 0])

    im = axes[0, 1].imshow(new['dE_dct'].T, origin='lower',
                            extent=[xr_new[0], xr_new[-1], zr_new[0], zr_new[-1]],
                            aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    axes[0, 1].set_title('dE/dct (new) [MeV/m]')
    plt.colorbar(im, ax=axes[0, 1])

    if leg['dE_dct'].shape == new['dE_dct'].shape:
        diff = new['dE_dct'] - leg['dE_dct']
        im = axes[0, 2].imshow(diff.T, origin='lower',
                                extent=[xr_leg[0], xr_leg[-1], zr_leg[0], zr_leg[-1]],
                                aspect='auto', cmap='RdBu_r')
        axes[0, 2].set_title('Δ(dE/dct) [MeV/m]')
        plt.colorbar(im, ax=axes[0, 2])

    # Transverse wake
    vmax = max(abs(leg['x_kick']).max(), abs(new['x_kick']).max())
    im = axes[1, 0].imshow(leg['x_kick'].T, origin='lower',
                            extent=[xr_leg[0], xr_leg[-1], zr_leg[0], zr_leg[-1]],
                            aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    axes[1, 0].set_title('x_kick (legacy) [MeV/m]')
    plt.colorbar(im, ax=axes[1, 0])

    im = axes[1, 1].imshow(new['x_kick'].T, origin='lower',
                            extent=[xr_new[0], xr_new[-1], zr_new[0], zr_new[-1]],
                            aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    axes[1, 1].set_title('x_kick (new) [MeV/m]')
    plt.colorbar(im, ax=axes[1, 1])

    if leg['x_kick'].shape == new['x_kick'].shape:
        diff = new['x_kick'] - leg['x_kick']
        im = axes[1, 2].imshow(diff.T, origin='lower',
                                extent=[xr_leg[0], xr_leg[-1], zr_leg[0], zr_leg[-1]],
                                aspect='auto', cmap='RdBu_r')
        axes[1, 2].set_title('Δ(x_kick) [MeV/m]')
        plt.colorbar(im, ax=axes[1, 2])

    for ax in axes[:, 0]:
        ax.set_ylabel('z (μm)')
    for ax in axes[1, :]:
        ax.set_xlabel('x (μm)')

    plt.tight_layout()
    plt.savefig(os.path.join(case_dir, f'wakes_step_{leg["step"]:03d}.png'), dpi=120)
    plt.close()


def run_detailed_benchmark(config_file, case_name):
    """Run detailed per-step benchmark for one case."""
    print(f"\n{'='*60}")
    print(f"DETAILED BENCHMARK: {case_name}")
    print(f"{'='*60}\n")

    case_dir = os.path.join(BENCHMARK_DIR, case_name + '_detailed')
    os.makedirs(case_dir, exist_ok=True)

    # Legacy run
    print("Running LEGACY...")
    t0 = time.time()
    leg_steps, leg_wakes, csr_leg = run_and_capture(config_file, method='legacy')
    time_legacy = time.time() - t0
    print(f"  Legacy: {len(leg_steps)} deposition steps, {len(leg_wakes)} wake steps, {time_legacy:.1f}s")

    # New method run
    print("Running NEW (B-spline + FFT)...")
    dep_config = {
        'method': 'bspline_fft',
        'xbins': 200, 'zbins': 200,
        'xlim': 5, 'zlim': 5,
        'smoothing_sigma': 3.0,
        'poly_degree': 3,
        'velocity_threhold': 1000,
    }
    t0 = time.time()
    new_steps, new_wakes, csr_new = run_and_capture(config_file, method='bspline_fft',
                                                     deposition_config=dep_config)
    time_new = time.time() - t0
    print(f"  New: {len(new_steps)} deposition steps, {len(new_wakes)} wake steps, {time_new:.1f}s")

    # Plot per-step comparisons (every other step)
    print(f"\nGenerating per-step field comparison plots (every other step)...")
    n_steps = min(len(leg_steps), len(new_steps))
    for i in range(0, n_steps, 2):
        plot_step_comparison(leg_steps, new_steps, i, case_dir)

    # Plot wake comparisons (every other step)
    print(f"Generating wake comparison plots (every other step)...")
    n_wakes = min(len(leg_wakes), len(new_wakes))
    for i in range(0, n_wakes, 2):
        plot_wake_comparison(leg_wakes, new_wakes, i, case_dir)

    # Summary log
    log_lines = [
        f"Detailed Benchmark: {case_name}",
        f"Config: {config_file}",
        f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"",
        f"Legacy: {time_legacy:.1f}s, {len(leg_steps)} deposit steps, {len(leg_wakes)} wake steps",
        f"New: {time_new:.1f}s, {len(new_steps)} deposit steps, {len(new_wakes)} wake steps",
        f"",
        f"Per-step field plots: fields_step_000.png to fields_step_{n_steps-1:03d}.png",
        f"Wake plots: wakes_step_*.png",
        f"",
    ]

    # Compute per-step L2 differences for density
    log_lines.append("Per-step density L2 norm (relative to legacy max):")
    for i in range(n_steps):
        if leg_steps[i]['density'].shape == new_steps[i]['density'].shape:
            diff_rho = np.linalg.norm(new_steps[i]['density'] - leg_steps[i]['density'])
            norm_rho = np.linalg.norm(leg_steps[i]['density'])
            rel = diff_rho / norm_rho if norm_rho > 0 else 0
            log_lines.append(f"  Step {i}: rel_diff = {rel:.4f}")
        else:
            log_lines.append(f"  Step {i}: grid mismatch ({leg_steps[i]['density'].shape} vs {new_steps[i]['density'].shape})")

    # Compute per-step L2 differences for wakes
    log_lines.append("")
    log_lines.append("Per-step wake L2 norm (relative to legacy):")
    n_wake_compare = min(len(leg_wakes), len(new_wakes))
    for i in range(n_wake_compare):
        leg_w = leg_wakes[i]
        new_w = new_wakes[i]
        if leg_w['dE_dct'].shape == new_w['dE_dct'].shape:
            # Longitudinal wake
            diff_dE = np.linalg.norm(new_w['dE_dct'] - leg_w['dE_dct'])
            norm_dE = np.linalg.norm(leg_w['dE_dct'])
            rel_dE = diff_dE / norm_dE if norm_dE > 0 else 0
            # Transverse wake
            diff_xk = np.linalg.norm(new_w['x_kick'] - leg_w['x_kick'])
            norm_xk = np.linalg.norm(leg_w['x_kick'])
            rel_xk = diff_xk / norm_xk if norm_xk > 0 else 0
            log_lines.append(f"  Step {leg_w['step']}: dE/dct rel_diff = {rel_dE:.4f}, x_kick rel_diff = {rel_xk:.4f}")
        else:
            log_lines.append(f"  Step {leg_w['step']}: wake grid mismatch ({leg_w['dE_dct'].shape} vs {new_w['dE_dct'].shape})")

    log_path = os.path.join(case_dir, 'benchmark_log.txt')
    with open(log_path, 'w') as f:
        f.write('\n'.join(log_lines))
    print(f"\nLog: {log_path}")
    print(f"Plots: {case_dir}/")


def run_chicane_benchmark(config_file, case_name):
    """Run chicane benchmark with selective plotting at element midpoints."""
    print(f"\n{'='*60}")
    print(f"CHICANE BENCHMARK: {case_name}")
    print(f"{'='*60}\n")

    case_dir = os.path.join(BENCHMARK_DIR, case_name + '_detailed')
    os.makedirs(case_dir, exist_ok=True)

    # Legacy run
    print("Running LEGACY...")
    t0 = time.time()
    leg_steps, leg_wakes, csr_leg = run_and_capture(config_file, method='legacy')
    time_legacy = time.time() - t0
    print(f"  Legacy: {len(leg_steps)} deposition steps, {len(leg_wakes)} wake steps, {time_legacy:.1f}s")

    # New method run
    print("Running NEW (B-spline + FFT)...")
    dep_config = {
        'method': 'bspline_fft',
        'xbins': 200, 'zbins': 200,
        'xlim': 5, 'zlim': 5,
        'smoothing_sigma': 3.0,
        'poly_degree': 3,
        'velocity_threhold': 1000,
    }
    t0 = time.time()
    new_steps, new_wakes, csr_new = run_and_capture(config_file, method='bspline_fft',
                                                     deposition_config=dep_config)
    time_new = time.time() - t0
    print(f"  New: {len(new_steps)} deposition steps, {len(new_wakes)} wake steps, {time_new:.1f}s")

    n_steps = min(len(leg_steps), len(new_steps))
    n_wakes = min(len(leg_wakes), len(new_wakes))

    # Determine approximate element midpoints based on step size
    # Chicane: D0(0.1) + B1(0.5) + D1(5.0) + B2(0.5) + D2(1.0) + B3(0.5) + D3(5.0) + B4(0.5) + Df(0.2)
    # Cumulative s at element midpoints:
    #   mid-B1: 0.1 + 0.25 = 0.35m
    #   mid-D1: 0.6 + 2.5 = 3.1m
    #   mid-B2: 5.6 + 0.25 = 5.85m
    #   mid-D2: 6.1 + 0.5 = 6.6m
    #   mid-B3: 7.1 + 0.25 = 7.35m
    #   mid-D3: 7.6 + 2.5 = 10.1m
    #   mid-B4: 12.6 + 0.25 = 12.85m
    # Step index = s / step_size (approximate)
    step_size = csr_leg.lattice.step_size if hasattr(csr_leg.lattice, 'step_size') else 0.1
    mid_s = [0.35, 3.1, 5.85, 6.6, 7.35, 10.1, 12.85]
    target_steps = [int(round(s / step_size)) for s in mid_s]
    target_steps.append(n_steps - 1)  # final step
    target_steps = [s for s in target_steps if s < n_steps]

    print(f"\nGenerating field plots at selected steps: {target_steps}")
    for i in target_steps:
        plot_step_comparison(leg_steps, new_steps, i, case_dir)

    # Wake indices correspond to wake_data entries (1-indexed from step count)
    target_wake_indices = [i for i in range(n_wakes) if leg_wakes[i]['step'] in
                           [3, 30, 58, 65, 73, 100, 128, n_wakes]]
    # Fallback: just use the same indices if step numbers don't match
    if not target_wake_indices:
        target_wake_indices = [i for i in target_steps if i < n_wakes]

    print(f"Generating wake plots at selected indices...")
    for i in target_wake_indices:
        plot_wake_comparison(leg_wakes, new_wakes, i, case_dir)

    # Summary log
    log_lines = [
        f"Chicane Benchmark: {case_name}",
        f"Config: {config_file}",
        f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"",
        f"Legacy: {time_legacy:.1f}s, {len(leg_steps)} deposit steps, {len(leg_wakes)} wake steps",
        f"New: {time_new:.1f}s, {len(new_steps)} deposit steps, {len(new_wakes)} wake steps",
        f"",
    ]

    # Per-step wake L2 norms
    log_lines.append("Per-step wake L2 norm (relative to legacy):")
    for i in range(n_wakes):
        leg_w = leg_wakes[i]
        new_w = new_wakes[i]
        if leg_w['dE_dct'].shape == new_w['dE_dct'].shape:
            diff_dE = np.linalg.norm(new_w['dE_dct'] - leg_w['dE_dct'])
            norm_dE = np.linalg.norm(leg_w['dE_dct'])
            rel_dE = diff_dE / norm_dE if norm_dE > 0 else 0
            diff_xk = np.linalg.norm(new_w['x_kick'] - leg_w['x_kick'])
            norm_xk = np.linalg.norm(leg_w['x_kick'])
            rel_xk = diff_xk / norm_xk if norm_xk > 0 else 0
            log_lines.append(f"  Step {leg_w['step']}: dE/dct rel_diff = {rel_dE:.4f}, x_kick rel_diff = {rel_xk:.4f}")
        else:
            log_lines.append(f"  Step {leg_w['step']}: wake grid mismatch ({leg_w['dE_dct'].shape} vs {new_w['dE_dct'].shape})")

    log_path = os.path.join(case_dir, 'benchmark_log.txt')
    with open(log_path, 'w') as f:
        f.write('\n'.join(log_lines))
    print(f"\nLog: {log_path}")
    print(f"Plots: {case_dir}/")


def main():
    os.makedirs(BENCHMARK_DIR, exist_ok=True)

    # Chicane fine step
    run_chicane_benchmark('input/chicane_config_fine.yaml', 'chicane_fine')


if __name__ == '__main__':
    main()
