"""
Test: how does x-z beam tilt affect wake agreement between legacy and new methods?

Hypothesis: when beam is highly tilted in x-z, the new method's wakes become noisy/incorrect,
while untilted beams show good agreement.

Approach: run a simple dipole with varying initial shear z:x values and compare wakes.
"""
import sys
import os
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import yaml
import tempfile
import shutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from pyDFCSR_2D.CSR import CSR2D
from pyDFCSR_2D.deposit_smooth import DF_tracker_smooth

RESULT_DIR = os.path.join(os.path.dirname(__file__), 'benchmark_results', 'tilt_sweep')
EXAMPLE_DIR = os.path.join(os.path.dirname(__file__), '..', 'example')


def create_beam_yaml(shear_zx, output_path):
    """Create a beam YAML with given shear z:x value."""
    beam = {
        'n_particle': 1000000,
        'species': 'electron',
        'px_dist': {'sigma_px': {'units': 'keV/c', 'value': 25.0}, 'type': 'gaussian'},
        'py_dist': {'sigma_py': {'units': 'keV/c', 'value': 25.0}, 'type': 'gaussian'},
        'pz_dist': {'avg_pz': {'units': 'GeV/c', 'value': 5},
                    'sigma_pz': {'units': 'MeV/c', 'value': 0.5}, 'type': 'gaussian'},
        'x_dist': {'sigma_x': {'units': 'um', 'value': 50}, 'type': 'gaussian'},
        'y_dist': {'sigma_y': {'units': 'um', 'value': 5}, 'type': 'gaussian'},
        'random': {'type': 'hammersley'},
        'start': {'tstart': {'units': 'sec', 'value': 0}, 'type': 'time'},
        'total_charge': {'units': 'nC', 'value': 1},
        'z_dist': {'avg_z': {'units': 'mm', 'value': 0},
                   'sigma_z': {'units': 'um', 'value': 50}, 'type': 'gaussian'},
    }
    if shear_zx != 0:
        beam['transforms'] = {
            's1': {
                'shear_coefficient': {'units': 'dimensionless', 'value': float(shear_zx)},
                'type': 'shear z:x'
            }
        }
    with open(output_path, 'w') as f:
        yaml.dump(beam, f, default_flow_style=False)


def create_config_yaml(beam_file, output_path, write_name):
    """Create a dipole config YAML pointing to the given beam file."""
    config = {
        'input_beam': {
            'style': 'distgen',
            'distgen_input_file': beam_file,
        },
        'input_lattice': {
            'lattice_input_file': 'input/dipole_lattice.yaml',
        },
        'particle_deposition': {
            'xbins': 100, 'zbins': 100,
            'xlim': 5, 'zlim': 5,
            'filter_order': 2, 'filter_window': 5,
            'velocity_threhold': 1000,
        },
        'CSR_integration': {
            'n_formation_length': 1.5,
            'zbins': 100, 'xbins': 100,
        },
        'CSR_computation': {
            'compute_CSR': 1, 'apply_CSR': 1, 'transverse_on': 1,
            'xbins': 10, 'zbins': 30,
            'xlim': 3, 'zlim': 3,
            'write_beam': [1],
            'write_wakes': True,
            'write_name': write_name,
            'workdir': './output',
        },
    }
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def run_with_capture(config_file, method='legacy', deposition_config=None):
    """Run and capture wakes + beam slope at each step."""
    os.chdir(EXAMPLE_DIR)
    csr = CSR2D(input_file=config_file)

    if method == 'bspline_fft' and deposition_config:
        csr.DF_tracker = DF_tracker_smooth(deposition_config)
        csr.use_smooth_deposit = True
        csr.DF_tracker.get_DF(x=csr.beam.x, z=csr.beam.z, px=csr.beam.px, t=csr.beam.position)
        csr.DF_tracker.append_DF()
        csr.DF_tracker.append_interpolant(formation_length=float('inf'),
                                           n_formation_length=csr.integration_params.n_formation_length)

    # Capture beam slope and wakes at each step
    step_info = []
    wake_data = []

    original_calculate = csr.calculate_2D_CSR

    def capture_calculate():
        # Record beam slope (x-z tilt) before computing wake
        x = csr.beam.x.copy()
        z = csr.beam.z.copy()
        # Linear fit: x = slope * z + intercept
        if len(z) > 10:
            slope, intercept = np.polyfit(z, x, 1)
        else:
            slope = 0.0
        step_info.append({
            'step': csr.beam.step,
            'position': csr.beam.position,
            'slope': slope,
            'sigma_x': np.std(x),
            'sigma_z': np.std(z),
        })

        original_calculate()

        wake_data.append({
            'step': csr.beam.step,
            'dE_dct': csr.dE_dct.copy(),
            'x_kick': csr.x_kick.copy(),
            'xrange': csr.CSR_xrange_transformed.copy(),
            'zrange': csr.CSR_zrange.copy(),
        })

    csr.calculate_2D_CSR = capture_calculate
    csr.run()
    return step_info, wake_data


def run_tilt_sweep():
    """Run dipole with multiple x-z tilt values and compare."""
    os.makedirs(RESULT_DIR, exist_ok=True)

    # Tilt values to test (shear z:x coefficient)
    tilt_values = [0, 2, 5, 10, 20, 40]

    dep_config = {
        'method': 'bspline_fft',
        'xbins': 200, 'zbins': 200,
        'xlim': 5, 'zlim': 5,
        'smoothing_sigma': 3.0,
        'poly_degree': 3,
        'velocity_threhold': 1000,
    }

    all_results = {}

    for tilt in tilt_values:
        print(f"\n{'='*60}")
        print(f"  TILT = {tilt} (shear z:x)")
        print(f"{'='*60}")

        # Create beam and config files
        beam_file = os.path.join(EXAMPLE_DIR, 'input', f'dipole_tilt{tilt}_beam.yaml')
        config_file = os.path.join(EXAMPLE_DIR, 'input', f'dipole_tilt{tilt}_config.yaml')
        create_beam_yaml(tilt, beam_file)
        create_config_yaml(f'input/dipole_tilt{tilt}_beam.yaml', config_file, f'dipole_tilt{tilt}')

        # Run legacy
        print(f"  Running legacy...")
        t0 = time.time()
        leg_info, leg_wakes = run_with_capture(f'input/dipole_tilt{tilt}_config.yaml', method='legacy')
        t_leg = time.time() - t0

        # Run new
        print(f"  Running new...")
        t0 = time.time()
        new_info, new_wakes = run_with_capture(f'input/dipole_tilt{tilt}_config.yaml',
                                                method='bspline_fft', deposition_config=dep_config)
        t_new = time.time() - t0

        print(f"  Legacy: {t_leg:.1f}s, New: {t_new:.1f}s")

        # Compute per-step wake differences
        n_wakes = min(len(leg_wakes), len(new_wakes))
        steps_data = []
        for i in range(n_wakes):
            lw = leg_wakes[i]
            nw = new_wakes[i]
            if lw['dE_dct'].shape == nw['dE_dct'].shape:
                diff_dE = np.linalg.norm(nw['dE_dct'] - lw['dE_dct'])
                norm_dE = np.linalg.norm(lw['dE_dct'])
                rel_dE = diff_dE / norm_dE if norm_dE > 0 else 0

                diff_xk = np.linalg.norm(nw['x_kick'] - lw['x_kick'])
                norm_xk = np.linalg.norm(lw['x_kick'])
                rel_xk = diff_xk / norm_xk if norm_xk > 0 else 0
            else:
                rel_dE = np.nan
                rel_xk = np.nan

            # Get slope from the corresponding step info
            slope = leg_info[i]['slope'] if i < len(leg_info) else 0
            position = leg_info[i]['position'] if i < len(leg_info) else 0

            steps_data.append({
                'step': lw['step'],
                'position': position,
                'slope': slope,
                'rel_dE': rel_dE,
                'rel_xk': rel_xk,
            })

        all_results[tilt] = {
            'steps': steps_data,
            'time_legacy': t_leg,
            'time_new': t_new,
            'leg_info': leg_info,
            'leg_wakes': leg_wakes,
            'new_wakes': new_wakes,
        }

        # Print summary
        print(f"  Step | Position | Slope    | dE/dct rel | x_kick rel")
        print(f"  -----|----------|----------|------------|----------")
        for sd in steps_data:
            print(f"  {sd['step']:4d} | {sd['position']:.3f}m   | {sd['slope']:.4f}  | {sd['rel_dE']:.4f}     | {sd['rel_xk']:.4f}")

        # Per-tilt wake comparison plots at mid-bend steps
        plot_steps = [3, 6, 9]  # early, mid, late bend
        for pi in plot_steps:
            if pi >= n_wakes:
                continue
            lw = leg_wakes[pi]
            nw = new_wakes[pi]
            if lw['dE_dct'].shape != nw['dE_dct'].shape:
                continue

            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            slope_val = leg_info[pi]['slope'] if pi < len(leg_info) else 0
            fig.suptitle(f'Shear={tilt}, Step {lw["step"]} (s={leg_info[pi]["position"]:.2f}m, slope={slope_val:.2f})', fontsize=13)

            xr = lw['xrange'] * 1e6
            zr = lw['zrange'] * 1e6

            # dE/dct
            vmax = max(abs(lw['dE_dct']).max(), abs(nw['dE_dct']).max())
            if vmax == 0:
                vmax = 1
            im = axes[0, 0].imshow(lw['dE_dct'].T, origin='lower',
                                    extent=[xr[0], xr[-1], zr[0], zr[-1]],
                                    aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
            axes[0, 0].set_title('dE/dct (legacy)')
            plt.colorbar(im, ax=axes[0, 0])

            im = axes[0, 1].imshow(nw['dE_dct'].T, origin='lower',
                                    extent=[xr[0], xr[-1], zr[0], zr[-1]],
                                    aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
            axes[0, 1].set_title('dE/dct (new)')
            plt.colorbar(im, ax=axes[0, 1])

            diff_dE = nw['dE_dct'] - lw['dE_dct']
            im = axes[0, 2].imshow(diff_dE.T, origin='lower',
                                    extent=[xr[0], xr[-1], zr[0], zr[-1]],
                                    aspect='auto', cmap='RdBu_r')
            axes[0, 2].set_title('Δ(dE/dct)')
            plt.colorbar(im, ax=axes[0, 2])

            # x_kick
            vmax_xk = max(abs(lw['x_kick']).max(), abs(nw['x_kick']).max())
            if vmax_xk == 0:
                vmax_xk = 1
            im = axes[1, 0].imshow(lw['x_kick'].T, origin='lower',
                                    extent=[xr[0], xr[-1], zr[0], zr[-1]],
                                    aspect='auto', cmap='RdBu_r', vmin=-vmax_xk, vmax=vmax_xk)
            axes[1, 0].set_title('x_kick (legacy)')
            plt.colorbar(im, ax=axes[1, 0])

            im = axes[1, 1].imshow(nw['x_kick'].T, origin='lower',
                                    extent=[xr[0], xr[-1], zr[0], zr[-1]],
                                    aspect='auto', cmap='RdBu_r', vmin=-vmax_xk, vmax=vmax_xk)
            axes[1, 1].set_title('x_kick (new)')
            plt.colorbar(im, ax=axes[1, 1])

            diff_xk = nw['x_kick'] - lw['x_kick']
            im = axes[1, 2].imshow(diff_xk.T, origin='lower',
                                    extent=[xr[0], xr[-1], zr[0], zr[-1]],
                                    aspect='auto', cmap='RdBu_r')
            axes[1, 2].set_title('Δ(x_kick)')
            plt.colorbar(im, ax=axes[1, 2])

            for ax in axes[:, 0]:
                ax.set_ylabel('z (μm)')
            for ax in axes[1, :]:
                ax.set_xlabel('x (μm)')

            plt.tight_layout()
            plt.savefig(os.path.join(RESULT_DIR, f'wakes_tilt{tilt}_step{lw["step"]:02d}.png'), dpi=120)
            plt.close()

    # Summary plot: wake disagreement vs tilt
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle('Wake Disagreement vs Beam x-z Tilt (Slope)', fontsize=14)

    # Plot 1: dE/dct rel_diff vs step for all tilts
    for tilt, result in all_results.items():
        steps = [sd['step'] for sd in result['steps']]
        rel_dE = [sd['rel_dE'] for sd in result['steps']]
        axes[0].plot(steps, rel_dE, 'o-', label=f'shear={tilt}', markersize=4)
    axes[0].set_xlabel('Step')
    axes[0].set_ylabel('dE/dct relative L2 difference')
    axes[0].set_yscale('log')
    axes[0].legend()
    axes[0].set_title('Longitudinal wake disagreement vs step')
    axes[0].grid(True, alpha=0.3)

    # Plot 2: mid-bend dE/dct vs tilt value
    # Use step ~6-8 (middle of bend) for each tilt
    mid_bend_dE = []
    mid_bend_xk = []
    mid_bend_slopes = []
    for tilt, result in all_results.items():
        # Average over in-bend steps (roughly steps 3-10 for this lattice)
        bend_steps = [sd for sd in result['steps'] if 3 <= sd['step'] <= 10]
        if bend_steps:
            avg_dE = np.mean([sd['rel_dE'] for sd in bend_steps if not np.isnan(sd['rel_dE'])])
            avg_xk = np.mean([sd['rel_xk'] for sd in bend_steps if not np.isnan(sd['rel_xk'])])
            avg_slope = np.mean([abs(sd['slope']) for sd in bend_steps])
            mid_bend_dE.append(avg_dE)
            mid_bend_xk.append(avg_xk)
            mid_bend_slopes.append(avg_slope)

    axes[1].plot(tilt_values[:len(mid_bend_dE)], mid_bend_dE, 'rs-', label='dE/dct', markersize=8)
    axes[1].plot(tilt_values[:len(mid_bend_xk)], mid_bend_xk, 'b^-', label='x_kick', markersize=8)
    axes[1].set_xlabel('Initial shear z:x')
    axes[1].set_ylabel('Average in-bend relative L2 difference')
    axes[1].set_yscale('log')
    axes[1].legend()
    axes[1].set_title('Mid-bend wake disagreement vs initial tilt')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, 'tilt_vs_wake_disagreement.png'), dpi=150)
    plt.close()

    # Plot: beam slope evolution for each tilt
    fig, ax = plt.subplots(figsize=(10, 6))
    for tilt, result in all_results.items():
        positions = [info['position'] for info in result['leg_info']]
        slopes = [abs(info['slope']) for info in result['leg_info']]
        ax.plot(positions, slopes, 'o-', label=f'shear={tilt}', markersize=3)
    ax.set_xlabel('Position along lattice (m)')
    ax.set_ylabel('|Beam slope| (x-z tilt)')
    ax.set_title('Beam x-z tilt evolution along dipole lattice')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, 'slope_evolution.png'), dpi=150)
    plt.close()

    # Write log
    log_lines = [
        "Tilt Sweep Benchmark",
        f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Lattice: dipole (0.1m drift + 1.0m bend + 0.5m drift, step_size=0.1m)",
        f"Legacy: CIC 100x100, New: B-spline 200x200 sigma=3",
        "",
    ]
    for tilt, result in all_results.items():
        log_lines.append(f"=== Shear z:x = {tilt} ===")
        log_lines.append(f"  Legacy: {result['time_legacy']:.1f}s, New: {result['time_new']:.1f}s")
        log_lines.append(f"  Step | Position | Slope      | dE/dct rel | x_kick rel")
        log_lines.append(f"  -----|----------|------------|------------|----------")
        for sd in result['steps']:
            log_lines.append(f"  {sd['step']:4d} | {sd['position']:.4f}  | {sd['slope']:+.6f} | {sd['rel_dE']:.4f}     | {sd['rel_xk']:.4f}")
        log_lines.append("")

    log_path = os.path.join(RESULT_DIR, 'tilt_sweep_log.txt')
    with open(log_path, 'w') as f:
        f.write('\n'.join(log_lines))
    print(f"\nLog: {log_path}")
    print(f"Plots: {RESULT_DIR}/")


if __name__ == '__main__':
    run_tilt_sweep()
