"""
Benchmark: energy-chirped beam in dipole, CSR OFF.

With CSR kicks disabled, the beam evolution is identical between methods.
Any wake difference is purely from the deposition/interpolation pipeline.
This isolates the effect of high x-z tilt on the new method's accuracy.

Usage:
    conda run -n pydfcsr python pyDFCSR_2D/test/benchmark_chirp_noCSR.py
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

BENCHMARK_DIR = os.path.join(os.path.dirname(__file__), 'benchmark_results', 'chirp_noCSR_highres_chirp1000')
EXAMPLE_DIR = os.path.join(os.path.dirname(__file__), '..', 'example')

DEP_CONFIG = {
    'method': 'bspline_fft',
    'xbins': 200, 'zbins': 200,
    'xlim': 5, 'zlim': 5,
    'smoothing_sigma': 3.0,
    'poly_degree': 3,
    'velocity_threhold': 1000,
}

CONFIG_FILE = 'input/dipole_chirp1000_config_fine_highres.yaml'


def run_and_capture(method):
    """Run with CSR off, capture wakes and beam slope at each step."""
    os.chdir(EXAMPLE_DIR)
    csr = CSR2D(input_file=CONFIG_FILE)
    csr.CSR_params.apply_CSR = 0  # CSR off — identical beam

    if method == 'bspline_fft':
        csr.DF_tracker = DF_tracker_smooth(DEP_CONFIG)
        csr.use_smooth_deposit = True
        csr.DF_tracker.get_DF(x=csr.beam.x, z=csr.beam.z, px=csr.beam.px, t=csr.beam.position)
        csr.DF_tracker.append_DF()
        csr.DF_tracker.append_interpolant(formation_length=float('inf'),
                                           n_formation_length=csr.integration_params.n_formation_length)

    step_info = []
    wake_data = []

    original_calculate = csr.calculate_2D_CSR

    def capture_calculate():
        x = csr.beam.x.copy()
        z = csr.beam.z.copy()
        slope = np.polyfit(z, x, 1)[0] if len(z) > 10 else 0.0
        step_info.append({
            'step': csr.beam.step,
            'position': csr.beam.position,
            'slope': slope,
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


def main():
    os.makedirs(BENCHMARK_DIR, exist_ok=True)

    print("=" * 70)
    print("Benchmark: chirp=500 dipole, CSR OFF, step_size=0.05")
    print("=" * 70)

    print("\nRunning LEGACY...")
    t0 = time.time()
    leg_info, leg_wakes = run_and_capture('legacy')
    t_leg = time.time() - t0
    print(f"  {len(leg_wakes)} wake steps, {t_leg:.1f}s")

    print("\nRunning NEW (poly_degree=3, B-spline+FFT)...")
    t0 = time.time()
    new_info, new_wakes = run_and_capture('bspline_fft')
    t_new = time.time() - t0
    print(f"  {len(new_wakes)} wake steps, {t_new:.1f}s")

    # Compare wakes
    n = min(len(leg_wakes), len(new_wakes))
    log_lines = [
        "Benchmark: chirp=500 dipole, CSR OFF",
        f"Config: {CONFIG_FILE}",
        f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Legacy: {t_leg:.1f}s, New: {t_new:.1f}s",
        f"New method: poly_degree=3, xbins=200, smoothing_sigma=3.0",
        "",
        "Step | Position | Slope      | dE/dct rel | x_kick rel",
        "-----|----------|------------|------------|----------",
    ]

    for i in range(n):
        lw, nw = leg_wakes[i], new_wakes[i]
        if lw['dE_dct'].shape == nw['dE_dct'].shape:
            rel_dE = np.linalg.norm(nw['dE_dct'] - lw['dE_dct']) / max(np.linalg.norm(lw['dE_dct']), 1e-30)
            rel_xk = np.linalg.norm(nw['x_kick'] - lw['x_kick']) / max(np.linalg.norm(lw['x_kick']), 1e-30)
        else:
            rel_dE = rel_xk = float('nan')
        slope = leg_info[i]['slope']
        pos = leg_info[i]['position']
        log_lines.append(f"  {lw['step']:2d}  | {pos:.4f}  | {slope:+.4f}   | {rel_dE:.4f}     | {rel_xk:.4f}")

    # Write log
    log_path = os.path.join(BENCHMARK_DIR, 'benchmark_log.txt')
    with open(log_path, 'w') as f:
        f.write('\n'.join(log_lines))
    print(f"\nLog: {log_path}")

    # Generate wake comparison plots at selected steps
    plot_indices = list(range(0, n, 2))  # every other step
    for i in plot_indices:
        lw, nw = leg_wakes[i], new_wakes[i]
        if lw['dE_dct'].shape != nw['dE_dct'].shape:
            continue

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        slope = leg_info[i]['slope']
        pos = leg_info[i]['position']
        fig.suptitle(f'Step {lw["step"]} (s={pos:.3f}m, slope={slope:.2f}) — CSR OFF', fontsize=13)

        xr = lw['xrange'] * 1e6
        zr = lw['zrange'] * 1e6

        # dE/dct
        vmax = max(abs(lw['dE_dct']).max(), abs(nw['dE_dct']).max(), 1e-30)
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
        vmax_xk = max(abs(lw['x_kick']).max(), abs(nw['x_kick']).max(), 1e-30)
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
        plt.savefig(os.path.join(BENCHMARK_DIR, f'wakes_step_{lw["step"]:02d}.png'), dpi=120)
        plt.close()

    print(f"Plots: {BENCHMARK_DIR}/")

    # Print summary to stdout
    print("\n" + "\n".join(log_lines))


if __name__ == '__main__':
    main()
