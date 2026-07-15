"""
Benchmark chirp sweep with high integration resolution (300x300).
Tests chirp = 10, 100, 500, 1000. CSR off, step_size=0.05.
"""
import sys, os, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..', 'example'))

from pyDFCSR_2D.CSR import CSR2D
from pyDFCSR_2D.deposit_smooth import DF_tracker_smooth

RESULT_BASE = os.path.join(os.path.dirname(os.path.abspath('.')), 'pyDFCSR_2D', 'test', 'benchmark_results')

DEP_CONFIG = {
    'method': 'bspline_fft',
    'xbins': 200, 'zbins': 200,
    'xlim': 5, 'zlim': 5,
    'smoothing_sigma': 3.0,
    'poly_degree': 3,
    'velocity_threhold': 1000,
}

CASES = [
    ('chirp10', 'input/dipole_chirp10_config_fine_highres.yaml'),
    ('chirp100', 'input/dipole_chirp100_config_fine_highres.yaml'),
    ('chirp500', 'input/dipole_chirp500_config_fine_highres.yaml'),
    ('chirp1000', 'input/dipole_chirp1000_config_fine_highres.yaml'),
]


def run_with_capture(config_file, method):
    csr = CSR2D(input_file=config_file)
    csr.CSR_params.apply_CSR = 0
    if method == 'bspline_fft':
        csr.DF_tracker = DF_tracker_smooth(DEP_CONFIG)
        csr.use_smooth_deposit = True
        csr.DF_tracker.get_DF(x=csr.beam.x, z=csr.beam.z, px=csr.beam.px, t=csr.beam.position)
        csr.DF_tracker.append_DF()
        csr.DF_tracker.append_interpolant(formation_length=float('inf'),
                                           n_formation_length=csr.integration_params.n_formation_length)
    step_info = []
    wake_data = []
    original = csr.calculate_2D_CSR
    def capture():
        x, z = csr.beam.x.copy(), csr.beam.z.copy()
        slope = np.polyfit(z, x, 1)[0] if len(z) > 10 else 0.0
        step_info.append({'step': csr.beam.step, 'position': csr.beam.position, 'slope': slope})
        original()
        wake_data.append({'step': csr.beam.step, 'dE_dct': csr.dE_dct.copy(), 'x_kick': csr.x_kick.copy(),
                          'xrange': csr.CSR_xrange_transformed.copy(), 'zrange': csr.CSR_zrange.copy()})
    csr.calculate_2D_CSR = capture
    csr.run()
    return step_info, wake_data


def plot_wakes(leg_wakes, new_wakes, leg_info, case_dir, plot_steps):
    for i in plot_steps:
        if i >= len(leg_wakes) or i >= len(new_wakes):
            continue
        lw, nw = leg_wakes[i], new_wakes[i]
        if lw['dE_dct'].shape != nw['dE_dct'].shape:
            continue
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        slope = leg_info[i]['slope']
        fig.suptitle(f'Step {lw["step"]} (s={leg_info[i]["position"]:.3f}m, slope={slope:.2f})', fontsize=13)
        xr = lw['xrange'] * 1e6
        zr = lw['zrange'] * 1e6
        vmax = max(abs(lw['dE_dct']).max(), abs(nw['dE_dct']).max(), 1e-30)
        im = axes[0,0].imshow(lw['dE_dct'].T, origin='lower', extent=[xr[0],xr[-1],zr[0],zr[-1]], aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        axes[0,0].set_title('dE/dct (legacy)'); plt.colorbar(im, ax=axes[0,0])
        im = axes[0,1].imshow(nw['dE_dct'].T, origin='lower', extent=[xr[0],xr[-1],zr[0],zr[-1]], aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        axes[0,1].set_title('dE/dct (new)'); plt.colorbar(im, ax=axes[0,1])
        diff = nw['dE_dct'] - lw['dE_dct']
        im = axes[0,2].imshow(diff.T, origin='lower', extent=[xr[0],xr[-1],zr[0],zr[-1]], aspect='auto', cmap='RdBu_r')
        axes[0,2].set_title('Δ(dE/dct)'); plt.colorbar(im, ax=axes[0,2])
        vmax_xk = max(abs(lw['x_kick']).max(), abs(nw['x_kick']).max(), 1e-30)
        im = axes[1,0].imshow(lw['x_kick'].T, origin='lower', extent=[xr[0],xr[-1],zr[0],zr[-1]], aspect='auto', cmap='RdBu_r', vmin=-vmax_xk, vmax=vmax_xk)
        axes[1,0].set_title('x_kick (legacy)'); plt.colorbar(im, ax=axes[1,0])
        im = axes[1,1].imshow(nw['x_kick'].T, origin='lower', extent=[xr[0],xr[-1],zr[0],zr[-1]], aspect='auto', cmap='RdBu_r', vmin=-vmax_xk, vmax=vmax_xk)
        axes[1,1].set_title('x_kick (new)'); plt.colorbar(im, ax=axes[1,1])
        diff_xk = nw['x_kick'] - lw['x_kick']
        im = axes[1,2].imshow(diff_xk.T, origin='lower', extent=[xr[0],xr[-1],zr[0],zr[-1]], aspect='auto', cmap='RdBu_r')
        axes[1,2].set_title('Δ(x_kick)'); plt.colorbar(im, ax=axes[1,2])
        for ax in axes[:,0]: ax.set_ylabel('z (μm)')
        for ax in axes[1,:]: ax.set_xlabel('x (μm)')
        plt.tight_layout()
        plt.savefig(os.path.join(case_dir, f'wakes_step_{lw["step"]:02d}.png'), dpi=120)
        plt.close()


for case_name, config_file in CASES:
    case_dir = os.path.join(RESULT_BASE, f'chirp_highres_{case_name}')
    os.makedirs(case_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  {case_name} (300x300 integration, CSR off)")
    print(f"{'='*60}")

    print("  Legacy...")
    leg_info, leg_wakes = run_with_capture(config_file, 'legacy')
    print("  New...")
    new_info, new_wakes = run_with_capture(config_file, 'bspline_fft')

    n = min(len(leg_wakes), len(new_wakes))
    print(f"\n  Step | Pos    | Slope    | dE/dct   | x_kick")
    print(f"  -----|--------|----------|----------|--------")
    for i in range(n):
        lw, nw = leg_wakes[i], new_wakes[i]
        if lw['dE_dct'].shape == nw['dE_dct'].shape:
            rel_dE = np.linalg.norm(nw['dE_dct'] - lw['dE_dct']) / max(np.linalg.norm(lw['dE_dct']), 1e-30)
            rel_xk = np.linalg.norm(nw['x_kick'] - lw['x_kick']) / max(np.linalg.norm(lw['x_kick']), 1e-30)
        else:
            rel_dE = rel_xk = float('nan')
        print(f"  {lw['step']:4d} | {leg_info[i]['position']:.3f} | {leg_info[i]['slope']:+.3f} | {rel_dE:.4f}  | {rel_xk:.4f}")

    # Plot at selected steps
    plot_steps = list(range(0, n, 4))  # every 4th
    plot_wakes(leg_wakes, new_wakes, leg_info, case_dir, plot_steps)
    print(f"  Plots: {case_dir}/")
