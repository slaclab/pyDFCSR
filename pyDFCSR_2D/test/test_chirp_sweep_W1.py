"""
Chirp sweep with CSR off: compare W1-only vs full (W1+W2+W3) transverse wake.
Isolates the effect of density derivatives on wake accuracy.
"""
import sys, os, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..', 'example'))

from pyDFCSR_2D.CSR import CSR2D
from pyDFCSR_2D.deposit_smooth import DF_tracker_smooth

DEP_CONFIG = {
    'method': 'bspline_fft',
    'xbins': 200, 'zbins': 200,
    'xlim': 5, 'zlim': 5,
    'smoothing_sigma': 3.0,
    'poly_degree': 3,
    'velocity_threhold': 1000,
}


def run_with_capture(config_file, method='legacy'):
    """Run with CSR off, capture wakes and beam slope."""
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
    original_calculate = csr.calculate_2D_CSR

    def capture():
        x, z = csr.beam.x.copy(), csr.beam.z.copy()
        slope = np.polyfit(z, x, 1)[0] if len(z) > 10 else 0.0
        step_info.append({'step': csr.beam.step, 'position': csr.beam.position, 'slope': slope})
        original_calculate()
        wake_data.append({
            'step': csr.beam.step,
            'dE_dct': csr.dE_dct.copy(),
            'x_kick': csr.x_kick.copy(),
        })

    csr.calculate_2D_CSR = capture
    csr.run()
    return step_info, wake_data


def main():
    chirp_values = [0, 80, 200, 500, 1000, 2000]
    config_file = 'input/dipole_config.yaml'  # base config, we'll create per-chirp

    print("=" * 80)
    print("Chirp Sweep: CSR OFF, W1-only vs Full (W1+W2+W3)")
    print("  x_kick comparison at mid-bend steps")
    print("=" * 80)

    # Header
    print(f"\n{'Chirp':<8} | {'Peak slope':<11} | {'xk full (avg 3-10)':<18} | {'xk full (avg 11-16)':<20}")
    print(f"{'':<8} | {'':<11} | {'(in-bend early)':<18} | {'(in-bend late)':<20}")
    print("-" * 70)

    all_results = {}

    for chirp in chirp_values:
        cfg = f'input/dipole_chirp{chirp}_config.yaml'
        if not os.path.exists(cfg):
            # Use base config for chirp=0
            if chirp == 0:
                cfg = 'input/dipole_config.yaml'
            else:
                continue

        # Run legacy
        leg_info, leg_wakes = run_with_capture(cfg, method='legacy')
        # Run new
        new_info, new_wakes = run_with_capture(cfg, method='bspline_fft')

        n = min(len(leg_wakes), len(new_wakes))
        steps_data = []
        for i in range(n):
            lw, nw = leg_wakes[i], new_wakes[i]
            if lw['dE_dct'].shape == nw['dE_dct'].shape:
                rel_dE = np.linalg.norm(nw['dE_dct'] - lw['dE_dct']) / max(np.linalg.norm(lw['dE_dct']), 1e-30)
                rel_xk = np.linalg.norm(nw['x_kick'] - lw['x_kick']) / max(np.linalg.norm(lw['x_kick']), 1e-30)
            else:
                rel_dE = rel_xk = float('nan')
            steps_data.append({
                'step': lw['step'], 'slope': leg_info[i]['slope'],
                'rel_dE': rel_dE, 'rel_xk': rel_xk,
            })

        # Compute averages for early and late bend
        early = [s for s in steps_data if 3 <= s['step'] <= 10]
        late = [s for s in steps_data if 11 <= s['step'] <= 16]
        peak_slope = max(abs(s['slope']) for s in steps_data) if steps_data else 0

        avg_xk_early = np.mean([s['rel_xk'] for s in early]) if early else 0
        avg_xk_late = np.mean([s['rel_xk'] for s in late]) if late else 0

        all_results[chirp] = steps_data
        print(f"{chirp:<8} | {peak_slope:<11.2f} | {avg_xk_early:<18.4f} | {avg_xk_late:<20.4f}")

    # Now print full table
    print("\n\nFull per-step results:")
    print(f"{'Chirp':<6} | {'Step':<4} | {'Slope':<8} | {'dE/dct':<10} | {'x_kick':<10}")
    print("-" * 50)
    for chirp in chirp_values:
        if chirp not in all_results:
            continue
        for sd in all_results[chirp]:
            print(f"{chirp:<6} | {sd['step']:<4} | {sd['slope']:+.3f} | {sd['rel_dE']:<10.4f} | {sd['rel_xk']:<10.4f}")
        print()


if __name__ == '__main__':
    main()
