"""Test: poly_degree=1 with finer time step on chirped beam in dipole."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..', 'example'))

from pyDFCSR_2D.test.test_chirp_sweep import run_with_capture
import numpy as np

dep_config_poly1 = {
    'method': 'bspline_fft',
    'xbins': 200, 'zbins': 200,
    'xlim': 5, 'zlim': 5,
    'smoothing_sigma': 3.0,
    'poly_degree': 1,
    'velocity_threhold': 1000,
}

dep_config_poly0 = {
    'method': 'bspline_fft',
    'xbins': 200, 'zbins': 200,
    'xlim': 5, 'zlim': 5,
    'smoothing_sigma': 3.0,
    'poly_degree': 0,
    'velocity_threhold': 1000,
}

# Use fine lattice (step_size=0.05)
config_file = 'input/dipole_chirp500_config_fine.yaml'

print("Running legacy chirp=500 (fine step)...")
leg_info, leg_wakes = run_with_capture(config_file, method='legacy')
print("Running new chirp=500 poly_degree=1 (fine step)...")
new1_info, new1_wakes = run_with_capture(config_file, method='bspline_fft', deposition_config=dep_config_poly1)
print("Running new chirp=500 poly_degree=0 (fine step)...")
new0_info, new0_wakes = run_with_capture(config_file, method='bspline_fft', deposition_config=dep_config_poly0)

print("\nResults (chirp=500, step_size=0.05):")
print("Step | Pos(m) | Slope    | dE poly1  | xk poly1  | dE poly0  | xk poly0")
print("-----|--------|----------|-----------|-----------|-----------|----------")
n = min(len(leg_wakes), len(new1_wakes), len(new0_wakes))
for i in range(n):
    lw = leg_wakes[i]
    nw1 = new1_wakes[i]
    nw0 = new0_wakes[i]
    if lw['dE_dct'].shape == nw1['dE_dct'].shape and lw['dE_dct'].shape == nw0['dE_dct'].shape:
        rel_dE_1 = np.linalg.norm(nw1['dE_dct'] - lw['dE_dct']) / max(np.linalg.norm(lw['dE_dct']), 1e-30)
        rel_xk_1 = np.linalg.norm(nw1['x_kick'] - lw['x_kick']) / max(np.linalg.norm(lw['x_kick']), 1e-30)
        rel_dE_0 = np.linalg.norm(nw0['dE_dct'] - lw['dE_dct']) / max(np.linalg.norm(lw['dE_dct']), 1e-30)
        rel_xk_0 = np.linalg.norm(nw0['x_kick'] - lw['x_kick']) / max(np.linalg.norm(lw['x_kick']), 1e-30)
        slope = leg_info[i]['slope'] if i < len(leg_info) else 0
        pos = leg_info[i]['position'] if i < len(leg_info) else 0
        step = lw['step']
        print(f"  {step:2d}  | {pos:.3f}  | {slope:+.4f} | {rel_dE_1:.4f}    | {rel_xk_1:.4f}    | {rel_dE_0:.4f}    | {rel_xk_0:.4f}")
