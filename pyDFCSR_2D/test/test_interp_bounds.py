"""Diagnose: are query points falling outside per-timestep grid bounds at high tilt?"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..', 'example'))

from pyDFCSR_2D.CSR import CSR2D
from pyDFCSR_2D.deposit_smooth import DF_tracker_smooth

dep_config = {
    'method': 'bspline_fft',
    'xbins': 200, 'zbins': 200,
    'xlim': 5, 'zlim': 5,
    'smoothing_sigma': 3.0,
    'poly_degree': 3,
    'velocity_threhold': 1000,
}

# Run chirp=500 with new method, capture DF_tracker state at a high-tilt step
csr = CSR2D(input_file='input/dipole_chirp500_config.yaml')
csr.DF_tracker = DF_tracker_smooth(dep_config)
csr.use_smooth_deposit = True
csr.DF_tracker.get_DF(x=csr.beam.x, z=csr.beam.z, px=csr.beam.px, t=csr.beam.position)
csr.DF_tracker.append_DF()
csr.DF_tracker.append_interpolant(formation_length=float('inf'),
                                   n_formation_length=csr.integration_params.n_formation_length)

# Monkey-patch to capture state at step 6 (slope ~ -9.9)
step_to_capture = 6
original_calc = csr.calculate_2D_CSR

def capture_calc():
    if csr.beam.step == step_to_capture:
        tracker = csr.DF_tracker
        n_t = len(tracker.DF_log)
        print(f"\n=== Step {csr.beam.step}, n_t={n_t} timesteps in history ===")
        print(f"Time range: [{tracker.min_x:.6f}, {tracker.max_x:.6f}], delta_t={tracker.delta_x:.6f}")
        print(f"\nPer-timestep grid extents:")
        print(f"{'k':>3} | {'min_xi':>12} | {'max_xi':>12} | {'min_z':>12} | {'max_z':>12} | {'poly[0]':>10} | {'poly[1]':>10}")
        print("-" * 90)
        for k in range(n_t):
            min_xi = tracker.min_xi_arr[k]
            max_xi = min_xi + tracker.delta_xi_arr[k] * 199  # 200 bins
            min_z = tracker.min_z_arr[k]
            max_z = min_z + tracker.delta_z_arr[k] * 199
            pc = tracker.poly_coeffs_interp[k]
            print(f"{k:3d} | {min_xi*1e6:12.2f} | {max_xi*1e6:12.2f} | {min_z*1e6:12.2f} | {max_z*1e6:12.2f} | {pc[0]:10.4f} | {pc[1]:10.4f}")

        # Check what query points look like
        # The CSR mesh is defined by CSR_xrange_transformed and CSR_zrange
        print(f"\nCSR mesh range (will be query points):")
        print(f"  x range: [{csr.CSR_xrange_transformed.min()*1e6:.2f}, {csr.CSR_xrange_transformed.max()*1e6:.2f}] um")
        print(f"  z range: [{csr.CSR_zrange.min()*1e6:.2f}, {csr.CSR_zrange.max()*1e6:.2f}] um")

        # For a sample query point at center of CSR mesh, check xi at each timestep
        x_query = csr.CSR_xrange_transformed[5]  # mid-x
        z_query = csr.CSR_zrange[15]  # mid-z
        print(f"\nSample query: x={x_query*1e6:.2f}um, z={z_query*1e6:.2f}um")
        print(f"  Per-timestep xi = x - poly(z):")
        for k in range(n_t):
            pc = tracker.poly_coeffs_interp[k]
            xi_k = x_query - np.polyval(pc, z_query)
            min_xi = tracker.min_xi_arr[k]
            max_xi = min_xi + tracker.delta_xi_arr[k] * 199
            in_bounds = "IN" if min_xi <= xi_k <= max_xi else "OUT"
            print(f"    k={k}: xi={xi_k*1e6:10.2f}um, grid=[{min_xi*1e6:.2f}, {max_xi*1e6:.2f}]um -> {in_bounds}")

    original_calc()

csr.calculate_2D_CSR = capture_calc
csr.run()
