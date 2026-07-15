"""
Measure cancellation ratio: |sum(integrand)| / sum(|integrand|)
for both the dipole chirp=500 and chicane cases.
A ratio near 0 = extreme cancellation; ratio near 1 = no cancellation.
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..', 'example'))

from pyDFCSR_2D.CSR import CSR2D

def measure_cancellation(config_file, label):
    """Run legacy, capture integrand at each wake step, compute cancellation ratio."""
    csr = CSR2D(input_file=config_file)
    csr.CSR_params.apply_CSR = 0

    ratios = []
    original = csr.get_CSR_wake

    def capture_wake(s, x, debug=False):
        result = original(s, x, debug=True)
        # Get the integrands from debug output
        if len(result) == 15:  # chirp_band mode
            iz1, ix1, iz2, ix2, iz3, ix3, iz4, ix4 = result[7:]
            sum_z = np.abs(iz1).sum() + np.abs(iz2).sum() + np.abs(iz3).sum() + np.abs(iz4).sum()
            net_z = abs(iz1.sum() + iz2.sum() + iz3.sum() + iz4.sum())
            sum_x = np.abs(ix1).sum() + np.abs(ix2).sum() + np.abs(ix3).sum() + np.abs(ix4).sum()
            net_x = abs(ix1.sum() + ix2.sum() + ix3.sum() + ix4.sum())
        else:  # no chirp_band
            iz1, ix1, iz2, ix2, iz3, ix3 = result[5:]
            sum_z = np.abs(iz1).sum() + np.abs(iz2).sum() + np.abs(iz3).sum()
            net_z = abs(iz1.sum() + iz2.sum() + iz3.sum())
            sum_x = np.abs(ix1).sum() + np.abs(ix2).sum() + np.abs(ix3).sum()
            net_x = abs(ix1.sum() + ix2.sum() + ix3.sum())

        ratio_z = net_z / sum_z if sum_z > 0 else 0
        ratio_x = net_x / sum_x if sum_x > 0 else 0
        ratios.append((csr.beam.step, ratio_z, ratio_x))

        # Return the actual wake (not debug)
        return original(s, x, debug=False)

    csr.get_CSR_wake = capture_wake
    csr.run()

    print(f"\n{label}:")
    print(f"  Step | Cancel_z | Cancel_x")
    print(f"  -----|----------|----------")
    for step, rz, rx in ratios:
        print(f"  {step:4d} | {rz:.6f} | {rx:.6f}")

# Dipole chirp=500
measure_cancellation('input/dipole_chirp500_config_fine.yaml', 'Dipole chirp=500 (angle=1.0rad)')

# Chicane
measure_cancellation('input/chicane_config_highres300_noCSR.yaml', 'Chicane (angle=0.0483rad)')
