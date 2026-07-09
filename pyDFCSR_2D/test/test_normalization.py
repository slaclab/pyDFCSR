"""Check normalization and resolution of both methods."""
import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..', 'example'))

from pyDFCSR_2D.CSR import CSR2D
from pyDFCSR_2D.deposit_smooth import DF_tracker_smooth

dep_config = {'method': 'bspline_fft', 'xbins': 200, 'zbins': 200, 'xlim': 5, 'zlim': 5,
              'smoothing_sigma': 3.0, 'poly_degree': 3, 'velocity_threhold': 1000}

# Legacy
csr = CSR2D(input_file='input/dipole_chirp500_config.yaml')
csr.CSR_params.apply_CSR = 0
step_data = []
orig = csr.DF_tracker.get_DF
def cap1(x, z, px, t):
    orig(x, z, px, t)
    tr = csr.DF_tracker
    integral = np.trapz(np.trapz(tr.density, tr.x_grids, axis=0), tr.z_grids)
    step_data.append({'t': t, 'integral': integral, 'peak': tr.density.max(),
                      'dx': tr.x_grids[1]-tr.x_grids[0], 'dz': tr.z_grids[1]-tr.z_grids[0],
                      'xrange': (tr.x_grids[0]*1e6, tr.x_grids[-1]*1e6)})
csr.DF_tracker.get_DF = cap1
csr.run(stop_time=0.75)

# New
csr2 = CSR2D(input_file='input/dipole_chirp500_config.yaml')
csr2.CSR_params.apply_CSR = 0
csr2.DF_tracker = DF_tracker_smooth(dep_config)
csr2.use_smooth_deposit = True
csr2.DF_tracker.get_DF(x=csr2.beam.x, z=csr2.beam.z, px=csr2.beam.px, t=csr2.beam.position)
csr2.DF_tracker.append_DF()
csr2.DF_tracker.append_interpolant(formation_length=float('inf'), n_formation_length=csr2.integration_params.n_formation_length)
step_data2 = []
orig2 = csr2.DF_tracker.get_DF
def cap2(x, z, px, t):
    orig2(x, z, px, t)
    tr = csr2.DF_tracker
    integral = np.trapz(np.trapz(tr.density, tr.xi_grids, axis=0), tr.z_grids)
    step_data2.append({'t': t, 'integral': integral, 'peak': tr.density.max(),
                       'dx': tr.xi_grids[1]-tr.xi_grids[0], 'dz': tr.z_grids[1]-tr.z_grids[0],
                       'xrange': (tr.xi_grids[0]*1e6, tr.xi_grids[-1]*1e6)})
csr2.DF_tracker.get_DF = cap2
csr2.run(stop_time=0.75)

print("Step | Legacy integral | New integral | Legacy peak | New peak | Legacy dx(um) | New dxi(um) | sigma/dx_leg | sigma/dxi_new")
print("-----|----------------|--------------|-------------|----------|--------------|-------------|-------------|-------------")
for i in range(min(len(step_data), len(step_data2))):
    d1, d2 = step_data[i], step_data2[i]
    sig_dx1 = 50e-6 / d1['dx']
    sig_dx2 = 50e-6 / d2['dx']
    print(f"  {i+1}  | {d1['integral']:.6f}       | {d2['integral']:.6f}     | {d1['peak']:.3e}  | {d2['peak']:.3e} | {d1['dx']*1e6:11.2f}  | {d2['dx']*1e6:11.2f} | {sig_dx1:11.2f} | {sig_dx2:11.2f}")
