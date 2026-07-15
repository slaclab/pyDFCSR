"""Run chicane with new B-spline+FFT method via MPI."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..', 'example'))
from pyDFCSR_2D.CSR import CSR2D
from pyDFCSR_2D.deposit_smooth import DF_tracker_smooth

dep_config = {
    'method': 'bspline_fft',
    'xbins': 100, 'zbins': 100,
    'xlim': 5, 'zlim': 5,
    'smoothing_sigma': 3.0,
    'poly_degree': 3,
    'velocity_threhold': 1000,
}

csr = CSR2D(input_file='input/chicane_config_highres300_noCSR.yaml', parallel=True)
csr.DF_tracker = DF_tracker_smooth(dep_config)
csr.use_smooth_deposit = True
csr.DF_tracker.get_DF(x=csr.beam.x, z=csr.beam.z, px=csr.beam.px, t=csr.beam.position)
csr.DF_tracker.append_DF()
csr.DF_tracker.append_interpolant(formation_length=float('inf'),
                                   n_formation_length=csr.integration_params.n_formation_length)
csr.run()
