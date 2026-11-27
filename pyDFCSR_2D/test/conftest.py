"""
pytest configuration for pyDFCSR_2D/test.

Only the modules in TEST_MODULES are unit tests. The other test_*.py files in this directory are
development scripts from earlier studies: most of them run simulations at import time, so they must
not be collected (pytest --collect-only would otherwise run them and write into example/output).
Run a script directly with python if it is needed.
"""
import os

TEST_MODULES = {
    'test_import.py',
    'test_quantile_frame.py',      # golden fixture (test/golden/golden_2d.npz): defaults must stay bit-identical
    'test_csr1d.py',
    'test_curvature.py',
    'test_drift_reach.py',
    'test_vx_model.py',
    'test_waist_scan.py',
    'test_band_jacobian.py',
    'test_tangent_bands.py',
    'test_tracking_edges.py',
    'test_frame_blend.py',
    'test_torch_backend.py',      # torch backend against numba (skipped without torch)
    'test_torch_diff.py',         # differentiable run: forward and gradients (skipped without torch)
    'test_integrand_kernel.py',   # the numba kernel against the numpy arithmetic it replaced
}

_HERE = os.path.dirname(os.path.abspath(__file__))
collect_ignore = sorted(f for f in os.listdir(_HERE)
                        if f.startswith('test_') and f.endswith('.py') and f not in TEST_MODULES)
collect_ignore_glob = ['*/*.py']   # subdirectories hold study scripts (facet_bc20, bc20_indranil, golden, ...)
