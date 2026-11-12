"""
Golden fixture for the default 2D path (frame rms, mesh uniform), made BEFORE the quantile
frame / charge mesh were added, so test_quantile_frame.py can check that the defaults stay
bit-identical.

    conda run -n pydfcsr python pyDFCSR_2D/test/golden/make_golden_2d.py

Case: 20 000 particles, a tilted, chirped Gaussian (fixed seed), through a 0.3 m dipole and a
0.2 m drift (so the bend, curvature shear and exit-transient paths all run), small grids so it
takes seconds. Writes golden_2d.npz: every kick's dE_dct, x_kick and mesh axes, the newest
snapshot's five fields and frame scalars, and the final beam.
"""
import contextlib
import io
import os
import sys
import tempfile

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', '..'))
from pyDFCSR_2D.CSR import CSR2D  # noqa: E402

P0C = 5e9


def beam_coords(n=20000, seed=7):
    rng = np.random.default_rng(seed)
    z = rng.normal(0, 50e-6, n)
    pz = rng.normal(0, 2e-4, n) - 20.0 * z            # chirp
    x = rng.normal(0, 30e-6, n) + 0.5 * z             # tilt in x-z
    px = rng.normal(0, 5e-6, n)
    y = rng.normal(0, 20e-6, n)
    py = rng.normal(0, 5e-6, n)
    return np.column_stack([x, px, y, py, z, pz])


def config(tmp, extra_dep=None, extra_csr=None):
    np.savetxt(os.path.join(tmp, 'beam.txt'), beam_coords())
    lat = {'step_size': 0.05,
           'B1': {'type': 'dipole', 'L': 0.3, 'angle': 0.05, 'E1': 0.0, 'E2': 0.0, 'nsep': 1},
           'D1': {'type': 'drift', 'L': 0.2, 'nsep': 1}}
    yaml.safe_dump(lat, open(os.path.join(tmp, 'lat.yaml'), 'w'), sort_keys=False)
    cfg = {'input_beam': {'style': 'from_file', 'beamfile': os.path.join(tmp, 'beam.txt'),
                          'charge': 1.0e-9, 'energy': P0C},
           'input_lattice': {'lattice_input_file': os.path.join(tmp, 'lat.yaml')},
           'particle_deposition': dict({'xbins': 64, 'zbins': 64, 'xlim': 5, 'zlim': 5,
                                        'smoothing_sigma': 2.0, 'velocity_threhold': 1000},
                                       **(extra_dep or {})),
           'CSR_integration': {'n_formation_length': 1, 'xbins': 40, 'zbins': 40},
           'CSR_computation': dict({'compute_CSR': 1, 'apply_CSR': 1, 'transverse_on': 1,
                                    'xbins': 6, 'zbins': 9, 'xlim': 4, 'zlim': 4,
                                    'write_beam': None, 'write_wakes': True,
                                    'workdir': tmp, 'write_name': 'golden'},
                                   **(extra_csr or {}))}
    f = os.path.join(tmp, 'cfg.yaml')
    yaml.safe_dump(cfg, open(f, 'w'), sort_keys=False)
    return f


def run_case(extra_dep=None, extra_csr=None, parallel=False):
    """Run the golden case; returns a dict of arrays (see module doc)."""
    with tempfile.TemporaryDirectory() as tmp:
        with contextlib.redirect_stdout(io.StringIO()):
            sim = CSR2D(input_file=config(tmp, extra_dep, extra_csr), parallel=parallel)
            rec = {'dE': [], 'xk': [], 'xr': [], 'zr': []}

            def record():
                rec['dE'].append(sim.dE_dct.copy())
                rec['xk'].append(sim.x_kick.copy())
                rec['xr'].append(np.asarray(sim.CSR_xrange_transformed).copy())
                rec['zr'].append(np.asarray(sim.CSR_zrange).copy())
            sim.write_wakes = record
            sim.run()
        tr = sim.DF_tracker
        p = sim.beam.particle
        out = {f'{k}_{i}': v for k in rec for i, v in enumerate(rec[k])}
        out.update(nkicks=len(rec['dE']),
                   fields=np.stack([tr.density, tr.density_u, tr.density_w, tr.vx, tr.vx_u]),
                   frame=np.r_[tr.poly_coeffs, tr.xi_bar, tr.sigma_xi, tr.z_bar, tr.sigma_z,
                               tr.curv_c2, tr.curv_c3],
                   beam=np.array([p.x, p.px, p.y, p.py, p.z, p.pz]))
    return out


if __name__ == '__main__':
    out = run_case()
    np.savez(os.path.join(HERE, 'golden_2d.npz'), **out)
    print(f'golden_2d.npz: {out["nkicks"]} kicks, mesh {out["dE_0"].shape}, '
          f'max |dE| {max(np.abs(out[f"dE_{i}"]).max() for i in range(out["nkicks"])):.3e}')
