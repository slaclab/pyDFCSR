"""Writes the beam/config YAMLs for the 1D-vs-2D tilt scan: 1 m / 1 rad dipole (0.1 m drift, bend,
0.5 m drift), the same 5 GeV, 1 nC, sigma_z = sigma_x = 50 um, sigma_E = 500 keV beam as the
slice-energy-spread study, with an initial x-z shear x += k z. 2D runs use the comoving deposition
with frame_blend 'orient' (correct through a longitudinal waist) and step_control auto.
Usage: tilt_scan_setup.py k [k ...]   ->  example/input/ts_{tag}_{beam,config}.yaml"""
import os, sys, yaml
HERE = os.path.dirname(os.path.abspath(__file__)); INP = os.path.join(HERE, '..', 'example', 'input')
def tag(k): return ('m' if k < 0 else 'p') + f'{abs(k):g}'.replace('.', 'p')
for k in map(float, sys.argv[1:]):
    b = yaml.safe_load(open(os.path.join(INP, 'dipole_chirp0_beam.yaml')))
    if k != 0:
        b['transforms'] = {'s1': {'type': 'shear z:x', 'shear_coefficient': {'units': 'dimensionless', 'value': k}}}
    yaml.safe_dump(b, open(os.path.join(INP, f'ts_{tag(k)}_beam.yaml'), 'w'))
    c = {'input_beam': {'style': 'distgen', 'distgen_input_file': f'input/ts_{tag(k)}_beam.yaml'},
         'input_lattice': {'lattice_input_file': 'input/dipole_lattice.yaml'},
         'particle_deposition': dict(method='bspline_comoving', xbins=200, zbins=200, xlim=5, zlim=5, smoothing_sigma=3.0,
                                     poly_degree=1, velocity_threhold=1000, frame_blend='orient'),
         'CSR_integration': dict(n_formation_length=1.5, xbins=100, zbins=100),
         # mesh widened from +-3 to +-4 sigma: at +-3, 1.4-2.1% of particles got a zero kick (slice_espread study)
         'CSR_computation': dict(compute_CSR=1, apply_CSR=1, transverse_on=1, xbins=12, zbins=40, xlim=4, zlim=4,
                                 write_beam=None, write_wakes=True, write_name=f'ts_{tag(k)}', workdir='./output'),
         'step_control': dict(mode='auto', kick_interval='midpoint', n_sub=400)}
    yaml.safe_dump(c, open(os.path.join(INP, f'ts_{tag(k)}_config.yaml'), 'w'))
    print('wrote', tag(k))
