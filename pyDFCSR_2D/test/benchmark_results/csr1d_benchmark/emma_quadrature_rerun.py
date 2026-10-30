"""
Closure test for the integrated 1D-vs-Emma differences. Reruns a benchmark case through pyDFCSR's 1D
mode (same bmad-x tracking, same binning) with the wake kernel replaced by a literal port of Emma's
csr_4case_calc quadrature (test_csr1d.emma_port: rectangle sum that skips the u < ds cell) and his
derivative rule. If this reproduces Emma's integrated CSR effects, the differences in the main
benchmark are the kernel quadrature, not tracking, binning, smoothing or geometry.
Usage: python emma_quadrature_rerun.py <case>   -> appends to emma_quadrature_rerun.txt
"""
import contextlib, io, os, sys, numpy as np, h5py
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, '..', '..'))
import benchmark_csr1d_vs_emma as B
import pyDFCSR_2D.csr1d as C
from test_csr1d import emma_port, emma_dlam
name = sys.argv[1]
def emma_wake(S, ds, lam, dlam, geom):
    if geom['case'] == 'none': return np.zeros_like(lam)
    return emma_port(S, lam, ds, abs(geom['R']), abs(geom['phi']), abs(geom['phim']), geom['x_R'] if geom['case'] == 'CD' else 0.0)
C.wake_1d = emma_wake                                    # swapped for this process only
case = B.CASES[name]; files = B.write_inputs(name + '_emmaq', case)
res = {}
for on in (0, 1):
    sim = C.CSR1D(input_file=files[on]); sim.CSR_params.write_wakes = False
    with contextlib.redirect_stdout(io.StringIO()): sim.run()
    p = sim.beam.particle; res[on] = dict(x=p.x, xp=p.px, z=p.z, d=p.pz); E0 = sim.beam.init_energy
B.G0 = E0 / B.MC2
q = B.metrics(res[1], res[0], E0)
em = {}
for on, suf in ((1, ''), (0, '_off')):
    X = h5py.File(os.path.join(HERE, f'1d_{name}{suf}.mat'))['Xs'][()]; em[on] = dict(x=X[:, 0], xp=X[:, 1], z=X[:, 2], d=X[:, 3])
e = B.metrics(em[1], em[0], E0)
L = [f'\n=== {name}: pyDFCSR 1D with EMMA\'S quadrature vs Emma ===', f'  {"":34s} {"pyDFCSR+Emma-quad":>17s} {"Emma":>10s} {"ratio":>7s}']
for k, lab in [('dE', 'mean CSR energy change [keV]'), ('dsEp', 'projected sigma_E change [keV]'), ('sEs', 'CSR-added slice sigma_E [keV]'), ('demit', 'emittance growth [um]'), ('sz', 'final sigma_z [um]')]:
    L.append(f'  {lab:34s} {q[k]:17.3f} {e[k]:10.3f} {q[k]/e[k] if e[k] else np.nan:7.3f}')
open(os.path.join(HERE, 'emma_quadrature_rerun.txt'), 'a').write('\n'.join(L) + '\n'); print('\n'.join(L))
