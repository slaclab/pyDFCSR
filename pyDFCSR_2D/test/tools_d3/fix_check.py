"""Frozen-state checks of (a) the wake mesh following the curvature ridge and (b) prev_bend_reach at a bend
entrance: wake map with/without, x-avg and the extreme points; B4 entrance compared with 1D at the same s."""
import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from d3_wake_diagnostic import build, wake_mesh, evaluate
from pyDFCSR_2D.csr1d import line_density, wake_in_bend, wake_after_bend, CSR1DGeometry
s_stop = float(sys.argv[1]); mode = sys.argv[2]
u = np.linspace(-4, 4, 12); wx = np.exp(-u ** 2 / 2); wx /= wx.sum()
V = {'mesh': [('curv off', None, {}), ('curv on', {'particle_deposition': dict(curvature=True)}, {})],
     'prev': [('prev off', None, {}), ('prev on', None, dict(prev_bend_reach=True))]}[mode]
for lab, dep, integ in V:
    sim = build(s_stop, dep); xm, zm = wake_mesh(sim)
    W = evaluate(sim, xm, zm, dict(band_jacobian_min=0.2, **integ)).reshape(12, 40)
    av = (wx[:, None] * W).sum(0)
    i, j = np.unravel_index(np.abs(W).argmax(), W.shape)
    print(f'{lab:9s} s = {sim.beam.position:.4f} ({sim.lattice.current_element}): x-avg z=-2,-1,0,1,2 ' + ' '.join(f'{v:+.4f}' for v in av[[10, 15, 20, 25, 30]])
          + f' | peak |W| {np.abs(W).max():.4f} at row {i} col {j} | row 11 at z +1.7: {W[11, 28]:+.4f} | x-spread rms {np.sqrt(np.mean((wx[:, None]*(W-av)**2).sum(0))):.4f}', flush=True)
g = CSR1DGeometry(sim.lattice.lattice_config, sim.lattice.distance).at(sim.beam.position, sim.lattice.current_element)
S, ds, lam, dl = line_density(sim.beam.z, 300, abs(sim.beam.charge) / 1.602176634e-19, 2)
W1 = (wake_in_bend(S, ds, lam, dl, g['R'], g['phi']) if g['case'] == 'AB' else wake_after_bend(S, ds, lam, dl, g['R'], g['phim'], g['x_R'])) / 1e6
Z = zm.reshape(12, 40)[0]
print(f'1D (case {g["case"]}, own bend only)  x-avg z=-2,-1,0,1,2 ' + ' '.join(f'{v:+.4f}' for v in np.interp(Z, S, W1)[[10, 15, 20, 25, 30]]))
