"""Per-particle kick bookkeeping for the slice energy spread study.
Wraps Beam.apply_wakes to accumulate, for every particle, its total CSR energy kick K and the
part of K that the wake would have given it at the beam's centre x (x_transform = mean):
    K = K_z + K_x,   K_z = sum_k L_k W_k(<x_t>, z_i,k),   K_x = K - K_z
K_z is what a wake with no x-dependence would give along the particle's own z path, so its spread
within a final slice measures slice mixing; K_x is the wake's own x-dependence (including particles
outside the +-xlim wake mesh, which get no kick). Saves per-particle arrays to kicks_<case>.npz."""
import os, sys, numpy as np
from scipy.interpolate import RegularGridInterpolator
from pyDFCSR_2D import CSR2D

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'benchmark_results', 'slice_espread_dipole')
os.chdir(os.path.join(HERE, '..', 'example'))
case = sys.argv[1]   # input/sesp_<case>_config.yaml
sim = CSR2D(input_file=f'input/sesp_{case}_config.yaml')
sim.CSR_params.write_wakes = False; sim.CSR_params.write_beam = None
b = sim.beam; N = len(b.particle.x); E0 = b.init_energy
K, Kz, out_mesh = np.zeros(N), np.zeros(N), np.zeros(N, bool)
x0, xp0, z0, d0 = b.particle.x.copy(), b.particle.px.copy(), b.particle.z.copy(), b.particle.pz.copy()
_apply = b.apply_wakes
def apply_wakes(dE_dct, x_kick, xrange, zrange, step_size, transverse_on):
    global K, Kz
    pz_before = b.particle.pz.copy()
    xt, z = b.x_transform, b.particle.z
    f = RegularGridInterpolator((xrange, zrange), step_size * dE_dct * 1e6 / E0, fill_value=0.0, bounds_error=False)
    Kz += f(np.array([np.full(N, xt.mean()), z]).T)
    out_mesh[(xt < xrange[0]) | (xt > xrange[-1]) | (z < zrange[0]) | (z > zrange[-1])] = True
    _apply(dE_dct, x_kick, xrange, zrange, step_size, transverse_on)
    K += b.particle.pz - pz_before
b.apply_wakes = apply_wakes
sim.run()
p = b.particle
np.savez(os.path.join(OUT, f'kicks_{case}.npz'), E0=E0, K=K, Kz=Kz, out_mesh=out_mesh,
         x0=x0, xp0=xp0, z0=z0, d0=d0, zf=p.z, df=p.pz, xf=p.x)
print('saved; fraction ever outside wake mesh:', out_mesh.mean())
