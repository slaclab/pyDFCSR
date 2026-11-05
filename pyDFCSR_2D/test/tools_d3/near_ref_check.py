"""Near region at D3 points where the +-40 sigma_xi reference and the default band disagree: is the reference
itself converged? Near-region total vs band margin (1..8) and x' nodes (200, 801, 2001), no Jacobian scaling."""
import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from d3_wake_diagnostic import build, wake_mesh
s_stop, r, c = float(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
sim = build(s_stop); t = sim.beam.position; ip = sim.integration_params; K = -sim.CSR_scaling
xm, zm = wake_mesh(sim); x, z = xm.reshape(12, 40)[r, c], zm.reshape(12, 40)[r, c]; s = t + z
(s1, s3), (_, s4) = sim._layout_bounds(s, x); radii = sim._near_patch_radii(s, s3, s4); tp = (*radii, False)
def tot(sp, nx, m):
    ip.xi_band_margin = m
    bands = sim._disjoint_bands(sim._retarded_xi_bands(s, x, t, sp)); frac = np.linspace(0, 1, nx); ci = np.zeros(sp.size)
    for lo, hi in bands:
        w = hi - lo; a = w > 0
        if not a.any(): continue
        xp = lo[None] + frac[:, None] * w[None]
        gz, _ = sim.get_CSR_integrand(s=s, t=t, x=x, xp=xp, sp=np.broadcast_to(sp[None], xp.shape).copy(), ignore_vx=False, taper=tp)
        ci += np.where(a, np.trapz(gz, axis=0) * w / (nx - 1), 0.0)
    ip.xi_band_margin = 2.0
    return K * np.trapz(ci, x=sp)
sp = sim._near_region_nodes(s, s3, s4)
ip.near_grade = 0.01; spf = sim._near_region_nodes(s, s3, s4); ip.near_grade = 0.05
print(f's = {t:.3f} r{r}c{c}, near region ({sp.size} / {spf.size} s\' nodes at near_grade 0.05 / 0.01):')
for m in (1.0, 2.0, 3.0, 5.0, 8.0):
    print(f'  margin {m:g} (+-{5*m:.0f} sxi):  ' + '  '.join(f'nx {nx}: {tot(sp, nx, m):+.5f}' for nx in (200, 801, 2001)) + f'   | fine s\' nx 801: {tot(spf, 801, m):+.5f}', flush=True)
