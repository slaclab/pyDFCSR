"""near_cell sweep at single points (near region only, band and x' fixed at default), with the near-region node count,
plus a full 12x40 map timing at each near_cell. Reference: near_cell 0.0625, xbins 800, margin 3."""
import sys, os, time, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from d3_wake_diagnostic import build, wake_mesh, evaluate
s_stop = float(sys.argv[1]); pts = [tuple(int(v) for v in p.split(':')) for p in sys.argv[2].split(',')]
sim = build(s_stop); t = sim.beam.position; ip = sim.integration_params; K = -sim.CSR_scaling
xm, zm = wake_mesh(sim); X, Z = xm.reshape(12, 40), zm.reshape(12, 40)
def near(s, x, cell, nx=200, margin=2.0):
    ip.near_cell, ip.xi_band_margin = cell, margin
    (s1, s3), (_, s4) = sim._layout_bounds(s, x); sp = sim._near_region_nodes(s, s3, s4); radii = sim._near_patch_radii(s, s3, s4)
    tp = None if radii is None else (*radii, False)
    bands = sim._disjoint_bands(sim._retarded_xi_bands(s, x, t, sp, scale_bands=False)); frac = np.linspace(0, 1, nx); ci = np.zeros(sp.size)
    for lo, hi in bands:
        w = hi - lo; a = w > 0
        if not a.any(): continue
        xp = lo[None] + frac[:, None] * w[None]
        gz, _ = sim.get_CSR_integrand(s=s, t=t, x=x, xp=xp, sp=np.broadcast_to(sp[None], xp.shape).copy(), ignore_vx=False, taper=tp)
        ci += np.where(a, np.trapz(gz, axis=0) * w / (nx - 1), 0.0)
    patch = sim._integrate_near_patch(s, x, t, *radii, False)[0] if radii else 0.0
    ip.near_cell, ip.xi_band_margin = 0.5, 2.0
    return K * np.trapz(ci, x=sp) + patch, sp.size
for r, c in pts:
    x, z = X[r, c], Z[r, c]; s = t + z
    ref, nr = near(s, x, 0.0625, 800, 3.0); full = sim.get_CSR_wake(s, x)[0]
    out = []
    for cell in (0.5, 0.25, 0.125, 0.0625):
        v, n = near(s, x, cell); out.append(f'cell {cell}: {(v - ref)/abs(full)*100:+.2f}% ({n} s\')')
    print(f'{sim.lattice.current_element} s={t:.3f} r{r}c{c}: near ref {ref:+.4f}, total {full:+.4f}; near error as % of TOTAL wake | ' + ', '.join(out), flush=True)
for cell in (0.5, 0.25, 0.125):
    t0 = time.time(); evaluate(sim, xm, zm, dict(near_cell=cell)); print(f'  12x40 map at near_cell {cell}: {time.time()-t0:.1f} s', flush=True)
