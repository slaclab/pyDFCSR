"""Near-region convergence at single wake points, now with the far band at its default (Jacobian-scaled).
The near region total (bands + polar patch) for: x' nodes (xbins 200/400/800/1600), band half-width (margin 2/3/4),
s' grading (near_grade 0.05/0.02/0.01), near_cell (0.5/0.25). Reference = finest in every direction at once.
Prints the error of the default and of each single refinement, so the limiting direction is visible."""
import sys, os, time, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from d3_wake_diagnostic import build, wake_mesh
s_stop = float(sys.argv[1]); pts = [tuple(int(v) for v in p.split(':')) for p in sys.argv[2].split(',')]
sim = build(s_stop); t = sim.beam.position; ip = sim.integration_params; K = -sim.CSR_scaling
xm, zm = wake_mesh(sim); X, Z = xm.reshape(12, 40), zm.reshape(12, 40)
def near(s, x, nx, margin, grade, cell):
    ip.xi_band_margin, ip.near_grade, ip.near_cell = margin, grade, cell
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
    ip.xi_band_margin, ip.near_grade, ip.near_cell = 2.0, 0.05, 0.5
    return K * np.trapz(ci, x=sp) + patch, sp.size
D = (200, 2.0, 0.05, 0.5)
V = [('default', D), ('xbins 400', (400, 2.0, 0.05, 0.5)), ('xbins 800', (800, 2.0, 0.05, 0.5)), ('margin 3', (200, 3.0, 0.05, 0.5)),
     ('grade 0.01', (200, 2.0, 0.01, 0.5)), ('cell 0.25', (200, 2.0, 0.05, 0.25)), ('xbins 800 + grade 0.01', (800, 2.0, 0.01, 0.5))]
REF = (1600, 4.0, 0.01, 0.25)
for r, c in pts:
    x, z = X[r, c], Z[r, c]; s = t + z
    ref, nref = near(s, x, *REF)
    out = []
    for lab, p in V:
        v, n = near(s, x, *p); out.append(f'{lab} {abs(v - ref) / max(abs(ref), 1e-12) * 100:.2f}%')
    full = sim.get_CSR_wake(s, x)[0]
    print(f'{sim.lattice.current_element} s={t:.3f} r{r}c{c} z/sz {(z - sim.beam._mean_z)/sim.beam._sigma_z:+.1f}: near ref {ref:+.4f} '
          f'(total wake {full:+.4f}; near error as % of total: default {abs(near(s, x, *D)[0] - ref)/max(abs(full),1e-12)*100:.2f}%) | ' + ', '.join(out), flush=True)
