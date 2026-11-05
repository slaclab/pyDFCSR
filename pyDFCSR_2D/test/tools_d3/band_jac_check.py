"""Jacobian-scaled bands (Integration_params.band_jacobian_min) at single wake points: far + near region totals
for the default band, the scaled band at several floors, and a wide reference (margin 8 = +-40 sigma_xi of x',
801 x' nodes, no scaling). Also checks the analytic du/dx' against a finite difference."""
import sys, os, time, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from d3_wake_diagnostic import build, wake_mesh
s_stop = float(sys.argv[1]); pts = [tuple(int(v) for v in p.split(':')) for p in sys.argv[2].split(',')]
sim = build(s_stop); t = sim.beam.position; ip = sim.integration_params; K = -sim.CSR_scaling
xm, zm = wake_mesh(sim); X, Z = xm.reshape(12, 40), zm.reshape(12, 40)
def region_total(s, x, sp, tp, nx, margin, jmin, far=True):
    ip.xi_band_margin, ip.band_jacobian_min = margin, jmin
    bands = sim._disjoint_bands(sim._retarded_xi_bands(s, x, t, sp, scale_bands=far)); frac = np.linspace(0, 1, nx); ci = np.zeros(sp.size)
    for lo, hi in bands:
        w = hi - lo; a = w > 0
        if not a.any(): continue
        xp = lo[None] + frac[:, None] * w[None]
        gz, _ = sim.get_CSR_integrand(s=s, t=t, x=x, xp=xp, sp=np.broadcast_to(sp[None], xp.shape).copy(), ignore_vx=False, taper=tp)
        ci += np.where(a, np.trapz(gz, axis=0) * w / (nx - 1), 0.0)
    ip.xi_band_margin, ip.band_jacobian_min = 2.0, 0.0
    return K * np.trapz(ci, x=sp)
V = [('default', 200, 2.0, 0.0), ('jac 0.2 far', 200, 2.0, 0.2), ('ref', 2001, 3.0, 0.0)]   # ref: +-15 sxi, 2001 nodes (converged: tools_d3/near_ref_check.py)
for r, c in pts:
    x, z = X[r, c], Z[r, c]; s = t + z
    (s1, s3), (_, s4) = sim._layout_bounds(s, x); radii = sim._near_patch_radii(s, s3, s4)
    regs = (('far', sim._far_region_nodes(s, x, t, s1, s3), None), ('near', sim._near_region_nodes(s, s3, s4), None if radii is None else (*radii, False)))
    tot = {v[0]: 0.0 for v in V}; parts = []
    for nm, sp, tp in regs:
        vals = {lab: region_total(s, x, sp, tp, nx, m, j, far=(nm == 'far')) for lab, nx, m, j in V}
        for k in tot: tot[k] += vals[k]
        parts.append(f"{nm}: " + ' '.join(f'{vals[k]:+.4f}' for k in vals))
    ref = tot['ref']
    print(f'{sim.lattice.current_element} s={t:.3f} r{r}c{c} | ' + ' | '.join(parts) + ' || total err vs ref: '
          + ', '.join(f'{k} {abs(tot[k]-ref)/max(abs(ref),1e-12)*100:.2f}%' for k in tot if not k.startswith('ref')) + f'  (ref {ref:+.4f} MeV/m)', flush=True)
# analytic J vs finite difference, far region of the first point
r, c = pts[0]; x, z = X[r, c], Z[r, c]; s = t + z; (s1, s3), _ = sim._layout_bounds(s, x)
sp = sim._far_region_nodes(s, x, t, s1, s3)
ip.band_jacobian_min = 1e-6; b1 = sim._retarded_xi_bands(s, x, t, sp); ip.band_jacobian_min = 0.0; b0 = sim._retarded_xi_bands(s, x, t, sp)
Jan = (b0[0][1] - b0[0][0]) / np.maximum(b1[0][1] - b1[0][0], 1e-300)
print('analytic |J| over the far columns of the first point: min', np.round(Jan[Jan > 0].min(), 3), 'max', np.round(Jan.max(), 3))
