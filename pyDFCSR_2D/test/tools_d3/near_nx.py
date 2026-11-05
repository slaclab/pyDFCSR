"""Near-region x' node count: total wake at single points vs xbins (far region with band_jacobian_min 0.2),
against a converged reference (near +-15 sxi, 2001 nodes; far +-15 sxi, 2001 nodes). Also times a full 12x40 map."""
import sys, os, time, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from d3_wake_diagnostic import build, wake_mesh, evaluate
s_stop = float(sys.argv[1]); pts = [tuple(int(v) for v in p.split(':')) for p in sys.argv[2].split(',')]
sim = build(s_stop); t = sim.beam.position; ip = sim.integration_params; K = -sim.CSR_scaling
xm, zm = wake_mesh(sim); X, Z = xm.reshape(12, 40), zm.reshape(12, 40)
def region(s, x, sp, tp, nx, m, jmin, far):
    ip.xi_band_margin, ip.band_jacobian_min = m, jmin
    bands = sim._disjoint_bands(sim._retarded_xi_bands(s, x, t, sp, scale_bands=far)); frac = np.linspace(0, 1, nx); ci = np.zeros(sp.size)
    for lo, hi in bands:
        w = hi - lo; a = w > 0
        if not a.any(): continue
        xp = lo[None] + frac[:, None] * w[None]
        gz, _ = sim.get_CSR_integrand(s=s, t=t, x=x, xp=xp, sp=np.broadcast_to(sp[None], xp.shape).copy(), ignore_vx=False, taper=tp)
        ci += np.where(a, np.trapz(gz, axis=0) * w / (nx - 1), 0.0)
    ip.xi_band_margin, ip.band_jacobian_min = 2.0, 0.0
    return K * np.trapz(ci, x=sp)
for r, c in pts:
    x, z = X[r, c], Z[r, c]; s = t + z
    (s1, s3), (_, s4) = sim._layout_bounds(s, x); radii = sim._near_patch_radii(s, s3, s4); tpn = None if radii is None else (*radii, False)
    spf, spn = sim._far_region_nodes(s, x, t, s1, s3), sim._near_region_nodes(s, s3, s4)
    patch = sim._integrate_near_patch(s, x, t, *radii, False)[0] if radii else 0.0
    far_ref = region(s, x, spf, None, 2001, 3.0, 0.0, True); near_ref = region(s, x, spn, tpn, 2001, 3.0, 0.0, False)
    ref = far_ref + near_ref + patch
    far_j = region(s, x, spf, None, 200, 2.0, 0.2, True)
    row = []
    for nx in (200, 400, 600, 800):
        tot = far_j + region(s, x, spn, tpn, nx, 2.0, 0.0, False) + patch
        row.append(f'near nx {nx}: {abs(tot-ref)/abs(ref)*100:.2f}%')
    print(f'{sim.lattice.current_element} s={t:.3f} r{r}c{c} ref {ref:+.4f} (far {far_ref:+.4f}, far jac 200 {far_j:+.4f}) | ' + ', '.join(row), flush=True)
for nx in (200, 400):
    ip.xbins = nx; ip.band_jacobian_min = 0.2
    t0 = time.time(); evaluate(sim, xm, zm); print(f'  12x40 map, xbins {nx} (both regions), band_jacobian 0.2: {time.time()-t0:.1f} s', flush=True)
ip.xbins = 200; ip.band_jacobian_min = 0.0
