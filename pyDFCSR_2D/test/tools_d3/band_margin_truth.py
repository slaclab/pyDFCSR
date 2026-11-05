"""D3: does the far-region column integral depend on the band half-width? Per s' column at one observer:
band integral at several xi_band_margin vs a brute-force lab-x' sum spanning +-W around each band centre."""
import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from d3_wake_diagnostic import build, wake_mesh
s_stop, row, col = float(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
dep = {'particle_deposition': dict(xlim=8, xbins=480)} if len(sys.argv) > 4 and sys.argv[4] == 'x8' else None
sim = build(s_stop, dep); t = sim.beam.position; ip = sim.integration_params; tr = sim.DF_tracker
xm, zm = wake_mesh(sim); x, z = xm.reshape(12, 40)[row, col], zm.reshape(12, 40)[row, col]; s = t + z
(s1, s3), (_, s4) = sim._layout_bounds(s, x)
K = -sim.CSR_scaling
for region, sp in (('far', sim._far_region_nodes(s, x, t, s1, s3)), ('near', sim._near_region_nodes(s, s3, s4))):
    radii = sim._near_patch_radii(s, s3, s4); tp = None if region == 'far' or radii is None else (*radii, False)
    out = {}
    for m in (1.0, 2.0, 3.2, 5.0, 8.0):
        ip.xi_band_margin = m
        bands = sim._disjoint_bands(sim._retarded_xi_bands(s, x, t, sp)); frac = np.linspace(0, 1, 801); ci = np.zeros(sp.size)
        for lo, hi in bands:
            width = hi - lo; alive = width > 0
            if not alive.any(): continue
            xp = lo[None, :] + frac[:, None] * width[None, :]
            gz, _ = sim.get_CSR_integrand(s=s, t=t, x=x, xp=xp, sp=np.broadcast_to(sp[None, :], xp.shape).copy(), ignore_vx=False, taper=tp)
            ci += np.where(alive, np.trapz(gz, axis=0) * width / 800, 0.0)
        out[m] = K * np.trapz(ci, x=sp)
    ip.xi_band_margin = 2.0
    print(f's = {t:.4f} row {row} col {col} grid xlim {tr.xlim}: {region:4s} region ({sp.size} cols) vs band margin '
          + ', '.join(f'{m:g} (±{m*tr.xlim:.0f} σξ): {v:+.5f}' for m, v in out.items()), flush=True)
