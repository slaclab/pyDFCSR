"""Curvature shear vs wide grid at one wake point: the total (far + near + patch) computed with the default band,
and with a WIDE band + many nodes (+-30 sigma_xi in u, i.e. margin 6 at xlim 5, 2001 x' nodes, far band
Jacobian-scaled) so the band cannot be the difference. Variants: default deposition, curvature, xlim 9."""
import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from d3_wake_diagnostic import build, wake_mesh
s_stop, r = float(sys.argv[1]), int(sys.argv[2]); zcs = [float(v) for v in sys.argv[3].split(',')]
for lab, dep, wide_m in (('default xlim 5', None, 6.0), ('curvature xlim 5', dict(curvature=True), 6.0), ('xlim 9', dict(xlim=9, xbins=540), 6.0 * 5 / 9)):
    sim = build(s_stop, {'particle_deposition': dep} if dep else None); t = sim.beam.position; ip = sim.integration_params; K = -sim.CSR_scaling
    xm, zm = wake_mesh(sim); X, Z = xm.reshape(12, 40), zm.reshape(12, 40); b = sim.beam
    out = []
    for zc in zcs:
        z = b._mean_z + zc * b._sigma_z; x = np.interp(z, Z[r], X[r]); s = t + z
        (s1, s3), (_, s4) = sim._layout_bounds(s, x); radii = sim._near_patch_radii(s, s3, s4)
        def reg(sp, tp, nx, m, far):
            ip.xi_band_margin, ip.band_jacobian_min = m, 0.2
            bands = sim._disjoint_bands(sim._retarded_xi_bands(s, x, t, sp, scale_bands=far)); frac = np.linspace(0, 1, nx); ci = np.zeros(sp.size)
            for lo, hi in bands:
                w = hi - lo; a = w > 0
                if not a.any(): continue
                xp = lo[None] + frac[:, None] * w[None]
                gz, _ = sim.get_CSR_integrand(s=s, t=t, x=x, xp=xp, sp=np.broadcast_to(sp[None], xp.shape).copy(), ignore_vx=False, taper=tp)
                ci += np.where(a, np.trapz(gz, axis=0) * w / (nx - 1), 0.0)
            ip.xi_band_margin, ip.band_jacobian_min = 2.0, 0.0
            return K * np.trapz(ci, x=sp)
        spf, spn = sim._far_region_nodes(s, x, t, s1, s3), sim._near_region_nodes(s, s3, s4)
        tpn = None if radii is None else (*radii, False)
        patch = sim._integrate_near_patch(s, x, t, *radii, False)[0] if radii else 0.0
        m0 = 2.0 if dep is None or 'xlim' not in dep else 2.0 * 5 / 9
        std = reg(spf, None, 200, m0, True) + reg(spn, tpn, 200, m0, False) + patch
        wide = reg(spf, None, 2001, wide_m, True) + reg(spn, tpn, 2001, wide_m, False) + patch
        out.append(f'z/sz {zc:+.1f}: band 10 sxi {std:+.5f}, wide/fine {wide:+.5f}')
    print(f's = {t:.3f} row {r} {lab:18s} clipped {sim.DF_tracker.clipped*100:.3f}% | ' + ' | '.join(out), flush=True)
