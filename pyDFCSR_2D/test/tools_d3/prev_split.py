"""Split the prev_bend_reach extra region at a B4 observer into the previous BEND (s' in B3) and the DRIFT between
(D3), with v_x on and off: which part carries the -0.3 MeV/m, and is the D3 part the self-field work?"""
import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from d3_wake_diagnostic import build, wake_mesh
s_stop = float(sys.argv[1]); b3 = (7.1062, 7.6064)
sim = build(s_stop, {'CSR_integration': dict(prev_bend_reach=True, band_jacobian_min=0.2)}); t = sim.beam.position
xm, zm = wake_mesh(sim); X, Z = xm.reshape(12, 40), zm.reshape(12, 40)
for r, c in ((6, 20), (6, 15), (1, 20)):
    x, z = X[r, c], Z[r, c]; s = t + z
    (s1, s3), (_, s4) = sim._layout_bounds(s, x); pb = sim._prev_bend_reach_start()
    out = []
    for lab, lo, hi in (('B3', pb, b3[1]), ('D3', b3[1], s1)):
        sp = sim._far_region_nodes(s, x, t, lo, hi, transient=True)
        on = sim._integrate_xi_region(s, x, t, sp, False, None, scale_bands=True)[0]
        off = sim._integrate_xi_region(s, x, t, sp, True, None, scale_bands=True)[0]
        out.append(f'{lab} [{lo:.3f},{hi:.3f}] v_x on {on:+.4f} / off {off:+.4f}')
    print(f'row {r} col {c} (z/sz {(z - sim.beam._mean_z)/sim.beam._sigma_z:+.1f}): ' + ' | '.join(out), flush=True)
