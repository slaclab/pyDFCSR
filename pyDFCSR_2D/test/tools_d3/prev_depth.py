"""How deep into the drift does the prev_bend_reach contribution reach? At a frozen B4 state, integrate the extra
region from the previous bend's entrance cut at s1 - L for L = 0.5, 1, 2, 3 m and the full region."""
import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from d3_wake_diagnostic import build, wake_mesh
s_stop = float(sys.argv[1])
sim = build(s_stop, {'CSR_integration': dict(prev_bend_reach=True)}); t = sim.beam.position
xm, zm = wake_mesh(sim); X, Z = xm.reshape(12, 40), zm.reshape(12, 40)
for r, c in ((6, 20), (6, 15), (1, 20), (10, 25)):
    x, z = X[r, c], Z[r, c]; s = t + z
    (s1, s3), _ = sim._layout_bounds(s, x); pb = sim._prev_bend_reach_start()
    full = sim._integrate_xi_region(s, x, t, sim._far_region_nodes(s, x, t, pb, s1, transient=True), False, None, True)[0]
    out = []
    for L in (0.5, 1.0, 2.0, 3.0):
        lo = max(pb, s1 - L)
        v = sim._integrate_xi_region(s, x, t, sim._far_region_nodes(s, x, t, lo, s1, transient=True), False, None, True)[0]
        out.append(f'last {L:g} m: {v:+.4f} ({v/full*100 if full else 0:.0f}%)')
    print(f's {t:.3f} r{r}c{c}: s1 {s1:.3f}, full [{pb:.3f},{s1:.3f}] {full:+.4f} | ' + ', '.join(out), flush=True)
print('layout test: exit-transient layout (default) vs pure log grading from s1 down to pb')
for r, c in ((6, 20), (6, 15), (1, 20), (10, 25)):
    x, z = X[r, c], Z[r, c]; s = t + z
    (s1, s3), _ = sim._layout_bounds(s, x); pb = sim._prev_bend_reach_start()
    spA = sim._far_region_nodes(s, x, t, pb, s1, transient=True)
    A = sim._integrate_xi_region(s, x, t, spA, False, None, True)[0]
    ref_sp = np.unique(np.concatenate([np.linspace(pb, s1, 4001), spA]))
    ref = sim._integrate_xi_region(s, x, t, ref_sp, False, None, True)[0]
    out = [f'transient {spA.size} nodes {A:+.4f}']
    for g in (0.05, 0.1, 0.2):
        u_hi, u_lo = s - pb, s - s1
        k = int(np.ceil(np.log(u_hi / u_lo) / np.log1p(g))); spB = s - np.minimum(u_lo * (1 + g) ** np.arange(k + 1), u_hi)
        B = sim._integrate_xi_region(s, x, t, np.unique(spB), False, None, True)[0]
        out.append(f'log {g} ({spB.size} nodes) {B:+.4f}')
    print(f'  r{r}c{c}: reference (4001 uniform) {ref:+.4f} | ' + ', '.join(out), flush=True)
print('log grading at 0.02 / 0.01, and the B3 part on its own with a fine window')
for r, c in ((6, 20), (10, 25)):
    x, z = X[r, c], Z[r, c]; s = t + z
    (s1, s3), _ = sim._layout_bounds(s, x); pb = sim._prev_bend_reach_start()
    ref_sp = np.unique(np.concatenate([np.linspace(pb, s1, 4001), sim._far_region_nodes(s, x, t, pb, s1, transient=True)]))
    ref = sim._integrate_xi_region(s, x, t, ref_sp, False, None, True)[0]; out = []
    for g in (0.02, 0.01):
        u_hi, u_lo = s - pb, s - s1
        k = int(np.ceil(np.log(u_hi / u_lo) / np.log1p(g))); spB = np.unique(s - np.minimum(u_lo * (1 + g) ** np.arange(k + 1), u_hi))
        out.append(f'log {g} ({spB.size}) {sim._integrate_xi_region(s, x, t, spB, False, None, True)[0]:+.4f}')
    print(f'  r{r}c{c}: ref {ref:+.4f} | ' + ', '.join(out), flush=True)
