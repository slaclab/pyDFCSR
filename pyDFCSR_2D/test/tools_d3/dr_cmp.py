import sys, os, time, numpy as np
sys.path.insert(0, os.getcwd())
from d3_wake_diagnostic import build, wake_mesh, evaluate, OUT
s_stop = float(sys.argv[1])
u = np.linspace(-4, 4, 12); wx = np.exp(-u ** 2 / 2); wx /= wx.sum()
d = np.load(os.path.join(OUT, f'dreach_{s_stop:.4f}.npz')); Wd = d['maps'][list(d['thins']).index(0.01)]
for lab, ov in (('n_fl 1, current defaults', dict(n_formation_length=1.0)), ('n_fl 6, current defaults', dict(n_formation_length=6.0)),
                ('n_fl 12, current defaults', dict(n_formation_length=12.0))):
    sim = build(s_stop, {'CSR_integration': ov}); xm, zm = wake_mesh(sim)
    t0 = time.time(); W = evaluate(sim, xm, zm).reshape(12, 40); dt = time.time() - t0
    (s1, s3), _ = sim._layout_bounds(sim.beam.position, 0.0)
    nodes = getattr(sim, '_last_region_nodes', None)
    avg = (wx[:, None] * W).sum(0)
    print(f'{lab:26s} reach {s1:.3f}, {len(sim.DF_tracker.time_log)} snaps, {dt/W.size*1e3:.0f} ms/pt, nodes {nodes} | x-avg z=-2..2 '
          + ' '.join(f'{v:+.4f}' for v in avg[[10, 15, 20, 25, 30]]) + f' | rel L2 drift_reach(thin .01) vs this {np.linalg.norm(Wd - W)/np.linalg.norm(W):.4f}', flush=True)
