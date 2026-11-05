import sys, os, time, numpy as np
sys.path.insert(0, os.getcwd())
from d3_wake_diagnostic import build, wake_mesh, evaluate, OUT
s_stop = float(sys.argv[1])
u = np.linspace(-4, 4, 12); wx = np.exp(-u ** 2 / 2); wx /= wx.sum()
sim = build(s_stop, {'CSR_integration': dict(n_formation_length=1.0, drift_reach=True, drift_snap_thin=0.01)})
xm, zm = wake_mesh(sim)
V = [('base far_cell 10, window 1', {}), ('far_cell 30', dict(far_cell=30.0)), ('far_cell 100', dict(far_cell=100.0)),
     ('far_cell 300', dict(far_cell=300.0)), ('window 0.5', dict(far_window=0.5)), ('window 0.5, cell 30', dict(far_window=0.5, far_cell=30.0)),
     ('window 0.3, cell 100', dict(far_window=0.3, far_cell=100.0)), ('far_zbins 100, cell 100', dict(far_zbins=100, far_cell=100.0)),
     ('ref: cell 3, window 2', dict(far_cell=3.0, far_window=2.0))]
res = {}; L = [f'far-region layout sweep at s = {sim.beam.position:.4f} ({sim.lattice.current_element}), drift_reach, thin 0.01']
for name, p in V:
    t0 = time.time(); W = evaluate(sim, xm, zm, p).reshape(12, 40); dt = time.time() - t0; res[name] = (W, dt)
    nf = sim._last_region_nodes[0]
    L.append(f'{name:28s} far nodes {nf:5d}  {dt/W.size*1e3:6.1f} ms/pt  x-avg z=0 {(wx[:, None]*W).sum(0)[20]:+.4f}'); print(L[-1], flush=True)
ref = res['ref: cell 3, window 2'][0]
L.append('rel L2 vs reference (cell 3, window 2): ' + ', '.join(f'{n}: {np.linalg.norm(w - ref)/np.linalg.norm(ref):.2e}' for n, (w, _) in res.items()))
print(L[-1]); open(os.path.join(OUT, f'farsweep_{s_stop:.4f}.txt'), 'w').write('\n'.join(L) + '\n')
