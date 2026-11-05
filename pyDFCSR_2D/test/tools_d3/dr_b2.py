import sys, os, time, numpy as np
sys.path.insert(0, os.getcwd())
from d3_wake_diagnostic import build, wake_mesh, evaluate
s_stop = 8.566
u = np.linspace(-4, 4, 12); wx = np.exp(-u ** 2 / 2); wx /= wx.sum()
sim = build(s_stop, {'CSR_integration': dict(n_formation_length=12.0)}); xm, zm = wake_mesh(sim)
t = sim.beam.position
# split the n_fl 12 far region at the B3 entrance: s' < 7.1062 (B2 and D2) vs s' >= 7.1062
parts = {}
for lab, cut in (('full', None), ('B3 onward only', 7.1062)):
    if cut is not None:
        lb = sim._layout_bounds
        sim._layout_bounds = lambda s, x, _lb=lb: ((max(_lb(s, x)[0][0], cut), _lb(s, x)[0][1]), _lb(s, x)[1])
    t0 = time.time(); W = evaluate(sim, xm, zm).reshape(12, 40); parts[lab] = W
    if cut is not None: sim._layout_bounds = lb
    avg = (wx[:, None] * W).sum(0)
    print(f'{lab:28s} x-avg z=-2..2 ' + ' '.join(f'{v:+.4f}' for v in avg[[10, 15, 20, 25, 30]]) + f' | {time.time()-t0:.0f} s', flush=True)
D = parts['full'] - parts['B3 onward only']
print('B2+D2 contribution: x-avg z=-2..2', ' '.join(f'{v:+.4f}' for v in (wx[:, None] * D).sum(0)[[10, 15, 20, 25, 30]]),
      f'; rms over map {np.sqrt(np.mean(D**2)):.4f}, peak {np.abs(D).max():.4f}; rel L2 of the full map {np.linalg.norm(D)/np.linalg.norm(parts["full"]):.3f}')
print('B2+D2 contribution rows at z=0:', np.round(D[:, 20], 3))
