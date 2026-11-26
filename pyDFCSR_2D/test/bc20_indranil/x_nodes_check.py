import os, sys, time, numpy as np
T = '/Users/jytang/Library/CloudStorage/GoogleDrive-jytang@stanford.edu/My Drive/Stanford/research/FEL/pyDFCSR_claude/pyDFCSR_near_adapt/pyDFCSR_2D/test'
sys.path.insert(0, os.path.join(T, '..', '..')); sys.path.insert(0, os.path.join(T, 'facet_bc20'))
import dump_kick_state as D
D.RUNS = os.path.join(T, 'benchmark_results', 'bc20_indranil', 'runs')
run, st = sys.argv[1], float(sys.argv[2]); stride = int(sys.argv[3]) if len(sys.argv) > 3 else 1
sim = D.load_state(D.state_path(run, st)); ip = sim.integration_params; pos = sim.beam.position
nx, nz = sim.CSR_params.xbins, sim.CSR_params.zbins
idx = [i * nz + j for i in range(nx) for j in range(0, nz, stride)]
xi = np.asarray(sim.CSR_xrange_transformed); h = np.gradient(xi); n, L = xi.size, xi[-1] - xi[0]
wx = np.clip(1.0 / ((n - 1) * h) - 0.3 / L, 0, None) * h; wx /= wx.sum()
print(f'{run} s={pos:.3f}, tau {sim._frame_tilt():+.3f}, {len(idx)} mesh points')
res = {}
for lab, kw in (('100 + adaptive (recipe)', dict(xbins=100)), ('200 + adaptive', dict(xbins=200)), ('400 + adaptive', dict(xbins=400)),
                ('400 uniform', dict(xbins=400, far_adapt_x=False, near_adapt_x=False)), ('800 uniform', dict(xbins=800, far_adapt_x=False, near_adapt_x=False)),
                ('1600 uniform (reference)', dict(xbins=1600, far_adapt_x=False, near_adapt_x=False))):
    old = {a: getattr(ip, a) for a in kw}
    for a, v in kw.items(): setattr(ip, a, v)
    t1 = time.time(); r = np.array([sim.get_CSR_wake(pos + sim.CSR_zmesh[k], sim.CSR_xmesh[k]) for k in idx]); dt = time.time() - t1
    for a, v in old.items(): setattr(ip, a, v)
    res[lab] = (r, dt)
ref = res['1600 uniform (reference)'][0]
W0, F0 = np.abs(ref[:, 0]).max(), np.abs(ref[:, 1]).max()
wrow = np.repeat(wx, len(range(0, nz, stride)))
for lab, (r, dt) in res.items():
    dW, dF = np.abs(r[:, 0] - ref[:, 0]), np.abs(r[:, 1] - ref[:, 1])
    mw, mw0 = (wrow * r[:, 0]).sum() / wrow.sum(), (wrow * ref[:, 0]).sum() / wrow.sum()
    print(f'   {lab:26s}: max|dW| {dW.max()/W0*100:5.2f}%  rms {np.sqrt(np.mean(dW**2))/W0*100:5.2f}%  max|dF| {dF.max()/F0*100:5.2f}%  rms {np.sqrt(np.mean(dF**2))/F0*100:5.2f}%  (of max|ref|); mean W {mw:+.4f} vs {mw0:+.4f}; {dt:4.0f} s')
