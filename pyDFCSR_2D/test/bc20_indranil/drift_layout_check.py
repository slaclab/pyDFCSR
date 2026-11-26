import os, sys, time, numpy as np
T = '/Users/jytang/Library/CloudStorage/GoogleDrive-jytang@stanford.edu/My Drive/Stanford/research/FEL/pyDFCSR_claude/pyDFCSR_near_adapt/pyDFCSR_2D/test'
sys.path.insert(0, os.path.join(T, '..', '..')); sys.path.insert(0, os.path.join(T, 'facet_bc20'))
import dump_kick_state as D
D.RUNS = os.path.join(T, 'benchmark_results', 'bc20_indranil', 'runs')
st = float(sys.argv[1]); run = sys.argv[2] if len(sys.argv) > 2 else 'b176_csroff_fl6'
sim = D.load_state(D.state_path(run, st)); ip = sim.integration_params; pos = sim.beam.position
nx, nz = sim.CSR_params.xbins, sim.CSR_params.zbins
xi = np.asarray(sim.CSR_xrange_transformed); h = np.gradient(xi); n, L = xi.size, xi[-1] - xi[0]
wx = np.clip(1.0 / ((n - 1) * h) - 0.3 / L, 0, None) * h; wx /= wx.sum()
zr = np.asarray(sim.CSR_zrange); hz = np.gradient(zr); Lz = zr[-1] - zr[0]
wz = np.clip(1.0 / ((zr.size - 1) * hz) - 0.3 / Lz, 0, None) * hz; wz /= wz.sum()
rows = np.argsort(-wx)[:3]; cols = np.arange(0, nz, 4); W8 = np.outer(wx[rows], wz[cols]); W8 /= W8.sum()
pts = [(a, b, i * nz + j) for a, i in enumerate(rows) for b, j in enumerate(cols)]
print(f'{run} s={pos:.3f}, tau {sim._frame_tilt():+.3f}')
def mean(fl, **kw):
    ip.n_formation_length, ip.far_zbins = fl, 200 * fl
    old = {a: getattr(ip, a) for a in kw}
    for a, v in kw.items(): setattr(ip, a, v)
    t1 = time.time(); m = sum(W8[a, b] * sim.get_CSR_wake(pos + sim.CSR_zmesh[k], sim.CSR_xmesh[k])[0] for a, b, k in pts)
    for a, v in old.items(): setattr(ip, a, v)
    return m, time.time() - t1
for lab, kw in (('recipe: xbins 100 + adaptive x\'', {}),
                ('xbins 400, adaptive off', dict(xbins=400, far_adapt_x=False, near_adapt_x=False)),
                ('xbins 400, adaptive off, uniform drift far (causal_clip off), far_zbins x4', dict(xbins=400, far_adapt_x=False, near_adapt_x=False, causal_clip=False, _fz4=True)),
                ('xbins 400, adaptive off, far_fine_max 4000', dict(xbins=400, far_adapt_x=False, near_adapt_x=False, far_fine_max=4000))):
    fz4 = kw.pop('_fz4', False)
    r = []
    for fl in (1, 6):
        ip.n_formation_length, ip.far_zbins = fl, 200 * fl * (4 if fz4 else 1)
        old = {a: getattr(ip, a) for a in kw}
        for a, v in kw.items(): setattr(ip, a, v)
        t1 = time.time(); m = sum(W8[a, b] * sim.get_CSR_wake(pos + sim.CSR_zmesh[k], sim.CSR_xmesh[k])[0] for a, b, k in pts)
        for a, v in old.items(): setattr(ip, a, v)
        r.append((m, time.time() - t1))
    print(f'   {lab:70s}: FL 1 {r[0][0]:+.5f}  FL 6 {r[1][0]:+.5f}  (FL6-FL1 {r[1][0]-r[0][0]:+.5f}; {r[0][1]:.0f}/{r[1][1]:.0f} s)', flush=True)
