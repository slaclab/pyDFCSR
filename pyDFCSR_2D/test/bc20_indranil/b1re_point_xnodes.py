import os, sys, time, numpy as np
T = '/Users/jytang/Library/CloudStorage/GoogleDrive-jytang@stanford.edu/My Drive/Stanford/research/FEL/pyDFCSR_claude/pyDFCSR_near_adapt/pyDFCSR_2D/test'
sys.path.insert(0, os.path.join(T, '..', '..')); sys.path.insert(0, os.path.join(T, 'facet_bc20'))
import dump_kick_state as D
D.RUNS = os.path.join(T, 'benchmark_results', 'bc20_indranil', 'runs')
sim = D.load_state(D.state_path('b176_csroff_fl6', 48.37)); ip = sim.integration_params; pos = sim.beam.position
nx, nz = sim.CSR_params.xbins, sim.CSR_params.zbins
k = (nx // 2) * nz + nz // 2; s, x = pos + sim.CSR_zmesh[k], sim.CSR_xmesh[k]
def W(fl, fz, **kw):
    ip.n_formation_length, ip.far_zbins = fl, fz
    old = {a: getattr(ip, a) for a in kw}
    for a, v in kw.items(): setattr(ip, a, v)
    t1 = time.time(); w = sim.get_CSR_wake(s, x)[0]
    for a, v in old.items(): setattr(ip, a, v)
    return w, time.time() - t1
print(f'point z={sim.CSR_zmesh[k]*1e6:+.1f} um x={x*1e6:+.0f} um at s={pos:.3f}')
for fl, fzs in ():
    for fz in fzs:
        w, dt = W(fl, fz); print(f'   FL {fl} far_zbins {fz:5d} (cell {fl*sim.formation_length/(2*fz)*1e3:.2f} mm): W {w:+.4f}  ({dt:.1f} s)', flush=True)



w, dt = W(1, 200, far_adapt_x=False); print(f'   FL 1, far_adapt_x off: {w:+.4f}')
w, dt = W(6, 1200, far_adapt_x=False); print(f'   FL 6, far_adapt_x off: {w:+.4f}')
print('--- brute-force x\' convergence (uniform x\' nodes per band, adaptive x\' off everywhere):')
for fl in (1, 6):
    for xb in (100, 400, 1600):
        w, dt = W(fl, 200 * fl, far_adapt_x=False, near_adapt_x=False, xbins=xb); print(f'   FL {fl} xbins {xb:5d}, adapt_x off: W {w:+.4f}  ({dt:.1f} s)', flush=True)
    w, dt = W(fl, 200 * fl, xbins=400); print(f'   FL {fl} xbins 400, adapt_x on: W {w:+.4f}', flush=True)
