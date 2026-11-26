import os, sys, time, numpy as np
T = '/Users/jytang/Library/CloudStorage/GoogleDrive-jytang@stanford.edu/My Drive/Stanford/research/FEL/pyDFCSR_claude/pyDFCSR_near_adapt/pyDFCSR_2D/test'
sys.path.insert(0, os.path.join(T, '..', '..')); sys.path.insert(0, os.path.join(T, 'facet_bc20'))
import dump_kick_state as D
D.RUNS = os.path.join(T, 'benchmark_results', 'bc20_indranil', 'runs')
run, st = sys.argv[1], float(sys.argv[2])
sim = D.load_state(D.state_path(run, st)); ip = sim.integration_params; pos = sim.beam.position
nx, nz = sim.CSR_params.xbins, sim.CSR_params.zbins
xi = np.asarray(sim.CSR_xrange_transformed); h = np.gradient(xi); n, L = xi.size, xi[-1] - xi[0]
wx = np.clip(1.0 / ((n - 1) * h) - 0.3 / L, 0, None) * h; wx /= wx.sum()
zr = np.asarray(sim.CSR_zrange); hz = np.gradient(zr); Lz = zr[-1] - zr[0]
wz = np.clip(1.0 / ((zr.size - 1) * hz) - 0.3 / Lz, 0, None) * hz; wz /= wz.sum()
rows = np.argsort(-wx)[:3]; cols = np.arange(0, nz, 4); W8 = np.outer(wx[rows], wz[cols]); W8 /= W8.sum()
pts = [(a, b, i * nz + j) for a, i in enumerate(rows) for b, j in enumerate(cols)]
sb = float(sim.bend_start_rec); h0 = float(sim.DF_tracker.time_log[0])
print(f'{run} s={pos:.3f}: inbend {sim.inbend}, bend start {sb:.3f}, history from {h0:.2f}, L_f {sim.formation_length:.3f}')
def mean(f):
    out = 0.0; extra = []
    for a, b, k in pts:
        v = f(pos + sim.CSR_zmesh[k], sim.CSR_xmesh[k])
        if isinstance(v, tuple): extra.append(v[1]); v = v[0]
        out += W8[a, b] * v
    return out, extra
ip.n_formation_length, ip.far_zbins = 1, 200
base, _ = mean(lambda s, x: sim.get_CSR_wake(s, x)[0])
ip.n_formation_length, ip.far_zbins = 6, 1200
ref, _ = mean(lambda s, x: sim.get_CSR_wake(s, x)[0])
ip.n_formation_length, ip.far_zbins = 1, 200
print(f'   FL 1 {base:+.5f}   FL 6 {ref:+.5f}   target extra {ref-base:+.5f} MeV/m')
cell = sim.formation_length / 400
for mode in ('uniform', 'grade0.02', 'grade0.005'):
    def f(s, x):
        w = sim.get_CSR_wake(s, x)[0]
        (s1, s3), _ = sim._layout_bounds(s, x)
        hi = min(s1, sb)
        lo = sim._causal_edge(s, x, pos, max(0.0, h0), hi, n_scan=128)
        if lo is None or lo >= hi - 1e-9:
            return w, (0, hi)
        if mode == 'uniform':
            sp = np.linspace(lo, hi, max(3, int(round((hi - lo) / cell)) + 1))
        else:
            g = float(mode[5:]); u_lo, u_hi = s - hi, s - lo
            k = int(np.ceil(np.log(u_hi / u_lo) / np.log1p(g))); sp = np.unique(s - np.minimum(u_lo * (1 + g) ** np.arange(k + 1), u_hi))
        e = sim._integrate_xi_region(s, x, pos, sp, False, None, scale_bands=True)[0]
        return w + e, (len(sp), lo)
    t1 = time.time(); m, ex = mean(f)
    print(f'   FL 1 + drift extension [{mode:10s}]: {m:+.5f} (extra {m-base:+.5f}, {(m-base)/(ref-base)*100 if ref != base else 0:.0f}% of target); nodes {np.median([e[0] for e in ex]):.0f}, lo {np.median([e[1] for e in ex]):.2f} m; {time.time()-t1:.0f} s', flush=True)
