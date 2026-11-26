"""
n_formation_length sensitivity at a pickled kick state, with the far cell fixed (far_zbins = 200 * FL), and
where along the lattice the extra wake comes from: every column's contribution -C int dx' K is attributed
to its source position s' and summed by lattice element (debug=True parts; the polar patch goes to s).
    python fl_decomp.py <run> <s_target> [FL list] [stride_z] [rows]
"""
import os, sys, time, numpy as np
T = '/Users/jytang/Library/CloudStorage/GoogleDrive-jytang@stanford.edu/My Drive/Stanford/research/FEL/pyDFCSR_claude/pyDFCSR_near_adapt/pyDFCSR_2D/test'
sys.path.insert(0, os.path.join(T, '..', '..')); sys.path.insert(0, os.path.join(T, 'facet_bc20'))
import dump_kick_state as D
D.RUNS = os.path.join(T, 'benchmark_results', 'bc20_indranil', 'runs')
run, st = sys.argv[1], float(sys.argv[2])
FLS = [float(v) for v in (sys.argv[3] if len(sys.argv) > 3 else '1,2,4,6').split(',')]
SZ = int(sys.argv[4]) if len(sys.argv) > 4 else 4
ROWS = int(sys.argv[5]) if len(sys.argv) > 5 else 4
sim = D.load_state(D.state_path(run, st)); ip = sim.integration_params; pos = sim.beam.position
nx, nz = sim.CSR_params.xbins, sim.CSR_params.zbins
xi = np.asarray(sim.CSR_xrange_transformed); h = np.gradient(xi); n, L = xi.size, xi[-1] - xi[0]
wx = np.clip(1.0 / ((n - 1) * h) - 0.3 / L, 0, None) * h; wx /= wx.sum()
zr = np.asarray(sim.CSR_zrange); hz = np.gradient(zr); Lz = zr[-1] - zr[0]
wz = np.clip(1.0 / ((zr.size - 1) * hz) - 0.3 / Lz, 0, None) * hz; wz /= wz.sum()
rows = np.argsort(-wx)[:ROWS]; cols = np.arange(0, nz, SZ)
W8 = np.outer(wx[rows], wz[cols]); W8 /= W8.sum()
lat = sim.lattice; dist = np.asarray(lat.distance); names = list(lat.lattice_config.keys())
names = [k for k in names if isinstance(lat.lattice_config[k], dict) and 'L' in lat.lattice_config[k]]
edges = np.concatenate([[0.0], dist])
print(f'history from {sim.DF_tracker.time_log[0]:.2f} m; '); print(f'{run} s={pos:.3f}: inbend {sim.inbend}, afterbend {sim.afterbend}, L_f {sim.formation_length:.3f} m, sigma_z {sim.beam._sigma_z*1e6:.1f} um, '
      f'bend start {getattr(sim, "bend_start_rec", None)}, prev bend start {getattr(sim, "prev_bend_start", None)}; {len(rows)}x{len(cols)} mesh points')
def element_of(sp):
    k = np.clip(np.searchsorted(edges, sp, side='right') - 1, 0, len(names) - 1)
    return np.where(sp < 0, -1, k)
res = {}
for fl in FLS:
    ip.n_formation_length = fl; ip.far_zbins = int(round(200 * fl))
    t1 = time.time(); tot = 0.0; by_el = np.zeros(len(names) + 1)
    for a, i in enumerate(rows):
        for b, j in enumerate(cols):
            k = i * nz + j; s, x = pos + sim.CSR_zmesh[k], sim.CSR_xmesh[k]
            wtrue = sim.get_CSR_wake(s, x)[0]
            dbg = sim.get_CSR_wake(s, x, debug=True)
            acc = np.zeros(len(names) + 1); dsum = 0.0
            for spm, xpm, iz in zip(dbg['sp_mesh'], dbg['xp_mesh'], dbg['integrand_z']):
                if np.ndim(iz) != 2 or iz.shape[1] < 2:
                    continue
                sp = spm[0]; colv = np.trapz(iz, xpm, axis=0)
                wsp = np.gradient(sp) if sp.size > 1 else np.ones(1)
                # trapezoid weights
                tw = np.zeros_like(sp); d = np.diff(sp); tw[:-1] += d / 2; tw[1:] += d / 2
                c = -sim.CSR_scaling * colv * tw
                if np.all(np.isclose(sp, sp[0])):     # polar patch part: no s' extent
                    continue
                e = element_of(sp)
                np.add.at(acc, e + 1, c); dsum += c.sum()
            # the remainder (polar patch + x' refinement) to the observer's element
            acc[element_of(np.array([s]))[0] + 1] += wtrue - dsum
            tot += W8[a, b] * wtrue; by_el += W8[a, b] * acc
    res[fl] = (tot, by_el)
    print(f'   FL {fl:g} (far_zbins {ip.far_zbins}): <W> {tot:+.5f} MeV/m  ({time.time()-t1:.0f} s)', flush=True)
base = res[FLS[0]]
print(f'   change vs FL {FLS[0]:g} by SOURCE element [MeV/m] (elements with |change| > 2% of |<W>|):')
lab = ['(s\'<0)'] + names
for fl in FLS[1:]:
    d = res[fl][1] - base[1]
    big = [(lab[i], d[i]) for i in np.argsort(-np.abs(d)) if abs(d[i]) > 0.02 * abs(base[0])][:8]
    print(f'     FL {fl:g}: total {res[fl][0]-base[0]:+.5f} ({(res[fl][0]-base[0])/abs(base[0])*100:+.1f}%); ' + ', '.join(f'{a} {v:+.4f}' for a, v in big))
np.savez(os.path.join(D.RUNS, f'fl_decomp_{run}_s{st:.2f}.npz'), fls=FLS, tot=[res[f][0] for f in FLS], by_el=np.array([res[f][1] for f in FLS]), names=np.array(lab))
