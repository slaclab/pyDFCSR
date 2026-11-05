"""Compare two chicane runs: exit metrics, energy per element, and per-element rel L2 of the wake maps at matched kicks."""
import sys, os, glob, re, numpy as np, h5py
sys.argv, ARGS = sys.argv[:1], sys.argv[1:]
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import chicane_2d_vs_1d as C
O = C.OUT; A, B = ARGS[0], ARGS[1]
off = dict(np.load(O + '/off.npz')); E0 = float(off['E0'])
for n in (A, B):
    m = C.metrics(dict(np.load(f'{O}/{n}.npz')), off, E0)
    print(f"{n:14s} dE {m['dE']:8.1f} keV | d proj sE {m['dsEp']:7.1f} | d slice sE {m['dsEs']:6.2f} keV | demit {m['demit']:.4f} um | "
          f"slice emit {m['semit']:.4f} | sz {m['sz']:.3f} | Ipk {m['Ipk']:.3f}")
ra, rb = np.load(f'{O}/{A}.npz'), np.load(f'{O}/{B}.npz'); K = lambda r: (r['pz'] - off['pz']) * E0
print(f'rms kick difference {np.std(K(rb) - K(ra))/1e3:.2f} keV of {np.std(K(ra))/1e3:.0f} keV')
eoff = C.per_element_loss(off)
print(f"{'energy/element':14s}" + ''.join(f'{e:>8s}' for e in C.ENAMES))
for n in (A, B):
    print(f'{n:14s}' + ''.join(f'{v:8.1f}' for v in C.per_element_loss(dict(np.load(f"{O}/{n}.npz"))) - eoff))
def load(run):
    f = sorted(glob.glob(f'{O}/runs/{run}-2*-wakes.h5'), key=os.path.getmtime)[-1]; h = h5py.File(f, 'r')
    return {round(float(h[k].attrs['position']), 3): (str(h[k].attrs['element']), h[k]['longitudinal/dE_dct'][()], h[k]['transverse/xkicks'][()]) for k in h}
WA, WB = load(A), load(B); stat = {}
for s, (e, wa, fa) in WA.items():
    if s not in WB: continue
    wb, fb = WB[s][1], WB[s][2]
    stat.setdefault(e, []).append((np.linalg.norm(wb - wa) / np.linalg.norm(wa), np.linalg.norm(fb - fa) / max(np.linalg.norm(fa), 1e-30)))
print('wake maps, rel L2 per kick, median / max over the element (longitudinal | transverse):')
for e in C.ENAMES:
    if e in stat:
        v = np.array(stat[e]); print(f'  {e:3s} {len(v):3d} kicks  long {np.median(v[:, 0]):.4f} / {v[:, 0].max():.4f}   trans {np.median(v[:, 1]):.4f} / {v[:, 1].max():.4f}')
E = [0.0, 0.1, 0.6002, 5.606, 6.1062, 7.1062, 7.6064, 12.6122, 13.1124, 13.3124]
for n in (A, B):
    st = [(float(s), float(t)) for s, t in re.findall(r'Finish step \d+, s = ([0-9.]+),\s+in ([0-9.]+) seconds', open(f'{O}/log_{n}.txt').read())]
    out = np.zeros(9)
    for s, dt in st: out[min(max(np.searchsorted(E, s, side='left') - 1, 0), 8)] += dt
    print(f'runtime {n:14s}' + ''.join(f'{v:7.1f}' for v in out / 60) + f'  total {out.sum()/60:.1f} min; kicks > 100 s: {[(round(s, 3), round(t)) for s, t in st if t > 100]}')
