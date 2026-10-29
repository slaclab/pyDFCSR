"""1D (csr_calc) calibration: how much does the far-drift CSR wake -- the part pyDFCSR's drift_cutoff = 1.0
discards (more than 1 R*phi past a bend exit) -- contribute to the final energy loss, energy spread
(projected and slice) and emittance?  Compares 1D runs with the cutoff off and on, each against the 1D
CSR-off run, at 5 mm and 1 mm bend slices.  Also splits the 1D wake integral along s by region, using
dE_dct on its bin grid and the per-slice length (the energy change a particle at the bunch centroid
collects), so the far-drift share can be read without rerunning.
Statistics are computed exactly as in compare_stats.py / compare_chicane.py."""
import numpy as np, h5py
G0 = 5e9 / 510998.95; E0 = 5e9; ZR = 3.0
def load(f):
    X = h5py.File(f)['Xs'][()]; return dict(x=X[:, 0], xp=X[:, 1], z=X[:, 2], d=X[:, 3])
def emit(x, xp, d=None):
    if d is not None:
        x = x - np.polyval(np.polyfit(d, x, 1), d); xp = xp - np.polyval(np.polyfit(d, xp, 1), d)
    return np.sqrt(max(np.linalg.det(np.cov(x, xp)), 0)) * G0
def slices(b, ns=80, sz_ref=None):
    z = b['z'] - b['z'].mean(); sz = z.std() if sz_ref is None else sz_ref
    ed = np.linspace(-ZR * sz, ZR * sz, ns + 1); idx = np.digitize(z, ed) - 1; v = {}; w = {}; em = {}
    for i in range(ns):
        m = idx == i
        if abs(0.5 * (ed[i] + ed[i + 1])) > 2 * sz or m.sum() < 500: continue
        e = b['d'][m] * E0; v[i] = np.var(e - np.polyval(np.polyfit(z[m], e, 2), z[m])); w[i] = m.sum()
        em[i] = emit(b['x'][m], b['xp'][m], b['d'][m])
    return v, w, em, sz
def stats(on, off):
    v1, w, em1, sz = slices(on); v0, _, em0, _ = slices(off, sz_ref=sz); k = [i for i in v1 if i in v0]
    ww = np.array([w[i] for i in k]); add = np.average([v1[i] - v0[i] for i in k], weights=ww)
    return dict(dE=(on['d'].mean() - off['d'].mean()) * E0 / 1e3, sEp=(on['d'].std() - off['d'].std()) * E0 / 1e3,
                sEs=np.sign(add) * np.sqrt(abs(add)) / 1e3, en=emit(on['x'], on['xp'], on['d']) * 1e6,
                den=(emit(on['x'], on['xp'], on['d']) - emit(off['x'], off['xp'], off['d'])) * 1e6,
                sen=np.average([em1[i] for i in k], weights=ww) * 1e6, sz=on['z'].std() * 1e6)
rows = [('dE', 'mean CSR energy change [keV]'), ('sEp', 'projected sigma_E change [keV]'), ('sEs', 'CSR-added core slice sigma_E [keV]'),
        ('den', 'emittance growth, dispersion-corrected [um]'), ('en', 'final emittance, dispersion-corrected [um]'),
        ('sen', 'core slice emittance [um]'), ('sz', 'final sigma_z [um]')]
L = [__doc__]
for nsb, suf in [(100, ''), (500, '_nsb500')]:
    off = load(f'1d_chic_off{suf}.mat'); full = stats(load(f'1d_chic{suf}.mat'), off); cut = stats(load(f'1d_chic{suf}_cut1.mat'), off)
    L += ['', f'== 1D, {1000*0.5002/nsb:.0f} mm bend slices ==', f"  {'':46s} {'full drift':>11s} {'cut 1 Rphi':>11s} {'far-drift part':>15s} {'share':>7s}"]
    for k, name in rows:
        a, b = full[k], cut[k]; L.append(f'  {name:46s} {a:11.3f} {b:11.3f} {a - b:15.3f} {100*(a-b)/a if a else np.nan:6.1f}%')
# wake integral along s, by region (1 mm run)
m = h5py.File('1d_chic_nsb500.mat'); s = m['s'][()].ravel() + 0.1; W = m['dE_dct'][()]; Si = m['Si'][()]; lam = m['fi'][()]
ds = np.diff(np.concatenate([[0.1], s]))
Wc = np.array([np.sum(W[:, j] * lam[:, j]) / np.sum(lam[:, j]) for j in range(len(s))])     # charge-weighted mean wake
exits = [(0.6002, 'B1'), (6.1062, 'B2'), (7.6064, 'B3'), (13.1124, 'B4')]; ents = [0.1, 5.606, 7.1062, 12.6122]; Rphi = 0.5002
def reg(x):
    if any(a <= x <= b + 1e-9 for a, b in zip(ents, [e for e, _ in exits])): return 'inside the dipoles'
    prev = max([e for e, _ in exits if e < x], default=None)
    return 'drift, within 1 Rphi of an exit' if x - prev <= Rphi else 'drift, beyond 1 Rphi (cut)'
acc = {}
for x, w_, l in zip(s, Wc, ds): acc[reg(x)] = acc.get(reg(x), 0) + w_ * l
tot = sum(acc.values())
L += ['', '1D charge-weighted wake integral along s, by region (1 mm run, no cutoff) [keV]:']
for k in ['inside the dipoles', 'drift, within 1 Rphi of an exit', 'drift, beyond 1 Rphi (cut)']:
    L.append(f'  {k:34s} {acc.get(k, 0)/1e3:9.1f}  ({100*acc.get(k, 0)/tot:5.1f}%)')
open('drift_tail_1d.txt', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L[1:]))
