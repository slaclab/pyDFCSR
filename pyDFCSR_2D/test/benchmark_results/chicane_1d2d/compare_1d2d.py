"""1D (Emma csr_calc) vs 2D (pyDFCSR, comoving) slice energy spread, tilt scan in a 1 m / 1 rad dipole.
Both codes start from the same 1e6 particles at the bend entrance (s = 0.1 m in pyDFCSR). Slices are
cut in final z (40 over +-3 sigma_z). The CSR-added slice variance is var_on - var_off, slice by slice,
with each code's own CSR-off run as reference (2D: pyDFCSR CSR off; 1D: csr_calc with fac = 0).
Usage: compare_1d2d.py [--nsb N] tag [tag ...]  -> compare_1d2d[_nsbN].txt/.png
  --nsb N: use the 1D runs with N slices in the bend (default 100 = 1 cm; 1000 = 1 mm, converged)"""
import sys, os, numpy as np, h5py, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
NS, ZR = 40, 3.0
args = sys.argv[1:]; NSB = ''
if args and args[0] == '--nsb': NSB = '' if args[1] == '100' else f'_nsb{args[1]}'; args = args[2:]
tags = args
def slices(z, e, zref=None):
    zc = z - z.mean(); sz = zc.std() if zref is None else zref
    edges = np.linspace(-ZR * sz, ZR * sz, NS + 1); idx = np.digitize(zc, edges) - 1
    n = np.bincount(idx[(idx >= 0) & (idx < NS)], minlength=NS)
    sE = np.array([e[idx == i].std() if n[i] > 2 else np.nan for i in range(NS)])
    mE = np.array([e[idx == i].mean() if n[i] > 2 else np.nan for i in range(NS)])
    return 0.5 * (edges[1:] + edges[:-1]), n, mE, sE, idx
def load1d(tag, suf=''):
    m = h5py.File(f'1d_{tag}{suf}{NSB}.mat'); X = m['Xs'][()]
    return dict(z=X[:, 2], d=X[:, 3], x=X[:, 0], s=m['s'][()].ravel() + 0.1, Si=m['Si'][()], W=m['dE_dct'][()],
                lam=m['fi'][()], E0=float(m['E0'][()].ravel()[0]))
def tlabel(t): return ('−' if t[0] == 'm' else '') + t[1:].replace('p', '.')
out = open(f'compare_1d2d{NSB}.txt', 'w')
out.write('case      | core CSR-added slice sigma_E [keV] (quadrature, |z|<2sz): 2D  1D  2D/1D | 2D x-dependence share | final sigma_z 2D/1D [um] | mean CSR dE 2D/1D [keV]\n')
fig, ax = plt.subplots(2, len(tags), figsize=(4.2 * len(tags), 7.5), layout='constrained', squeeze=False)
summary = []
for j, t in enumerate(tags):
    A, B = np.load(f'2d_{t}_on.npz'), np.load(f'2d_{t}_off.npz'); E0 = float(A['E0'])
    o1, f1 = load1d(t), load1d(t, '_off')
    # common slicing: each code's CSR-on beam is sliced on its own final z; its CSR-off beam with the same bin width
    r = {}
    for code, zon, eon, zoff, eoff in [('2D', A['zf'], A['df'] * E0, B['zf'], B['df'] * E0),
                                       ('1D', o1['z'], o1['d'] * o1['E0'], f1['z'], f1['d'] * f1['E0'])]:
        zc, n, mE, sE, _ = slices(zon, eon); _, n0, mE0, sE0, _ = slices(zoff, eoff, zref=(zon - zon.mean()).std())
        add = np.sign(sE**2 - sE0**2) * np.sqrt(np.abs(sE**2 - sE0**2))
        core = (np.abs(zc) < 2 * (zon - zon.mean()).std()) & (n > 1000)
        r[code] = dict(zc=zc * 1e6, sE=sE / 1e3, sE0=sE0 / 1e3, add=add / 1e3, n=n, core=core, mloss=(eon.mean() - eoff.mean()) / 1e3,
                       sz=(zon - zon.mean()).std() * 1e6, addc=np.sqrt(max(np.average(sE[core]**2 - sE0[core]**2, weights=n[core]), 0)) / 1e3)
    # 2D: share of CSR-added slice variance from the wake's x-dependence (var K_x + 2cov(K_z,K_x)), in-mesh particles
    K, Kz = A['K'], A['Kz']; Kx = K - Kz; z = A['zf'] - A['zf'].mean(); sz = z.std()
    ed = np.linspace(-ZR * sz, ZR * sz, NS + 1); idx = np.digitize(z, ed) - 1; ok = ~A['out_mesh']
    xs, tot = [], []
    for i in range(NS):
        m = (idx == i) & ok
        if m.sum() < 1000 or abs(0.5 * (ed[i] + ed[i + 1])) > 2 * sz: continue
        xs.append((np.var(Kx[m]) + 2 * np.mean((Kz[m] - Kz[m].mean()) * (Kx[m] - Kx[m].mean()))) * m.sum())
        tot.append((np.var(A['df'][m]) - np.var(A['d0'][m])) * m.sum())
    xshare = np.sum(xs) / np.sum(tot)
    a2, a1 = r['2D']['addc'], r['1D']['addc']
    summary.append((t, a2, a1, xshare))
    out.write(f"{t:>8}     | {a2:7.1f} {a1:7.1f} {a2/a1 if a1 > 0 else np.nan:6.2f} | {100*xshare:6.1f}% | {r['2D']['sz']:6.1f} / {r['1D']['sz']:6.1f} | {r['2D']['mloss']:7.0f} / {r['1D']['mloss']:7.0f}\n")
    for code, c, ls in [('2D', 'C3', '-'), ('1D', 'C0', '-')]:
        q = r[code]; m = q['n'] > 1000
        ax[0, j].plot(q['zc'][m], q['sE'][m], ls, c=c, label=f'{code} CSR on'); ax[0, j].plot(q['zc'][m], q['sE0'][m], ':', c=c, label=f'{code} CSR off')
        ax[1, j].plot(q['zc'][m], q['add'][m], 'o-', ms=3, c=c, label=f'{code}: core {q["addc"]:.0f} keV')
    ax[0, j].set(title=f'{t}', xlabel='final z − ⟨z⟩ [µm]', ylabel='final slice σE [keV]'); ax[0, j].legend(fontsize=7)
    ax[1, j].axhline(0, c='k', lw=.5); ax[1, j].set(xlabel='final z − ⟨z⟩ [µm]', ylabel='CSR-added slice σE [keV] (signed √|Δσ²|)'); ax[1, j].legend(fontsize=8)
plt.savefig(f'compare_1d2d{NSB}.png', dpi=110); plt.close(); out.close()
print(open(f'compare_1d2d{NSB}.txt').read())
