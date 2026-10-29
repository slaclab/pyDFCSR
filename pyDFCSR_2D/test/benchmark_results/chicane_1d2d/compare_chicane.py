"""Chicane: 1D (Emma csr_calc) vs 2D (pyDFCSR) slice energy spread.
Both codes start from the same 1e6 particles at the B1 entrance (s = 0.1 m); each is compared with its own
CSR-off run. The chicane leaves a large residual chirp (36 MeV across sigma_z ~ 19 um), so a raw slice rms
is dominated by the chirp across the slice width. Slice spread here is therefore the rms after removing a
quadratic fit in z within each slice, and is checked against the slice count (it must not depend on it).
Core = |z| < 2 sigma_z. CSR-added = sqrt(<var_on - var_off>), slice-population weighted."""
import numpy as np, h5py, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
ZR, DEG = 3.0, 2
def slices(z, e, ns, sz_ref=None):
    zc = z - z.mean(); sz = zc.std() if sz_ref is None else sz_ref
    ed = np.linspace(-ZR * sz, ZR * sz, ns + 1); idx = np.digitize(zc, ed) - 1
    n = np.zeros(ns, int); v = np.full(ns, np.nan)
    for i in range(ns):
        m = idx == i; n[i] = m.sum()
        if n[i] > 50: v[i] = np.var(e[m] - np.polyval(np.polyfit(zc[m], e[m], DEG), zc[m]))
    return 0.5 * (ed[1:] + ed[:-1]), n, v, sz
def compare(on, off, ns):
    zc, n, v1, sz = slices(*on, ns); _, _, v0, _ = slices(*off, ns, sz_ref=sz)
    core = (np.abs(zc) < 2 * sz) & (n > 500)
    add = np.average(v1[core] - v0[core], weights=n[core])
    return dict(zc=zc, n=n, v1=v1, v0=v0, sz=sz, on=np.sqrt(np.average(v1[core], weights=n[core])),
                off=np.sqrt(np.average(v0[core], weights=n[core])), add=np.sign(add) * np.sqrt(abs(add)))
A, B = np.load('2d_chic_on.npz'), np.load('2d_chic_off.npz'); E0 = float(A['E0'])
def beams(where):
    if where == 'exit':
        out = {'2D': ((A['zf'], A['df'] * E0), (B['zf'], B['df'] * E0))}; files = [('', '1D 5 mm'), ('_nsb500', '1D 1 mm')]; pre = '1d_chic'
    else:
        out = {}; files = [('', '1D 5 mm'), ('_nsb500', '1D 1 mm')]; pre = '1d_chicB4'
    for suf, lab in files:
        X, Y = h5py.File(f'{pre}{suf}.mat')['Xs'][()], h5py.File(f'{pre}_off{suf}.mat')['Xs'][()]
        out[lab] = ((X[:, 2], X[:, 3] * E0), (Y[:, 2], Y[:, 3] * E0))
    return out
fo = open('compare_chicane.txt', 'w')
fo.write(__doc__ + '\n')
res = {}
for where, title in [('exit', 'chicane exit, s = 13.31 m'), ('B4', 'B4 entrance, s = 12.61 m (1D lattice cut after D3)')]:
    fo.write(f'\n== {title} ==\n  {"code":9s} {"slices":>6s}  {"sigma_z um":>10s}  {"slice sE off":>12s}  {"on":>8s}  {"CSR-added":>9s}  [keV]\n')
    for lab, (on, off) in beams(where).items():
        for ns in [40, 80, 160]:
            r = compare(on, off, ns); res[(where, lab, ns)] = r
            fo.write(f"  {lab:9s} {ns:6d}  {r['sz']*1e6:10.2f}  {r['off']/1e3:12.1f}  {r['on']/1e3:8.1f}  {r['add']/1e3:9.1f}\n")
fo.write('\n  2D at the B4 entrance comes from the per-node slice table (40 slices, LINEAR detrend only):\n')
s = A['s']; iB = np.argmin(abs(s - 12.6122))
for i, name in [(iB, 'B4 entrance'), (len(s) - 1, 'exit')]:
    c = (np.abs(A['zc'][i]) < np.abs(A['zc'][i]).max() * 2 / 3) & (A['n'][i] > 500)
    v = np.average(A['sEd'][i][c]**2 - B['sEd'][i][c]**2, weights=A['n'][i][c])
    fo.write(f"    {name:12s} s = {s[i]:.3f}: off {np.sqrt(np.average(B['sEd'][i][c]**2, weights=A['n'][i][c]))/1e3:.1f}, CSR-added {np.sign(v)*np.sqrt(abs(v))/1e3:.1f} keV\n")
fo.write('\n  mean CSR energy change [MeV]: 2D exit %.3f, B4 entr %.3f' % ((A['df'].mean() - B['df'].mean()) * E0 / 1e6,
         (np.average(A['mE'][iB], weights=A['n'][iB]) - np.average(B['mE'][iB], weights=B['n'][iB])) / 1e6))
for pre, lab in [('1d_chic', '1D exit'), ('1d_chicB4', '1D B4 entr')]:
    for suf in ['', '_nsb500']:
        fo.write(f"; {lab}{suf or ' 5mm'} {h5py.File(f'{pre}{suf}.mat')['Ebar_CSR'][()].ravel()[-1]*E0/1e6:.3f}")
fo.write('\n'); fo.close(); print(open('compare_chicane.txt').read())
# plots: exit, 80 slices
fig, ax = plt.subplots(1, 2, figsize=(13, 4.6), layout='constrained')
for lab, c in [('2D', 'C3'), ('1D 5 mm', 'C0'), ('1D 1 mm', 'C2')]:
    r = res[('exit', lab, 80)]; m = r['n'] > 500; zc = r['zc'][m] * 1e6
    ax[0].plot(zc, np.sqrt(r['v1'][m]) / 1e3, c=c, label=f'{lab} CSR on'); ax[0].plot(zc, np.sqrt(r['v0'][m]) / 1e3, ':', c=c, label=f'{lab} CSR off')
    d = r['v1'][m] - r['v0'][m]; ax[1].plot(zc, np.sign(d) * np.sqrt(np.abs(d)) / 1e3, 'o-', ms=3, c=c, label=f"{lab}: core {r['add']/1e3:.0f} keV")
ax[0].set(xlabel='final z − ⟨z⟩ [µm]', ylabel='slice σE, quadratic chirp removed [keV]', title='Chicane exit, 80 slices'); ax[0].legend(fontsize=8)
ax[1].axhline(0, c='k', lw=.5); ax[1].set(xlabel='final z − ⟨z⟩ [µm]', ylabel='CSR-added slice σE [keV]', title='CSR-added (signed √|Δσ²|)'); ax[1].legend(fontsize=8)
plt.savefig('compare_chicane.png', dpi=110)
