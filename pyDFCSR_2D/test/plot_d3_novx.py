"""Plot d3_wake_diagnostic.py novx results: the drift wake with the transverse-velocity terms on and off."""
import os, sys, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'benchmark_results', 'd3_diagnostic')
tags = sys.argv[1:] or ['8.5660', '10.0580', '12.3000']
fig, ax = plt.subplots(3, len(tags), figsize=(5.8 * len(tags), 13), layout='constrained', squeeze=False)
for j, s in enumerate(tags):
    d = np.load(os.path.join(OUT, f'novx_{s}.npz')); names, maps, xm, zm, W1 = d['names'], d['maps'], d['xm'], d['zm'], d['W1z']
    nx, nz = maps.shape[1:]
    X, Z = (xm.reshape(nx, nz) - xm.mean()) * 1e6, (zm.reshape(nx, nz) - zm.mean()) * 1e6
    u = np.linspace(-4, 4, nx); wx = np.exp(-u ** 2 / 2); wx /= wx.sum()
    v = np.abs(maps).max()
    for i, (n, W) in enumerate(zip(names, maps)):
        a = ax[i, j]; pc = a.pcolormesh(Z, X, W, cmap='RdBu_r', vmin=-v, vmax=v, shading='gouraud')
        a.set(title=f's = {float(s):.2f} m: {n}', xlabel='z − ⟨z⟩ [µm]', ylabel='x − ⟨x⟩ [µm]'); fig.colorbar(pc, ax=a, label='MeV/m')
    a = ax[2, j]
    for n, W, c in zip(names, maps, ['C3', 'C0']):
        for r in range(nx):
            a.plot(Z[r], W[r], c=c, lw=0.5, alpha=0.35)
        a.plot(Z[0], (wx[:, None] * W).sum(0), c=c, lw=2.2, label=f'{n}: x-average (thin: rows)')
    a.plot(Z[0], W1, 'k--', lw=1.8, label='1D line charge')
    a.axhline(0, c='0.6', lw=0.5); a.set(xlabel='z − ⟨z⟩ [µm]', ylabel='dE/d(ct) [MeV/m]', title='rows and x-average vs 1D'); a.legend(fontsize=7)
fig.suptitle('Chicane D3, reach covering B3, CSR-off history: the x-structure is carried entirely by the transverse-velocity (v_x) terms')
f = os.path.join(OUT, 'novx.png'); plt.savefig(f, dpi=100); print(f)
