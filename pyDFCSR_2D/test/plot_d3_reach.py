"""Plot the D3 wake maps from d3_wake_diagnostic.py reach: n_formation_length 1 vs 6, with the 1D wake."""
import os, sys, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'benchmark_results', 'd3_diagnostic')
s = sys.argv[1] if len(sys.argv) > 1 else '10.0580'
d = np.load(os.path.join(OUT, f'reach_{s}.npz')); nfl, maps, xm, zm, W1 = d['nfl'], d['maps'], d['xm'], d['zm'], d['W1z']
nx, nz = maps.shape[1:]
X, Z = (xm.reshape(nx, nz) - xm.mean()) * 1e6, (zm.reshape(nx, nz) - zm.mean()) * 1e6
u = np.linspace(-4, 4, nx); wx = np.exp(-u ** 2 / 2); wx /= wx.sum()
fig = plt.figure(figsize=(17, 9), layout='constrained'); gs = fig.add_gridspec(2, len(nfl) + 1)
v = np.abs(maps).max()
for j, (n, W) in enumerate(zip(nfl, maps)):
    a = fig.add_subplot(gs[0, j])
    pc = a.pcolormesh(Z, X, W, cmap='RdBu_r', vmin=-v, vmax=v, shading='gouraud')
    a.set(title=f'2D wake, n_formation_length = {n:g}',
          xlabel='z − ⟨z⟩ [µm]', ylabel='x − ⟨x⟩ [µm]')
    fig.colorbar(pc, ax=a, label='dE/d(ct) [MeV/m]')
    a = fig.add_subplot(gs[1, j])
    for r, c in zip(range(nx), plt.cm.viridis(np.linspace(0, 1, nx))):
        a.plot(Z[r], W[r], c=c, lw=0.8)
    a.plot(Z[0], (wx[:, None] * W).sum(0), c='C3', lw=2.2, label='2D, x-average')
    a.plot(Z[0], W1, 'k--', lw=1.8, label='1D line charge')
    a.axhline(0, c='0.6', lw=0.5); a.set(xlabel='z − ⟨z⟩ [µm]', ylabel='dE/d(ct) [MeV/m]', ylim=(-v * 1.05, v * 1.05),
                                         title=f'rows (colour −4σ → +4σ in x), n_fl = {n:g}'); a.legend(fontsize=8)
a = fig.add_subplot(gs[:, -1])
for n, W, c in zip(nfl, maps, ['C1', 'C3']):
    a.plot(Z[0], (wx[:, None] * W).sum(0), c=c, lw=2, label=f'2D x-average, n_fl = {n:g}')
a.plot(Z[0], W1, 'k--', lw=1.8, label='1D line charge')
a.axhline(0, c='0.6', lw=0.5); a.set(xlabel='z − ⟨z⟩ [µm]', ylabel='dE/d(ct) [MeV/m]', title='x-averaged wake vs 1D'); a.legend(fontsize=8)
txt = open(os.path.join(OUT, f'reach_{s}.txt')).read().splitlines()
cover = ', '.join(f"n_fl {l.split()[1]}: reach covers the bend exit = {'covers its exit: True' in l}" for l in txt if l.startswith('n_fl'))
fig.suptitle(f'Chicane drift, s = {float(s):.3f} m, CSR-off history. {cover}')
f = os.path.join(OUT, f'reach_{s}.png'); plt.savefig(f, dpi=105); print(f)
