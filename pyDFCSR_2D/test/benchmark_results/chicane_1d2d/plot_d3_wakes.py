"""2D wake maps through the chicane's D3 drift (B3 exit -> B4 entrance), from the drift_cutoff = 0 run.
One panel per kick in D3, longitudinal and transverse, on the 12 x 40 wake mesh (lab x and z, um), with the
1D (csr_calc, 1 mm bend slices) longitudinal wake at the nearest 1D step overlaid as a profile. Colour scale
per panel (amplitude changes 60x across D3); the title gives the distance past the B3 exit in units of R*phi.
Outputs d3_wakes_longitudinal.png, d3_wakes_transverse.png, d3_wake_profiles.png."""
import numpy as np, h5py, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
sB3, sB4, Rphi = 7.6064, 12.6122, 0.5002
g = h5py.File('2d_chic_on_wakes.h5'); ks = sorted(g.keys(), key=lambda k: g[k].attrs['position'])
ks = [k for k in ks if sB3 - 0.01 <= g[k].attrs['position'] <= sB4 + 1e-6]
m = h5py.File('1d_chic_nsb500.mat'); s1 = m['s'][()].ravel() + 0.1; Si = m['Si'][()]; W1 = m['dE_dct'][()] / 1e6; lam = m['fi'][()]
nc = 6; nr = int(np.ceil(len(ks) / nc))
for comp, key, fname in [('longitudinal', 'dE_dct', 'd3_wakes_longitudinal.png'), ('transverse', 'xkicks', 'd3_wakes_transverse.png')]:
    fig, ax = plt.subplots(nr, nc, figsize=(3.3 * nc, 2.9 * nr), layout='constrained'); ax = ax.ravel()
    for a, k in zip(ax, ks):
        G = g[k]; p = G.attrs['position']; W = G[f'{comp}/{key}'][()]; X = G[f'{comp}/x_grids'][()] * 1e6; Z = G[f'{comp}/z_grids'][()] * 1e6
        Z = Z - Z.mean(); X = X - X.mean(); v = np.abs(W).max() or 1
        pc = a.pcolormesh(Z, X, W, cmap='RdBu_r', vmin=-v, vmax=v, shading='gouraud')
        a.set_title(f's = {p:.3f} m  ({(p - sB3) / Rphi:.1f} Rφ)', fontsize=8); a.tick_params(labelsize=6)
        cb = fig.colorbar(pc, ax=a, pad=0.01); cb.ax.tick_params(labelsize=6)
    for a in ax[len(ks):]: a.axis('off')
    for a in ax[::nc]: a.set_ylabel('x − ⟨x⟩ [µm]', fontsize=7)
    for a in ax[nc * (nr - 1):]: a.set_xlabel('z − ⟨z⟩ [µm]', fontsize=7)
    fig.suptitle(f'2D {comp} wake [MeV/m] at every kick in D3 (drift_cutoff = 0; default 1.0 stops at 1 Rφ = s 8.107 m)')
    plt.savefig(fname, dpi=110); plt.close()
# profiles: every 2D x-row vs the 1D wake
fig, ax = plt.subplots(nr, nc, figsize=(3.3 * nc, 2.7 * nr), layout='constrained'); ax = ax.ravel()
for a, k in zip(ax, ks):
    G = g[k]; p = G.attrs['position']; W = G['longitudinal/dE_dct'][()]; Z = G['longitudinal/z_grids'][()]; zc = Z.mean()
    for r, c in zip(range(W.shape[0]), plt.cm.viridis(np.linspace(0, 1, W.shape[0]))): a.plot((Z[r] - zc) * 1e6, W[r], c=c, lw=0.8)
    j = np.argmin(abs(s1 - p)); z1 = (Si[:, j] - np.sum(Si[:, j] * lam[:, j]) / np.sum(lam[:, j])) * 1e6
    a.plot(z1, W1[:, j], 'k--', lw=1.5); a.set_xlim((Z.min() - zc) * 1e6, (Z.max() - zc) * 1e6)
    a.axhline(0, c='0.6', lw=0.5); a.set_title(f's = {p:.3f} m  ({(p - sB3) / Rphi:.1f} Rφ)', fontsize=8); a.tick_params(labelsize=6)
for a in ax[len(ks):]: a.axis('off')
for a in ax[::nc]: a.set_ylabel('dE/d(ct) [MeV/m]', fontsize=7)
for a in ax[nc * (nr - 1):]: a.set_xlabel('z − ⟨z⟩ [µm]', fontsize=7)
fig.suptitle('D3: every x-row of the 2D wake (colour: −4σ → +4σ) against the 1D wake (black dashed)')
plt.savefig('d3_wake_profiles.png', dpi=110); plt.close()
print(len(ks), 'kicks in D3 plotted')
