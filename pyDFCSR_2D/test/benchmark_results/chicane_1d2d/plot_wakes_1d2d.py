"""Wake maps for the 1D-vs-2D tilt scan.
For each case:
  wakes/wakemap_<tag>.png        -- wake vs (s, z - <z>): 1D at every 1-cm slice (all 150), 2D on-axis row at every kick
  wakes/wake_steps_<tag>.png     -- 1D W(z) overlaid on every row of the 2D mesh W(x, z), at 16 positions along s
  wakes/1d_<tag>/1d_wake_step_NNN.png -- the 1D wake (and line density) at every one of its 150 steps
Usage: plot_wakes_1d2d.py [--nsb N] tag [tag ...]   (N = 1D slices in the bend; outputs get suffix _nsbN)"""
import sys, os, numpy as np, h5py, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib import cm
os.makedirs('wakes', exist_ok=True)
BENDS = [(0.1, 0.6002), (5.606, 6.1062), (7.1062, 7.6064), (12.6122, 13.1124)]   # chicane B1-B4
args = sys.argv[1:]; NSB = ''
if args and args[0] == '--nsb': NSB = '' if args[1] == '100' else f'_nsb{args[1]}'; args = args[2:]
for t in args:
    m = h5py.File(f'1d_{t}{NSB}.mat'); s1 = m['s'][()].ravel() + 0.1; Si = m['Si'][()]; W1 = m['dE_dct'][()] / 1e6; lam = m['fi'][()]
    g = h5py.File(f'2d_{t}_on_wakes.h5'); ks = sorted(g.keys(), key=lambda k: g[k].attrs['position'])
    s2 = np.array([g[k].attrs['position'] for k in ks])
    # (s, z) maps, z relative to the bunch centroid; colour scale shared, clipped at the 99th percentile of |W|
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5), layout='constrained', sharey=True)
    zc1 = (Si - np.sum(Si * lam, 0) / np.sum(lam, 0)) * 1e6
    W2c = [g[k]['longitudinal/dE_dct'][()][g[k]['longitudinal/dE_dct'].shape[0] // 2] for k in ks]
    Z2 = [g[k]['longitudinal/z_grids'][()][0] for k in ks]; Z2 = [(z - z.mean()) * 1e6 for z in Z2]
    v = np.nanpercentile(np.abs(np.concatenate([W1.ravel(), np.concatenate(W2c)])), 95)   # 95th: the waist spike would otherwise wash out the rest
    for j in range(len(s1)):
        ax[0].scatter(np.full(Si.shape[0], s1[j]), zc1[:, j], c=W1[:, j], cmap='RdBu_r', vmin=-v, vmax=v, s=4, marker='s', lw=0)
    for j in range(len(ks)):
        sc = ax[1].scatter(np.full(len(Z2[j]), s2[j]), Z2[j], c=W2c[j], cmap='RdBu_r', vmin=-v, vmax=v, s=14, marker='s', lw=0)
    for a, ttl in zip(ax, ['1D (Emma csr_calc), every step', '2D (pyDFCSR), on-axis row of the mesh, every kick']):
        [a.axvspan(*bb, color='0.93', zorder=-1) for bb in BENDS]; a.set(xlabel='s [m]', title=ttl)
    ax[0].set_ylabel('z − ⟨z⟩ [µm]'); fig.colorbar(sc, ax=ax, label='dE/d(ct) [MeV/m]')
    fig.suptitle(f'{t}: longitudinal wake along the lattice'); plt.savefig(f'wakes/wakemap_{t}{NSB}.png', dpi=110); plt.close()
    # 16 positions: 1D line on top of all 2D rows
    want = [0.35, 0.6, 1.0, 3.0, 5.85, 6.1, 6.6, 7.1, 7.35, 7.6, 8.1, 10.0, 12.86, 13.1, 13.2, 13.31]
    fig, ax = plt.subplots(4, 4, figsize=(16, 12), layout='constrained')
    for a, w in zip(ax.ravel(), want):
        k = ks[np.argmin(abs(s2 - w))]; G = g[k]; W2 = G['longitudinal/dE_dct'][()]; Z = G['longitudinal/z_grids'][()]
        zc = np.mean(Z)
        for r, c in zip(range(W2.shape[0]), cm.viridis(np.linspace(0, 1, W2.shape[0]))):
            a.plot((Z[r] - zc) * 1e6, W2[r], c=c, lw=0.8)
        j = np.argmin(abs(s1 - G.attrs['position']))
        a.plot((Si[:, j] - np.sum(Si[:, j] * lam[:, j]) / np.sum(lam[:, j])) * 1e6, W1[:, j], 'k--', lw=1.8)
        a.set_title(f"s = {G.attrs['position']:.3f} m (1D at {s1[j]:.2f})", fontsize=9); a.set_xlim(np.array([Z.min(), Z.max()]) * 1e6 - zc * 1e6)
    for a in ax[-1]: a.set_xlabel('z − ⟨z⟩ [µm]')
    for a in ax[:, 0]: a.set_ylabel('dE/d(ct) [MeV/m]')
    fig.suptitle(f'{t}: 1D wake (black dashed) vs every x-row of the 2D wake mesh (colour: −4σ → +4σ in tilt-removed x)')
    plt.savefig(f'wakes/wake_steps_{t}{NSB}.png', dpi=100); plt.close()
    # every 1D step
    d = f'wakes/1d_{t}{NSB}'; os.makedirs(d, exist_ok=True)
    for j in range(len(s1)):
        fig, a = plt.subplots(figsize=(6, 3.6), layout='constrained'); z = zc1[:, j]
        a.plot(z, W1[:, j], c='C3'); a.set(xlabel='z − ⟨z⟩ [µm]', ylabel='dE/d(ct) [MeV/m]', title=f'1D, {t}, step {j+1}, s = {s1[j]:.2f} m')
        b = a.twinx(); b.fill_between(z, lam[:, j] * 1e-3, color='0.85', zorder=-1); b.set_ylabel('λ(z) [1/mm]', color='0.5'); a.set_zorder(b.get_zorder() + 1); a.patch.set_visible(False)
        plt.savefig(f'{d}/1d_wake_step_{j+1:04d}.png', dpi=80); plt.close()
    print('done', t)
