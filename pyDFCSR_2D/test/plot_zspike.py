"""Before/after figure for the +-3 sigma_z wake spikes in B2/B3: z-cuts (mesh row 6) on frozen chicane states
for the deposition grid xlim 5 (default) vs wider grids at the same cell size. Data: tools_d3/zspike_fix.py."""
import os, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'benchmark_results', 'd3_diagnostic')
SURF, INK, INK2, MUTED, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9'
COL = {'default': '#e34948', 'xlim 7, xbins 420': '#2a78d6', 'xlim 9, xbins 540': '#1baf7a'}
LAB = {'default': 'xlim 5 (default): 0.28–0.30% of the charge off the grid',
       'xlim 7, xbins 420': 'xlim 7, same cell size: 0.05% off', 'xlim 9, xbins 540': 'xlim 9, same cell size: 0.01% off'}
plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 9, 'figure.facecolor': SURF, 'axes.facecolor': SURF,
                     'axes.edgecolor': '#c3c2b7', 'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.labelcolor': INK2,
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.titlelocation': 'left',
                     'axes.titleweight': 'bold', 'axes.titlecolor': INK})
fig, axs = plt.subplots(2, 2, figsize=(12.5, 7.2), gridspec_kw=dict(height_ratios=[2, 1]), sharex=True)
for j, (s, lab) in enumerate(((5.854, 'B2, s = 5.86 m (τ = −12)'), (7.385, 'B3, s = 7.39 m (τ = −32)'))):
    d = np.load(os.path.join(D, f'zspike_fix_{s:.4f}.npz')); names = list(d['names']); zc = d['zc']
    ax, ax2 = axs[0, j], axs[1, j]
    for k in ('default', 'xlim 7, xbins 420', 'xlim 9, xbins 540'):
        if k in names:
            w = d['cuts'][names.index(k)]
            ax.plot(zc, w, color=COL[k], lw=2 if k == 'default' else 1.6, label=LAB[k], zorder=3 if k == 'default' else 4)
    base = d['cuts'][names.index('default')]
    best = d['cuts'][names.index('xlim 9, xbins 540' if 'xlim 9, xbins 540' in names else 'xlim 7, xbins 420')]
    ax2.axhline(0, color=GRID, lw=0.8)
    ax2.plot(zc, base - best, color=COL['default'], lw=1.6)
    for a in (ax, ax2):
        for z0 in (-3, 3): a.axvspan(z0 - 0.35, z0 + 0.35, color='#f0efec', zorder=0, lw=0)
    ax.axhline(0, color=GRID, lw=0.8)
    ax.set_title(lab); ax.set_ylabel('dE/ds on one mesh row [MeV/m]')
    ax2.set_ylabel('default − widest grid [MeV/m]'); ax2.set_xlabel('z / σz')
    ax.legend(frameon=False, fontsize=8.5, loc='lower left', labelcolor=INK2)
fig.suptitle('The ±3σz spikes: the bunch tails bend off the deposition grid (straight tilt fit on a curved beam),\n'
             'so ρ drops to zero there and ∂ρ/∂z has a false step. A wider grid at the same cell size removes them', x=0.01, ha='left', fontsize=11, fontweight='bold', color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.92))
fig.savefig(os.path.join(D, 'zspike_before_after.png'), dpi=150); print('wrote zspike_before_after.png')
