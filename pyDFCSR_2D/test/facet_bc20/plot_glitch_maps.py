"""
2D wake maps W(z, xi) at the D5 x-waist: the pickled 19.110 m state, the hand-stepped 19.123 m state
(the one-snapshot glitch), the latter with 3x wider ribbons, and the difference 19.123 - 19.110.
From glitch_1911.npz / glitch_19123.npz (glitch_1911.py). Writes glitch_1911_maps.png.
"""
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, '..', 'benchmark_results', 'facet_bc20')
a = np.load(os.path.join(HERE, 'glitch_1911.npz'))
b = np.load(os.path.join(HERE, 'glitch_19123.npz'))
cmap = LinearSegmentedColormap.from_list('div', ['#0d366b', '#2a78d6', '#9ec5f4', '#f0efec', '#f4a3a2', '#e34948', '#7a1f1e'])
panels = [(a['z'], a['xi'], a['W_base'], 's = 19.110 m (τ = −1.15), default bands'),
          (b['z'], b['xi'], b['W_base'], 's = 19.123 m (τ = −1.00), default bands: the glitch'),
          (b['z'], b['xi'], b['W_margin3'], 's = 19.123 m, ribbons 3× wider'),
          (b['z'], b['xi'], b['W_base'] - a['W_base'], 'difference 19.123 − 19.110 (default)')]
vmax = max(np.abs(p[2]).max() for p in panels[:3])
fig, axs = plt.subplots(1, 4, figsize=(17, 4.6))
for ax, (z, xi, W, title) in zip(axs, panels):
    xi0 = (xi - xi.mean()) * 1e6
    pc = ax.pcolormesh(z * 1e6, xi0, W, cmap=cmap, vmin=-vmax, vmax=vmax, shading='nearest', rasterized=True)
    zz, xx = np.meshgrid(z * 1e6, xi0)
    ax.plot(zz.ravel(), xx.ravel(), '.', color='#52514e', ms=0.8)
    pw = (a['wx'][:, None] * W).sum(0) if W.shape[0] == a['wx'].size else None
    ax.text(0.03, 0.04, f'{W.min():+.3g} … {W.max():+.3g} MeV/m' + (f';  ⟨W⟩ {pw.mean():+.3f}' if pw is not None else ''),
            transform=ax.transAxes, fontsize=7.5, bbox=dict(boxstyle='round,pad=0.1', fc='#fcfcfb', ec='none', alpha=0.85))
    ax.set_title(title, loc='left', fontsize=9)
    ax.set_xlabel('z [µm]  (head at +z)')
    ax.tick_params(labelsize=7)
axs[0].set_ylabel('ξ − ⟨ξ⟩ [µm]')
fig.colorbar(pc, ax=axs.tolist(), fraction=0.012, pad=0.01, label='W [MeV/m], one scale for all panels')
fig.suptitle('BC20 drift D5 (sextupole section, x-waist): the longitudinal wake map one snapshot before and at the 19.123 m glitch; '
             'dots are the 12 × 40 mesh nodes', x=0.01, ha='left', fontsize=10)
fig.savefig(os.path.join(RES, 'glitch_1911_maps.png'), dpi=140, bbox_inches='tight')
print('wrote glitch_1911_maps.png')
