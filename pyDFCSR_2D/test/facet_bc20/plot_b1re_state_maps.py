"""
2D wake maps at the pickled B1RE kick (s = 48.39 m) from b1re_state_fullmesh.npz: the fixed layout,
near_adapt + near_adapt_x at eps 0.1, and at the defaults eps 0.02 / 4 x' rounds, plus the
difference fixed - defaults. One colour scale per row (F; W). Writes b1re_state_maps.png.

    conda run -n pydfcsr python pyDFCSR_2D/test/facet_bc20/plot_b1re_state_maps.py
"""
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, '..', 'benchmark_results', 'facet_bc20')
d = np.load(os.path.join(HERE, 'b1re_state_fullmesh.npz'))
z, xi = d['z'] * 1e6, (d['xi'] - d['xi'].mean()) * 1e6
cmap = LinearSegmentedColormap.from_list('div', ['#0d366b', '#2a78d6', '#9ec5f4', '#f0efec', '#f4a3a2', '#e34948', '#7a1f1e'])
cols = [('base', 'fixed layout'), ('adapt_x', 'near_adapt + near_adapt_x, ε 0.1'),
        ('adapt_x_e02_r4', 'ε 0.02, 4 x′ rounds (defaults)'), (None, 'fixed − defaults')]
fig, axs = plt.subplots(2, 4, figsize=(15, 7.2))
for r, (key, lab) in enumerate((('F', 'transverse force F'), ('W', 'longitudinal W'))):
    M = {k: d[f'{key}_{k}'] for k, _ in cols if k}
    M[None] = M['base'] - M['adapt_x_e02_r4']
    a = max(np.abs(M[k]).max() for k in ('adapt_x_e02_r4', 'base'))
    for c, (k, title) in enumerate(cols):
        ax = axs[r, c]
        pc = ax.pcolormesh(z, xi, M[k], cmap=cmap, vmin=-a, vmax=a, shading='nearest', rasterized=True)
        zz, xx = np.meshgrid(z, xi)
        ax.plot(zz.ravel(), xx.ravel(), '.', color='#52514e', ms=0.7)
        ax.text(0.03, 0.04, f'{M[k].min():+.3g} … {M[k].max():+.3g} MeV/m', transform=ax.transAxes, fontsize=7,
                bbox=dict(boxstyle='round,pad=0.1', fc='#fcfcfb', ec='none', alpha=0.85))
        ax.tick_params(labelsize=7)
        if r == 0:
            ax.set_title(title, loc='left', fontsize=9)
        if c == 0:
            ax.set_ylabel(lab + '\nξ − ⟨ξ⟩ [µm]', fontsize=8)
        if r == 1:
            ax.set_xlabel('z [µm]  (head at +z)', fontsize=8)
    fig.colorbar(pc, ax=axs[r, :].tolist(), fraction=0.012, pad=0.01, label='MeV/m (scale shared along the row)')
fig.suptitle('BC20, B1RE kick at s = 48.39 m (pickled state, same beam): 12 × 40 wake mesh with the fixed and the adaptive '
             'near-region quadratures; dots are the mesh nodes', x=0.01, ha='left', fontsize=10)
fig.savefig(os.path.join(RES, 'b1re_state_maps.png'), dpi=140, bbox_inches='tight')
print('wrote b1re_state_maps.png')
