"""
Chirp scan on the closed chicane: wakes compared ACROSS chirps.
  chirp_scan_wakes_zproj.png : 2D x-projection W(z) (solid) and 1D (dashed) at representative kicks
                               (mid-B2, mid-B3, B3 exit, mid-D3, mid-B4, B4 exit), one row per chirp
  chirp_scan_wakes_maps.png  : 2D (z, xi) longitudinal map at the same kicks, one row per chirp
  chirp_scan_wakes_along_s.png: |W| at bunch centre along s for every chirp, 2D vs 1D (log scale)
Runs: closed/ (nominal, h = -36) and shear<S>/closed/. Usage: python plot_chirp_scan_wakes.py
"""
import os, glob
import numpy as np, h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

HERE = os.path.dirname(os.path.abspath(__file__))
B = os.path.join(HERE, 'benchmark_results', 'chicane_2d_vs_1d')
SH = [(-100, -20, 2.0), (-140, -28, 3.3), (-160, -32, 5.0), (-180, -36, 9.8), (-190, -38, 18.6), (-210, -42, 19.3), (-195, -39, 32.1)]
FOLD = {-190, -195, -210}
PICK = [('mid B2', 5.854), ('mid B3', 7.354), ('B3 exit', 7.606), ('mid D3', 10.06), ('mid B4', 12.86), ('B4 exit', 13.112)]
SURF, INK, INK2, MUTED, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9'
RED, BLUE = '#e34948', '#2a78d6'
CMAP = LinearSegmentedColormap.from_list('div', ['#0d366b', '#2a78d6', '#9ec5f4', '#f0efec', '#f4a3a2', '#e34948', '#7a1f1e'])
plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 8, 'figure.facecolor': SURF, 'axes.facecolor': SURF,
                     'axes.edgecolor': '#c3c2b7', 'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.labelcolor': INK2,
                     'axes.titlesize': 8.5, 'axes.titlecolor': INK, 'axes.spines.top': False, 'axes.spines.right': False})


def folder(sh):
    return os.path.join(B, 'closed') if sh == -180 else os.path.join(B, f'shear{sh}', 'closed')


def newest(pat):
    f = sorted(glob.glob(pat), key=os.path.getmtime); return f[-1] if f else None


def load(sh):
    d = folder(sh)
    h = h5py.File(newest(os.path.join(d, 'runs', '2dC-2*-wakes.h5')), 'r')
    W2 = {}
    for k in h:
        g = h[k]; Z = g['longitudinal/z_grids'][()]; X = g['longitudinal/x_grids'][()]
        z = Z[0]; sz = np.ptp(z) / 8.0; xi = (X - X.mean(0, keepdims=True))[:, len(z) // 2]
        u = xi / (np.ptp(xi) / 8.0); wx = np.exp(-u ** 2 / 2); wx /= wx.sum()
        M = g['longitudinal/dE_dct'][()]
        W2[round(float(g.attrs['position']), 3)] = dict(ele=str(g.attrs['element']), M=M, av=(wx[:, None] * M).sum(0),
                                                       zc=(z - z.mean()) / sz, z=z, sz=sz, xi=xi)
    h1 = h5py.File(newest(os.path.join(d, 'runs', '1d_full-2*-wakes1d.h5')), 'r')
    W1 = {round(float(h1[k].attrs['position']), 3): (h1[k]['longitudinal_1d/z_grid'][()], h1[k]['longitudinal_1d/dE_dct'][()] / 1e6) for k in h1}
    return W2, W1


def nearest(W, s):
    k = min(W, key=lambda q: abs(q - s)); return k, W[k]


D = {sh: load(sh) for sh, _, _ in SH}

# ---------------- x-projection at representative kicks ----------------
fig, axs = plt.subplots(len(SH), len(PICK), figsize=(2.3 * len(PICK), 1.55 * len(SH) + 1.0), squeeze=False)
for i, (sh, h, C) in enumerate(SH):
    W2, W1 = D[sh]
    for j, (lab, s) in enumerate(PICK):
        ax = axs[i, j]; k, r = nearest(W2, s)
        ax.axhline(0, color=GRID, lw=0.7)
        ax.plot(r['zc'], r['av'], color=RED, lw=1.5)
        if k in W1:
            z1, w1 = W1[k]; zz = (z1 - r['z'].mean()) / r['sz']; m = np.abs(zz) <= 4
            ax.plot(zz[m], w1[m], color=BLUE, lw=1.2, ls=(0, (3, 1.5)))
        ax.set_xlim(-4, 4); ax.tick_params(labelsize=6, length=1.5)
        if i == 0: ax.set_title(f'{lab}\n', loc='center', fontweight='bold')
        ax.text(0.03, 0.95, f's = {k:.2f}', transform=ax.transAxes, fontsize=6, color=MUTED, va='top')
        if j == 0: ax.set_ylabel(f'h {h}, C {C}' + ('\n(folded)' if sh in FOLD else '') + '\nMeV/m', fontsize=7)
        if i == len(SH) - 1: ax.set_xlabel('z / σz')
        else: ax.set_xticklabels([])
h_ = [plt.Line2D([], [], color=RED, lw=1.5), plt.Line2D([], [], color=BLUE, lw=1.2, ls=(0, (3, 1.5)))]
fig.legend(h_, ['2D, x-projection (charge-weighted over ξ), all current defaults', '1D, no cutoff, same kick'],
           loc='upper right', ncol=2, frameon=False, fontsize=8.5, labelcolor=INK2)
fig.suptitle('Chirp scan, closed chicane: longitudinal wake W(z) at representative kicks, one row per chirp (h, CSR-off compression C);'
             ' each panel its own scale', x=0.01, ha='left', fontsize=10, fontweight='bold', color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.965), h_pad=0.3, w_pad=0.4)
fig.savefig(os.path.join(B, 'chirp_scan_wakes_zproj.png'), dpi=130); plt.close(fig); print('wrote chirp_scan_wakes_zproj.png')

# ---------------- 2D maps at the same kicks ----------------
fig, axs = plt.subplots(len(SH), len(PICK), figsize=(2.3 * len(PICK), 1.55 * len(SH) + 1.0), squeeze=False)
for i, (sh, h, C) in enumerate(SH):
    W2, _ = D[sh]
    for j, (lab, s) in enumerate(PICK):
        ax = axs[i, j]; k, r = nearest(W2, s); a = np.abs(r['M']).max() or 1
        ax.pcolormesh(r['zc'], r['xi'] * 1e6, r['M'] / a, cmap=CMAP, vmin=-1, vmax=1, shading='nearest', rasterized=True)
        ax.text(0.03, 0.04, f'{r["M"].min():+.2g}…{r["M"].max():+.2g}', transform=ax.transAxes, fontsize=5.5, color=INK,
                bbox=dict(boxstyle='round,pad=0.1', fc=SURF, ec='none', alpha=0.85))
        ax.tick_params(labelsize=5.5, length=1.5)
        if i == 0: ax.set_title(f'{lab}\n', loc='center', fontweight='bold')
        if j == 0: ax.set_ylabel(f'h {h}, C {C}' + ('\n(folded)' if sh in FOLD else '') + '\nξ [µm]', fontsize=7)
        if i == len(SH) - 1: ax.set_xlabel('z / σz')
        else: ax.set_xticklabels([])
fig.suptitle('Chirp scan, closed chicane: 2D longitudinal wake maps (z/σz, ξ = offset from the tilted axis) at the same kicks;'
             ' each panel normalised to its own max |W|, range printed in MeV/m', x=0.01, ha='left', fontsize=10, fontweight='bold', color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.965), h_pad=0.3, w_pad=0.4)
fig.savefig(os.path.join(B, 'chirp_scan_wakes_maps.png'), dpi=130); plt.close(fig); print('wrote chirp_scan_wakes_maps.png')

# ---------------- wake at bunch centre along s ----------------
fig, ax = plt.subplots(1, 1, figsize=(13, 5))
cm = plt.get_cmap('viridis')
for i, (sh, h, C) in enumerate(SH):
    W2, W1 = D[sh]; col = cm(i / (len(SH) - 1))
    ks = sorted(W2); s = np.array(ks)
    c2 = np.array([np.interp(0.0, W2[k]['zc'], W2[k]['av']) for k in ks])
    c1 = np.array([np.interp(W2[k]['z'].mean(), *W1[k]) if k in W1 else np.nan for k in ks])
    ax.plot(s, -c2, color=col, lw=1.6, marker='o', ms=2.5, label=f'h {h} (C {C})' + (' folded' if sh in FOLD else ''))
    ax.plot(s, -c1, color=col, lw=1.0, ls=(0, (3, 1.5)))
for a, b in ((0.1, 0.6002), (5.606, 6.1062), (7.1062, 7.6064), (12.6122, 13.1124)):
    ax.axvspan(a, b, color='#f0efec', zorder=0, lw=0)
ax.set_yscale('symlog', linthresh=0.01)
ax.set_xlabel('s [m]'); ax.set_ylabel('−W at bunch centre, x-averaged [MeV/m]  (energy loss > 0)')
ax.legend(frameon=False, fontsize=8, ncol=2, labelcolor=INK2, title='solid: 2D   dashed: 1D (no cutoff)', title_fontsize=8)
ax.set_title('Chirp scan, closed chicane: wake at bunch centre along the lattice (grey: dipoles)', loc='left', fontweight='bold')
fig.tight_layout(); fig.savefig(os.path.join(B, 'chirp_scan_wakes_along_s.png'), dpi=130); plt.close(fig); print('wrote chirp_scan_wakes_along_s.png')
