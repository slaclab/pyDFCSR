"""
Closed benchmark chicane: the BC20 recipe (2dP: peak frame, capped z smoothing, charge mesh 12 x 40 with
the x axis on the 0.2% quantiles) against the rms-frame 2D run (2dC) and 1D (1d_full), at the same kicks.

    conda run -n pydfcsr python pyDFCSR_2D/test/chicane_compare_2dP.py

Projected wakes are charge-weighted over the x rows: for the charge mesh from its node density
(as plot_bc20_wakes.load2d), for the uniform +-4 sigma mesh a Gaussian (as chicane_2d_vs_1d.wakes2d).
Writes benchmark_results/chicane_2d_vs_1d/closed/compare_2dP_zproj.png, compare_2dP_maps.png (2D maps
W(z, xi) and F(z, xi) of both runs at the same kicks, one colour scale per kick and quantity) and prints the roughness of
the projected wake (rms deviation from a 3-node running median / rms wake), median by element.
"""
import glob
import os

import h5py
import numpy as np
from scipy.ndimage import median_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'benchmark_results', 'chicane_2d_vs_1d', 'closed')
BENDS = [(0.1, 0.6002, 'B1'), (5.606, 6.1062, 'B2'), (7.1062, 7.6064, 'B3'), (12.6122, 13.1124, 'B4')]
EDGES = [0.0, 0.1, 0.6002, 5.606, 6.1062, 7.1062, 7.6064, 12.6122, 13.1124, 13.3124]
ENAMES = ['D0', 'B1', 'D1', 'B2', 'D2', 'B3', 'D3', 'B4', 'Df']
PICK = [(0.35, 'B1 mid'), (0.6, 'B1 exit'), (3.0, 'D1'), (5.86, 'B2 mid'), (6.6, 'D2'), (7.36, 'B3 mid'),
        (7.6, 'B3 exit'), (8.5, 'D3 early'), (11.0, 'D3 late'), (12.86, 'B4 mid'), (13.1, 'B4 exit')]
SERIES = [('2D rms frame, uniform 12×40 mesh (2dC)', '2dC', '#1baf7a', 1.4),
          ('2D BC20 recipe: peak frame, capped smoothing, charge 12×40 mesh (2dP)', '2dP', '#4a3aa7', 1.6)]


def load2d(name):
    prefix = str(np.load(os.path.join(OUT, f'{name}.npz'))['prefix'])
    f = h5py.File(glob.glob(os.path.join(OUT, 'runs', f'{prefix}-wakes.h5'))[0], 'r')
    out = {}
    for k in f:
        g = f[k]
        W = g['longitudinal/dE_dct'][()]
        F = g['transverse/xkicks'][()]
        z = g['longitudinal/z_grids'][()][0]
        X = g['longitudinal/x_grids'][()]
        if 'x_axis_transformed' in g:
            xi = g['x_axis_transformed'][()]
            h = np.gradient(xi)
            n, L = xi.size, xi[-1] - xi[0]
            wx = np.clip(1.0 / ((n - 1) * h) - 0.3 / L, 0, None) * h
        else:
            u = np.linspace(-4, 4, W.shape[0])
            wx = np.exp(-u ** 2 / 2)
            xi = (X - X.mean(0, keepdims=True))[:, len(z) // 2]
        wx = wx / wx.sum()
        out[round(float(g.attrs['position']), 6)] = dict(z=z, W=(wx[:, None] * W).sum(0), F=(wx[:, None] * F).sum(0),
                                                        Wmap=W, Fmap=F, xi=xi - xi.mean())
    return out


def load1d(name):
    prefix = str(np.load(os.path.join(OUT, f'{name}.npz'))['prefix'])
    fs = glob.glob(os.path.join(OUT, 'runs', f'{prefix}-wakes1d.h5'))
    if not fs:
        return {}
    f = h5py.File(fs[0], 'r')
    return {round(float(f[k].attrs['position']), 6): (f[k]['longitudinal_1d/z_grid'][()],
                                                      f[k]['longitudinal_1d/dE_dct'][()] / 1e6) for k in f}


def element(s):
    return ENAMES[min(np.searchsorted(EDGES, s, side='left') - 1, len(ENAMES) - 1)] if s > 0 else 'D0'


def main():
    D = {n: load2d(n) for _, n, _, _ in SERIES}
    D1 = load1d('1d_full')
    common = sorted(set(D['2dC']) & set(D['2dP']))
    print(f'{len(common)} common kicks')
    print('roughness of the projected W (rms deviation from a 3-node running median / rms W), median by element; '
          'and rel. L2 difference 2dP vs 2dC of the projected W (on the 2dC nodes)')
    for e in ENAMES:
        ks = [s for s in common if element(s) == e]
        if not ks:
            continue
        row = []
        for _, n, _, _ in SERIES:
            r = [np.sqrt(np.mean((D[n][s]['W'] - median_filter(D[n][s]['W'], 3, mode='nearest')) ** 2))
                 / np.sqrt(np.mean(D[n][s]['W'] ** 2)) for s in ks]
            row.append(f'{n} {np.median(r):.3f}')
        dl = [np.linalg.norm(np.interp(D['2dC'][s]['z'], D['2dP'][s]['z'], D['2dP'][s]['W']) - D['2dC'][s]['W'])
              / np.linalg.norm(D['2dC'][s]['W']) for s in ks]
        print(f'  {e:3s} {len(ks):3d} kicks: ' + '   '.join(row) + f'   rel L2 2dP vs 2dC {np.median(dl):.3f}')

    picks = [(min(common, key=lambda s: abs(s - p)), lab) for p, lab in PICK]
    fig, axs = plt.subplots(2, len(picks), figsize=(2.6 * len(picks), 5.4))
    for j, (s, lab) in enumerate(picks):
        for _, n, c, lw in SERIES:
            r = D[n][s]
            axs[0, j].plot(r['z'] * 1e6, r['W'], color=c, lw=lw)
            axs[1, j].plot(r['z'] * 1e6, r['F'], color=c, lw=lw)
        d = D1.get(round(s, 6))
        if d is not None:
            axs[0, j].plot(d[0] * 1e6, d[1], color='#eb6834', ls=(0, (3, 1.5)), lw=1.1)
        axs[0, j].set_title(f'{lab}, s = {s:.2f} m', loc='left', fontsize=8)
        for a in axs[:, j]:
            a.axhline(0, color='#e1e0d9', lw=0.8)
            a.tick_params(labelsize=6)
        axs[1, j].set_xlabel('z [µm]', fontsize=7)
    axs[0, 0].set_ylabel('W, charge-weighted over x [MeV/m]', fontsize=7)
    axs[1, 0].set_ylabel('transverse force [MeV/m]', fontsize=7)
    h = [plt.Line2D([], [], color=c, lw=lw) for _, _, c, lw in SERIES] + \
        [plt.Line2D([], [], color='#eb6834', ls=(0, (3, 1.5)))]
    fig.legend(h, [lab for lab, _, _, _ in SERIES] + ['1D, no drift cutoff (1d_full)'], loc='upper right', ncol=3,
               frameon=False, fontsize=8)
    fig.suptitle('Closed benchmark chicane: projected wakes at the same kicks', x=0.01, ha='left', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(os.path.join(OUT, 'compare_2dP_zproj.png'), dpi=130)
    print('wrote compare_2dP_zproj.png')
    maps(D, picks)


def maps(D, picks):
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list('div', ['#0d366b', '#2a78d6', '#9ec5f4', '#f0efec',
                                                     '#f4a3a2', '#e34948', '#7a1f1e'])
    rows = [('Wmap', '2dC', 'W, rms frame (2dC)'), ('Wmap', '2dP', 'W, BC20 recipe (2dP)'),
            ('Fmap', '2dC', 'F, rms frame (2dC)'), ('Fmap', '2dP', 'F, BC20 recipe (2dP)')]
    fig, axs = plt.subplots(len(rows), len(picks), figsize=(2.15 * len(picks), 1.95 * len(rows) + 0.9))
    for j, (s, lab) in enumerate(picks):
        for key in ('Wmap', 'Fmap'):
            a = max(np.abs(D[n][s][key]).max() for n in ('2dC', '2dP')) or 1.0
            for i, (k, n, rl) in enumerate(rows):
                if k != key:
                    continue
                r = D[n][s]
                ax = axs[i, j]
                ax.pcolormesh(r['z'] * 1e6, r['xi'] * 1e6, r[k], cmap=cmap, vmin=-a, vmax=a, shading='nearest',
                              rasterized=True)
                zz, xx = np.meshgrid(r['z'] * 1e6, r['xi'] * 1e6)
                ax.plot(zz.ravel(), xx.ravel(), '.', color='#52514e', ms=0.6)
                ax.text(0.03, 0.04, f"{r[k].min():+.3g} … {r[k].max():+.3g}", transform=ax.transAxes, fontsize=5.5,
                        bbox=dict(boxstyle='round,pad=0.1', fc='#fcfcfb', ec='none', alpha=0.85))
                ax.tick_params(labelsize=5.5, length=1.5)
                if j == 0:
                    ax.set_ylabel(rl + '\nξ [µm]', fontsize=6.5)
                if i == 0:
                    ax.set_title(f'{lab}, s = {s:.2f} m', loc='left', fontsize=7)
                if i == len(rows) - 1:
                    ax.set_xlabel('z [µm]', fontsize=6.5)
        # same xi range for the two runs of a column (the larger of the two meshes)
        lo = min(D[n][s]['xi'].min() for n in ('2dC', '2dP')) * 1e6
        hi = max(D[n][s]['xi'].max() for n in ('2dC', '2dP')) * 1e6
        for i in range(len(rows)):
            axs[i, j].set_ylim(lo, hi)
    fig.suptitle('Closed benchmark chicane: 2D wake maps (MeV/m; red > 0, blue < 0; one colour scale per kick for W and '
                 'one for F, shared by the two runs); grey dots are the mesh nodes', x=0.01, ha='left', fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(OUT, 'compare_2dP_maps.png'), dpi=140)
    print('wrote compare_2dP_maps.png')


if __name__ == '__main__':
    main()
