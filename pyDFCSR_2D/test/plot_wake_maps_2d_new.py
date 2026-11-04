"""
Wake maps (x, z) at every kick of the chicane run 2d_new (both 2026-11-04 fixes, drift_cutoff 1.0) and of the
FODO check (default vs both fixes). Coordinates: z/sigma_z along the bunch, xi = offset from the beam axis at
that z (the mesh follows the tilted beam, so lab x would show a thin tilted strip at |tau| ~ 100).
Each panel is normalised to its own max |W| (diverging, grey = 0) and labelled with its range in MeV/m.
Writes into benchmark_results/chicane_2d_vs_1d/ and benchmark_results/fodo_auto/newfix_check/.
"""
import os, glob
import numpy as np, h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

HERE = os.path.dirname(os.path.abspath(__file__))
CH = os.path.join(HERE, 'benchmark_results', 'chicane_2d_vs_1d')
FO = os.path.join(HERE, 'benchmark_results', 'fodo_auto', 'newfix_check')
SURF, INK, INK2, MUTED = '#fcfcfb', '#0b0b0b', '#52514e', '#898781'
CMAP = LinearSegmentedColormap.from_list('div', ['#0d366b', '#2a78d6', '#9ec5f4', '#f0efec', '#f4a3a2', '#e34948', '#7a1f1e'])
plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 8, 'figure.facecolor': SURF, 'axes.facecolor': SURF,
                     'axes.edgecolor': '#c3c2b7', 'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.labelcolor': INK2,
                     'axes.titlesize': 8, 'axes.titlecolor': INK})


def load(prefix):
    f = h5py.File(glob.glob(os.path.join(CH, 'runs', f'{prefix}*-wakes.h5'))[0], 'r')
    ks = sorted(f, key=lambda k: f[k].attrs['position'])
    out = []
    for k in ks:
        g = f[k]
        out.append(dict(s=float(g.attrs['position']), ele=g.attrs['element'], W=g['longitudinal/dE_dct'][()],
                        F=g['transverse/xkicks'][()], X=g['longitudinal/x_grids'][()], Z=g['longitudinal/z_grids'][()]))
    return out


def coords(X, Z):
    z = Z[0]; zc = (z - z.mean()) / (np.ptp(z) / 8.0)          # mesh spans +-4 sigma_z
    xi = X - X.mean(0, keepdims=True)                           # offset from the axis at each z
    return zc, xi[:, len(z) // 2] * 1e6


def panel(ax, M, zc, xi_um, title, show_x, show_y):
    a = np.abs(M).max()
    ax.pcolormesh(zc, xi_um, M / (a if a > 0 else 1), cmap=CMAP, vmin=-1, vmax=1, shading='nearest', rasterized=True)
    ax.set_title(title, loc='left', pad=2)
    ax.text(0.03, 0.04, f'{M.min():+.3g} … {M.max():+.3g}', transform=ax.transAxes, fontsize=6.5, color=INK,
            bbox=dict(boxstyle='round,pad=0.15', fc=SURF, ec='none', alpha=0.85))
    ax.tick_params(labelsize=6, length=2)
    ax.set_xlabel('z / σz' if show_x else ''); ax.set_ylabel('ξ [µm]' if show_y else '')
    if not show_x: ax.set_xticklabels([])


def all_kicks(run, key, label, fname):
    R = load(run); n = len(R); nc = 8; nr = int(np.ceil(n / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(2.05 * nc, 1.75 * nr + 0.9), squeeze=False)
    for i, ax in enumerate(axs.flat):
        if i >= n:
            ax.axis('off'); continue
        r = R[i]; zc, xi = coords(r['X'], r['Z'])
        panel(ax, r[key], zc, xi, f"{r['ele']}  s = {r['s']:.3f} m", i // nc == nr - 1 or i + nc >= n, i % nc == 0)
    fig.suptitle(f'{label} at every kick — chicane run 2d_new (tangent_bands + vx_model affine, drift_cutoff 1.0)\n'
                 f'each panel normalised to its own max |value|; red > 0, blue < 0; range printed in MeV/m; '
                 f'no kicks in D3 beyond 1 R·φ (8.107 m) because of the drift cutoff',
                 x=0.01, ha='left', fontsize=10.5, fontweight='bold', color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.965), h_pad=0.6, w_pad=0.4)
    fig.savefig(os.path.join(CH, fname), dpi=130); plt.close(fig); print('wrote', fname)


def old_vs_new(fname):
    A, B = load('2d-2026-10-30T13_17_33'), load('2d_new')
    pick = [9, 20, 25, 33, 40, 45, 49, 51, 53, 55]                 # B2 B2 B2 B3 B3 B3 D3 D3 B4 B4
    fig, axs = plt.subplots(3, len(pick), figsize=(1.95 * len(pick), 6.2), squeeze=False)
    for j, i in enumerate(pick):
        a, b = A[i], B[i]; assert abs(a['s'] - b['s']) < 1e-6
        zc, xi = coords(b['X'], b['Z'])
        panel(axs[0, j], a['W'], zc, xi, f"{b['ele']}  s = {b['s']:.3f}", False, j == 0)
        panel(axs[1, j], b['W'], zc, xi, '', False, j == 0)
        panel(axs[2, j], b['W'] - a['W'], zc, xi, '', True, j == 0)
    for r, t in enumerate(('old default\n(columns dropped)', 'both fixes\n(new default)', 'new − old')):
        axs[r, 0].annotate(t, xy=(-0.62, 0.5), xycoords='axes fraction', ha='right', va='center', fontsize=9,
                           fontweight='bold', color=INK)
    fig.suptitle('Chicane longitudinal wake dE/ds [MeV/m]: old default vs both fixes, same kicks '
                 '(each panel normalised to its own max |W|; range printed)', x=0.01, ha='left', fontsize=10.5,
                 fontweight='bold', color=INK)
    fig.tight_layout(rect=(0.06, 0, 1, 0.95), h_pad=0.5, w_pad=0.4)
    fig.savefig(os.path.join(CH, fname), dpi=140); plt.close(fig); print('wrote', fname)


def fodo(fname):
    A, B = np.load(os.path.join(FO, 'base.npz')), np.load(os.path.join(FO, 'new.npz'))
    full = np.load(os.path.join(HERE, 'benchmark_results', 'fodo_auto', 'full_run.npz'), allow_pickle=True)
    maps = list(full['maps']); n = len(B['s'])
    fig, axs = plt.subplots(3, n, figsize=(1.55 * n, 5.6), squeeze=False)
    for j in range(n):
        mp = maps[j]; assert abs(mp['s'] - B['s'][j]) < 1e-6
        zz, xx = mp['zz'], mp['xx']
        zc = (zz[0] - zz[0].mean()) / mp['sigma_z']; xi = (xx[:, 0] - xx[:, 0].mean()) * 1e6
        panel(axs[0, j], A['dE'][j], zc, xi, f"s = {B['s'][j]:.3f}", False, j == 0)
        panel(axs[1, j], B['dE'][j], zc, xi, '', False, j == 0)
        panel(axs[2, j], B['dE'][j] - A['dE'][j], zc, xi, '', True, j == 0)
    for r, t in enumerate(('old default', 'both fixes', 'new − old')):
        axs[r, 0].annotate(t, xy=(-0.7, 0.5), xycoords='axes fraction', ha='right', va='center', fontsize=9,
                           fontweight='bold', color=INK)
    fig.suptitle('FODO longitudinal wake dE/ds [MeV/m] at every kick (the whole wake is the div v term; transverse is '
                 'identically 0)\nthe fix removes a ~6% smoothing shrinkage of ∂vₓ/∂x', x=0.01, ha='left',
                 fontsize=10.5, fontweight='bold', color=INK)
    fig.tight_layout(rect=(0.05, 0, 1, 0.92), h_pad=0.5, w_pad=0.3)
    fig.savefig(os.path.join(FO, fname), dpi=140); plt.close(fig); print('wrote', fname)


if __name__ == '__main__':
    all_kicks('2d_new', 'W', 'Longitudinal wake dE/ds', 'wakes_2d_new_longitudinal.png')
    all_kicks('2d_new', 'F', 'Transverse force', 'wakes_2d_new_transverse.png')
    old_vs_new('wakes_old_vs_new.png')
    fodo('fodo_wakes_old_vs_new.png')
