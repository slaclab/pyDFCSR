"""
Wake maps along the chicane for one run, every kick, three views:
  (1) wakes_<run>_maps_long.png / _maps_trans.png : 2D (z, xi) map at every kick (each panel own colour scale)
  (2) wakes_<run>_zproj.png : the x-projection (charge-weighted average over xi, i.e. the wake as a function of z),
      every kick, against 1D at the same kick where the 1D run (same schedule) has one; and the x-spread
  (3) wakes_<run>_along_s.png : the projections laid out along s as waterfalls -- W(z) vs s (x-averaged) and
      W(xi) vs s (z-averaged over the core, charge-weighted) -- with the lattice drawn underneath
Usage: python plot_wakes_along_lattice.py <run> [1d run]     (runs from benchmark_results/chicane_2d_vs_1d/runs)
"""
import os, sys, glob
import numpy as np, h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

HERE = os.path.dirname(os.path.abspath(__file__))
CH = os.path.join(HERE, 'benchmark_results', 'chicane_2d_vs_1d')
if '--out' in sys.argv:
    _i = sys.argv.index('--out'); CH = os.path.normpath(os.path.join(CH, sys.argv[_i + 1])); del sys.argv[_i:_i + 2]
TITLE = ''
if '--title' in sys.argv:
    _i = sys.argv.index('--title'); TITLE = sys.argv[_i + 1]; del sys.argv[_i:_i + 2]
RUN = sys.argv[1] if len(sys.argv) > 1 else '2d_reach'
RUN1 = sys.argv[2] if len(sys.argv) > 2 else '1d_full'
SURF, INK, INK2, MUTED, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9'
RED, BLUE = '#e34948', '#2a78d6'
CMAP = LinearSegmentedColormap.from_list('div', ['#0d366b', '#2a78d6', '#9ec5f4', '#f0efec', '#f4a3a2', '#e34948', '#7a1f1e'])
BENDS = [(0.1, 0.6002, 'B1'), (5.606, 6.1062, 'B2'), (7.1062, 7.6064, 'B3'), (12.6122, 13.1124, 'B4')]
plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 8, 'figure.facecolor': SURF, 'axes.facecolor': SURF,
                     'axes.edgecolor': '#c3c2b7', 'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.labelcolor': INK2,
                     'axes.titlesize': 8, 'axes.titlecolor': INK, 'axes.spines.top': False, 'axes.spines.right': False})


def newest(pattern):
    f = sorted(glob.glob(os.path.join(CH, 'runs', pattern)), key=os.path.getmtime)
    return f[-1] if f else None


def load2d(run):
    f = h5py.File(newest(f'{run}-2*-wakes.h5'), 'r')
    out = []
    for k in sorted(f, key=lambda k: f[k].attrs['position']):
        g = f[k]
        Z = g['longitudinal/z_grids'][()]; X = g['longitudinal/x_grids'][()]
        z = Z[0]; sz = np.ptp(z) / 8.0; xi = (X - X.mean(0, keepdims=True))[:, len(z) // 2]
        out.append(dict(s=float(g.attrs['position']), ele=str(g.attrs['element']), W=g['longitudinal/dE_dct'][()],
                        F=g['transverse/xkicks'][()], z=z, zc=(z - z.mean()) / sz, xi=xi, sz=sz, sxi=np.ptp(xi) / 8.0))
    return out


def load1d(run):
    p = newest(f'{run}-2*-wakes1d.h5')
    if p is None:
        return {}
    f = h5py.File(p, 'r')
    return {round(float(f[k].attrs['position']), 4): (f[k]['longitudinal_1d/z_grid'][()], f[k]['longitudinal_1d/dE_dct'][()] / 1e6)
            for k in f}


def proj(r, M):
    u = (r['xi'] - r['xi'].mean()) / r['sxi']; wx = np.exp(-u ** 2 / 2); wx /= wx.sum()
    wz = np.exp(-r['zc'] ** 2 / 2); wz /= wz.sum()
    av = (wx[:, None] * M).sum(0)
    return av, np.sqrt((wx[:, None] * (M - av) ** 2).sum(0)), (wz[None, :] * M).sum(1)


def maps(R, key, label, fname):
    n = len(R); nc = 10; nr = int(np.ceil(n / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(1.75 * nc, 1.5 * nr + 0.9), squeeze=False)
    for i, ax in enumerate(axs.flat):
        if i >= n:
            ax.axis('off'); continue
        r = R[i]; M = r[key]; a = np.abs(M).max() or 1
        ax.pcolormesh(r['zc'], r['xi'] * 1e6, M / a, cmap=CMAP, vmin=-1, vmax=1, shading='nearest', rasterized=True)
        ax.set_title(f"{r['ele']}  {r['s']:.2f} m", loc='left', pad=1.5)
        ax.text(0.03, 0.04, f'{M.min():+.2g}…{M.max():+.2g}', transform=ax.transAxes, fontsize=6, color=INK,
                bbox=dict(boxstyle='round,pad=0.1', fc=SURF, ec='none', alpha=0.85))
        ax.tick_params(labelsize=5.5, length=1.5)
        if i % nc == 0: ax.set_ylabel('ξ [µm]')
        if i // nc == nr - 1 or i + nc >= n: ax.set_xlabel('z / σz')
        else: ax.set_xticklabels([])
    fig.suptitle(f'{TITLE}{label}, run {RUN}, every kick (z/σz, ξ = offset from the tilted axis); each panel normalised to its own '
                 f'max |value|, red > 0, blue < 0, range in MeV/m', x=0.01, ha='left', fontsize=10, fontweight='bold', color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.975), h_pad=0.4, w_pad=0.3)
    fig.savefig(os.path.join(CH, fname), dpi=130); plt.close(fig); print('wrote', fname)


def zproj(R, D1, fname):
    n = len(R); nc = 10; nr = int(np.ceil(n / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(1.75 * nc, 1.4 * nr + 1.0), squeeze=False)
    for i, ax in enumerate(axs.flat):
        if i >= n:
            ax.axis('off'); continue
        r = R[i]; av, sp, _ = proj(r, r['W'])
        ax.axhline(0, color=GRID, lw=0.8)
        ax.fill_between(r['zc'], av - sp, av + sp, color=RED, alpha=0.18, lw=0)
        ax.plot(r['zc'], av, color=RED, lw=1.4)
        d = D1.get(round(r['s'], 4))
        if d is not None:
            zz = (d[0] - r['z'].mean()) / r['sz']; m = np.abs(zz) <= 4
            ax.plot(zz[m], d[1][m], color=BLUE, lw=1.2, ls=(0, (3, 1.5)))
        ax.set_xlim(-4, 4)
        ax.set_title(f"{r['ele']}  {r['s']:.2f} m", loc='left', pad=1.5)
        ax.tick_params(labelsize=5.5, length=1.5)
        if i % nc == 0: ax.set_ylabel('MeV/m')
        if i // nc == nr - 1 or i + nc >= n: ax.set_xlabel('z / σz')
        else: ax.set_xticklabels([])
    h = [plt.Line2D([], [], color=RED, lw=1.4), plt.Rectangle((0, 0), 1, 1, color=RED, alpha=0.18),
         plt.Line2D([], [], color=BLUE, lw=1.2, ls=(0, (3, 1.5)))]
    fig.legend(h, [f'2D, x-projection (charge-weighted average over ξ), run {RUN}', '± rms over ξ at that z',
                   f'1D at the same kick, run {RUN1}'], loc='upper right', ncol=3, frameon=False, fontsize=8.5, labelcolor=INK2)
    fig.suptitle(f'{TITLE}Longitudinal wake dE/ds projected on z, every kick', x=0.01, ha='left', fontsize=10, fontweight='bold', color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.965), h_pad=0.4, w_pad=0.3)
    fig.savefig(os.path.join(CH, fname), dpi=130); plt.close(fig); print('wrote', fname)


def along_s(R, D1, fname):
    s = np.array([r['s'] for r in R])
    zc = np.linspace(-4, 4, 81); xu = np.linspace(-4, 4, 41)
    Wz = np.array([np.interp(zc, r['zc'], proj(r, r['W'])[0]) for r in R])
    Wx = np.array([np.interp(xu, (r['xi'] - r['xi'].mean()) / r['sxi'], proj(r, r['W'])[2]) for r in R])
    Fz = np.array([np.interp(zc, r['zc'], proj(r, r['F'])[0]) for r in R])
    W1 = np.full_like(Wz, np.nan)
    for i, r in enumerate(R):
        d = D1.get(round(r['s'], 4))
        if d is not None:
            W1[i] = np.interp(zc, (d[0] - r['z'].mean()) / r['sz'], d[1], left=np.nan, right=np.nan)
    # kicks are unevenly spaced (dense in bends): plot against kick INDEX, label with s and element
    idx = np.arange(len(s))
    fig = plt.figure(figsize=(15, 10.5))
    gs = fig.add_gridspec(5, 1, height_ratios=[0.25, 1, 1, 1, 1], hspace=0.12)
    axL = fig.add_subplot(gs[0]); axs = [fig.add_subplot(gs[i], sharex=axL) for i in range(1, 5)]
    ele = [r['ele'] for r in R]
    for i, e in enumerate(ele):
        if e.startswith('B'):
            axL.axvspan(i - 0.5, i + 0.5, color='#c3c2b7', lw=0)
    for e in sorted(set(ele), key=ele.index):
        j = [i for i, x in enumerate(ele) if x == e]
        axL.text(np.mean(j), 0.5, e, ha='center', va='center', fontsize=9, color=INK, fontweight='bold' if e.startswith('B') else 'normal')
    axL.set_yticks([]); axL.set_title('lattice (grey: dipoles); horizontal axis is the kick number, s [m] below', loc='left', fontsize=9)
    plt.setp(axL.get_xticklabels(), visible=False)
    for ax, M, lab in ((axs[0], Wz, f'2D longitudinal, x-projection W(z)  [{RUN}]'), (axs[1], W1, f'1D longitudinal W(z)  [{RUN1}]'),
                       (axs[2], Wx, '2D longitudinal, z-projection over the core W(ξ)'), (axs[3], Fz, '2D transverse force, x-projection F(z)')):
        lim = np.nanmax(np.abs(M)) or 1
        y = xu if M is Wx else zc
        pc = ax.pcolormesh(idx, y, M.T, cmap=CMAP, norm=TwoSlopeNorm(0, -lim, lim), shading='nearest', rasterized=True)
        ax.set_ylabel('ξ / σξ' if M is Wx else 'z / σz'); ax.set_title(lab, loc='left', fontsize=9)
        cb = fig.colorbar(pc, ax=ax, pad=0.01, fraction=0.025); cb.set_label('MeV/m'); cb.ax.tick_params(labelsize=7)
        if ax is not axs[-1]: plt.setp(ax.get_xticklabels(), visible=False)
    t = np.arange(0, len(s), 4)
    axs[-1].set_xticks(t); axs[-1].set_xticklabels([f'{s[i]:.2f}' for i in t], rotation=60, fontsize=7); axs[-1].set_xlabel('s [m]')
    fig.suptitle(f'{TITLE}Chicane wakes along the lattice, run {RUN} (signed, one colour scale per panel; bands of the bend kicks are dense)',
                 x=0.01, ha='left', fontsize=11, fontweight='bold', color=INK, y=0.995)
    fig.savefig(os.path.join(CH, fname), dpi=130, bbox_inches='tight'); plt.close(fig); print('wrote', fname)


if __name__ == '__main__':
    R = load2d(RUN); D1 = load1d(RUN1)
    print(f'{RUN}: {len(R)} kicks; 1D {RUN1}: {len(D1)} kicks, {sum(round(r["s"], 4) in D1 for r in R)} matched')
    maps(R, 'W', 'Longitudinal wake dE/ds', f'wakes_{RUN}_maps_long.png')
    maps(R, 'F', 'Transverse force', f'wakes_{RUN}_maps_trans.png')
    zproj(R, D1, f'wakes_{RUN}_zproj.png')
    along_s(R, D1, f'wakes_{RUN}_along_s.png')
