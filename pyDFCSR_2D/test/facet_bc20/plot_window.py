"""
Beam current and the quantile-frame deposition window along BC20 (run dfcsr_2d_q).

    conda run -n pydfcsr python pyDFCSR_2D/test/facet_bc20/plot_window.py

The current is from the run's own beam at every element exit (the beam is not saved at kicks,
so the compression inside B2RE is not sampled); the window is the per-kick log of the run
(statistics h5, group quantile_frame): [z_q, z_1-q] (T = 1) and the grid edges, which add the
taper margin on each side. Writes benchmark_results/facet_bc20/window_along_s.png and
window_profiles.png.
"""
import glob
import os

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.ndimage import gaussian_filter1d

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, '..', 'benchmark_results', 'facet_bc20')
RUNS = os.path.join(RES, 'runs')
INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
Q, C0, DZ = 1.6e-9, 299792458.0, 0.5e-6
NB_DEP = 1200


def current(z, edges):
    h, _ = np.histogram(z, bins=edges)
    return gaussian_filter1d(h * Q / z.size / (DZ / C0), 1.0)


def main():
    d = np.load(os.path.join(RUNS, 'dfcsr_2d_q.npz'))
    X = np.load(os.path.join(HERE, 'bc20_export.npz'))
    coords = np.concatenate([X['coords_fine'][:1, :6], d['coords']])     # BEGBC20 + element exits
    s_ex = np.r_[0.0, d['s']]
    names = ['BEGBC20'] + list(d['names'])
    q = h5py.File(glob.glob(os.path.join(RUNS, 'dfcsr_2d_q-*statistics.h5'))[0], 'r')['quantile_frame']
    L = {k: q[k][()] for k in q}
    s = L['s']
    lo, hi, mg = L['window_lo'], L['window_hi'], L['window_margin']

    edges = np.arange(-400e-6, 300e-6 + DZ, DZ)
    zc = 0.5 * (edges[1:] + edges[:-1])
    I = np.array([current(c[4], edges) for c in coords]) / 1e3          # kA

    plt.rcParams.update({'font.size': 9, 'axes.edgecolor': INK2, 'axes.labelcolor': INK,
                         'xtick.color': INK2, 'ytick.color': INK2, 'text.color': INK})
    fig, axs = plt.subplots(3, 1, figsize=(11, 9.5), sharex=True, facecolor=SURF,
                            gridspec_kw=dict(height_ratios=(1.6, 1, 1)))
    ax = axs[0]
    # each element-exit profile drawn as a column reaching halfway to its neighbours
    s_edges = np.r_[s_ex[0], 0.5 * (s_ex[1:] + s_ex[:-1]), s_ex[-1]]
    pc = ax.pcolormesh(s_edges, edges * 1e6, np.maximum(I, 1e-2).T, norm=LogNorm(1e-2, 50), cmap='magma_r',
                       shading='flat', rasterized=True)
    ax.plot(s, lo * 1e6, color='#2a78d6', lw=1.3, label='window [z_q, z_(1-q)], T = 1')
    ax.plot(s, hi * 1e6, color='#2a78d6', lw=1.3)
    ax.plot(s, (lo - mg) * 1e6, color='#2a78d6', lw=1.0, ls='--', label='grid edges (taper to 0)')
    ax.plot(s, (hi + mg) * 1e6, color='#2a78d6', lw=1.0, ls='--')
    ax.set_ylim(-380, 280)
    ax.set_ylabel('z [µm]  (head at +z)')
    cb = fig.colorbar(pc, cax=ax.inset_axes([0.985, 0.08, 0.012, 0.84]))
    cb.set_label('current [kA]')
    ax.legend(loc='lower left', fontsize=8, frameon=False)
    ax.set_title('BC20, run dfcsr_2d_q: current profile at every element exit, and the deposition '
                 'window at every kick', loc='left', fontsize=10)

    ax = axs[1]
    ax.semilogy(s_ex, I.max(1), color=INK2, lw=1.6, marker='o', ms=3, label='peak current [kA]')
    ax.semilogy(s_ex, coords[:, 4].std(1) * 1e6, color='#eb6834', lw=1.4, label='rms σz [µm]')
    ax.semilogy(s, (hi - lo) * 1e6, color='#2a78d6', lw=1.4, label='window width z_(1-q) − z_q [µm]')
    ax.legend(fontsize=8, frameon=False, loc='upper left')
    ax.set_ylabel('kA  /  µm')

    ax = axs[2]
    dz = (hi - lo + 2 * mg) / NB_DEP
    ax.semilogy(s, L['peak_fwhm'] * 1e6, color=INK2, lw=1.6, label='FWHM of the highest peak')
    ax.semilogy(s, dz * 1e6, color='#2a78d6', lw=1.4, label=f'deposition bin ({NB_DEP} bins over the grid)')
    ax.semilogy(s, L['res_z'] * 1e6, color='#2a78d6', lw=1.4, ls='--', label='deposition resolution (bin × √(σ_s² + 2/3))')
    ax.semilogy(s, L['mesh_dz_min'] * 1e6, color='#1baf7a', lw=1.4, label='finest wake-mesh z spacing (80 nodes)')
    ax.set_ylabel('µm')
    ax.set_xlabel('s from BEGBC20 [m]')
    ax2 = ax.twinx()
    ax2.plot(s, (1 - L['f_window']) * 100, color='#e87ba4', lw=1.0)
    ax2.set_ylabel('dropped charge 1 − f_window [%]', color='#e87ba4')
    ax2.tick_params(axis='y', colors='#e87ba4')
    ax2.set_ylim(0, 0.5)
    ax.legend(fontsize=8, frameon=False, loc='center left', bbox_to_anchor=(0.0, 0.52))
    for a in axs:
        a.grid(color=GRID, lw=0.5)
        for k, n in enumerate(names):
            if n.startswith('B') and n != 'BEGBC20':
                a.axvspan(s_ex[k - 1], s_ex[k], color='#d9d8d3', lw=0, zorder=0)
    fig.savefig(os.path.join(RES, 'window_along_s.png'), dpi=140, facecolor=SURF, bbox_inches='tight')
    plt.close(fig)

    # line-outs with the window shaded
    picks = ['BEGBC20', 'B2LE', 'B3RE', 'Q2ER', 'B2RE', 'B1RE']
    fig, axs = plt.subplots(2, 3, figsize=(12, 6), facecolor=SURF)
    for ax, nm in zip(axs.flat, picks):
        k = names.index(nm)
        j = int(np.argmin(np.abs(s - s_ex[k])))
        ax.axvspan((lo[j] - mg[j]) * 1e6, (hi[j] + mg[j]) * 1e6, color='#cde2fb', lw=0, label='grid (taper)')
        ax.axvspan(lo[j] * 1e6, hi[j] * 1e6, color='#86b6ef', lw=0, label='T = 1')
        ax.plot(zc * 1e6, I[k], color=INK, lw=1.2)
        ax.set_yscale('log')
        ax.set_ylim(1e-2, 80)
        zz = coords[k][4]
        ax.set_xlim((zz.min() - 60e-6) * 1e6, (zz.max() + 60e-6) * 1e6)
        ax.set_title(f'{nm}, s = {s_ex[k]:.2f} m: window {(hi[j]-lo[j])*1e6:.0f} µm, '
                     f'bin {(hi[j]-lo[j]+2*mg[j])/NB_DEP*1e6:.2f} µm', loc='left', fontsize=9)
        ax.set_xlabel('z [µm]')
        ax.set_ylabel('current [kA]')
        ax.grid(color=GRID, lw=0.5)
    axs[0, 0].legend(fontsize=8, frameon=False, loc='upper left')
    fig.tight_layout()
    fig.savefig(os.path.join(RES, 'window_profiles.png'), dpi=140, facecolor=SURF)
    plt.close(fig)
    print('wrote window_along_s.png, window_profiles.png')


if __name__ == '__main__':
    main()
