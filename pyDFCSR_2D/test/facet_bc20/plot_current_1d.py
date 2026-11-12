"""
Current profile along BC20 from a 1D run's line density at every kick (wakes1d.h5, 'lam').

    conda run -n pydfcsr python pyDFCSR_2D/test/facet_bc20/plot_current_1d.py [run]   (default dfcsr_1d_1M)

Overlaid: a candidate peak-centred deposition window -- centre at the current peak, full weight
over +-A_FULL * FWHM, tapered to zero over a further A_TAPER * FWHM -- and the charge it holds.
Writes benchmark_results/facet_bc20/current_along_s_<run>.png.
"""
import glob
import os
import sys

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, '..', 'benchmark_results', 'facet_bc20')
E, C0 = 1.602176634e-19, 299792458.0
A_FULL, A_TAPER = 3.0, 2.0
SURF, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'


def fwhm(z, I):
    k = int(np.argmax(I))
    above = I >= 0.5 * I[k]
    a = k
    while a > 0 and above[a - 1]:
        a -= 1
    b = k
    while b < I.size - 1 and above[b + 1]:
        b += 1
    return z[k], z[b] - z[a] + (z[1] - z[0])


def main(run='dfcsr_1d_1M'):
    f = h5py.File(sorted(glob.glob(os.path.join(HERE, '..', 'benchmark_results', 'facet_bc20', 'runs',
                                                f'{run}-2*-wakes1d.h5')))[-1], 'r')
    K = sorted(f, key=lambda k: f[k].attrs['position'])
    rows = []
    for k in K:
        g = f[k]['longitudinal_1d']
        z, lam = g['z_grid'][()], g['lam'][()]
        I = lam * E * C0
        zp, fw = fwhm(z, I)
        lo, hi = zp - A_FULL * fw, zp + A_FULL * fw
        T = np.clip(1 - np.maximum(np.maximum(lo - z, z - hi), 0) / (A_TAPER * fw), 0, 1)
        q = np.trapz(lam, z)
        rows.append(dict(s=float(f[k].attrs['position']), ele=str(f[k].attrs['element']), z=z, I=I,
                         zp=zp, fw=fw, f_in=np.trapz(lam * T, z) / q,
                         sz=np.sqrt(np.trapz(lam * (z - np.trapz(lam * z, z) / q) ** 2, z) / q)))
    s = np.array([r['s'] for r in rows])

    plt.rcParams.update({'font.size': 9, 'axes.edgecolor': INK2, 'axes.labelcolor': INK,
                         'xtick.color': INK2, 'ytick.color': INK2, 'text.color': INK})
    fig = plt.figure(figsize=(13, 10), facecolor=SURF)
    gs = fig.add_gridspec(3, 2, height_ratios=(1.5, 1, 1), width_ratios=(1.6, 1))
    ax = fig.add_subplot(gs[0, :])
    # every kick as a column, z relative to the current peak
    zg = np.linspace(-300e-6, 60e-6, 721)
    M = np.array([np.interp(zg, r['z'] - r['zp'], r['I'], left=0, right=0) for r in rows])
    idx = np.arange(len(s))
    pc = ax.pcolormesh(idx, zg * 1e6, np.maximum(M, 1e1).T, norm=LogNorm(1e1, 6e4), cmap='magma_r',
                       shading='nearest', rasterized=True)
    for sgn in (-1, 1):
        ax.plot(idx, sgn * A_FULL * np.array([r['fw'] for r in rows]) * 1e6, color='#2a78d6', lw=1.2)
        ax.plot(idx, sgn * (A_FULL + A_TAPER) * np.array([r['fw'] for r in rows]) * 1e6, color='#2a78d6',
                lw=1.0, ls='--')
    ax.set_ylim(zg[0] * 1e6, zg[-1] * 1e6)
    ax.set_ylabel('z − z_peak [µm]  (head at +z)')
    cb = fig.colorbar(pc, cax=ax.inset_axes([1.005, 0.08, 0.012, 0.84]))
    cb.set_label('current [A]')
    ele = [r['ele'] for r in rows]
    for e in ('B1LE', 'B2LE', 'B3LE', 'B3RE', 'B2RE', 'B1RE'):
        j = [i for i, x in enumerate(ele) if x == e]
        if j:
            ax.axvspan(j[0] - 0.5, j[-1] + 0.5, color='#2a78d6', alpha=0.06, lw=0)
            ax.text(np.mean(j), 45, e, ha='center', fontsize=8, color=INK2)
    t = np.arange(0, len(s), 15)
    ax.set_xticks(t)
    ax.set_xticklabels([f'{s[i]:.1f}' for i in t], rotation=60, fontsize=7)
    ax.set_xlabel('s [m] at the kick (columns are kicks; bends shaded)')
    ax.set_title(f'{run}: current along BC20 at every kick, centred on the current peak; blue: a '
                 f'peak window, full weight within ±{A_FULL:g} FWHM (solid), taper to ±{A_FULL + A_TAPER:g} FWHM (dashed)',
                 loc='left', fontsize=9)

    ax = fig.add_subplot(gs[1, 0])
    ax.semilogy(s, [r['I'].max() / 1e3 for r in rows], color=INK, lw=1.6, label='peak current [kA]')
    ax.semilogy(s, [r['fw'] * 1e6 for r in rows], color='#2a78d6', lw=1.4, label='FWHM of the peak [µm]')
    ax.semilogy(s, [r['sz'] * 1e6 for r in rows], color='#eb6834', lw=1.4, label='rms σz [µm]')
    ax.legend(fontsize=8, frameon=False)
    ax.set_ylabel('kA  /  µm')
    ax2 = fig.add_subplot(gs[2, 0], sharex=ax)
    ax2.plot(s, [100 * r['f_in'] for r in rows], color='#2a78d6', lw=1.6)
    ax2.set_ylabel(f'charge inside the peak window [%]')
    ax2.set_xlabel('s from BEGBC20 [m]')
    for a in (ax, ax2):
        a.grid(color=GRID, lw=0.5)

    ax = fig.add_subplot(gs[1:, 1])
    for sp, c in ((1.06, '#898781'), (24.1, '#1baf7a'), (36.6, '#eda100'), (38.65, '#e34948'),
                  (45.5, '#4a3aa7'), (49.08, INK)):
        r = rows[int(np.argmin(np.abs(s - sp)))]
        ax.semilogy((r['z'] - r['zp']) * 1e6, np.maximum(r['I'], 1), color=c, lw=1.2,
                    label=f"{r['ele']} s = {r['s']:.2f} m (FWHM {r['fw']*1e6:.1f} µm)")
    ax.set_xlim(-320, 60)
    ax.set_ylim(10, 1e5)
    ax.set_xlabel('z − z_peak [µm]')
    ax.set_ylabel('current [A]')
    ax.legend(fontsize=7, frameon=False, loc='upper left')
    ax.grid(color=GRID, lw=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, f'current_along_s_{run}.png'), dpi=140, facecolor=SURF)
    plt.close(fig)
    print(f'wrote current_along_s_{run}.png')
    for sp in (1.06, 12.24, 24.1, 36.6, 38.0, 38.65, 41.5, 48.0, 49.08):
        r = rows[int(np.argmin(np.abs(s - sp)))]
        print(f"  s = {r['s']:6.2f} {r['ele']:14s} I_pk {r['I'].max()/1e3:6.2f} kA, FWHM {r['fw']*1e6:7.2f} um, "
              f"sigma_z {r['sz']*1e6:6.1f} um, charge in the peak window {100*r['f_in']:5.1f}%")


if __name__ == '__main__':
    main(*sys.argv[1:])
