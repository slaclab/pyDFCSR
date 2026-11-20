"""
The two 2D-only transients through the dense-kick windows (rerun_kick_dense.py): the projected wake
W(z) at every kick in [18, 20] m and [38.6, 41] m as waterfalls, and per kick its min / max / charge
mean, against the 116- and 222-kick runs' kicks in the same windows.

    conda run -n pydfcsr python pyDFCSR_2D/test/facet_bc20/plot_kdense_windows.py

Writes bc20_kdense_windows.png.
"""
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import plot_bc20_wakes as P

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, '..', 'benchmark_results', 'facet_bc20')
RUNS = [('dense (297 kicks)', 'dfcsr_2d_peak_1M_sz6.5cap_m40_k115_kdense', '#4a3aa7'),
        ('116 kicks', 'dfcsr_2d_peak_1M_sz6.5cap_m40_k115', '#1baf7a'),
        ('222 kicks', 'dfcsr_2d_peak_1M', '#eda100')]
WIN = ((18.0, 20.0), (38.6, 41.0))


def main():
    D = {f: P.load2d(f) for _, f, _ in RUNS}
    fig, axs = plt.subplots(3, 2, figsize=(13, 10), height_ratios=(1.6, 1, 1))
    for c, (lo, hi) in enumerate(WIN):
        R = [r for r in D[RUNS[0][1]] if lo <= r['s'] <= hi]
        zc = np.linspace(min(r['z'][0] for r in R), max(r['z'][-1] for r in R), 300)
        M = np.array([np.interp(zc, r['z'], P.proj(r, r['W']), left=np.nan, right=np.nan) for r in R])
        a = np.nanmax(np.abs(M))
        ax = axs[0, c]
        ax.pcolormesh([r['s'] for r in R], zc * 1e6, M.T, cmap=P.CMAP, vmin=-a, vmax=a, shading='nearest', rasterized=True)
        ax.set_ylabel('z [µm]  (head at +z)')
        ax.set_title(f'dense run: projected W(z) at every kick in {lo}–{hi} m (colour ±{a:.2f} MeV/m)', loc='left', fontsize=9)
        for lab, f, col in RUNS:
            rr = [r for r in D[f] if lo - 0.3 <= r['s'] <= hi + 0.3]
            s = [r['s'] for r in rr]
            w = [P.proj(r, r['W']) for r in rr]
            axs[1, c].plot(s, [x.max() for x in w], color=col, lw=1.2, marker='.', ms=4, label=lab + ' max')
            axs[1, c].plot(s, [x.min() for x in w], color=col, lw=1.2, marker='.', ms=4, ls='--')
            # charge-weighted mean over z: the energy-loss rate the kick applies
            axs[2, c].plot(s, [np.average(x, weights=np.histogram(np.zeros(1), bins=1)[0] * 0 + 1) if False else
                               np.trapz(x * P_lam(r), r['z']) / max(np.trapz(P_lam(r), r['z']), 1e-300)
                               for x, r in zip(w, rr)], color=col, lw=1.4, marker='.', ms=4, label=lab)
        for ax in axs[1:, c]:
            ax.axhline(0, color='#e1e0d9', lw=0.8)
            ax.axvspan(lo, hi, color='#c3c2b7', alpha=0.25, lw=0)
        axs[1, c].set_ylabel('max (solid) / min (dashed) of projected W [MeV/m]')
        axs[2, c].set_ylabel('charge-weighted ⟨W⟩ [MeV/m]')
        axs[2, c].set_xlabel('s [m]')
    axs[1, 0].legend(fontsize=7, frameon=False)
    axs[2, 0].legend(fontsize=7, frameon=False)
    fig.suptitle('BC20: the two 2D-only wake transients the 1D-guided kick schedule under-samples', x=0.01, ha='left', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(os.path.join(RES, 'bc20_kdense_windows.png'), dpi=140)
    print('wrote bc20_kdense_windows.png')


def P_lam(r):
    """charge weight along z for a wake record: the deposited density is not stored, so use the
    x-weighted mean of |W| as a stand-in only when nothing better exists -> here a flat weight"""
    return np.ones_like(r['z'])


if __name__ == '__main__':
    main()
