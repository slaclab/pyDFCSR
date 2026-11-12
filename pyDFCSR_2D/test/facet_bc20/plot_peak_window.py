"""
Current profile in the middle of each BC20 dipole, and the deposition window the peak frame would
use there (from a 1D run's line density at the nearest kick).

    conda run -n pydfcsr python pyDFCSR_2D/test/facet_bc20/plot_peak_window.py [run]   (default dfcsr_1d_1M)

The window follows DF_tracker_comoving._peak_window on the line density: quantile window
[z_q, z_1-q] + margins, the highest peak's centroid and FWHM (peakfind.peak_center_fwhm, on
particles sampled from lambda), full weight within centre +- peak_halfwidth * FWHM, taper over
peak_taper * FWHM, intersected with the quantile window. The rate limit between snapshots is not
applied (it binds only while the spike forms in B2RE). Writes current_peak_window_<run>.png.
"""
import glob
import os
import sys

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', '..'))
from pyDFCSR_2D.peakfind import peak_center_fwhm, window_weight  # noqa: E402
import yaml  # noqa: E402

RES = os.path.join(HERE, '..', 'benchmark_results', 'facet_bc20')
E, C0 = 1.602176634e-19, 299792458.0
Q, MARGIN, HW, TAP = 0.005, 0.1, 3.0, 2.0
SURF, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'


def window(z, lam, n_sample=400_000, seed=0):
    """the peak frame's (lo, hi, m_lo, m_hi) from a line density, via particles sampled from it"""
    rng = np.random.default_rng(seed)
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (lam[1:] + lam[:-1]) * np.diff(z))])
    cdf /= cdf[-1]
    zs = np.interp(rng.uniform(0, 1, n_sample), cdf, z)
    lo_q, hi_q = np.interp([Q, 1 - Q], cdf, z)
    m_q = MARGIN * (hi_q - lo_q)
    c, fw = peak_center_fwhm(zs)
    H, tap = HW * fw, TAP * fw
    lo, m_lo = (c - H, tap) if c - H > lo_q else (lo_q, m_q)
    hi, m_hi = (c + H, tap) if c + H < hi_q else (hi_q, m_q)
    return lo, hi, m_lo, m_hi, c, fw


def main(run='dfcsr_1d_1M'):
    lat = yaml.safe_load(open(os.path.join(HERE, '..', '..', 'example', 'input', 'facet_bc20_lattice.yaml')))
    s0, dips = 0.0, []
    for k, v in lat.items():
        if k == 'step_size':
            continue
        if v['type'] == 'dipole':
            dips.append((k, s0 + 0.5 * v['L']))
        s0 += v['L']
    f = h5py.File(sorted(glob.glob(os.path.join(RES, 'runs', f'{run}-2*-wakes1d.h5')))[-1], 'r')
    K = sorted(f, key=lambda k: f[k].attrs['position'])
    S = np.array([f[k].attrs['position'] for k in K])

    plt.rcParams.update({'font.size': 8, 'axes.edgecolor': INK2, 'axes.labelcolor': INK,
                         'xtick.color': INK2, 'ytick.color': INK2, 'text.color': INK})
    fig, axs = plt.subplots(2, len(dips), figsize=(3.0 * len(dips), 5.6), facecolor=SURF)
    lines = []
    for j, (name, sm) in enumerate(dips):
        g = f[K[int(np.argmin(np.abs(S - sm)))]]
        z, lam = g['longitudinal_1d/z_grid'][()], g['longitudinal_1d/lam'][()]
        lo, hi, m_lo, m_hi, c, fw = window(z, lam)
        T = window_weight(z, lo, hi, m_lo, m_hi)
        I = lam * E * C0 / 1e3
        q_in = np.trapz(lam * T, z) / np.trapz(lam, z)
        zr = (z - c) * 1e6
        for row, ax in enumerate(axs[:, j]):
            ax.axvspan((lo - m_lo - c) * 1e6, (lo - c) * 1e6, color='#cde2fb', lw=0)
            ax.axvspan((hi - c) * 1e6, (hi + m_hi - c) * 1e6, color='#cde2fb', lw=0)
            ax.axvspan((lo - c) * 1e6, (hi - c) * 1e6, color='#86b6ef', alpha=0.45, lw=0)
            ax.plot(zr, I, color=INK2, lw=1.0, label='current')
            ax.plot(zr, I * T, color='#e34948', lw=1.3, label='current × T (what is deposited)')
            ax.set_facecolor(SURF)
            ax.grid(color=GRID, lw=0.5)
            if row == 1:
                ax.set_xlim((lo - m_lo - c) * 1e6 - 2 * fw * 1e6 * (row == 1) * 0.2,
                            (hi + m_hi - c) * 1e6 + 2 * fw * 1e6 * 0.2)
                ax.set_xlabel('z − z_peak [µm]  (head at +z)')
        axs[0, j].set_title(f"{name} middle, s = {float(g.attrs['position']):.2f} m\n"
                            f"FWHM {fw*1e6:.1f} µm, I_pk {I.max():.1f} kA; window holds {100*q_in:.0f}%",
                            loc='left', fontsize=8)
        axs[1, j].set_title('zoom on the window', loc='left', fontsize=8)
        lines.append(f"{name:6s} s = {float(g.attrs['position']):6.2f}: FWHM {fw*1e6:7.2f} um, I_pk {I.max():6.2f} kA, "
                     f"full weight {(lo-c)*1e6:+8.1f} .. {(hi-c)*1e6:+7.1f} um, taper to "
                     f"{(lo-m_lo-c)*1e6:+8.1f} / {(hi+m_hi-c)*1e6:+7.1f} um, charge in window {100*q_in:5.1f}%")
    axs[0, 0].set_ylabel('current [kA]')
    axs[1, 0].set_ylabel('current [kA]')
    axs[0, 0].legend(fontsize=7, frameon=False, loc='upper left')
    fig.suptitle(f'BC20 current in the middle of each dipole ({run}) and the peak-frame window: full weight '
                 f'(blue), taper (light blue)', x=0.01, ha='left', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(RES, f'current_peak_window_{run}.png'), dpi=140, facecolor=SURF)
    plt.close(fig)
    print('\n'.join(lines))


if __name__ == '__main__':
    main(*sys.argv[1:])
