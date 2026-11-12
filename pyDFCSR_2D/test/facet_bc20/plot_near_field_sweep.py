"""
The fine observer-z sweep of near_field_check.py, against the beam's line density.

    conda run -n pydfcsr python pyDFCSR_2D/test/facet_bc20/plot_near_field_sweep.py [run]

Top: the wake along the worst mesh row at the first kick (s = 0.42 m), far / near / total at
0.5 um observer spacing, with the wake-mesh nodes and the stored mesh values on that row, and the
1D wake (dfcsr_1d_1M) at the same kick. Bottom: the line density of the input beams (20k and the
1M upsample) on 2 um bins; over the first 0.42 m of B1LE z changes by < 0.1 um, so the input
profile is the profile at the kick. Writes near_field_sweep_<run>.png.
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
RES = os.path.join(HERE, '..', 'benchmark_results', 'facet_bc20')
INP = os.path.join(HERE, '..', '..', 'example', 'input')


def main(run='dfcsr_2d_peak_1M'):
    d = np.load(os.path.join(HERE, f'near_field_check_{run}.npz'))
    z, sw, zr, W, r0 = d['sweep_z'], d['sweep'], d['z'], d['W'], int(d['rows'][0])
    f = h5py.File(sorted(glob.glob(os.path.join(RES, 'runs', 'dfcsr_1d_1M-2*-wakes1d.h5')))[-1], 'r')
    g = f[min(f, key=lambda k: abs(f[k].attrs['position'] - float(d['s'])))]['longitudinal_1d']
    z1, W1 = g['z_grid'][()], g['dE_dct'][()] / 1e6
    lo, hi, m_lo, m_hi = d['window']
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(9, 6.4), sharex=True, height_ratios=(1.4, 1))
    zu = z * 1e6
    ax.plot(zu, sw[:, 0, 0], color='#4a3aa7', lw=1.2, label="far s' region")
    ax.plot(zu, sw[:, 1, 0], color='#1baf7a', lw=1.2, label="near s' region (+ polar patch, ~1e-9)")
    ax.plot(zu, sw[:, :, 0].sum(1), color='#0b0b0b', lw=1.8, label=f'total, row ix = {r0}, 0.5 µm observer spacing')
    m = (zr >= z[0]) & (zr <= z[-1])
    ax.plot(zr[m] * 1e6, W[r0, m], 'o', color='#e34948', ms=5, label='wake-mesh nodes (what the run stores)')
    m1 = (z1 >= z[0]) & (z1 <= z[-1])
    ax.plot(z1[m1] * 1e6, W1[m1], 's--', color='#eb6834', ms=4, lw=1.0, label='1D 1M, same kick (18 µm bins)')
    ax.axhline(0, color='#e1e0d9', lw=0.8)
    ax.set_ylabel('dE/ds [MeV/m]')
    ax.legend(fontsize=7, frameon=False, loc='lower right')
    ax.set_title(f'{run}, first kick s = {float(d["s"]):.3f} m (B1LE): wake along the mesh row with the '
                 'worst "outlier", resolved in observer z', loc='left', fontsize=9)
    for fn, lab, c in (('facet_bc20_beam.txt', '20k input beam', '#898781'),
                       ('facet_bc20_beam_1000000.txt', '1M upsample', '#2a78d6')):
        zz = np.loadtxt(os.path.join(INP, fn), usecols=4)
        e = np.arange(z[0], z[-1] + 2e-6, 2e-6)
        h, _ = np.histogram(zz, bins=e)
        bx.step(0.5 * (e[1:] + e[:-1]) * 1e6, h / zz.size / 2e-6 / 1e3, where='mid', color=c, lw=1.2, label=lab)
    for a in (ax, bx):
        a.axvspan((lo - m_lo) * 1e6, lo * 1e6, color='#cde2fb', lw=0, zorder=0)
    bx.text(lo * 1e6 - 2, bx.get_ylim()[1] * 0.9, 'window taper', ha='right', fontsize=7, color='#52514e')
    bx.set_ylabel('line density [1/mm]')
    bx.set_xlabel('z [µm]  (head at +z)')
    bx.legend(fontsize=7, frameon=False, loc='upper left')
    fig.tight_layout()
    fig.savefig(os.path.join(RES, f'near_field_sweep_{run}.png'), dpi=140)
    print(f'wrote near_field_sweep_{run}.png')


if __name__ == '__main__':
    main(*sys.argv[1:])
