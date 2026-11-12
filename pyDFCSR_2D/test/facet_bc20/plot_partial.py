"""
Quick look at a 2D BC20 run while it is still running: copies its wakes.h5 (the writer keeps it
open) to <run>partial-<suffix>, and plots the kicks computed so far.

    HDF5_USE_FILE_LOCKING=FALSE python pyDFCSR_2D/test/facet_bc20/plot_partial.py <run>

Writes bc20_wake_maps_long_<run>_partial.png, _trans_, bc20_wake_zproj_<run>_partial.png (this
run, 2D 20k adaptive, the old rms grid and 1D 1M at the same s) and prints the roughness table
(rms deviation of the projected wake from a 20 um smoothing, relative to the wake, median by
section) against the same runs.
"""
import glob
import os
import shutil
import sys

import numpy as np
from scipy.ndimage import gaussian_filter1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import plot_bc20_wakes as P

CMP = (('2D 20k, adaptive (dfcsr_2d_qa_kp)', 'dfcsr_2d_qa_kp', '#eda100', 1.0),
       ('2D 20k, old rms grid (dfcsr_2d)', 'dfcsr_2d', '#1baf7a', 1.2))
SECTIONS = ((0, 12.3, 'B1LE..B2LE'), (12.3, 24.0, 'quads/sext, D5'), (24.0, 36.65, 'B3 pair..Q2ER'),
            (36.65, 38.67, 'B2RE'), (38.67, 48.02, 'drift+wiggler'), (48.02, 50, 'B1RE'))


def rough(r):
    z = r['z']
    w = P.proj(r, r['W'])
    g = np.arange(z[0], z[-1], 2e-6)
    if g.size < 20:
        return np.nan
    wi = np.interp(g, z, w)
    sm = gaussian_filter1d(wi, 10.0, mode='nearest')
    core = slice(int(0.1 * g.size), int(0.9 * g.size))
    return np.sqrt(np.mean((wi - sm)[core] ** 2)) / np.sqrt(np.mean(sm[core] ** 2))


def main(run):
    src = sorted(glob.glob(os.path.join(P.RUNS, f'{run}-2*-wakes.h5')), key=os.path.getmtime)[-1]
    part = f'{run}partial'
    for old in glob.glob(os.path.join(P.RUNS, f'{part}-2*-wakes.h5')):
        os.remove(old)
    shutil.copy(src, os.path.join(P.RUNS, f'{part}-' + os.path.basename(src)[len(run) + 1:]))
    R = P.load2d(part)
    P.RUN = part
    P.LABEL[part] = f'{run} (in progress)'
    smax = R[-1]['s']
    picks = [p for p in P.PICK if p[0] <= smax + 1e-6]
    P.maps(R, 'W', f'Longitudinal wake dE/ds, {len(R)} kicks so far', f'bc20_wake_maps_long_{run}_partial.png',
           picks=picks)
    P.maps(R, 'F', f'Transverse force, {len(R)} kicks so far', f'bc20_wake_maps_trans_{run}_partial.png',
           picks=picks)
    others = [(lab, P.load2d(f), c, lw) for lab, f, c, lw in CMP]
    D1 = P.load1d('dfcsr_1d_1M')
    nc = 5
    nr = int(np.ceil(len(picks) / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(3.0 * nc, 2.3 * nr + 0.8), squeeze=False)
    for ax, (s, lab) in zip(axs.flat, picks):
        r = R[P.nearest(R, s)]
        zm = P.zmid(r)
        for _, RR, c, lw in others:
            q = RR[P.nearest(RR, r['s'])]
            ax.plot((q['z'] - zm) * 1e6, P.proj(q, q['W']), color=c, lw=lw)
        d = D1.get(round(r['s'], 6))
        if d is not None:
            ax.plot((d[0] - zm) * 1e6, d[1], color='#eb6834', ls=(0, (3, 1.5)), lw=1.2)
        ax.plot((r['z'] - zm) * 1e6, P.proj(r, r['W']), color='#4a3aa7', lw=1.6)
        ax.axhline(0, color='#e1e0d9', lw=0.8)
        ax.set_title(f'{lab}, s = {r["s"]:.2f} m', loc='left', fontsize=8)
        ax.set_xlim((r['z'][0] - zm) * 1e6, (r['z'][-1] - zm) * 1e6)
        ax.tick_params(labelsize=6)
    for ax in axs.flat[len(picks):]:
        ax.axis('off')
    h = [plt.Line2D([], [], color='#4a3aa7', lw=1.6)] + [plt.Line2D([], [], color=c, lw=lw) for _, _, c, lw in others] \
        + [plt.Line2D([], [], color='#eb6834', lw=1.2, ls=(0, (3, 1.5)))]
    fig.legend(h, [run] + [lab for lab, _, _, _ in others] + ['1D 1M, same s'], loc='upper right', ncol=4,
               frameon=False, fontsize=8)
    fig.suptitle(f'BC20 longitudinal wake projected on z, the {len(R)} kicks computed so far by {run}',
                 x=0.01, ha='left', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(os.path.join(P.RES, f'bc20_wake_zproj_{run}_partial.png'), dpi=130)
    plt.close(fig)
    print(f'{run}: {len(R)} kicks, last s = {smax:.2f} m')
    print('roughness (rms deviation from a 20 um smoothing / rms wake), median over kicks')
    for lo, hi, lab in SECTIONS:
        if lo >= smax:
            break
        row = {run: np.median([rough(r) for r in R if lo < r['s'] <= hi])}
        for lab2, RR, _, _ in others:
            row[lab2.split(' (')[0]] = np.median([rough(r) for r in RR if lo < r['s'] <= min(hi, smax)])
        print(f'  {lab:16s} ' + '  '.join(f'{k}: {v:.3f}' for k, v in row.items()))


if __name__ == '__main__':
    main(sys.argv[1])
