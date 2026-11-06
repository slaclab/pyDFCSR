"""Spike / outlier measures on the stored wake maps of chicane runs: per element, the |z|~3 sigma_z excess of the
x-avg wake (median-filter, % of peak), and the single worst point vs a smooth neighbourhood (median over 3x5)."""
import sys, os, glob, numpy as np, h5py
from scipy.ndimage import median_filter
O = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'benchmark_results', 'chicane_2d_vs_1d', 'runs')
def load(run):
    f = sorted(glob.glob(f'{O}/{run}-2*-wakes.h5'), key=os.path.getmtime)[-1]; h = h5py.File(f, 'r')
    return {round(float(h[k].attrs['position']), 3): (str(h[k].attrs['element']), h[k]['longitudinal/dE_dct'][()]) for k in h}
runs = sys.argv[1:]; R = {n: load(n) for n in runs}
u = np.linspace(-4, 4, 12); wx = np.exp(-u ** 2 / 2); wx /= wx.sum(); zc = np.linspace(-4, 4, 40); sp = np.abs(np.abs(zc) - 3) < 0.5
print('per element: |z|~3 spike (median / max %) | worst isolated point vs 3x5 median, % of map peak (max over kicks, at s)')
for e in ('D1', 'B2', 'D2', 'B3', 'D3', 'B4', 'Df'):
    cells = []
    for n in runs:
        a, o = [], []
        for s, (el, M) in R[n].items():
            if el != e: continue
            av = (wx[:, None] * M).sum(0); d = av - median_filter(av, 7, mode='nearest'); a.append(np.abs(d[sp]).max() / np.abs(av).max() * 100)
            iso = np.abs(M - median_filter(M, (3, 5), mode='nearest')).max() / np.abs(M).max() * 100; o.append((iso, s))
        w = max(o)
        cells.append(f'{np.median(a):5.1f}/{max(a):5.1f} | {w[0]:5.1f} @{w[1]:.2f}')
    print(f'  {e:3s} ' + '   '.join(f'{n}: {c}' for n, c in zip(runs, cells)))
