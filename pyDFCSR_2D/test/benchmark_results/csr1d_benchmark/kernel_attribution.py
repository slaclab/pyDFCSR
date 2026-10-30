"""
Attribute the per-kick wake difference between pyDFCSR's 1D mode and Emma's csr_calc to the kernel
quadrature. At sampled kicks of each case, both kernels are evaluated on EMMA'S OWN line density
(his smoothed fi, his bin grid Si, his derivative rule) and compared with a reference: pyDFCSR's
product-integrated kernel on an 8x finer grid through a cubic spline of that same density. Same
lambda for all three, so the difference is the quadrature alone. Geometry from the pyDFCSR run.
Writes kernel_attribution.txt.
"""
import glob, sys, numpy as np, h5py
from scipy.interpolate import CubicSpline
from pyDFCSR_2D.csr1d import wake_in_bend, wake_after_bend
def emma_dl(lam, ds):
    d1 = np.diff(lam) / ds; d = np.empty_like(lam); d[1:-1] = 0.5 * (d1[:-1] + d1[1:]); d[0], d[-1] = d1[0], d1[-1]; return d
def kernel(S, ds, lam, dl, g):
    return wake_in_bend(S, ds, lam, dl, g['R'], g['phi']) if g['x_R'] == 0 else wake_after_bend(S, ds, lam, dl, g['R'], g['phim'], g['x_R'])
out = [__doc__]
for name in sys.argv[1:]:
    m = h5py.File(f'1d_{name}.mat'); s0 = float(h5py.File(glob.glob(f'runs/{name}_on-*-wakes1d.h5')[0])[sorted(h5py.File(glob.glob(f'runs/{name}_on-*-wakes1d.h5')[0]).keys())[0]].attrs['position'])
    f = h5py.File(glob.glob(f'runs/{name}_on-*-wakes1d.h5')[0]); pos = {round(float(f[k].attrs['position']), 6): k for k in f}
    first = min(pos); es = m['s'][()].ravel() + first - m['s'][()].ravel()[0]
    N = float(m['N'][()].ravel()[0]); Wall = m['dE_dct'][()]; amp = np.abs(Wall).max(0)
    idx = [j for j in np.linspace(0, len(es) - 1, 40).astype(int) if amp[j] > 0.05 * amp.max()]
    e_em, e_py, gap = [], [], []
    for j in idx:
        k = pos.get(round(float(es[j]), 6)) or pos[min(pos, key=lambda p: abs(p - es[j]))]
        g = {a: float(f[k].attrs[a]) for a in ('R', 'phi', 'phim', 'x_R')}
        Si = m['Si'][:, j]; lam = m['fi'][:, j] * N; ds = Si[1] - Si[0]; W = Wall[:, j]
        cs = CubicSpline(Si, lam); Sf = np.linspace(Si[0], Si[-1], 8 * Si.size); ref = np.interp(Si, Sf, kernel(Sf, Sf[1] - Sf[0], cs(Sf), cs(Sf, 1), g))
        mine = kernel(Si, ds, lam, emma_dl(lam, ds), g); n = np.linalg.norm(ref)
        e_em.append(np.linalg.norm(W - ref) / n); e_py.append(np.linalg.norm(mine - ref) / n); gap.append(np.linalg.norm(mine - W) / n)
    e_em, e_py, gap = map(np.array, (e_em, e_py, gap))
    out.append(f'{name:14s} {len(idx):3d} kicks | Emma vs ref: median {np.median(e_em):.4f} max {e_em.max():.4f} | '
               f'pyDFCSR kernel vs ref: median {np.median(e_py):.4f} max {e_py.max():.4f} | pyDFCSR vs Emma: median {np.median(gap):.4f}')
open('kernel_attribution.txt', 'a').write('\n'.join(out) + '\n'); print('\n'.join(out[1:]))
