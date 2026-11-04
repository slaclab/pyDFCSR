"""
Is the spurious z-trend of the deposited d v_x/dx a smoothing artefact? Freeze the chicane beam at s (D3),
redeposit with the SAME DF_tracker_comoving settings, but with px fields whose exact d px/dx|z is known:

  true         px of the particles                         (truth: per-slice fit)
  linear_only  b0 * x                                      (truth: b0 everywhere)
  lin+g(z)     b0 * x + g(z), g = cubic fit of <px>(z)     (truth: b0 everywhere)
  control      deposit px - [b_hat x + g(z)], add b_hat analytically   (truth: per-slice fit)

d v_x/dx|z = vx_u / sigma_xi on the newest snapshot, read at the particles by bilinear lookup in (u, w).
Usage: python d3_vx_synthetic.py <s> [smoothing_sigma]
"""
import os, sys, copy
import numpy as np
from scipy.interpolate import RegularGridInterpolator
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from d3_wake_diagnostic import build, OUT

s_stop = float(sys.argv[1]); ov = {}
if len(sys.argv) > 2:
    ov = {'particle_deposition': {'smoothing_sigma': float(sys.argv[2])}}
sim = build(s_stop, ov)
b = sim.beam; tr0 = sim.DF_tracker
x, z, px = b.particle.x.copy(), b.particle.z.copy(), b.particle.px.copy()
zc = z - z.mean(); sz = zc.std()
NS = 80; ed = np.linspace(-3 * sz, 3 * sz, NS + 1); idx = np.digitize(zc, ed) - 1

def deposit(pxf):
    tr = copy.copy(tr0)
    tr.get_DF(x, z, pxf, b.position)
    f = RegularGridInterpolator((tr.u_grids, tr.w_grids), tr.vx_u / tr.sigma_xi, bounds_error=False, fill_value=0.0)
    xi = x - np.polyval(tr.poly_coeffs, z)
    return f(np.c_[(xi - tr.xi_bar) / tr.sigma_xi, (z - tr.z_bar) / tr.sigma_z]), tr

def per_slice(vxx, pxf):
    out = []
    for i in range(NS):
        m = idx == i
        if m.sum() < 2000:
            continue
        out.append((0.5 * (ed[i] + ed[i + 1]) / sz, np.polyfit(x[m], pxf[m], 1)[0], vxx[m].mean(), m.sum()))
    return np.array(out)

b0 = -0.28
g = np.polyval(np.polyfit(zc / sz, px, 3), zc / sz)            # smooth longitudinal part of <px>
A = np.c_[x - x.mean(), np.ones_like(x)]
bhat_cols = np.linalg.lstsq(np.c_[A, zc / sz, (zc / sz) ** 2, (zc / sz) ** 3], px, rcond=None)[0]
model = np.c_[A, zc / sz, (zc / sz) ** 2, (zc / sz) ** 3] @ bhat_cols; bhat = bhat_cols[0]

cases = {}
cases['true'] = (px, *deposit(px))
cases['linear_only'] = (b0 * x, *deposit(b0 * x))
cases['lin+g(z)'] = (b0 * x + g, *deposit(b0 * x + g))
v, tr = deposit(px - model); cases['control'] = (px, v + bhat, tr)

L = [f's = {b.position:.4f}, smoothing {tr0.smoothing_sigma}, sigma_xi {tr0.sigma_xi*1e6:.1f} um, sigma_z {sz*1e6:.2f} um, '
     f'g(z) spans {g.min():+.2e}..{g.max():+.2e} rad, b_hat {bhat:+.4f} /m',
     'per-slice d v_x/dx at fixed z [1/m]: truth | code;  trend = slope along the bunch per sigma_z (core |z|<2)']
for name, (pxf, vxx, _) in cases.items():
    r = per_slice(vxx, pxf); core = np.abs(r[:, 0]) < 2
    tt = np.polyfit(r[core, 0], r[core, 1], 1)[0]; tc = np.polyfit(r[core, 0], r[core, 2], 1)[0]
    pick = [np.argmin(abs(r[:, 0] - zz)) for zz in (-2.4, -1.2, 0, 1.2, 1.8)]
    L.append(f'{name:12s} ' + '  '.join(f'z{r[k,0]:+.1f}: {r[k,1]:+.3f}|{r[k,2]:+.3f}' for k in pick) +
             f'   trend truth {tt:+.4f} code {tc:+.4f}   wtd-mean err {np.average(r[core,2]-r[core,1], weights=r[core,3]):+.4f}')
tag = f'_sm{tr0.smoothing_sigma:g}'
open(os.path.join(OUT, f'vx_synthetic_{s_stop:.4f}{tag}.txt'), 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))
