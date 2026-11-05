"""Is the +-3 sigma_z spike in the input fields? At a frozen B2 state: (1) the particles' line density vs a
Gaussian, (2) the deposited density's line integral and its w-derivative, (3) the z-cut with the v_x terms off."""
import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from d3_wake_diagnostic import build, wake_mesh, evaluate
from scipy.ndimage import median_filter
s_stop = float(sys.argv[1])
sim = build(s_stop); b = sim.beam; tr = sim.DF_tracker
z = b.z; w = (z - z.mean()) / z.std()
h, e = np.histogram(w, 160, (-4, 4)); c = 0.5 * (e[1:] + e[:-1])
g = len(w) * np.diff(e)[0] * np.exp(-c ** 2 / 2) / np.sqrt(2 * np.pi)
print(f's = {b.position:.4f}: particles per 0.05-sigma bin / Gaussian at |w| = 2.5..3.5:')
for lo in (-3.5, 2.5):
    m = (c >= lo) & (c <= lo + 1.0)
    print('   ' + ' '.join(f'{c_:+.2f}:{r:.2f}' for c_, r in zip(c[m][::2], (h / g)[m][::2])))
print(f'   fraction beyond 3 sigma: {(np.abs(w) > 3).mean():.5f} (Gaussian 0.00270)')
lam = tr.density.sum(0) * tr.delta_u; dl = tr.density_w.sum(0) * tr.delta_u
wg = tr.w_grids; m = np.abs(np.abs(wg) - 3) < 0.6
print('deposited line density / Gaussian, and d lam/dw - Gaussian derivative, near |w| = 3:')
G = np.exp(-wg ** 2 / 2) / np.sqrt(2 * np.pi)
print('   ' + ' '.join(f'{x:+.2f}:{r:.2f}/{d:+.4f}' for x, r, d in zip(wg[m][::3], (lam / G)[m][::3], (dl + wg * G)[m][::3])))
xm, zm = wake_mesh(sim); X, Z = xm.reshape(12, 40), zm.reshape(12, 40)
zc = np.linspace(-4, 4, 81); zz = b._mean_z + zc * b._sigma_z; xx = np.interp(zz, Z[6], X[6])
orig = sim.get_CSR_integrand
def spikes(v):
    d = v - median_filter(v, 9, mode='nearest'); k = np.argsort(-np.abs(d))[:2]; return ', '.join(f'{zc[i]:+.2f}: {d[i]:+.4f}' for i in sorted(k))
for name, flag in (('v_x on', False), ('v_x off', True)):
    sim.get_CSR_integrand = (lambda *a, _f=flag, **k: orig(*a, **{**k, 'ignore_vx': _f}))
    v = evaluate(sim, xx, zz); print(f'z-cut, {name:8s}: max |W| {np.abs(v).max():.4f}; spikes {spikes(v)}', flush=True)
sim.get_CSR_integrand = orig
