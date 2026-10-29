"""2D chicane: which part of the CSR kick makes the slice energy spread at the exit?
Per particle (tilt_scan_run.py): K = total CSR energy kick, K_z = the kick from the wake taken at the
beam-centre x along the particle's own z path, K_x = K - K_z (the wake's x-dependence). Within each final-z
slice, each quantity has a quadratic fit in z removed (the residual chirp), then its variance is taken.
Particles that ever left the wake mesh (zero kick, fill_value=0) are excluded and counted."""
import numpy as np
A, B = np.load('2d_chic_on.npz'), np.load('2d_chic_off.npz'); E0 = float(A['E0']) / 1e3   # keV per unit delta
K, Kz = A['K'], A['Kz']; Kx = K - Kz; ok = ~A['out_mesh']
z = A['zf'] - A['zf'].mean(); sz = z.std(); NS = 80; ed = np.linspace(-3 * sz, 3 * sz, NS + 1); idx = np.digitize(z, ed) - 1
def resid(y, m):
    return y[m] - np.polyval(np.polyfit(z[m], y[m], 2), z[m])
acc = {k: 0.0 for k in ['tot', 'd0', 'K', 'Kz', 'Kx', 'cross', 'covd0K']}; W = 0
for i in range(NS):
    if abs(0.5 * (ed[i] + ed[i + 1])) > 2 * sz: continue
    m = (idx == i) & ok
    if m.sum() < 500: continue
    rf, r0, rK, rz, rx = resid(A['df'], m), resid(A['d0'], m), resid(K, m), resid(Kz, m), resid(Kx, m)
    w = m.sum(); W += w
    acc['tot'] += w * rf.var(); acc['d0'] += w * r0.var(); acc['K'] += w * rK.var(); acc['Kz'] += w * rz.var(); acc['Kx'] += w * rx.var()
    acc['cross'] += w * 2 * np.mean(rz * rx); acc['covd0K'] += w * 2 * np.mean(r0 * rK)
a = {k: v / W * E0**2 for k, v in acc.items()}; added = a['tot'] - a['d0']
lines = [__doc__, f"particles ever outside the wake mesh: {100*(~ok).mean():.2f}% (excluded)",
         f"identity check: d0 + 2cov(d0,K) + var K = {a['d0']+a['covd0K']+a['K']:.0f} vs total {a['tot']:.0f} keV^2",
         f"CSR-added slice variance (in-mesh, 80 slices, |z|<2sz): {added:.0f} keV^2 -> {np.sqrt(max(added,0)):.0f} keV",
         f"  2 cov(delta0, K)             {a['covd0K']:10.0f}  ({100*a['covd0K']/added:6.1f}%)",
         f"  var K_z  (no x-dependence)   {a['Kz']:10.0f}  ({100*a['Kz']/added:6.1f}%)",
         f"  var K_x  (x-dependence)      {a['Kx']:10.0f}  ({100*a['Kx']/added:6.1f}%)",
         f"  2 cov(K_z, K_x)              {a['cross']:10.0f}  ({100*a['cross']/added:6.1f}%)",
         f"  -> wake x-dependence, net     {a['Kx']+a['cross']:10.0f}  ({100*(a['Kx']+a['cross'])/added:6.1f}%)"]
open('kick_decomposition_chicane.txt', 'w').write('\n'.join(lines) + '\n'); print('\n'.join(lines[1:]))
