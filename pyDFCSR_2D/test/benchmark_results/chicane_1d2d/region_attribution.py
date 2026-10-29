"""Attribute the 2D chicane's CSR-added exit slice spread to lattice regions, and to the single s = 8.317 m
D3 kick that progress_comoving_interp.md §11v flags as defective (roughness 58x its neighbours).
Kr[r] = per-particle CSR energy kick accumulated in region r (the 8.317 kick is its own region and is
excluded from D3). Within each final-z slice (80 over +-3 sigma_z, |z| < 2 sigma_z, in-mesh particles only)
each quantity has a quadratic-in-z fit removed. Since delta_f = delta_0 + sum_r Kr,
    var(delta_f) - var(delta_0) = sum_r [ var Kr + 2 cov(delta_0, Kr) ] + sum_{r != q} cov(Kr, Kq)
so 'own' is var Kr + 2cov(delta_0, Kr), and the pairwise cross terms are listed separately.
Counterfactual: the exit CSR-added spread with the 8.317 kick removed (delta_f - K_8317)."""
import numpy as np
A, B = np.load('2d_chic_on_regions.npz'), np.load('2d_chic_off.npz'); E0 = float(A['E0']) / 1e3   # keV
names = ['B1', 'D1', 'B2', 'D2', 'B3', 'D3*', 'B4', 'Df', 'k8317']
Kr = A['Kr'].astype(float); ok = ~A['out_mesh']; z = A['zf'] - A['zf'].mean(); sz = z.std()
assert np.allclose(Kr.sum(0), A['K'], atol=1e-4 * np.abs(A['K']).max()), 'regions do not add up to K'   # Kr is stored float32
ed = np.linspace(-3 * sz, 3 * sz, 81); idx = np.digitize(z, ed) - 1
R = len(names); own = np.zeros(R); C = np.zeros((R, R)); tot = d0v = 0.0; W = 0; noK8 = 0.0; Kx8 = 0.0
for i in range(80):
    if abs(0.5 * (ed[i] + ed[i + 1])) > 2 * sz: continue
    m = (idx == i) & ok
    if m.sum() < 500: continue
    rs = lambda y: y[m] - np.polyval(np.polyfit(z[m], y[m], 2), z[m])
    rk = np.array([rs(Kr[r]) for r in range(R)]); r0 = rs(A['d0']); rf = rs(A['df']); w = m.sum(); W += w
    tot += w * rf.var(); d0v += w * r0.var(); noK8 += w * rs(A['df'] - Kr[-1]).var()
    C += w * (rk @ rk.T) / m.sum(); own += w * np.array([2 * np.mean(r0 * rk[r]) for r in range(R)])
tot, d0v, noK8 = tot / W * E0**2, d0v / W * E0**2, noK8 / W * E0**2; C *= E0**2 / W; own = own * E0**2 / W + np.diag(C)
added = tot - d0v
L = [__doc__, f'in-mesh particles; CSR-added slice variance {added:.0f} keV^2 -> {np.sqrt(max(added, 0)):.0f} keV',
     f'  identity check: sum(own) + sum(off-diagonal) = {own.sum() + (C.sum() - np.trace(C)):.0f}',
     '', '  region   own = var K_r + 2cov(d0,K_r)   share     var K_r alone']
for r in range(R): L.append(f'  {names[r]:7s} {own[r]:14.0f}            {100*own[r]/added:7.1f}%   {C[r, r]:12.0f}')
L.append(''); L.append('  largest cross terms 2cov(K_r, K_q):')
pairs = sorted([(abs(C[a, b]), a, b) for a in range(R) for b in range(a + 1, R)], reverse=True)[:6]
for _, a, b in pairs: L.append(f'    {names[a]:6s} x {names[b]:6s}: {2*C[a, b]:12.0f}  ({100*2*C[a, b]/added:6.1f}%)')
na = noK8 - d0v
L += ['', f'COUNTERFACTUAL without the s = 8.317 kick: CSR-added {na:.0f} keV^2 -> {np.sign(na)*np.sqrt(abs(na)):.0f} keV (with it: {np.sqrt(max(added,0)):.0f} keV)',
      '  (first-order: removes the kick but not its effect on later tracking/wakes)']
open('region_attribution.txt', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L[1:]))

# ---- D3 (excluding the 8.317 kick): x-dependence vs path-in-z ----
# Kzr[r] = kick from the wake at the beam-centre x along the particle's own z path, accumulated in region r.
iD3 = names.index('D3*'); Kd, Kzd = Kr[iD3], A['Kzr'][iD3].astype(float); Kxd = Kd - Kzd
acc = np.zeros(4); W = 0
for i in range(80):
    if abs(0.5 * (ed[i] + ed[i + 1])) > 2 * sz: continue
    m = (idx == i) & ok
    if m.sum() < 500: continue
    rs = lambda y: y[m] - np.polyval(np.polyfit(z[m], y[m], 2), z[m])
    a, b = rs(Kzd), rs(Kxd); w = m.sum(); W += w
    acc += w * np.array([a.var(), b.var(), 2 * np.mean(a * b), np.corrcoef(rs(A['x0'] * 0 + A['d0']), b)[0, 1]])
acc[:3] *= E0**2 / W; acc[3] /= W
# where the beam sits in D3: size and tilt
s = A['s']; j = np.argmin(abs(s - 10.0))
extra = [f'', f'D3 (without 8.317) split, slice variance after quadratic detrend [keV^2]:',
         f'  var K_z (wake at centre x, own z path)  {acc[0]:10.0f}',
         f'  var K_x (wake x-dependence)            {acc[1]:10.0f}',
         f'  2cov(K_z, K_x)                         {acc[2]:10.0f}',
         f'  mean in-slice corr(delta0, K_x)        {acc[3]:10.3f}',
         f'beam in D3 (s = {s[j]:.2f} m, CSR off): sigma_x {B["projx"][j]*1e3:.2f} mm, sigma_z {B["projz"][j]*1e6:.1f} um;',
         f'  B3 radius R = {0.5002/0.0483:.2f} m -> (R sigma_z^2)^(1/3) = {(0.5002/0.0483*B["projz"][j]**2)**(1/3)*1e3:.2f} mm, '
         f'sigma_x / (R sigma_z^2)^(1/3) = {B["projx"][j]/(0.5002/0.0483*B["projz"][j]**2)**(1/3):.2f}']
open('region_attribution.txt', 'a').write('\n'.join(extra) + '\n'); print('\n'.join(extra))

# ---- counterfactuals: remove whole regions (first order: the removed kicks' effect on later tracking/wakes is kept) ----
def cf(drop):
    keep = A['df'] - sum(Kr[names.index(n)] for n in drop); t = d = 0.0; W = 0
    for i in range(80):
        if abs(0.5 * (ed[i] + ed[i + 1])) > 2 * sz: continue
        m = (idx == i) & ok
        if m.sum() < 500: continue
        rs = lambda y: y[m] - np.polyval(np.polyfit(z[m], y[m], 2), z[m]); w = m.sum(); W += w
        t += w * rs(keep).var(); d += w * rs(A['d0']).var()
    v = (t - d) / W * E0**2; return np.sign(v) * np.sqrt(abs(v))
out = ['', 'COUNTERFACTUAL exit CSR-added slice sigma_E with regions removed [keV]:']
for drop in [[], ['k8317'], ['D3*', 'k8317'], ['D3*', 'k8317', 'B4', 'Df']]:
    out.append(f"  without {'+'.join(drop) or 'nothing':22s}: {cf(drop):7.1f}")
open('region_attribution.txt', 'a').write('\n'.join(out) + '\n'); print('\n'.join(out))
