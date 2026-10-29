"""Exact decomposition of the final slice energy variance from per-particle kicks (kicks_<case>.npz).
delta_f = delta_0 + K (tracking does not change pz), so within a final-z slice
    var(delta_f) = var(delta_0) + var(K) + 2 cov(delta_0, K).
K = K_z + K_x: K_z = kick along the particle's own z path from the wake taken at the beam-centre x
(mixing + chirp within the slice); K_x = the wake's own x-dependence.
K_z is split again: the part linear in final z inside the slice (chirp within the slice) and the rest
(mixing). The rest is regressed on (x0, x0', delta0) to attribute it to the R51, R52, R56 channels."""
import sys, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
NS, ZR = 40, 3.0
EXCL = '--in-mesh' in sys.argv   # drop particles that ever left the wake mesh (they get zero kick: fill_value=0)
SUF = '_inmesh' if EXCL else ''
out = open(f'kick_decomposition{SUF}.txt', 'w')
fig, axes = plt.subplots(1, 2, figsize=(15, 4.8), layout='constrained')
for ax, case, lab in zip(axes, ['cm', 'cm_sx5_sE50'], ['baseline σx=50 µm, σE=500 keV', 'null test σx=5 µm, σE=50 keV']):
    d = np.load(f'kicks_{case}.npz'); E0 = float(d['E0']) / 1e3   # keV per unit delta
    K, Kz = d['K'], d['Kz']; Kx = K - Kz; d0, df, zf = d['d0'], d['df'], d['zf']
    assert np.allclose(df, d0 + K, atol=1e-12), 'pz changed by something other than CSR kicks'
    z = zf - zf.mean(); sz = z.std(); edges = np.linspace(-ZR*sz, ZR*sz, NS+1); idx = np.digitize(z, edges) - 1
    rows = []
    for i in range(NS):
        m = (idx == i) & (~d['out_mesh'] if EXCL else True)
        if m.sum() < 1000: continue
        zi = z[m]; A = np.vstack([np.ones(m.sum()), zi]).T
        Kz_chirp = A @ np.linalg.lstsq(A, Kz[m], rcond=None)[0]; Kz_mix = Kz[m] - Kz_chirp
        B = np.vstack([np.ones(m.sum()), zi, d['x0'][m], d['xp0'][m], d0[m]]).T
        c = np.linalg.lstsq(B, Kz_mix, rcond=None)[0]
        part = lambda col: np.var(c[col] * B[:, col])
        cv = lambda a, b: 2 * np.mean((a - a.mean()) * (b - b.mean()))
        rows.append(dict(z=0.5*(edges[i]+edges[i+1])*1e6, n=m.sum(), tot=np.var(df[m]), d0=np.var(d0[m]),
                         cov_d0K=cv(d0[m], K[m]), chirp=np.var(Kz_chirp), mix=np.var(Kz_mix),
                         Kx=np.var(Kx[m]), cross=cv(Kz[m], Kx[m]),
                         r51=part(2), r52=part(3), r56=part(4)))
    R = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    core = np.abs(R['z']) < 2 * sz * 1e6; w = R['n'][core]
    avg = lambda k: np.average(R[k][core], weights=w) * E0**2
    added = avg('tot') - avg('d0')
    out.write(f"\n{lab}  (core |z|<2 sigma_z, slice variances in keV^2, CSR-added total = {added:.0f} keV^2 -> {np.sqrt(max(added,0)):.0f} keV)\n")
    for k, name in [('d0', 'initial spread of the slice members'), ('cov_d0K', '2 cov(delta0, K)  [R56 compression term]'),
                    ('mix', 'var K_z, mixing part'), ('chirp', 'var K_z, chirp within slice'), ('Kx', 'var K_x  [wake x-dependence]'),
                    ('cross', '2 cov(K_z, K_x)')]:
        out.write(f"  {name:45s} {avg(k):12.0f}  ({100*avg(k)/added:6.1f}% of added)\n" if k != 'd0' else f"  {name:45s} {avg(k):12.0f}\n")
    out.write(f"  mixing part attributed by regression: R51 (x0) {avg('r51'):.0f}, R52 (x0') {avg('r52'):.0f}, R56 (delta0) {avg('r56'):.0f} keV^2\n")
    out.write(f"  check: d0 + cov + mix + chirp + Kx + cross = {avg('d0')+avg('cov_d0K')+avg('mix')+avg('chirp')+avg('Kx')+avg('cross'):.0f} vs total {avg('tot'):.0f}\n")
    out.write(f"  particles ever outside the wake mesh: {100*d['out_mesh'].mean():.2f}%; var K_x of those vs the rest: "
              f"{np.var(Kx[d['out_mesh']])*E0**2:.0f} vs {np.var(Kx[~d['out_mesh']])*E0**2:.0f} keV^2\n")
    for k, c_, name in [('cov_d0K', 'C0', '2cov(δ0,K): R56 compression'), ('mix', 'C3', 'var K_z mixing'), ('chirp', 'C2', 'var K_z chirp in slice'),
                        ('Kx', 'C4', 'var K_x: wake x-dependence'), ('cross', '0.5', '2cov(K_z,K_x)')]:
        ax.plot(R['z'], R[k] * E0**2 / 1e3, 'o-', ms=3, c=c_, label=name)
    ax.plot(R['z'], (R['tot'] - R['d0']) * E0**2 / 1e3, 'k--', lw=2, label='total CSR-added variance')
    ax.axhline(0, c='k', lw=0.5); ax.set(xlabel='final z − ⟨z⟩ [µm]', ylabel='slice variance [10³ keV²]', title=lab + (' (in-mesh particles only)' if EXCL else '')); ax.legend(fontsize=8)
plt.savefig(f'kick_decomposition{SUF}.png', dpi=110); out.close()
print(open(f'kick_decomposition{SUF}.txt').read())
