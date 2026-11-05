"""Along each s' column, where is the beam in lab x'? u_ret(x') = (x' - p(z_ret) - xi_bar)/sigma_xi at the retarded
frame, with z_ret depending on x' through |r - r'|. Reports du/dx' (in units of 1/sigma_xi) at the band centre, the
x'-extent of the ribbon |u| < xlim, and how much of that extent the default band (half-width xlim*margin*sigma_xi) covers;
plus the per-column integral at margin 2 vs 8, to show which columns lose charge."""
import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from d3_wake_diagnostic import build, wake_mesh
s_stop, row, col = float(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
sim = build(s_stop); t = sim.beam.position; ip = sim.integration_params; tr = sim.DF_tracker
xm, zm = wake_mesh(sim); x, z = xm.reshape(12, 40)[row, col], zm.reshape(12, 40)[row, col]; s = t + z
(s1, s3), (_, s4) = sim._layout_bounds(s, x)
sp = sim._far_region_nodes(s, x, t, s1, s3)
X0s, Y0s, nsx, nsy = (v[0] for v in sim._lattice_at(np.array([s])))
X0p, Y0p, npx, npy = sim._lattice_at(sp)
qx = X0s - X0p + x * nsx; qy = Y0s - Y0p + x * nsy
def u_of(xp):                                    # xp: (m, ncol)
    dx = qx[None] - xp * npx[None]; dy = qy[None] - xp * npy[None]; r = np.sqrt(dx * dx + dy * dy)
    tret = t - r; zret = sp[None] - tret
    out = np.empty_like(xp)
    for j in range(xp.shape[1]):
        poly, xb, zb, sxi, sz = sim._comoving_frame_at(tret[:, j])
        out[:, j] = (xp[:, j] - sim._eval_poly_rows(poly, zret[:, j]) - xb) / sxi
    return out, sxi
bands = sim._disjoint_bands(sim._retarded_xi_bands(s, x, t, sp))
lo, hi = bands[0]; c = 0.5 * (lo + hi); half = 0.5 * (hi - lo)
alive = half > 0
_, sxi = u_of(c[None, :])
sxi_ret = np.array([sim._comoving_frame_at(np.array([t - np.hypot(qx[j] - c[j] * npx[j], qy[j] - c[j] * npy[j])]))[3][0] for j in range(sp.size)])
h = 1e-3 * sxi_ret
up, _ = u_of((c + h)[None, :]); um, _ = u_of((c - h)[None, :])
J = (up - um)[0] / 2 / 1e-3                       # du per sigma_xi of x'
ext = tr.xlim / np.maximum(np.abs(J), 1e-6)      # ribbon half-extent in x', units of sigma_xi
cov = np.minimum(1.0, (half / sxi_ret) / ext)    # fraction of the ribbon the band covers
# per-column contribution, margin 2 vs 8
def colint(m):
    ip.xi_band_margin = m
    bs = sim._disjoint_bands(sim._retarded_xi_bands(s, x, t, sp)); frac = np.linspace(0, 1, 801); ci = np.zeros(sp.size)
    for l, hh in bs:
        w = hh - l; a = w > 0
        if not a.any(): continue
        xp = l[None] + frac[:, None] * w[None]
        gz, _ = sim.get_CSR_integrand(s=s, t=t, x=x, xp=xp, sp=np.broadcast_to(sp[None], xp.shape).copy(), ignore_vx=False, taper=None)
        ci += np.where(a, np.trapz(gz, axis=0) * w / 800, 0.0)
    ip.xi_band_margin = 2.0
    return -sim.CSR_scaling * ci
c2, c8 = colint(2.0), colint(8.0)
d = (c8 - c2) * np.gradient(sp)
print(f'observer s = {s:.4f} ({sim.lattice.current_element}), x = {x*1e6:+.0f} um; far region s\' = {sp.min():.3f}..{sp.max():.3f} m, {sp.size} columns; '
      f'grid half-width xlim = {tr.xlim} sigma_xi; default band half-width = {tr.xlim*2:.0f} sigma_xi (in x\')')
print(f"{'s-prime':>8} {'element':>7} | {'du/dx_prime':>11} {'ribbon half-extent in x_prime':>30} {'band covers':>11} | {'missing contribution':>20}")
from bisect import bisect
E = [0.0, 0.1, 0.6002, 5.606, 6.1062, 7.1062, 7.6064, 12.6122, 13.1124]; N = ['D0', 'B1', 'D1', 'B2', 'D2', 'B3', 'D3', 'B4', 'Df']
order = np.argsort(-np.abs(d))
for j in sorted(order[:3].tolist() + list(np.linspace(0, sp.size - 1, 7).astype(int))):
    print(f'{sp[j]:8.3f} {N[min(bisect(E, sp[j]) - 1, 8)]:>7} | {J[j]:+11.3f} {ext[j]:27.1f} sxi {cov[j]*100:10.0f}% | {d[j]*1e3:+17.3f} keV/m')
tot = d.sum()
print(f'missing in total (margin 8 - margin 2): {tot:+.4f} MeV/m; from columns in B3: {d[(sp>=7.1062)&(sp<7.6064)].sum():+.4f}, '
      f'in D3: {d[sp>=7.6064].sum():+.4f}; columns where the band covers < 90% of the ribbon: {(cov[alive] < 0.9).mean()*100:.0f}% of live columns')
