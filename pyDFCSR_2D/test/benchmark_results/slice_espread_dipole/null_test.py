"""Null test: sigma_x = 5 um AND sigma_E = 50 keV (comoving deposition, CSR on vs off).
Turning both off removes most of the z mixing. Also measures the wake's own x-dependence at fixed z."""
import numpy as np, h5py, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
C, O = np.load('beam_cm_sx5_sE50_on.npz'), np.load('beam_sx5_sE50_off.npz')
n = C['n'][-1] > 1000; zc = C['zc'][-1] * 1e6
dE = (C['mE'][-1] - O['mE'][-1]) / 1e3; g = np.gradient(dE, zc)
add = np.sqrt(np.maximum(C['sE'][-1]**2 - O['sE'][-1]**2, 0)) / 1e3
core = np.abs(C['zc'][-1]) < np.abs(C['zc'][-1]).max() * 2 / 3; w = C['n'][-1][core]
R, th, E0 = 1.0, 1.0, float(C['p0c'])
# residual rms z-shift through the full bend from first-order optics (R51 x0, R52 x0', R56 delta0)
sx, sxp, sd = 5e-6, 25e3 / E0, 50e3 / E0
dz51, dz52, dz56 = np.sin(th) * sx, R * (1 - np.cos(th)) * sxp, R * (th - np.sin(th)) * sd
dz = np.sqrt(dz51**2 + dz52**2 + dz56**2) * 1e6
# slice width and chirp-within-slice term
wslice = np.diff(C['zc'][-1][:2])[0] * 1e6
# the wake's own x-dependence: at each step, rms over the beam of (W(x,z) - W(x0,z)) at fixed z, then summed over steps
f = h5py.File('wakes_cm_sx5_sE50_on.h5'); xdep = []
for i in range(1, 17):
    G = f[f'step_{i}']; W = G['longitudinal/dE_dct'][()]; X = G['longitudinal/x_grids'][()]
    # mesh is (xbins, zbins) in tilt-removed x; at fixed z the rows span +-3 sigma_x, so the rms over a
    # Gaussian beam is approx the rms over rows weighted by exp(-u^2/2)
    u = np.linspace(-3, 3, W.shape[0]); wx = np.exp(-u**2 / 2); wx /= wx.sum()
    Wm = (wx[:, None] * W).sum(0); xdep.append(np.sqrt((wx[:, None] * (W - Wm)**2).sum(0)).max())   # MeV/m, worst z
xdep = np.array(xdep); dE_xdep_max = xdep.sum() * 0.1 * 1e3   # keV, coherent upper bound over 1.6 m
with open('null_test.txt', 'w') as fo:
    fo.write(f"sigma_x=5um + sigma_E=50keV (comoving, CSR on vs off), final s = {C['s'][-1]:.1f} m\n"
             f"  core mean slice sE: init {np.average(C['sE'][0][core], weights=C['n'][0][core])/1e3:.1f}, off {np.average(O['sE'][-1][core], weights=w)/1e3:.1f}, on {np.average(C['sE'][-1][core], weights=w)/1e3:.1f} keV\n"
             f"  peak slice sE on: {C['sE'][-1][n].max()/1e3:.1f} keV at z = {zc[n][np.argmax(C['sE'][-1][n])]:.0f} um; slices on<off: {np.sum((C['sE'][-1] < O['sE'][-1]) & n)} of {n.sum()}\n"
             f"  CSR-added (quadrature) max {add[n].max():.1f} keV, core-weighted rms {np.sqrt(np.average(add[core]**2, weights=w)):.1f} keV\n"
             f"  max |dDeltaE/dz| = {np.abs(g[n]).max():.1f} keV/um; peak CSR loss {dE.min():.0f} keV\n"
             f"  residual full-bend z-shift: R51*sx = {dz51*1e6:.2f}, R52*sx' = {dz52*1e6:.2f}, R56*sd = {dz56*1e6:.2f} -> rms {dz:.2f} um\n"
             f"  chirp-within-slice term: |dE/dz| * w/sqrt(12) <= {np.abs(g[n]).max()*wslice/np.sqrt(12):.1f} keV (slice width {wslice:.1f} um)\n"
             f"  wake x-dependence at fixed z: worst-z rms over beam per step [MeV/m]: {np.round(xdep, 3).tolist()}\n"
             f"     coherent upper bound on accumulated spread from it: {dE_xdep_max:.1f} keV\n")
    k = np.polyfit(np.abs(g[n]), add[n], 1)
    fo.write(f"  fit CSR-added = a + b*|dDeltaE/dz|: a = {k[1]:.1f} keV, b = {k[0]:.2f} um, r = {np.corrcoef(np.abs(g[n]), add[n])[0,1]:.3f}\n")
fig, ax = plt.subplots(1, 2, figsize=(12, 4.3), layout='constrained')
ax[0].plot(zc, C['sE'][-1] / 1e3, c='C3', label='CSR on'); ax[0].plot(zc, O['sE'][-1] / 1e3, ':', c='C0', label='CSR off')
ax[0].set(xlabel='z − ⟨z⟩ [µm]', ylabel='final slice σE [keV]', title='σx = 5 µm and σE = 50 keV (comoving)'); ax[0].legend()
ax[1].plot(zc[n], add[n], 'o-', c='C3', ms=3, label='CSR-added slice σE (quadrature)')
ax[1].plot(zc[n], np.abs(g[n]) * dz, '--', c='0.4', label=f'|dΔE/dz| × {dz:.1f} µm (full-bend residual shift)')
ax[1].set(xlabel='z − ⟨z⟩ [µm]', ylabel='keV', title='What is left, vs the residual-mixing estimate'); ax[1].legend(fontsize=8)
plt.savefig('null_test.png', dpi=110)
print(open('null_test.txt').read())
