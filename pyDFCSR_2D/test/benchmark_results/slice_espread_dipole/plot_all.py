"""Plots for the unchirped-dipole slice energy spread study (reads beam_{on,off}.npz, wakes_on.h5)."""
import os, h5py, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib import cm, colors
os.makedirs('wakes', exist_ok=True); os.makedirs('phasespace', exist_ok=True)
on, off = np.load('beam_on.npz'), np.load('beam_off.npz'); E0 = float(on['p0c'])
s = on['s']; ns = len(s)
L_BEND = (0.1, 1.1)
scol = cm.viridis(np.linspace(0, 0.95, ns)); snorm = colors.Normalize(s[0], s[-1])

# ---------- 1. 2D wake maps per step ----------
f = h5py.File('wakes_on.h5'); steps = sorted(f.keys(), key=lambda k: int(k.split('_')[1]))
for k in steps:
    g = f[k]; X = g['longitudinal/x_grids'][()] * 1e6; Z = g['longitudinal/z_grids'][()] * 1e6
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for a, (W, name) in zip(ax, [(g['longitudinal/dE_dct'][()], 'Longitudinal wake dE/d(ct)'),
                                 (g['transverse/xkicks'][()], 'Transverse wake F_x')]):
        v = np.abs(W).max()
        pc = a.pcolormesh(Z, X, W, cmap='RdBu_r', vmin=-v, vmax=v, shading='gouraud')
        a.set(xlabel='z [µm]', ylabel='x [µm]', title=name)
        fig.colorbar(pc, ax=a, label='MeV/m')
    fig.suptitle(f"{k}, s = {g.attrs['position']:.2f} m ({g.attrs['element']})")
    plt.tight_layout(); plt.savefig(f"wakes/wakes_{k}.png", dpi=110); plt.close()

# ---------- 2. slice profiles vs z at each step, and beam stats vs s ----------
fig, ax = plt.subplots(2, 3, figsize=(16, 8), sharex=True, layout='constrained')
for row, d, lab in [(0, on, 'CSR on'), (1, off, 'CSR off')]:
    for i in range(ns):
        zc = d['zc'][i] * 1e6
        ax[row, 0].plot(zc, (d['mE'][i] - d['mE'][0].mean()) / 1e3, c=scol[i], lw=1.2)
        ax[row, 1].plot(zc, d['sE'][i] / 1e3, c=scol[i], lw=1.2)
        ax[row, 2].plot(zc, d['sx'][i] * 1e6, c=scol[i], lw=1.2)
    ax[row, 0].set(ylabel=f'{lab}\nslice mean ΔE [keV]'); ax[row, 1].set(ylabel='slice σE [keV]')
    ax[row, 2].set(ylabel='slice σx [µm]')
for a in ax[1]: a.set(xlabel='z − ⟨z⟩ [µm]')
for a in ax[:, 1]: a.set_ylim(350, 1000)
for a in ax[:, 2]: a.set_ylim(0, max(on['sx'].max(), off['sx'].max()) * 1.05e6)
fig.colorbar(cm.ScalarMappable(snorm, 'viridis'), ax=ax, label='s [m]', shrink=0.8)
fig.suptitle('Slice profiles along the bunch at every step (40 slices, ±3σz)')
plt.savefig('slice_profiles_vs_z.png', dpi=110); plt.close()

def core_mean(d, k):  # weighted by slice population, |z| < 2 sigma_z (slices span +-3 sigma_z)
    out = []
    for i in range(ns):
        m = np.abs(d['zc'][i]) < np.abs(d['zc'][i]).max() * 2 / 3; w = d['n'][i][m]
        out.append(np.sum(d[k][i][m] * w) / w.sum())
    return np.array(out)
def proj(d, H, e, axis):  # projected rms from the histograms
    out = []
    for i in range(ns):
        c = 0.5 * (d[e][i][1:] + d[e][i][:-1]); w = d[H][i].sum(axis=axis)
        m = np.sum(w * c) / w.sum(); out.append(np.sqrt(np.sum(w * (c - m) ** 2) / w.sum()))
    return np.array(out)
fig, ax = plt.subplots(1, 3, figsize=(16, 4.3))
for d, lab, c in [(on, 'CSR on', 'C3'), (off, 'CSR off', 'C0')]:
    ax[0].plot(s, proj(d, 'H_zd', 'ed', 0) * E0 / 1e3, 'o-', c=c, label=f'{lab}: projected')
    ax[0].plot(s, core_mean(d, 'sE') / 1e3, 's--', c=c, label=f'{lab}: mean slice (|z|<2σz)')
    ax[1].plot(s, proj(d, 'H_xxp', 'ex', 1) * 1e6, 'o-', c=c, label=f'{lab}: projected σx')
    ax[1].plot(s, core_mean(d, 'sx') * 1e6, 's--', c=c, label=f'{lab}: mean slice σx')
    ax[2].plot(s, proj(d, 'H_zd', 'ez', 1) * 1e6, 'o-', c=c, label=f'{lab}: σz')
ax[0].set(ylabel='σE [keV]', title='Energy spread vs s'); ax[1].set(ylabel='σx [µm]', title='Transverse size vs s')
ax[2].set(ylabel='σz [µm]', title='Bunch length vs s')
for a in ax:
    a.axvspan(*L_BEND, color='0.92', zorder=0); a.set(xlabel='s [m]'); a.legend(fontsize=8)
plt.tight_layout(); plt.savefig('beam_stats_vs_s.png', dpi=110); plt.close()

# ---------- 3. phase-space density per step (CSR on | CSR off) ----------
def show(a, d, i, H, e1, e2, s1, s2, xl, yl):
    pc = a.pcolormesh(d[e1][i] * s1, d[e2][i] * s2, d[H][i].T / d[H][i].max(), cmap='Blues', shading='flat')
    a.set(xlabel=xl, ylabel=yl); return pc
for i in range(ns):
    fig, ax = plt.subplots(2, 2, figsize=(11, 8.5), layout='constrained')
    for col, d, lab in [(0, on, 'CSR on'), (1, off, 'CSR off')]:
        show(ax[0, col], d, i, 'H_zd', 'ez', 'ed', 1e6, 1e4, 'z [µm]', 'δ [10⁻⁴]').axes.set_title(f'{lab}: (z, δ)')
        pc = show(ax[1, col], d, i, 'H_xxp', 'ex', 'exp', 1e6, 1e6, 'x [µm]', "x' [µrad]"); ax[1, col].set_title(f"{lab}: (x, x')")
    fig.colorbar(pc, ax=ax, label='charge density (normalized to peak)', shrink=0.7)
    fig.suptitle(f'step {i}, s = {s[i]:.2f} m')
    plt.savefig(f'phasespace/phasespace_step_{i:02d}.png', dpi=100); plt.close()

# ---------- 4. initial vs final: energy spectrum + phase spaces ----------
fig, ax = plt.subplots(2, 3, figsize=(16, 8.5))
for d, lab, c, ls in [(on, 'final, CSR on', 'C3', '-'), (off, 'final, CSR off', 'C0', '--'), (on, 'initial', 'k', ':')]:
    i = 0 if lab == 'initial' else ns - 1
    c_ = 0.5 * (d['ed'][i][1:] + d['ed'][i][:-1]); w = d['H_zd'][i].sum(0) * np.diff(d['ez'][i])[0]
    ax[0, 0].plot(c_ * E0 / 1e6, w / w.max(), ls, c=c, label=lab, lw=1.5)
    if lab != 'final, CSR off':
        ax[1, 0].plot(d['zc'][i] * 1e6, d['sE'][i] / 1e3, ls, c=c, label=lab, lw=1.5)
ax[1, 0].plot(off['zc'][-1] * 1e6, off['sE'][-1] / 1e3, '--', c='C0', label='final, CSR off')
ax[0, 0].set(xlabel='ΔE [MeV]', ylabel='dQ/dE (normalized)', title='Energy spectrum'); ax[0, 0].legend(fontsize=8)
ax[1, 0].set(xlabel='z − ⟨z⟩ [µm]', ylabel='slice σE [keV]', title='Slice energy spread'); ax[1, 0].legend(fontsize=8)
for col, (i, d, lab) in zip([1, 2], [(0, on, 'initial'), (ns - 1, on, f'final (s = {s[-1]:.1f} m), CSR on')]):
    show(ax[0, col], d, i, 'H_zd', 'ez', 'ed', 1e6, 1e4, 'z [µm]', 'δ [10⁻⁴]'); ax[0, col].set_title(f'{lab}: (z, δ)')
    show(ax[1, col], d, i, 'H_xxp', 'ex', 'exp', 1e6, 1e6, 'x [µm]', "x' [µrad]"); ax[1, col].set_title(f"{lab}: (x, x')")
plt.tight_layout(); plt.savefig('initial_vs_final.png', dpi=110); plt.close()

# ---------- summary numbers ----------
with open('analysis.txt', 'w') as fo:
    sEp_on, sEp_off = proj(on, 'H_zd', 'ed', 0) * E0, proj(off, 'H_zd', 'ed', 0) * E0
    cs_on, cs_off = core_mean(on, 'sE'), core_mean(off, 'sE')
    sx_on, sx_off = proj(on, 'H_xxp', 'ex', 1), proj(off, 'H_xxp', 'ex', 1)
    fo.write(f"{'s[m]':>5} {'projE_on':>9} {'projE_off':>9} {'sliceE_on':>9} {'sliceE_off':>10} {'sx_on':>7} {'sx_off':>7}   [keV, um]\n")
    for i in range(ns):
        fo.write(f"{s[i]:5.2f} {sEp_on[i]/1e3:9.1f} {sEp_off[i]/1e3:9.1f} {cs_on[i]/1e3:9.1f} {cs_off[i]/1e3:10.1f} {sx_on[i]*1e6:7.1f} {sx_off[i]*1e6:7.1f}\n")
    fo.write(f"final peak slice sigma_E on: {on['sE'][-1].max()/1e3:.1f} keV, min {on['sE'][-1][on['n'][-1]>1000].min()/1e3:.1f} keV\n")
print(open('analysis.txt').read())
