"""Rerun of the three slice-energy-spread cases with the bspline_comoving deposition (rotating
frame, no history re-grid), compared with the legacy CIC+SG runs (CSR-on legacy uses --guard)."""
import numpy as np, h5py, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
m = lambda d, i: np.average(d['mE'][i], weights=d['n'][i]) / 1e3
cases = [('cm', 'on_guard', 'off', 'baseline: σx=50 µm, σE=500 keV'), ('cm_sx5', 'sx5_on_guard', 'sx5_off', 'σx=5 µm (no R51 term)'),
         ('cm_sE50', 'sE50_on_guard', 'sE50_off', 'σE=50 keV (no R56 term)')]
fig, ax = plt.subplots(1, 3, figsize=(16, 4.5), layout='constrained')
with open('compare_comoving.txt', 'w') as fo:
    for j, (cm, leg, off, lab) in enumerate(cases):
        C, L, O = np.load(f'beam_{cm}_on.npz'), np.load(f'beam_{leg}.npz'), np.load(f'beam_{off}.npz')
        f = h5py.File(f'wakes_{cm}_on.h5'); W = [f[f'step_{i}']['longitudinal/dE_dct'][()] for i in range(1, 17)]
        core = np.abs(C['zc'][-1]) < np.abs(C['zc'][-1]).max() * 2 / 3; w = C['n'][-1][core]
        cs = lambda d: np.average(d['sE'][-1][core], weights=w) / 1e3
        n = C['n'][-1] > 1000; zc = C['zc'][-1] * 1e6
        g = np.gradient((C['mE'][-1] - O['mE'][-1]) / 1e3, zc); r = C['sE'][-1] / O['sE'][-1]; k = np.polyfit(g[n], r[n], 1)
        fo.write(f"{lab}\n  comoving wake range over all steps: {min(x.min() for x in W):.2f} / {max(x.max() for x in W):.2f} MeV/m\n"
                 f"  final mean CSR dE: comoving {m(C,-1)-m(O,-1):.0f}, legacy {m(L,-1)-m(O,-1):.0f} keV\n"
                 f"  core mean slice sE: off {cs(O):.1f}, legacy {cs(L):.1f}, comoving {cs(C):.1f} keV; max |comoving-legacy| over slices {np.abs(C['sE'][-1]-L['sE'][-1])[n].max()/1e3:.1f} keV\n"
                 f"  comoving: slices with on<off {np.sum((r<1)&n)} of {n.sum()}, min ratio {r[n].min():.2f}; ratio fit a={k[1]:.2f} b={k[0]:.4f}/(keV/um) r={np.corrcoef(g[n],r[n])[0,1]:.3f}\n")
        ax[j].plot(zc, C['sE'][-1] / 1e3, c='C3', label='CSR on, comoving'); ax[j].plot(L['zc'][-1] * 1e6, L['sE'][-1] / 1e3, '--', c='C1', label='CSR on, legacy (guarded)')
        ax[j].plot(O['zc'][-1] * 1e6, O['sE'][-1] / 1e3, ':', c='C0', label='CSR off'); ax[j].set(title=lab, xlabel='z − ⟨z⟩ [µm]', ylabel='final slice σE [keV]'); ax[j].legend(fontsize=8)
plt.savefig('compare_comoving.png', dpi=110)
print(open('compare_comoving.txt').read())
