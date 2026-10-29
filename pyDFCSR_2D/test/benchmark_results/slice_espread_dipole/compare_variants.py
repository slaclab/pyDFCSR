"""Separate the R51 (x0) and R56 (delta0) contributions to the CSR slice energy spread.
Compares baseline (sx=50um, sE=500keV) with sx=5um and sE=50keV, each with CSR on and off. CSR-on runs use --guard (forced history re-grid; see slice_espread_dipole.py)."""
import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
cases = [('', 'baseline: σx=50 µm, σE=500 keV'), ('sx5_', 'σx=5 µm (no R51 mixing)'), ('sE50_', 'σE=50 keV (no R56 mixing)')]
fig, ax = plt.subplots(2, 3, figsize=(16, 8), layout='constrained')
with open('compare_variants.txt', 'w') as fo:
    for j, (c, lab) in enumerate(cases):
        on, off = np.load(f'beam_{c}on_guard.npz'), np.load(f'beam_{c}off.npz')
        zc, n = on['zc'][-1] * 1e6, on['n'][-1]; core = np.abs(on['zc'][-1]) < np.abs(on['zc'][-1]).max() * 2 / 3
        d = (on['sE'][-1] - off['sE'][-1]) / 1e3
        q = (np.sqrt(np.maximum(on['sE'][-1]**2 - off['sE'][-1]**2, 0)) - np.sqrt(np.maximum(off['sE'][-1]**2 - on['sE'][-1]**2, 0))) / 1e3
        ax[0, j].plot(zc, on['sE'][-1] / 1e3, '-', c='C3', label='CSR on'); ax[0, j].plot(zc, off['sE'][-1] / 1e3, '--', c='C0', label='CSR off')
        ax[0, j].plot(on['zc'][0] * 1e6, on['sE'][0] / 1e3, ':', c='k', label='initial')
        ax[0, j].set(title=lab, xlabel='z − ⟨z⟩ [µm]', ylabel='final slice σE [keV]'); ax[0, j].legend(fontsize=8)
        ax[1, j].plot(zc, q, '-', c='C3', label='signed √|σ²on − σ²off|')
        a2 = ax[1, j].twinx(); a2.plot(zc, (on['mE'][-1] - off['mE'][-1]) / 1e3, c='0.5', lw=1); a2.set_ylabel('slice ΔE, on − off [keV]', color='0.4')
        ax[1, j].axhline(0, c='k', lw=0.5); ax[1, j].set(xlabel='z − ⟨z⟩ [µm]', ylabel='CSR-added slice σE [keV]'); ax[1, j].legend(fontsize=8, loc='upper left')
        w = n[core]
        fo.write(f"{lab}\n  final sigma_z on/off: {np.sqrt(np.average(on['zc'][-1]**2, weights=n))*1e6:.1f}\n"
                 f"  core mean slice sE: init {np.average(on['sE'][0][core], weights=on['n'][0][core])/1e3:.1f}, off {np.average(off['sE'][-1][core], weights=w)/1e3:.1f}, on {np.average(on['sE'][-1][core], weights=w)/1e3:.1f} keV\n"
                 f"  CSR-added in quadrature (core, weighted): {np.sqrt(max(np.average(on['sE'][-1][core]**2 - off['sE'][-1][core]**2, weights=w), 0))/1e3:.1f} keV\n"
                 f"  min / max of (on - off) over slices with n>1000: {d[n>1000].min():.1f} / {d[n>1000].max():.1f} keV; slices with on<off: {np.sum((d<0)&(n>1000))} of {np.sum(n>1000)}\n"
                 f"  peak slice CSR energy loss: {((on['mE'][-1]-off['mE'][-1])/1e3).min():.0f} keV\n")
plt.savefig('compare_variants.png', dpi=110)
print(open('compare_variants.txt').read())
