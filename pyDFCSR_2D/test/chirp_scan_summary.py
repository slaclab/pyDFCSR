"""
Chirp scan of the benchmark chicane, 2D vs 1D CSR in pyDFCSR (same bmad-x tracking, beam, lattice and
auto schedule per chirp). Collects the per-chirp runs written by chicane_2d_vs_1d.py --shear S (and the
reference h = -36 /m in the base folder) into chirp_scan.txt and chirp_scan.png.
"""
import os, sys, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.argv = sys.argv[:1]
import chicane_2d_vs_1d as C
BASE = os.path.join(HERE, 'benchmark_results', 'chicane_2d_vs_1d')
SHEARS = [-100, -140, -160, -180, -190, -195, -210]
rows = []
for sh in SHEARS:
    d = BASE if sh == -180 else os.path.join(BASE, f'shear{sh}')
    R = {n: dict(np.load(os.path.join(d, f'{n}.npz'))) for n in ('off', '2d', '1d', '1d_full')}
    off = R['off']; E0 = float(off['E0'])
    M = {n: C.metrics(R[n], off, E0) for n in ('2d', '1d', '1d_full')}
    Moff = C.metrics(off, off, E0)
    s0 = np.sqrt(off['slice_var'][-1])
    rel = {n: 100 * (np.sqrt(R[n]['slice_var'][-1]) - s0) / s0 for n in ('2d', '1d', '1d_full')}
    el = {n: C.per_element_loss(R[n]) - C.per_element_loss(off) for n in ('2d', '1d')}
    rows.append(dict(sh=sh, h=sh / 5.0, C=200.0 / Moff['sz'], sz_off=Moff['sz'], szmin=off['sigma_z'].min() * 1e6,
                     M=M, slice_off=s0 / 1e3, rel=rel, el=el, emit_off=Moff['semit'],
                     kicks=int(np.sum(np.diff(R['2d']['s']) > 0)),   # tracking steps (nodes - 1), not CSR kicks
                     rt=float(R['2d']['runtime']) / 60))
L = [__doc__.strip(), '']
hdr = f"{'h [1/m]':>8s} {'C':>6s} {'sz_min':>7s} | {'dE 2D':>8s} {'1D':>8s} {'2D/1D':>6s} | {'emit 2D':>8s} {'1D':>7s} | {'slice emit 2D':>13s} {'1D':>6s} | {'slice sE 2D':>11s} {'1D':>7s} | {'B2+B3 2D':>9s} {'1D':>7s} | {'B4 2D':>7s} {'1D':>7s}"
L += ['chicane exit, CSR effect vs CSR off (1D: same schedule, cutoff 1 R*phi as 2D)',
      'C = compression factor 200 um / final sigma_z (CSR off); sz_min = smallest sigma_z along the lattice (CSR off) [um]',
      'dE = mean CSR energy change [keV]; emit = eta-corrected emittance growth [um]; slice emit = core slice emittance [um]',
      'slice sE = relative change of the core slice sigma_E at the exit [%]; B2+B3, B4 = energy collected there [keV]', '', hdr]
for r in rows:
    M = r['M']
    L.append(f"{r['h']:8.1f} {r['C']:6.1f} {r['szmin']:7.2f} | {M['2d']['dE']:8.0f} {M['1d']['dE']:8.0f} {M['2d']['dE']/M['1d']['dE']:6.2f} | "
             f"{M['2d']['demit']:8.3f} {M['1d']['demit']:7.3f} | {M['2d']['semit']:13.3f} {M['1d']['semit']:6.3f} | "
             f"{r['rel']['2d']:+10.1f}% {r['rel']['1d']:+6.1f}% | {r['el']['2d'][3] + r['el']['2d'][5]:9.0f} {r['el']['1d'][3] + r['el']['1d'][5]:7.0f} | "
             f"{r['el']['2d'][7]:7.0f} {r['el']['1d'][7]:7.0f}")
L += ['', f"{'h [1/m]':>8s} {'2D steps':>9s} {'2D runtime [min]':>17s} {'slice emit off':>15s} {'final sz 2D':>12s} {'1D':>6s} {'off':>6s}"]
for r in rows:
    L.append(f"{r['h']:8.1f} {r['kicks']:9d} {r['rt']:17.1f} {r['emit_off']:15.3f} {r['M']['2d']['sz']:12.2f} {r['M']['1d']['sz']:6.2f} {r['sz_off']:6.2f}")
open(os.path.join(BASE, 'chirp_scan.txt'), 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))

import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
C_ = np.array([r['C'] for r in rows]); h = np.array([r['h'] for r in rows])
g = lambda n, k: np.array([r['M'][n][k] for r in rows])
fig, ax = plt.subplots(2, 3, figsize=(17, 9), layout='constrained'); ax = ax.ravel()
for n, c, lab in [('2d', 'C3', '2D'), ('1d', 'C0', '1D (cutoff, as 2D)'), ('1d_full', 'C2', '1D, full drift')]:
    ax[0].plot(h, -g(n, 'dE') / 1e3, 'o-', c=c, label=lab)
    ax[1].plot(h, g(n, 'demit'), 'o-', c=c, label=lab)
    ax[2].plot(h, g(n, 'semit'), 'o-', c=c, label=lab)
    ax[3].plot(h, [r['rel'][n] for r in rows], 'o-', c=c, label=lab)
    ax[4].plot(h, g(n, 'sz'), 'o-', c=c, label=lab)
ax[2].plot(h, [r['emit_off'] for r in rows], 'k:', label='CSR off')
ax[4].plot(h, [r['sz_off'] for r in rows], 'k:', label='CSR off')
ax[5].plot(h, [r['el']['2d'][3] + r['el']['2d'][5] for r in rows], 'o-', c='C3', label='2D, B2+B3')
ax[5].plot(h, [r['el']['1d'][3] + r['el']['1d'][5] for r in rows], 'o-', c='C0', label='1D, B2+B3')
ax[5].plot(h, [r['el']['2d'][7] for r in rows], 's--', c='C3', label='2D, B4')
ax[5].plot(h, [r['el']['1d'][7] for r in rows], 's--', c='C0', label='1D, B4')
for a, yl, t in zip(ax, ['−⟨ΔE⟩ [MeV]', 'Δεn, η-corrected [µm]', 'core slice εn [µm]', 'Δσ_E,slice / σ_E,slice(off) [%]', 'final σz [µm]', 'CSR energy change [keV]'],
                    ['mean CSR energy loss', 'emittance growth', 'core slice emittance', 'CSR change of core slice σE (exit)', 'final bunch length', 'energy collected per region']):
    a.set(xlabel='chirp h [1/m]', ylabel=yl, title=t); a.legend(fontsize=8)
    for r in rows: a.axvline(r['h'], c='0.93', lw=0.5, zorder=-1)
ax[4].set_yscale('log'); ax[0].set_yscale('log'); ax[1].set_yscale('symlog', linthresh=0.1)
sec = ax[0].secondary_xaxis('top', functions=(lambda x: np.interp(x, h[::-1], C_[::-1]), lambda c: np.interp(c, C_, h)))
sec.set_xlabel('compression factor (CSR off)')
fig.suptitle('Benchmark chicane, chirp scan: 2D vs 1D CSR on identical tracking')
plt.savefig(os.path.join(BASE, 'chirp_scan.png'), dpi=110)
