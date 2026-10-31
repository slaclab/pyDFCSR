"""
Benchmark chicane, chirp scan: 1D vs 2D vs 2D without transverse force, all on identical bmad-x tracking,
beam, lattice and auto schedule (both CSR models with drift_cutoff = 1 R*phi). Every quantity is at the
chicane exit; changes are against the CSR-off run at the same chirp.

Definitions (core average = population-weighted over the slices whose centre is within +-2 sigma_z, of 80
slices over +-3 sigma_z, slices with < 500 particles dropped):
  mean energy loss   <dE> of CSR on minus CSR off                                           [keV]
  emittance growth   eta-corrected normalized emittance (x, x' with the linear fit in delta
                     removed), CSR on minus CSR off                                          [um]
  slice emittance    core average of the per-slice eta-corrected normalized emittance        [um]
  slice sigma_E      core average: sqrt of the population-weighted mean of the per-slice
                     energy variance after a quadratic fit in z                              [keV]
  sigma_x            rms beam size (projected)                                               [um]
  sigma_z            rms bunch length                                                        [um]
  peak current       max of a 400-bin z histogram of the 1 nC bunch                          [kA]

Folded phase space: from h ~ -38 /m on, the longitudinal phase space of even the CSR-off beam folds over
(z(delta) is no longer monotonic: second-order compression from T566 turns the head back). A z-slice then
holds two energy branches up to ~100 MeV apart, which a quadratic fit in z cannot remove, so the core slice
sigma_E measures the fold (8.0, 27.8, 3.9 MeV with CSR off at h = -38, -39, -42), not a slice energy spread.
Those values are flagged FOLDED in the table and drawn hollow in the plot; slice emittance in a folded slice
is likewise inflated by mixing the two branches (flagged the same way).

Usage: python chirp_scan_models.py   -> chirp_scan_models.txt, chirp_scan_models.png
"""
FOLDED = {-190, -195, -210}          # shears whose CSR-off phase space is folded (see docstring)
import os
import sys

import numpy as np

sys.argv = sys.argv[:1]
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import chicane_2d_vs_1d as C   # noqa: E402

BASE = os.path.join(HERE, 'benchmark_results', 'chicane_2d_vs_1d')
SHEARS = [-100, -140, -160, -180, -190, -195, -210]
MODELS = [('1d', '1D'), ('2d', '2D'), ('2d_noT', '2D, transverse off')]
COLS = {'off': 'k', '1d': 'C0', '2d': 'C3', '2d_noT': 'C1'}


def folder(sh):
    return BASE if sh == -180 else os.path.join(BASE, f'shear{sh}')


def beam_metrics(r, off, E0):
    """All exit quantities for one run, with the core-average slice definitions above."""
    m = C.metrics(r, off, E0)
    return dict(dE=m['dE'], demit=m['demit'], semit=m['semit'], sEs=m['sEs'], dsEs=m['dsEs'], rsEs=m['rsEs'],
                sx=r['x'].std() * 1e6, sz=m['sz'], Ipk=m['Ipk'])


rows = []
for sh in SHEARS:
    d = folder(sh)
    off = dict(np.load(os.path.join(d, 'off.npz'))); E0 = float(off['E0'])
    row = dict(sh=sh, h=sh / 5.0, off=beam_metrics(off, off, E0))
    for n, _ in MODELS:
        f = os.path.join(d, f'{n}.npz')
        row[n] = beam_metrics(dict(np.load(f)), off, E0) if os.path.isfile(f) else None
    row['C'] = 200.0 / row['off']['sz']
    rows.append(row)

QTY = [('dE', 'mean CSR energy loss [keV]', '{:+.0f}'),
       ('demit', 'emittance growth, eta-corrected [um]', '{:+.3f}'),
       ('semit', 'core slice emittance [um]', '{:.3f}'),
       ('sEs', 'core slice sigma_E [keV]', '{:.1f}'),
       ('rsEs', '  change vs CSR off [%]', '{:+.1f}'),
       ('sx', 'beam size sigma_x [um]', '{:.2f}'),
       ('sz', 'bunch length sigma_z [um]', '{:.2f}'),
       ('Ipk', 'peak current [kA]', '{:.2f}')]
L = [__doc__.split('Usage')[0].strip(), '']
for r in rows:
    L.append(f"=== h = {r['h']:.0f} /m (shear {r['sh']} GeV/c/m), compression {r['C']:.1f} (CSR off) ===")
    L.append(f"{'':38s}" + ''.join(f'{lab:>20s}' for _, lab in MODELS) + f"{'CSR off':>12s}")
    for k, lab, fmt in QTY:
        flag = ' FOLDED' if (r['sh'] in FOLDED and k in ('sEs', 'rsEs', 'semit')) else ''
        vals = ''.join(f"{((fmt.format(r[n][k]) + flag) if r[n] else 'not run'):>20s}" for n, _ in MODELS)
        offv = '' if k in ('dE', 'demit', 'rsEs') else fmt.format(r['off'][k]) + flag
        L.append(f'{lab:38s}{vals}{offv:>12s}')
    L.append('')
open(os.path.join(BASE, 'chirp_scan_models.txt'), 'w').write('\n'.join(L) + '\n')
print('\n'.join(L))

# ---------------------------------------------------------------------------------- plot
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt   # noqa: E402,E401

h = np.array([r['h'] for r in rows])
fig, ax = plt.subplots(1, 3, figsize=(17, 4.8), layout='constrained')
fold = np.array([r['sh'] in FOLDED for r in rows])
for n, lab in MODELS:
    ok = np.array([bool(r[n]) for r in rows])
    for j, k in ((0, 'semit'), (1, 'demit'), (2, 'sEs')):
        y = np.array([r[n][k] if r[n] else np.nan for r in rows])
        good = ok & ~fold if k != 'demit' else ok
        ax[j].plot(h[good], y[good], 'o-', c=COLS[n], label=lab)
        if k != 'demit':
            bad = ok & fold
            ax[j].plot(h[bad], y[bad], 'o', mfc='none', c=COLS[n])
for j, k in ((0, 'semit'), (2, 'sEs')):
    y = np.array([r['off'][k] for r in rows])
    ax[j].plot(h[~fold], y[~fold], 'k:', label='CSR off')
    ax[j].plot(h[fold], y[fold], 'kx', ms=5, label='folded phase space (not a slice spread)')
    ax[j].axvspan(-43, -37.3, color='0.95', zorder=-2)
for a, yl, t in zip(ax, ['core slice εn [µm]', 'Δεn, η-corrected [µm]', 'core slice σE [keV]'],
                    ['core slice emittance', 'emittance growth (CSR on − off)', 'core slice energy spread']):
    a.set(xlabel='initial chirp h [1/m]', ylabel=yl, title=t); a.legend(fontsize=8)
    a.set_yscale('log')
    # compression factor on top, and a mark at the nominal chirp
    a.axvline(-36, c='0.85', lw=1, zorder=-1)
    sec = a.secondary_xaxis('top', functions=(lambda x: np.interp(x, h[::-1], np.array([r['C'] for r in rows])[::-1]),
                                              lambda c: np.interp(c, np.array([r['C'] for r in rows]), h)))
    sec.set_xlabel('compression factor (CSR off)')
    sec.set_ticks([2, 3.4, 5.2, 10.9, 23, 45])
ax[1].set_yscale('symlog', linthresh=0.01)
fig.suptitle('Benchmark chicane, exit, vs initial chirp: 1D vs 2D vs 2D without transverse force (core averages, |z| < 2σz; shaded: phase space folded, slice quantities hollow)')
plt.savefig(os.path.join(BASE, 'chirp_scan_models.png'), dpi=110)
