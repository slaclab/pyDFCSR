"""
BC20 with CSR: pyDFCSR 1D / 2D / 2D without transverse force vs Tao 1D CSR, all from the same
20 000 particles (runs from run_dfcsr_bc20.py and tao_bc20.py, CSR-off reference from
bc20_export.npz, the Tao beam with every collective effect off).

    conda run -n pydfcsr python pyDFCSR_2D/test/facet_bc20/compare_csr_runs.py

Writes
    test/facet_bc20/compare_csr_runs.txt                       (tables, also printed)
    test/benchmark_results/facet_bc20/csr_along_s.png         (element exits through BC20)
    test/benchmark_results/facet_bc20/csr_at_end.png          (along the bunch at ENDBC20)
    test/benchmark_results/facet_bc20/csr_resolution_1d.png   (1D energy loss vs z bin width)

The particles are in the same order in every file, so the CSR effect is taken particle by
particle as run - CSR off. eps_nx is dispersion-corrected: x and px with their fit in delta
removed, to first order (as twiss.py) and, in the tables, also to third order (that removes most
of the chromatic growth BC20 has with CSR off, which otherwise hides the CSR part).
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
from pyDFCSR_2D.csr1d import bin_width  # noqa: E402
from run_dfcsr_bc20 import half_indices  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, '..', 'benchmark_results', 'facet_bc20')
RUNS = os.path.join(RES, 'runs')
ME = 0.51099895e6
Q = 1.6e-9
C0 = 299792458.0
# label, file, color (reference categorical slots in fixed order), line style
SERIES = [('Tao 1D CSR', 'tao_csr', '#2a78d6', '-'),
          ('pyDFCSR 1D', 'dfcsr_1d', '#eb6834', '--'),
          ('pyDFCSR 2D', 'dfcsr_2d', '#1baf7a', '-'),
          ('pyDFCSR 2D, no transverse', 'dfcsr_2d_noT', '#eda100', ':'),
          ('Tao, all effects (ISR, SC, wakes, 1D CSR)', 'tao_all', '#e87ba4', '-.')]
# the quantile-frame 2D runs (particle_deposition.frame quantile, mesh_mode charge), if present
for _s in (('pyDFCSR 2D quantile frame', 'dfcsr_2d_q', '#4a3aa7', '-'),
           ('pyDFCSR 2D quantile, no transverse', 'dfcsr_2d_q_noT', '#008300', ':')):
    if os.path.isfile(os.path.join(RUNS, _s[1] + '.npz')):
        SERIES.insert(-1, _s)
SER_CSR = [s for s in SERIES if s[1] != 'tao_all']      # the CSR-only runs, compared like for like
INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'


def emit_n(c, p0c, order=1):
    dl = c[5] - c[5].mean()
    A = np.stack([np.ones_like(dl)] + [dl ** k for k in range(1, order + 1)], 1)
    res = lambda u: u - A @ np.linalg.lstsq(A, u, rcond=None)[0]
    g = (1 + c[5].mean()) * p0c / ME
    return np.sqrt(np.linalg.det(np.cov(res(c[0]), res(c[1])))) * g


def peak_current(z, dz=0.5e-6):
    h, _ = np.histogram(z, bins=np.arange(z.min(), z.max() + dz, dz))
    I = gaussian_filter1d(h * Q / len(z) / (dz / C0), 1.0)
    k = I.argmax()
    half = np.flatnonzero(I > I[k] / 2)
    return I[k], (half[-1] - half[0] + 1) * dz


def stats(c, off, p0c):
    dE = (c[5] - off[5]) * p0c
    return dict(sz=c[4].std(), Ipk=peak_current(c[4])[0], E=c[5].mean() * p0c, dE=dE.mean(),
                sdE=dE.std(), sd=c[5].std(), en1=emit_n(c, p0c, 1), en3=emit_n(c, p0c, 3),
                sx=c[0].std())


def load():
    X = np.load(os.path.join(HERE, 'bc20_export.npz'))
    els = json.loads(str(X['elements']))
    s_tao = np.array([e['s'] for e in els]) - els[0]['s']
    ref = np.load(os.path.join(RUNS, 'dfcsr_1d.npz'))
    s, names = ref['s'], ref['names']
    # last Tao element at each pyDFCSR element exit (zero-length ones after it are inert)
    jt = [int(np.flatnonzero(np.abs(s_tao - x) < 1e-9)[-1]) for x in s]
    p0c = float(X['p0c'])
    data = {'off': X['coords_fine'][jt][:, :6]}
    runtime = {}
    for _, f, _, _ in SERIES:
        d = np.load(os.path.join(RUNS, f + '.npz'))
        data[f] = d['coords'][jt][:, :6] if f.startswith('tao') else d['coords']
        runtime[f] = float(d['runtime'])
    extra = [scan_file(nb) for nb in NB_SCAN] + [f'dfcsr_1d_q{n}' for n in Q_SCAN] + \
        [f'dfcsr_1d_pk{m}' for m in PK_SCAN] + [f'dfcsr_1d_{v}_half{h}' for v in HALF for h in 'AB']
    for f in extra:
        data.setdefault(f, np.load(os.path.join(RUNS, f + '.npz'))['coords'])
    for f in [tao_file(nb) for nb in TAO_NB] + [f'tao_csr_{v}' for v in TAO_SPAN]:
        data.setdefault(f, np.load(os.path.join(RUNS, f + '.npz'))['coords'][jt][:, :6])
    init = X['coords_fine'][0][:6]
    types = np.load(os.path.join(HERE, 'compare_tracking.npz'))['types']
    assert (np.load(os.path.join(HERE, 'compare_tracking.npz'))['names'] == names).all()
    return s, names, types, p0c, data, init, runtime


NB_SCAN = (1200, 600, 300, 150, 80, 40, 20)     # dfcsr_1d_nb<N>; 300 is dfcsr_1d itself
Q_SCAN = (1200, 600, 300, 150, 75)               # dfcsr_1d_q<N>: bin_mode quantile, q = 0.05
PK_SCAN = (80, 40, 20, 10, 5)                    # dfcsr_1d_pk<M>: bin_mode peak, M bins per FWHM
HALF = ('nb300', 'nb1200', 'pk10', 'pk20', 'pk40', 'pk80')   # also run on halves A and B
TAO_NB = (64, 128, 256, 512, 1024, 2048)         # tao_csr_nb<N> (128 is tao_csr), particle_bin_span 2
TAO_SPAN = ('span1', 'span4', 'span8', 'nb1024_span1', 'nb1024_span4', 'nb1024_span8')


def tao_file(nb):
    return 'tao_csr' if nb == 128 else f'tao_csr_nb{nb}'


def tao_dz(z, n_bin, span=2):
    """Bmad csr_bin_particles: dz_slice = (max z - min z) / (n_bin - 2 - (particle_bin_span + 1))"""
    return np.ptp(z) / (n_bin - 3 - span)
SECTIONS = (('BEGBC20 -> Q2ER exit', 'Q2ER'), ('B2RE', 'B2RE'),
            ('B2RE exit -> D1ET2B end', 'DM11Q_D1ET2B'), ('B1RE', 'B1RE'))


def scan_file(nb):
    return 'dfcsr_1d' if nb == 300 else f'dfcsr_1d_nb{nb}'


def section_loss(d, off, names, p0c):
    """mean CSR energy change [MeV] in each of SECTIONS, and the total"""
    cum = [0.0] + [(d[i][5].mean() - off[i][5].mean()) * p0c / 1e6
                   for i in (int(np.flatnonzero(names == nm)[0]) for _, nm in SECTIONS)]
    return np.diff(cum), cum[-1]


def scan_rows(p0c, data):
    """(label, file, mode, bin width at ENDBC20 [m]) of every 1D bin scan run"""
    zend = lambda f: data[f][-1][4]
    rows = [(f'extent {nb}', scan_file(nb), 'extent', bin_width(zend(scan_file(nb)), nb, 'extent'))
            for nb in NB_SCAN]
    rows += [(f'quantile {n}', f'dfcsr_1d_q{n}', 'quantile',
              bin_width(zend(f'dfcsr_1d_q{n}'), n, 'quantile', 0.05)) for n in Q_SCAN]
    rows += [(f'peak {m}/FWHM', f'dfcsr_1d_pk{m}', 'peak',
              bin_width(zend(f'dfcsr_1d_pk{m}'), 300, 'peak', bins_per_fwhm=m)) for m in PK_SCAN]
    return rows


def noise_corrected(v, full, off, names, p0c, data):
    """2 x (20k result) - mean of the two 10k halves: removes a shot-noise term that scales as 1/N.
    Returns (full, half A, half B, corrected), each (sections..., total)."""
    flat = lambda d, o: np.r_[section_loss(d, o, names, p0c)[0], section_loss(d, o, names, p0c)[1]]
    F = flat(data[full], off)
    A, B = (flat(data[f'dfcsr_1d_{v}_half{h}'], off[:, :, half_indices(h)]) for h in 'AB')
    return F, A, B, 2 * F - 0.5 * (A + B)


HALF_FULL = {'nb300': 'dfcsr_1d', 'nb1200': 'dfcsr_1d_nb1200', 'pk10': 'dfcsr_1d_pk10',
             'pk20': 'dfcsr_1d_pk20', 'pk40': 'dfcsr_1d_pk40', 'pk80': 'dfcsr_1d_pk80'}


def resolution_table(names, p0c, data):
    off = data['off']
    L = ['', 'z resolution, pyDFCSR 1D: mean CSR energy change [MeV] per section; bin width at ENDBC20',
         f'{"run":>22s} {"bin@end":>8s}' + ''.join(f'{t[:24]:>25s}' for t, _ in SECTIONS) +
         f'{"total":>8s} {"eps1":>7s}']
    rows = [(lab, f, ds * 1e6) for lab, f, _, ds in scan_rows(p0c, data)]
    rows += [('Tao 1D (n_bin 128)', 'tao_csr', None), ('2D (wake dz ~11 um)', 'dfcsr_2d', None)]
    for lab, f, bw in rows:
        sec, tot = section_loss(data[f], off, names, p0c)
        e1 = emit_n(data[f][-1], p0c, 1) * 1e6
        L.append(f'{lab:>22s} {(f"{bw:.2f}" if bw else ""):>8s}' + ''.join(f'{v:25.3f}' for v in sec) +
                 f'{tot:8.3f} {e1:7.2f}')
    L += ['', 'Tao 1D CSR, binning scan (Bmad bins the full z extent: dz = (max z - min z) / (n_bin - 3 - particle_bin_span),'
          ' particles are triangles of base particle_bin_span * dz)',
          f'{"run":>22s} {"dz@end":>8s}' + ''.join(f'{t[:24]:>25s}' for t, _ in SECTIONS) + f'{"total":>8s}']
    for lab, f, nb, sp in ([(f'n_bin {nb}, span 2', tao_file(nb), nb, 2) for nb in TAO_NB] +
                           [(f'n_bin {v.split("_")[0][2:] if "nb" in v else 128}, span {v[-1]}', f'tao_csr_{v}',
                             1024 if 'nb' in v else 128, int(v[-1])) for v in TAO_SPAN]):
        sec, tot = section_loss(data[f], off, names, p0c)
        L.append(f'{lab:>22s} {tao_dz(data[f][-1][4], nb, sp) * 1e6:8.2f}' +
                 ''.join(f'{v:25.3f}' for v in sec) + f'{tot:8.3f}')
    L += ['', 'shot noise: the same runs on two complementary random halves (10 000 particles, same '
          'charge); corrected = 2 full - mean(halves)',
          f'{"run":>8s} {"beam":>6s}' + ''.join(f'{t[:24]:>25s}' for t, _ in SECTIONS) + f'{"total":>8s}']
    for v, full in HALF_FULL.items():
        for lab, r in zip(('20k', 'A 10k', 'B 10k', 'corr'), noise_corrected(v, full, off, names, p0c, data)):
            L.append(f'{v:>8s} {lab:>6s}' + ''.join(f'{x:25.3f}' for x in r[:-1]) + f'{r[-1]:8.3f}')
    return L


def plot_resolution(names, p0c, data):
    off = data['off']
    rows = scan_rows(p0c, data)
    style_of = {'extent': ('bin_mode extent (nbins scan)', '#eb6834', 'o'),
                'quantile': ('bin_mode quantile, q = 0.05', '#008300', 's'),
                'peak': ('bin_mode peak (bins per FWHM scan)', '#e87ba4', 'D')}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=SURF)
    for col, ax in enumerate(axes):
        pick = (lambda sec, tot: tot) if col == 0 else (lambda sec, tot: sec[3])
        for mode, (lab, c, mk) in style_of.items():
            r = [(ds, pick(*section_loss(data[f], off, names, p0c))) for _, f, m, ds in rows if m == mode]
            r.sort()
            ax.plot([x[0] * 1e6 for x in r], [x[1] for x in r], color=c, lw=2, marker=mk, ms=7, label=lab)
        pk = [(bin_width(data[HALF_FULL[v]][-1][4], 300, 'peak', bins_per_fwhm=int(v[2:])),
               noise_corrected(v, HALF_FULL[v], off, names, p0c, data)[3]) for v in HALF if v.startswith('pk')]
        pk.sort(key=lambda x: x[0])
        ax.plot([x[0] * 1e6 for x in pk], [x[1][-1] if col == 0 else x[1][3] for x in pk], color='#e87ba4',
                lw=1.5, ls='--', marker='D', ms=7, mfc=SURF, label='peak, shot noise removed (2·20k − 10k)')
        tz = [(tao_dz(data[tao_file(nb)][-1][4], nb), pick(*section_loss(data[tao_file(nb)], off, names, p0c)))
              for nb in TAO_NB]
        ax.plot([x[0] * 1e6 for x in tz], [x[1] for x in tz], color='#2a78d6', lw=2, marker='^', ms=8,
                label='Tao 1D CSR (n_bin scan, particle_bin_span 2)')
        ax.axhline(pick(*section_loss(data['dfcsr_2d'], off, names, p0c)), color='#1baf7a', lw=2,
                   label='pyDFCSR 2D (wake mesh dz ≈ 11 µm)')
        if 'dfcsr_2d_q' in data:
            ax.axhline(pick(*section_loss(data['dfcsr_2d_q'], off, names, p0c)), color='#4a3aa7',
                       lw=2, ls='--', label='pyDFCSR 2D quantile frame (0.26 µm resolution)')
        ax.set_xscale('log')
        ax.set_xlabel('1D bin width at ENDBC20 [µm]  (Tao: dz_slice)')
        ax.set_ylabel('CSR mean energy change [MeV]')
        ax.set_title('total through BC20' if col == 0 else 'in B1RE (the last bend)', fontsize=10,
                     loc='left', color=INK)
        style(ax)
    axes[1].legend(fontsize=8, frameon=False, loc='upper left')
    fig.tight_layout()
    fig.savefig(os.path.join(RES, 'csr_resolution_1d.png'), dpi=140, facecolor=SURF)
    plt.close(fig)


def tables(s, names, p0c, data, init, runtime):
    off = data['off']
    keys = ['off'] + [f for _, f, _, _ in SERIES]
    kc = [f for _, f, _, _ in SER_CSR]
    lab = {'off': 'CSR off', **{f: l for l, f, _, _ in SERIES}}
    L = ['BC20 with CSR, 20 000 particles, 1.6 nC, P0C = %.2f MeV' % (p0c / 1e6), '',
         'CSR effect = run - CSR off, particle by particle. dE in MeV; eps_nx in um, dispersion-'
         'corrected to 1st / 3rd order in delta.', '']
    st0 = stats(init, init, p0c)
    L.append(f'at BEGBC20: sigma_z {st0["sz"]*1e6:.2f} um, I_pk {st0["Ipk"]/1e3:.2f} kA, '
             f'sigma_delta {st0["sd"]*100:.4f} %, eps_nx {st0["en1"]*1e6:.3f} / {st0["en3"]*1e6:.3f} um')
    L += ['', 'at ENDBC20:',
          f'{"run":>44s} {"sigma_z":>8s} {"I_pk":>6s} {"<dE>":>7s} {"rms dE":>7s} {"sig_dlt":>8s} '
          f'{"eps1":>7s} {"eps3":>7s} {"sigma_x":>8s} {"runtime":>9s}',
          f'{"":>44s} {"[um]":>8s} {"[kA]":>6s} {"[MeV]":>7s} {"[MeV]":>7s} {"[%]":>8s} '
          f'{"[um]":>7s} {"[um]":>7s} {"[um]":>8s} {"[min]":>9s}']
    for k in keys:
        t = stats(data[k][-1], off[-1], p0c)
        rt = f'{runtime[k]/60:9.2f}' if k in runtime else f'{"":>9s}'
        L.append(f'{lab[k]:>44s} {t["sz"]*1e6:8.3f} {t["Ipk"]/1e3:6.1f} {t["dE"]/1e6:7.3f} '
                 f'{t["sdE"]/1e6:7.3f} {t["sd"]*100:8.4f} {t["en1"]*1e6:7.3f} {t["en3"]*1e6:7.3f} '
                 f'{t["sx"]*1e6:8.2f} {rt}')
    L += ['', 'mean CSR energy change <dE> [MeV] gained in each part of the line (negative = loss):']
    parts = [('BEGBC20 -> B2LE exit', 'B2LE'), ('-> B3RE exit (B3 pair)', 'B3RE'),
             ('-> Q2ER exit', 'Q2ER'), ('B2RE', 'B2RE'), ('B2RE exit -> D1ET2B end (drift, wiggler)',
                                                        'DM11Q_D1ET2B'), ('B1RE', 'B1RE')]
    hdr = f'{"":>42s}' + ''.join(f'{lab[k][:12]:>13s}' for k in kc)
    L.append(hdr)
    cum = lambda k, i: (data[k][i][5].mean() - off[i][5].mean()) * p0c / 1e6 if i >= 0 else 0.0
    prev = -1
    for txt, nm in parts:
        i = int(np.flatnonzero(names == nm)[0])
        L.append(f'{txt:>42s}' + ''.join(f'{cum(k, i) - cum(k, prev):13.3f}' for k in kc))
        prev = i
    L += ['', 'along the line, at element exits (bends and the last elements of each section):',
          f'{"element":>14s} {"s":>6s} {"I_pk off":>8s} | {"<dE> [MeV]":^31s} | '
          f'{"eps_nx 3rd order [um]":^38s}',
          f'{"":>14s} {"[m]":>6s} {"[kA]":>8s} | ' + ' '.join(f'{lab[k][:7]:>7s}' for k in kc) +
          ' | ' + ' '.join(f'{lab[k][:7]:>7s}' for k in ['off'] + kc)]
    for i, nm in enumerate(names):
        if not nm.startswith(('B', 'Q2', 'D5', 'Q5', 'WIGE3', 'Q1ER', 'DM11')):
            continue
        dE = [(data[k][i][5].mean() - off[i][5].mean()) * p0c / 1e6 for k in kc]
        e3 = [emit_n(data[k][i], p0c, 3) * 1e6 for k in ['off'] + kc]
        L.append(f'{nm:>14s} {s[i]:6.2f} {peak_current(off[i][4])[0]/1e3:8.1f} | ' +
                 ' '.join(f'{v:7.3f}' for v in dE) + ' | ' + ' '.join(f'{v:7.2f}' for v in e3))
    L += resolution_table(names, p0c, data)
    txt = '\n'.join(L)
    print(txt)
    with open(os.path.join(HERE, 'compare_csr_runs.txt'), 'w') as f:
        f.write(txt + '\n')


def style(ax):
    ax.set_facecolor(SURF)
    ax.grid(color=GRID, lw=0.6)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)


def shade(ax, s, types):
    s0 = np.concatenate([[0.0], s[:-1]])
    for a, b, t in zip(s0, s, types):
        if t in ('dipole', 'vbend'):
            ax.axvspan(a, b, color='#d9d8d3' if t == 'dipole' else '#ecebe7', lw=0, zorder=0)


def plot_along_s(s, names, types, p0c, data):
    off = data['off']
    n = len(s)
    fig, axes = plt.subplots(5, 1, figsize=(10, 11), sharex=True, facecolor=SURF)
    Ipk = np.array([peak_current(off[i][4])[0] for i in range(n)]) / 1e3
    axes[0].plot(s, Ipk, color=INK2, lw=2, marker='o', ms=3)
    axes[0].set_ylabel('peak current,\nCSR off [kA]')
    for _, f, c, ls in SERIES:
        d = data[f]
        dE = np.array([(d[i][5] - off[i][5]).mean() for i in range(n)]) * p0c / 1e6
        sdE = np.array([(d[i][5] - off[i][5]).std() for i in range(n)]) * p0c / 1e6
        dsz = np.array([d[i][4].std() - off[i][4].std() for i in range(n)]) * 1e6
        e1 = np.array([emit_n(d[i], p0c, 1) for i in range(n)]) * 1e6
        for ax, y in zip(axes[1:], (dE, sdE, dsz, e1)):
            ax.plot(s, y, color=c, ls=ls, lw=2)
    e1off = np.array([emit_n(off[i], p0c, 1) for i in range(n)]) * 1e6
    axes[4].plot(s, e1off, color=INK2, lw=1.2, ls='-', label='CSR off')
    axes[1].set_ylabel('CSR mean energy\nchange [MeV]')
    axes[2].set_ylabel('CSR-induced rms\nenergy spread [MeV]')
    axes[3].set_ylabel('σz − σz,off [µm]')
    axes[4].set_ylabel('εnx, dispersion-\ncorrected [µm]')
    axes[4].set_yscale('log')
    axes[4].set_xlabel('s from BEGBC20 [m]')
    for ax in axes:
        style(ax)
        shade(ax, s, types)
    for (lab, f, c, ls) in SERIES:
        axes[1].plot([], [], color=c, ls=ls, lw=2, label=lab)
    axes[1].plot([], [], color=INK2, lw=1.2, label='CSR off (εnx panel)')
    axes[1].legend(loc='lower left', fontsize=8.5, frameon=False)
    s0 = np.concatenate([[0.0], s[:-1]])
    for nm, dy in (('B1LE', 0), ('B2LE', 0), ('B3LE', 0), ('B3RE', -11), ('B2RE', 0), ('B1RE', 0)):
        i = int(np.flatnonzero(names == nm)[0])
        axes[0].annotate(nm, (0.5 * (s0[i] + s[i]), axes[0].get_ylim()[1]), xytext=(0, -10 + dy),
                         textcoords='offset points', fontsize=8, color=INK2, ha='center')
    axes[0].set_title('FACET-II BC20 with CSR: element exits (dark bands: bends, light: vertical '
                      'wiggler)', fontsize=10, color=INK, loc='left')
    fig.tight_layout()
    fig.savefig(os.path.join(RES, 'csr_along_s.png'), dpi=140, facecolor=SURF)
    plt.close(fig)


def binned(z, y, edges):
    k = np.digitize(z, edges) - 1
    ok = (k >= 0) & (k < len(edges) - 1)
    n = np.bincount(k[ok], minlength=len(edges) - 1)
    m = np.bincount(k[ok], weights=y[ok], minlength=len(edges) - 1)
    with np.errstate(invalid='ignore'):
        return np.where(n >= 20, m / np.maximum(n, 1), np.nan)


def plot_at_end(p0c, data):
    off = data['off'][-1]
    zc = off[4].mean()
    fig, axes = plt.subplots(3, 2, figsize=(11, 8.4), sharex='col', facecolor=SURF,
                             gridspec_kw=dict(width_ratios=(1.6, 1)))
    for col, (zlo, zhi, dz) in enumerate(((-130e-6, 90e-6, 2e-6), (-12e-6, 12e-6, 0.5e-6))):
        # zoom window centred on the spike of the CSR-off beam
        h, e = np.histogram(off[4], bins=np.arange(off[4].min(), off[4].max() + 0.5e-6, 0.5e-6))
        zpk = e[np.argmax(gaussian_filter1d(h.astype(float), 1.0))] if col == 1 else zc
        edges = zpk + np.arange(zlo, zhi + dz, dz)
        mid = 0.5 * (edges[1:] + edges[:-1])
        ax = axes[0, col]
        for lab, f, c, ls in [('CSR off', 'off', INK2, '-')] + SER_CSR:
            hh, _ = np.histogram(data[f][-1][4], bins=edges)
            ax.plot((mid - zc) * 1e6, hh * Q / len(off[4]) / (dz / C0) / 1e3, color=c, ls=ls, lw=1.6,
                    label=lab)
        ax.set_ylabel('current [kA]')
        for lab, f, c, ls in SER_CSR:
            dE = (data[f][-1][5] - off[5]) * p0c / 1e6
            dpx = (data[f][-1][1] - off[1]) * 1e6
            axes[1, col].plot((mid - zc) * 1e6, binned(off[4], dE, edges), color=c, ls=ls, lw=1.6)
            axes[2, col].plot((mid - zc) * 1e6, binned(off[4], dpx, edges), color=c, ls=ls, lw=1.6)
        axes[1, col].set_ylabel('CSR energy change\n(bin mean) [MeV]')
        axes[2, col].set_ylabel('CSR change of px\n(bin mean) [µrad]')
        axes[2, col].set_xlabel('z − ⟨z⟩ of the CSR-off particle [µm]  (head at +z)')
        for ax in axes[:, col]:
            style(ax)
    axes[0, 0].legend(fontsize=8.5, frameon=False, loc='upper left')
    axes[0, 0].set_title('at ENDBC20, whole bunch (2 µm bins)', fontsize=10, loc='left', color=INK)
    axes[0, 1].set_title('the current spike (0.5 µm bins)', fontsize=10, loc='left', color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, 'csr_at_end.png'), dpi=140, facecolor=SURF)
    plt.close(fig)


def main():
    s, names, types, p0c, data, init, runtime = load()
    tables(s, names, p0c, data, init, runtime)
    plt.rcParams.update({'font.size': 10, 'axes.edgecolor': INK2, 'axes.labelcolor': INK,
                         'xtick.color': INK2, 'ytick.color': INK2, 'text.color': INK})
    plot_along_s(s, names, types, p0c, data)
    plot_at_end(p0c, data)
    plot_resolution(names, p0c, data)


if __name__ == '__main__':
    main()
