"""
The benchmark chicane with 2D CSR (CSR2D) and 1D CSR (CSR1D), both in pyDFCSR, on the SAME bmad-x
tracking, beam (chicane_init_beam.yaml, 1e6 particles) and lattice. With CSR off the two classes give
bit-identical particles (test_csr1d.py), so ONE CSR-off run is the reference for every CSR-on run.

Runs (name: class, schedule, CSR settings):
  off        CSR1D  auto (= the 2D schedule: 489 nodes, 105 midpoint kicks)   CSR off   (shared reference)
  2d         CSR2D  auto                                                      drift_cutoff 1.0 (default)
  2d_noT     CSR2D  auto                                                      drift_cutoff 1.0, transverse off
  1d         CSR1D  auto                                                      drift_cutoff 1.0  (like-for-like)
  1d_full    CSR1D  auto                                                      no drift cutoff
  1d_fine    CSR1D  manual: 1 mm in bends, 5 mm in drifts, kick every step    drift_cutoff 1.0
  1d_fine_full  as 1d_fine                                                    no drift cutoff
The 2D run with drift_cutoff = 0 is NOT rerun: its far-drift wake is the known failure mode
(progress.md, top TODO), and the 2D code has not changed since chicane_1d2d/ recorded it.

Usage:
  python chicane_2d_vs_1d.py run <name>     # one run -> benchmark_results/chicane_2d_vs_1d/<name>.npz
  python chicane_2d_vs_1d.py analyze        # compare all available runs -> summary.txt + plots
"""
import contextlib
import glob
import io
import os
import sys
import time

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
INP = os.path.join(HERE, '..', 'example', 'input')
OUT = os.path.join(HERE, 'benchmark_results', 'chicane_2d_vs_1d')
MC2 = 510998.95
BENDS = [(0.1, 0.6002, 'B1'), (5.606, 6.1062, 'B2'), (7.1062, 7.6064, 'B3'), (12.6122, 13.1124, 'B4')]
EDGES = [0.0, 0.1, 0.6002, 5.606, 6.1062, 7.1062, 7.6064, 12.6122, 13.1124, 13.3124]
ENAMES = ['D0', 'B1', 'D1', 'B2', 'D2', 'B3', 'D3', 'B4', 'Df']

# the 2D configuration of chicane_1d2d/ (ts_chiccut_config.yaml), with absolute paths
BASE = {
    'input_beam': {'style': 'distgen', 'distgen_input_file': os.path.join(INP, 'chicane_init_beam.yaml')},
    'input_lattice': {'lattice_input_file': os.path.join(INP, 'chicane_lattice.yaml')},
    'particle_deposition': {'method': 'bspline_comoving', 'xbins': 300, 'zbins': 300, 'xlim': 5, 'zlim': 5,
                            'smoothing_sigma': 3.0, 'poly_degree': 1, 'velocity_threhold': 1000},
    'CSR_integration': {'n_formation_length': 1, 'xbins': 200, 'zbins': 200, 'drift_cutoff': 1.0},
    'CSR_computation': {'compute_CSR': 1, 'apply_CSR': 1, 'transverse_on': 1, 'xbins': 12, 'zbins': 40,
                        'xlim': 4, 'zlim': 4, 'write_beam': None, 'write_wakes': True},
    'step_control': {'mode': 'auto', 'kick_interval': 'midpoint', 'n_sub': 400},
}
RUNS = {
    'off': dict(cls='1d', csr=False),
    '2d': dict(cls='2d'),
    '2d_noT': dict(cls='2d', transverse=0),
    '1d': dict(cls='1d', cutoff=1.0),
    '1d_full': dict(cls='1d', cutoff=0.0),
    '1d_fine': dict(cls='1d', cutoff=1.0, fine=True),
    '1d_fine_full': dict(cls='1d', cutoff=0.0, fine=True),
}
LABEL = {'2d': '2D (cutoff 1 Rφ)', '2d_noT': '2D, no transverse force', '1d': '1D, same schedule (cutoff 1 Rφ)',
         '1d_full': '1D, same schedule, full drift', '1d_fine': '1D, fine steps (cutoff 1 Rφ)',
         '1d_fine_full': '1D, fine steps, full drift'}


def config(name):
    r = RUNS[name]
    cfg = yaml.safe_load(yaml.safe_dump(BASE))
    cfg['CSR_computation'].update(workdir=os.path.join(OUT, 'runs'), write_name=name)
    if r.get('csr') is False:
        cfg['CSR_computation'].update(compute_CSR=0, apply_CSR=0, write_wakes=False)
    if 'transverse' in r:
        cfg['CSR_computation']['transverse_on'] = r['transverse']
    if r['cls'] == '1d':
        cfg['CSR_1D'] = {'nbins': 300, 'nsmooth': 2, 'drift_cutoff': r.get('cutoff', 0.0)}
    if r.get('fine'):
        lat = yaml.safe_load(open(BASE['input_lattice']['lattice_input_file']))
        for k, v in lat.items():
            if k != 'step_size':
                v['steps'] = max(1, int(round(v['L'] / (0.001 if v['type'] == 'dipole' else 0.005))))
        lf = os.path.join(OUT, 'inputs', 'chicane_lattice_fine.yaml')
        yaml.safe_dump(lat, open(lf, 'w'), sort_keys=False)
        cfg['input_lattice']['lattice_input_file'] = lf
        cfg['step_control'] = {'mode': 'manual', 'default_kick_every': 1, 'kick_interval': 'midpoint'}
    f = os.path.join(OUT, 'inputs', f'{name}.yaml')
    yaml.safe_dump(cfg, open(f, 'w'))
    return f


def run(name):
    os.makedirs(os.path.join(OUT, 'runs'), exist_ok=True); os.makedirs(os.path.join(OUT, 'inputs'), exist_ok=True)
    f = config(name)
    if RUNS[name]['cls'] == '2d':
        from pyDFCSR_2D import CSR2D as Cls
    else:
        from pyDFCSR_2D.csr1d import CSR1D as Cls
    t0 = time.time()
    sim = Cls(input_file=f)
    sim.run()
    p = sim.beam.particle
    st = {k: v for k, v in sim.statistics.items() if not isinstance(v, dict)}
    tw = {f'twiss_{k}': v for k, v in sim.statistics['twiss'].items()}
    np.savez(os.path.join(OUT, f'{name}.npz'), x=p.x, px=p.px, z=p.z, pz=p.pz, E0=sim.beam.init_energy,
             s=np.array(sim.lattice.steps_record), runtime=time.time() - t0, prefix=sim.prefix, **st, **tw)
    print(f'{name}: done in {(time.time() - t0) / 60:.1f} min', flush=True)


# ---------------------------------------------------------------------------------- analysis
def emit(x, xp, d, G):
    x = x - np.polyval(np.polyfit(d, x, 1), d); xp = xp - np.polyval(np.polyfit(d, xp, 1), d)
    return np.sqrt(max(np.linalg.det(np.cov(x, xp)), 0)) * G


def slices(z, d, ns=80, sz=None):
    zc = z - z.mean(); sz = zc.std() if sz is None else sz
    ed = np.linspace(-3 * sz, 3 * sz, ns + 1); idx = np.digitize(zc, ed) - 1
    return zc, sz, ed, idx


def metrics(b, off, E0):
    G = E0 / MC2
    zc, sz, ed, idx = slices(b['z'], b['pz'])
    zo, _, _, io_ = slices(off['z'], off['pz'], sz=sz)
    v, w, se = [], [], []
    for i in range(len(ed) - 1):
        c = 0.5 * (ed[i] + ed[i + 1])
        m, mo = idx == i, io_ == i
        if abs(c) > 2 * sz or m.sum() < 500 or mo.sum() < 500:
            continue
        r = lambda zz, e: np.var(e - np.polyval(np.polyfit(zz, e, 2), zz))
        v.append(r(zc[m], b['pz'][m] * E0) - r(zo[mo], off['pz'][mo] * E0)); w.append(m.sum())
        se.append(emit(b['x'][m], b['px'][m], b['pz'][m], G))
    a = np.average(v, weights=w)
    h, e = np.histogram(zc, 400)
    return dict(dE=(b['pz'].mean() - off['pz'].mean()) * E0 / 1e3,
                dsEp=(b['pz'].std() - off['pz'].std()) * E0 / 1e3,
                sEs=np.sign(a) * np.sqrt(abs(a)) / 1e3,
                emit=emit(b['x'], b['px'], b['pz'], G) * 1e6,
                demit=(emit(b['x'], b['px'], b['pz'], G) - emit(off['x'], off['px'], off['pz'], G)) * 1e6,
                semit=np.average(se, weights=w) * 1e6,
                sz=sz * 1e6, Ipk=h.max() / np.diff(e)[0] * 1e-9 / len(zc) * 2.998e8 / 1e3)


def per_element_loss(b):
    s, mE = b['s'], b['mean_energy']
    at = lambda x: np.interp(x, s, mE)
    return np.array([(at(EDGES[i + 1]) - at(EDGES[i])) / 1e3 for i in range(len(ENAMES))])


def wakes2d(prefix):
    import h5py
    f = h5py.File(glob.glob(os.path.join(OUT, 'runs', f'{prefix}-wakes.h5'))[0])
    u = np.linspace(-4, 4, 12); wx = np.exp(-u ** 2 / 2); wx /= wx.sum()
    out = {}
    for k in f:
        g = f[k]; W = g['longitudinal/dE_dct'][()] * 1e6; Z = g['longitudinal/z_grids'][()]
        out[round(float(g.attrs['position']), 6)] = dict(z=Z[0], W=W, Wavg=(wx[:, None] * W).sum(0))
    return out


def wakes1d(prefix):
    import h5py
    f = h5py.File(os.path.join(OUT, 'runs', f'{prefix}-wakes1d.h5'))
    return {round(float(f[k].attrs['position']), 6): dict(z=f[k]['longitudinal_1d/z_grid'][()],
                                                          W=f[k]['longitudinal_1d/dE_dct'][()],
                                                          lam=f[k]['longitudinal_1d/lam'][()]) for k in f}


def analyze():
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    R = {n: dict(np.load(os.path.join(OUT, f'{n}.npz'))) for n in RUNS if os.path.isfile(os.path.join(OUT, f'{n}.npz'))}
    off = R['off']; E0 = float(off['E0'])
    L = [__doc__.split('Usage:')[0].strip(), '', f"runs available: {', '.join(R)}"]
    # 0. shared reference: identical tracking
    for n in ('2d', '1d'):
        pass
    prev = sorted(glob.glob(os.path.join(HERE, 'benchmark_results', 'chicane_1d2d', '2d_chic_off.npz')))
    if prev:
        q = np.load(prev[0])
        L.append(f"CSR-off reference (CSR1D) vs the earlier CSR2D CSR-off run: z bit-identical = "
                 f"{np.array_equal(np.sort(q['zf']), np.sort(off['z']))}, pz bit-identical = {np.array_equal(np.sort(q['df']), np.sort(off['pz']))}")
    M = {n: metrics(r, off, E0) for n, r in R.items() if n != 'off'}
    rows = [('dE', 'mean CSR energy change [keV]'), ('dsEp', 'projected sigma_E change [keV]'),
            ('sEs', 'CSR-added core slice sigma_E [keV]'), ('demit', 'emittance growth, eta-corr. [um]'),
            ('semit', 'core slice emittance [um]'), ('sz', 'final sigma_z [um]'), ('Ipk', 'peak current [kA]')]
    names = [n for n in LABEL if n in M]
    Moff = metrics(off, off, E0)
    L += ['', f"{'chicane exit, vs CSR off':38s}" + ''.join(f'{n:>14s}' for n in names) + f"{'CSR off':>12s}"]
    for k, lab in rows:
        L.append(f'{lab:38s}' + ''.join(f'{M[n][k]:14.3f}' for n in names) + (f'{Moff[k]:12.3f}' if k in ('semit', 'sz', 'Ipk') else ''))
    L.append(f"{'runtime [min]':38s}" + ''.join(f"{float(R[n]['runtime']) / 60:14.1f}" for n in names))
    # per element energy loss
    L += ['', 'mean CSR energy change collected in each element [keV] (from mean_energy along s, minus CSR off):']
    eoff = per_element_loss(off)
    L.append(f"{'':16s}" + ''.join(f'{e:>9s}' for e in ENAMES))
    for n in names:
        L.append(f'{n:16s}' + ''.join(f'{v:9.1f}' for v in per_element_loss(R[n]) - eoff))
    # wakes at matched kicks: 1D vs the x-average of the 2D mesh, same schedule
    if '2d' in R and '1d' in R:
        w2, w1 = wakes2d(str(R['2d']['prefix'])), wakes1d(str(R['1d']['prefix']))
        common = sorted(set(w2) & set(w1))
        L += ['', f'wakes at the {len(common)} common kicks (2D: Gaussian-weighted average over the 12 x-rows):']
        by = {}
        for p in common:
            a, b = w2[p], w1[p]
            zc = np.sum(b['z'] * b['lam']) / np.sum(b['lam'])
            g = np.linspace(a['z'].min(), a['z'].max(), 200)
            W1 = np.interp(g, b['z'], b['W']); W2 = np.interp(g, a['z'], a['Wavg'])
            sp = np.sqrt(np.mean(np.var(a['W'], axis=0)))                 # 2D spread over x, rms over z
            el = next((e for lo, hi, e in BENDS if lo <= p <= hi + 1e-9), 'drift')
            by.setdefault(el, []).append((np.linalg.norm(W2 - W1) / np.linalg.norm(W1), np.abs(W2).max() / np.abs(W1).max(),
                                          sp / np.abs(W1).max()))
        for el, v in by.items():
            v = np.array(v)
            L.append(f'  {el:6s} {len(v):3d} kicks: rel L2(2D avg - 1D) median {np.median(v[:, 0]):.3f}; peak 2D/1D median {np.median(v[:, 1]):.3f}; '
                     f'2D x-spread / 1D peak median {np.median(v[:, 2]):.3f}')
    open(os.path.join(OUT, 'summary.txt'), 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))

    # ---- plots
    cols = {'2d': 'C3', '2d_noT': 'C1', '1d': 'C0', '1d_full': 'C2', '1d_fine': 'C9', '1d_fine_full': 'C4'}
    fig, ax = plt.subplots(2, 2, figsize=(14, 8.5), layout='constrained'); ax = ax.ravel()
    for n in ['off'] + names:
        r = R[n]; c = cols.get(n, 'k'); ls = ':' if n == 'off' else '-'; lab = LABEL.get(n, 'CSR off')
        ax[0].plot(r['s'], r['sigma_z'] * 1e6, ls, c=c, label=lab)
        ax[1].plot(r['s'], (r['mean_energy'] - off['mean_energy'][0]) / 1e3 if n != 'off' else 0 * r['s'], ls, c=c, label=lab)
        ax[2].plot(r['s'], r['twiss_norm_emit_x'] * 1e6, ls, c=c, label=lab)
        ax[3].plot(r['s'], r['sigma_energy'] / 1e6, ls, c=c, label=lab)
    for a, yl, t in zip(ax, ['σz [µm]', 'mean CSR energy change [keV]', 'εn,x, η-corrected [µm]', 'σE [MeV]'],
                        ['bunch length', 'energy loss', 'emittance', 'energy spread']):
        [a.axvspan(lo, hi, color='0.92', zorder=-1) for lo, hi, _ in BENDS]; a.set(xlabel='s [m]', ylabel=yl, title=t)
    ax[0].set_yscale('log'); ax[1].legend(fontsize=8)
    plt.savefig(os.path.join(OUT, 'along_s.png'), dpi=110); plt.close()

    if '2d' in R and '1d' in R:
        pick = []
        for lo, hi, e in BENDS:
            for x in (lo + 0.5 * (hi - lo), hi):
                pick.append(min(common, key=lambda p: abs(p - x)))
        pick += [min(common, key=lambda p: abs(p - x)) for x in (1.0, 6.4, 7.9, 13.25)]
        pick = sorted(set(pick))
        nc = 4; nr = int(np.ceil(len(pick) / nc))
        fig, ax = plt.subplots(nr, nc, figsize=(4.2 * nc, 3.2 * nr), layout='constrained'); ax = ax.ravel()
        for a, p in zip(ax, pick):
            A, Bw = w2[p], w1[p]
            zc = np.sum(Bw['z'] * Bw['lam']) / np.sum(Bw['lam'])
            for i, c in zip(range(A['W'].shape[0]), plt.cm.viridis(np.linspace(0, 1, A['W'].shape[0]))):
                a.plot((A['z'] - zc) * 1e6, A['W'][i] / 1e6, c=c, lw=0.6, alpha=0.7)
            a.plot((A['z'] - zc) * 1e6, A['Wavg'] / 1e6, c='C3', lw=2, label='2D, x-average')
            a.plot((Bw['z'] - zc) * 1e6, Bw['W'] / 1e6, 'k--', lw=1.6, label='1D')
            a.set_xlim((A['z'].min() - zc) * 1e6, (A['z'].max() - zc) * 1e6)
            el = next((e for lo, hi, e in BENDS if lo <= p <= hi + 1e-9), 'drift')
            a.set_title(f's = {p:.3f} m ({el})', fontsize=9); a.set_xlabel('z − ⟨z⟩ [µm]', fontsize=8); a.set_ylabel('MeV/m', fontsize=8)
        ax[0].legend(fontsize=7)
        for a in ax[len(pick):]: a.axis('off')
        fig.suptitle('Longitudinal wake at matched kicks: 2D x-rows (colour, −4σ → +4σ), 2D x-average (red), 1D (black dashed)')
        plt.savefig(os.path.join(OUT, 'wakes_matched.png'), dpi=100); plt.close()

    fig, ax = plt.subplots(1, 3, figsize=(17, 4.5), layout='constrained')
    for n in ['off'] + names:
        r = R[n]; c = cols.get(n, 'k'); ls = ':' if n == 'off' else '-'
        zc, sz, ed, idx = slices(r['z'], r['pz'])
        ctr, se, me = [], [], []
        for i in range(len(ed) - 1):
            m = idx == i
            if m.sum() < 500: continue
            e = r['pz'][m] * E0; zz = zc[m]
            ctr.append(0.5 * (ed[i] + ed[i + 1]) * 1e6); me.append(e.mean() / 1e6)
            se.append(np.std(e - np.polyval(np.polyfit(zz, e, 2), zz)) / 1e3)
        ax[0].plot(ctr, se, ls, c=c, label=LABEL.get(n, 'CSR off'))
        ax[1].plot(ctr, me, ls, c=c)
    ax[0].set(xlabel='z − ⟨z⟩ [µm]', ylabel='slice σE, quadratic chirp removed [keV]', title='final slice energy spread'); ax[0].legend(fontsize=7)
    ax[1].set(xlabel='z − ⟨z⟩ [µm]', ylabel='slice mean ΔE [MeV]', title='final slice energy (chirp)')
    x = np.arange(len(ENAMES)); wdt = 0.8 / max(len(names), 1)
    for i, n in enumerate(names):
        ax[2].bar(x + i * wdt, per_element_loss(R[n]) - eoff, wdt, color=cols[n], label=LABEL[n])
    ax[2].set_xticks(x + 0.4 - wdt / 2, ENAMES); ax[2].set(ylabel='CSR energy change [keV]', title='energy loss per element'); ax[2].legend(fontsize=7)
    plt.savefig(os.path.join(OUT, 'final_beam.png'), dpi=110); plt.close()


if __name__ == '__main__':
    if sys.argv[1] == 'run':
        run(sys.argv[2])
    else:
        analyze()
