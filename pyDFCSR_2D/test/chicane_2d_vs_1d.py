"""
The benchmark chicane with 2D CSR (CSR2D) and 1D CSR (CSR1D), both in pyDFCSR, on the SAME bmad-x
tracking, beam (chicane_init_beam.yaml, 1e6 particles) and lattice. With CSR off the two classes give
bit-identical particles (test_csr1d.py), so ONE CSR-off run is the reference for every CSR-on run.

Runs (name: class, schedule, CSR settings):
  off        CSR1D  auto (= the 2D schedule: 489 nodes, 105 midpoint kicks)   CSR off   (shared reference)
  2d         CSR2D  auto                                                      drift_cutoff 1.0 (default)
  2d_noT     CSR2D  auto                                                      drift_cutoff 1.0, transverse off
  2d_tb      CSR2D  auto   as 2d, + CSR_integration.tangent_bands             (integration_region.md 5a)
  2d_vx      CSR2D  auto   as 2d, + particle_deposition.vx_model affine
  2d_new     CSR2D  auto   as 2d, both fixes
  2d_new_noT CSR2D  auto   both fixes, transverse off
  1d         CSR1D  auto                                                      drift_cutoff 1.0  (like-for-like)
  1d_full    CSR1D  auto                                                      no drift cutoff
  1d_fine    CSR1D  manual: 1 mm in bends, 5 mm in drifts, kick every step    drift_cutoff 1.0
  1d_fine_full  as 1d_fine                                                    no drift cutoff
The 2D run with drift_cutoff = 0 is NOT rerun: its far-drift wake is the known failure mode
(progress.md, top TODO), and the 2D code has not changed since chicane_1d2d/ recorded it.

Usage:
  python chicane_2d_vs_1d.py run <name> [--shear S]   # one run -> <out>/<name>.npz
  python chicane_2d_vs_1d.py analyze [--shear S]      # compare all available runs -> summary.txt + plots
--shear S replaces the beam's z:pz shear (GeV/c/m; the reference is -180, i.e. h = -36 /m) and puts
everything in benchmark_results/chicane_2d_vs_1d/shear<S>/ instead of the base folder.
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
SHEAR = float(sys.argv[sys.argv.index('--shear') + 1]) if '--shear' in sys.argv else None
if SHEAR is not None:
    OUT = os.path.join(OUT, f'shear{SHEAR:g}')
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
    # CSR off on the fine schedule: the slice-sigma_E reference for the fine runs at their OWN nodes.
    # Interpolating the auto-schedule 'off' onto fine nodes is wrong where slice sigma_E rises fast
    # (10 -> 35 keV over the first 2 cm of B1) and showed up as a spurious -41% spike there.
    'off_fine': dict(cls='1d', csr=False, fine=True),
    '2d': dict(cls='2d'),
    '2d_noT': dict(cls='2d', transverse=0),
    # 2026-11-03 fixes (integration_region.md 5a; progress.md "How the v_x fix works"), cutoff 1.0 as '2d'
    '2d_tb': dict(cls='2d', tangent=True),
    '2d_vx': dict(cls='2d', vx='affine'),
    '2d_new': dict(cls='2d', tangent=True, vx='affine'),
    '2d_new_noT': dict(cls='2d', tangent=True, vx='affine', transverse=0),
    # B4 slice-sigma_E diagnosis: the 2D longitudinal wake replaced by its x-average (Gaussian-weighted over the
    # 12 rows, the same average as the 1D comparisons) before it is applied, everywhere or only from B4 on.
    # The transverse kick is left as computed. Separates the wake's x-dependence from its z-profile.
    '2d_new_xavg': dict(cls='2d', tangent=True, vx='affine', xavg=0.0),
    '2d_new_xavgB4': dict(cls='2d', tangent=True, vx='affine', xavg=12.6122),
    # D3 reach fix: no drift cutoff, the reach goes back to the last bend's entrance, drift history thinned
    '2d_reach': dict(cls='2d', tangent=True, vx='affine', cutoff2d=0.0, reach=True, thin=0.01),
    '2d_reach_noT': dict(cls='2d', tangent=True, vx='affine', cutoff2d=0.0, reach=True, thin=0.01, transverse=0),
    # same, with the far fine-window node cap (Integration_params.far_fine_max)
    '2d_reach_cap': dict(cls='2d', tangent=True, vx='affine', cutoff2d=0.0, reach=True, thin=0.01, cap=500),
    '2d_reach_cap_noT': dict(cls='2d', tangent=True, vx='affine', cutoff2d=0.0, reach=True, thin=0.01, cap=500, transverse=0),
    # the current 2D defaults (both fixes + drift_reach, thin 0.01, cap 500) with drift_cutoff kept at 1.0:
    # what an existing config now gets, against 2d_new (same but the old reach)
    '2d_def_cut1': dict(cls='2d', tangent=True, vx='affine', reach=True, thin=0.01, cap=500),
    # the +-3 sigma_z spikes: current defaults, no drift cutoff, deposition grid xlim 8 in u (xbins 300 -> 480 keeps
    # the cell; 300 halves the u resolution but measured the same on frozen B2)
    '2d_x8': dict(cls='2d', tangent=True, vx='affine', cutoff2d=0.0, reach=True, thin=0.01, cap=500, dep=dict(xlim=8, xbins=480)),
    '2d_x8c': dict(cls='2d', tangent=True, vx='affine', cutoff2d=0.0, reach=True, thin=0.01, cap=500, dep=dict(xlim=8, xbins=300)),
    # timing pair, run one after the other on an idle machine (the four runs above shared it with 3 others)
    '2d_time_old': dict(cls='2d'),
    '2d_time_new': dict(cls='2d', tangent=True, vx='affine'),
    '1d': dict(cls='1d', cutoff=1.0),
    '1d_full': dict(cls='1d', cutoff=0.0),
    # drift-cutoff calibration in 1D (units of the last bend's |R*phi|; D3 is 10.0 R*phi long)
    '1d_c0.25': dict(cls='1d', cutoff=0.25), '1d_c0.5': dict(cls='1d', cutoff=0.5), '1d_c2': dict(cls='1d', cutoff=2.0),
    '1d_c4': dict(cls='1d', cutoff=4.0), '1d_c6': dict(cls='1d', cutoff=6.0),
    '1d_fine': dict(cls='1d', cutoff=1.0, fine=True),
    '1d_fine_full': dict(cls='1d', cutoff=0.0, fine=True),
}
LABEL = {'2d': '2D (cutoff 1 Rφ)', '2d_noT': '2D, no transverse force',
         '2d_tb': '2D + tangent_bands', '2d_vx': '2D + vx_model affine', '2d_new': '2D, both fixes',
         '2d_new_noT': '2D, both fixes, no transverse force',
         '2d_x8': '2D, defaults, no cutoff, deposition xlim 8', '2d_x8c': '2D, defaults, no cutoff, xlim 8 xbins 300',
         '2d_reach_cap': '2D, reach fix + node cap, no cutoff', '2d_reach_cap_noT': '2D, reach fix + cap, no cutoff, no transverse',
         '2d_reach': '2D, reach fix, no drift cutoff', '2d_reach_noT': '2D, reach fix, no cutoff, no transverse force',
         '2d_new_xavg': '2D, both fixes, x-averaged wake', '2d_new_xavgB4': '2D, both fixes, x-averaged wake in B4+', '1d': '1D, same schedule (cutoff 1 Rφ)',
         '1d_full': '1D, same schedule, full drift', '1d_fine': '1D, fine steps (cutoff 1 Rφ)',
         '1d_fine_full': '1D, fine steps, full drift'}


def config(name):
    r = RUNS[name]
    cfg = yaml.safe_load(yaml.safe_dump(BASE))
    if SHEAR is not None:
        b = yaml.safe_load(open(BASE['input_beam']['distgen_input_file']))
        b['transforms']['s1']['shear_coefficient']['value'] = SHEAR
        bf = os.path.join(OUT, 'inputs', 'beam.yaml'); yaml.safe_dump(b, open(bf, 'w'))
        cfg['input_beam']['distgen_input_file'] = bf
    cfg['CSR_computation'].update(workdir=os.path.join(OUT, 'runs'), write_name=name)
    if r.get('csr') is False:
        cfg['CSR_computation'].update(compute_CSR=0, apply_CSR=0, write_wakes=False)
    # explicit either way, so these runs do not change meaning if the defaults change
    if r['cls'] == '2d':
        cfg['CSR_integration']['tangent_bands'] = bool(r.get('tangent', False))
        cfg['particle_deposition']['vx_model'] = r.get('vx', 'none')
        cfg['CSR_integration']['drift_reach'] = bool(r.get('reach', False))
        cfg['CSR_integration']['drift_snap_thin'] = float(r.get('thin', 0.0))
        cfg['CSR_integration']['far_fine_max'] = int(r.get('cap', 0))
        cfg['particle_deposition'].update(r.get('dep', {}))
        if 'cutoff2d' in r:
            cfg['CSR_integration']['drift_cutoff'] = float(r['cutoff2d'])
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


def slice_variance(z, pz, E0, ns=80, zr=3.0, core=2.0, nmin=500):
    """
    Core slice energy variance [eV^2]: ns slices over +-zr sigma_z, a quadratic fit in z removed within
    each slice (the chicane leaves a large chirp across a slice), averaged over the slices with
    |centre| < core sigma_z, weighted by population. Same definition as metrics() below, but from
    per-slice moments (bincount), so it is cheap enough to call at every step on 1e6 particles.
    """
    zc = z - z.mean(); sz = zc.std()
    ed = np.linspace(-zr * sz, zr * sz, ns + 1); w = ed[1] - ed[0]
    idx = np.digitize(zc, ed) - 1
    ok = (idx >= 0) & (idx < ns)
    i = idx[ok]
    u = (zc[ok] - 0.5 * (ed[i] + ed[i + 1])) / w                     # local coordinate in the slice
    e = pz[ok] * E0
    n = np.bincount(i, minlength=ns).astype(float)
    em = np.bincount(i, e, ns) / np.maximum(n, 1)
    e = e - em[i]                                                     # remove each slice's mean first
    M = np.stack([np.bincount(i, u ** k, ns) for k in range(5)])
    Y = np.stack([np.bincount(i, e * u ** k, ns) for k in range(3)])
    ee = np.bincount(i, e * e, ns)
    ctr = 0.5 * (ed[1:] + ed[:-1])
    use = (np.abs(ctr) < core * sz) & (n > nmin)
    A = np.stack([[M[0], M[1], M[2]], [M[1], M[2], M[3]], [M[2], M[3], M[4]]]).transpose(2, 0, 1)[use]
    coef = np.linalg.solve(A, Y.T[use][..., None])[..., 0]
    ss = ee[use] - np.sum(coef * Y.T[use], axis=1)                   # residual sum of squares per slice
    return ss.sum() / n[use].sum()


def run(name):
    os.makedirs(os.path.join(OUT, 'runs'), exist_ok=True); os.makedirs(os.path.join(OUT, 'inputs'), exist_ok=True)
    f = config(name)
    if RUNS[name]['cls'] == '2d':
        from pyDFCSR_2D import CSR2D as Cls
    else:
        from pyDFCSR_2D.csr1d import CSR1D as Cls
    t0 = time.time()
    sim = Cls(input_file=f)
    # core slice energy variance along s: every node for the auto schedule, ~600 points for fine ones
    E0 = sim.beam.init_energy
    n_nodes = len(sim.lattice.schedule.s_nodes)
    stride = max(1, int(np.ceil(n_nodes / 600)))
    rec_s, rec_v = [sim.beam.position], [slice_variance(sim.beam.z, sim.beam.particle.pz, E0)]
    upd = sim.update_statistics

    def update_statistics(step):
        upd(step)
        if step % stride == 0 or step == n_nodes - 1:
            rec_s.append(sim.beam.position)
            rec_v.append(slice_variance(sim.beam.z, sim.beam.particle.pz, E0))
    sim.update_statistics = update_statistics
    if RUNS[name].get('xavg') is not None:
        s_from = RUNS[name]['xavg']; u = np.linspace(-4, 4, sim.CSR_params.xbins); wx = np.exp(-u ** 2 / 2); wx /= wx.sum()
        aw = sim.beam.apply_wakes

        def apply_wakes(dE_dct, x_kick, *a, **kw):
            if sim.beam.position >= s_from - 1e-9:
                dE_dct = np.broadcast_to((wx[:, None] * dE_dct).sum(0), dE_dct.shape).copy()
            return aw(dE_dct, x_kick, *a, **kw)
        sim.beam.apply_wakes = apply_wakes
    sim.run()
    p = sim.beam.particle
    st = {k: v for k, v in sim.statistics.items() if not isinstance(v, dict)}
    tw = {f'twiss_{k}': v for k, v in sim.statistics['twiss'].items()}
    np.savez(os.path.join(OUT, f'{name}.npz'), x=p.x, px=p.px, z=p.z, pz=p.pz, E0=E0,
             s=np.array(sim.lattice.steps_record), runtime=time.time() - t0, prefix=sim.prefix,
             slice_s=np.array(rec_s), slice_var=np.array(rec_v), **st, **tw)
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
    """
    Final-beam CSR effects against the CSR-off beam. The core slice energy spread is computed for each
    beam on its OWN slicing (slice_variance: 80 slices over +-3 sigma_z of that beam, quadratic chirp
    removed, |z| < 2 sigma_z), the same definition as the along_s.png panels, and reported as the
    LINEAR change dsEs = sigma_CSR - sigma_off [keV] and the relative change rsEs [%]. (It used to be
    sqrt(<var_CSR - var_off>) with the off beam sliced on the CSR beam's grid -- an 'added in
    quadrature' spread, 80.6 keV for the nominal 2D run where the linear change is +20.5 keV.)
    """
    G = E0 / MC2
    zc, sz, ed, idx = slices(b['z'], b['pz'])
    w, se = [], []
    for i in range(len(ed) - 1):
        c = 0.5 * (ed[i] + ed[i + 1])
        m = idx == i
        if abs(c) > 2 * sz or m.sum() < 500:
            continue
        w.append(m.sum()); se.append(emit(b['x'][m], b['px'][m], b['pz'][m], G))
    s_on = np.sqrt(slice_variance(b['z'], b['pz'], E0))
    s_off = np.sqrt(slice_variance(off['z'], off['pz'], E0))
    h, e = np.histogram(zc, 400)
    return dict(dE=(b['pz'].mean() - off['pz'].mean()) * E0 / 1e3,
                dsEp=(b['pz'].std() - off['pz'].std()) * E0 / 1e3,
                sEs=s_on / 1e3, dsEs=(s_on - s_off) / 1e3, rsEs=100 * (s_on - s_off) / s_off,
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
    if prev and SHEAR is None:
        q = np.load(prev[0])
        L.append(f"CSR-off reference (CSR1D) vs the earlier CSR2D CSR-off run: z bit-identical = "
                 f"{np.array_equal(np.sort(q['zf']), np.sort(off['z']))}, pz bit-identical = {np.array_equal(np.sort(q['df']), np.sort(off['pz']))}")
    M = {n: metrics(r, off, E0) for n, r in R.items() if n not in ('off', 'off_fine')}
    rows = [('dE', 'mean CSR energy change [keV]'), ('dsEp', 'projected sigma_E change [keV]'),
            ('sEs', 'core slice sigma_E [keV]'), ('dsEs', 'change of core slice sigma_E [keV]'),
            ('rsEs', 'change of core slice sigma_E [%]'), ('demit', 'emittance growth, eta-corr. [um]'),
            ('semit', 'core slice emittance [um]'), ('sz', 'final sigma_z [um]'), ('Ipk', 'peak current [kA]')]
    names = [n for n in LABEL if n in M]
    Moff = metrics(off, off, E0)
    L += ['', f"{'chicane exit, vs CSR off':38s}" + ''.join(f'{n:>14s}' for n in names) + f"{'CSR off':>12s}"]
    for k, lab in rows:
        L.append(f'{lab:38s}' + ''.join(f'{M[n][k]:14.3f}' for n in names) + (f'{Moff[k]:12.3f}' if k in ('sEs', 'semit', 'sz', 'Ipk') else ''))
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
    cols = {'2d': 'C3', '2d_noT': 'C1', '1d': 'C0', '1d_full': 'C2', '1d_fine': 'C9', '1d_fine_full': 'C4',
            '2d_tb': 'C5', '2d_vx': 'C7', '2d_new': 'C6', '2d_new_noT': 'C8', '2d_new_xavg': 'C9', '2d_new_xavgB4': 'C4',
            '2d_reach': 'C3', '2d_reach_noT': 'C1', '2d_reach_cap': 'C6', '2d_reach_cap_noT': 'C8', '2d_x8': 'C2', '2d_x8c': 'C0'}
    have_slice = all('slice_var' in R[n] for n in ['off'] + names)
    if have_slice:
        fig = plt.figure(figsize=(14, 16.5), layout='constrained')
        gs = fig.add_gridspec(4, 2)
        ax = [fig.add_subplot(gs[i // 2, i % 2]) for i in range(6)] + [fig.add_subplot(gs[3, :])]
    else:
        fig, ax = plt.subplots(2, 2, figsize=(14, 8.5), layout='constrained')
        ax = ax.ravel()
    for n in ['off'] + names:
        r = R[n]; c = cols.get(n, 'k'); ls = ':' if n == 'off' else '-'; lab = LABEL.get(n, 'CSR off')
        ax[0].plot(r['s'], r['sigma_z'] * 1e6, ls, c=c, label=lab)
        ax[1].plot(r['s'], (r['mean_energy'] - off['mean_energy'][0]) / 1e3 if n != 'off' else 0 * r['s'], ls, c=c, label=lab)
        ax[2].plot(r['s'], r['twiss_norm_emit_x'] * 1e6, ls, c=c, label=lab)
        ax[3].plot(r['s'], r['sigma_energy'] / 1e6, ls, c=c, label=lab)
        if have_slice:
            ax[4].plot(r['slice_s'], np.sqrt(r['slice_var']) / 1e3, ls, c=c, label=lab)
            if n != 'off':
                # actual change of the core slice sigma_E against the CSR-off run at the same s, in keV
                # and relative. NOT signed sqrt|sigma^2 - sigma0^2|: inside the chicane sigma0 is ~2 MeV
                # (the slice cuts across the dispersed, chirped ribbon), and that form turned a 0.3% change
                # into a -155 keV "dip"
                ref = R['off_fine'] if (RUNS.get(n, {}).get('fine') and 'off_fine' in R) else off
                if not np.array_equal(np.round(r['slice_s'], 9), np.round(ref['slice_s'], 9)):
                    print(f'  [along_s] {n}: no CSR-off run on the same nodes; slice change interpolated')
                s0 = np.sqrt(np.interp(r['slice_s'], ref['slice_s'], ref['slice_var']))
                ds = np.sqrt(r['slice_var']) - s0
                ax[5].plot(r['slice_s'], ds / 1e3, ls, c=c, label=lab)
                ax[6].plot(r['slice_s'], 100 * ds / s0, ls, c=c, label=lab)
    titles = ['bunch length', 'energy loss', 'emittance', 'projected energy spread']
    ylabs = ['σz [µm]', 'mean CSR energy change [keV]', 'εn,x, η-corrected [µm]', 'projected σE [MeV]']
    if have_slice:
        titles += ['core slice energy spread (|z| < 2σz, 80 slices, quadratic chirp removed)',
                   'change of core slice σE vs CSR off at the same s',
                   'relative change of core slice σE vs CSR off at the same s']
        ylabs += ['slice σE [keV]', 'σE(CSR) − σE(off) [keV]', '[σE(CSR) − σE(off)] / σE(off) [%]']
    for a, yl, t in zip(ax, ylabs, titles):
        [a.axvspan(lo, hi, color='0.92', zorder=-1) for lo, hi, _ in BENDS]; a.set(xlabel='s [m]', ylabel=yl, title=t)
    ax[0].set_yscale('log'); ax[1].legend(fontsize=8)
    if have_slice:
        for a in (ax[5], ax[6]):
            a.axhline(0, c='k', lw=0.5)
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
