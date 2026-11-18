"""
The new 2D options on the standard Gaussian beam in a dipole: are they still correct there?

Case: example/input/dipole_chirp0_config.yaml (5 GeV, 1 nC, sigma_z = sigma_x = 50 um, no chirp,
1M Hammersley particles; 0.1 m drift, 1 m / 1 rad bend, 0.5 m drift), the beam of the recorded
single-dipole study (progress.md, 2026-10-28; default 2D: mean energy loss ~ -1380 keV, core slice
sigma_E ~ 701 keV with CSR, 486 keV without).

    cd pyDFCSR_2D/example && python ../test/quantile_frame/dipole_gauss_check.py <variant>
    then: python ../test/quantile_frame/dipole_gauss_check.py --compare

Variants (VARIANTS below): default; quantile (frame quantile + charge mesh at the default
resolution); quantile_adapt (+ smoothing_adaptive); quantile_adapt_kp (+ kick_placement csr1d);
bc20set (the BC20 settings: 150 x 1200, smoothing_sigma_z 1, adaptive, charge mesh 12 x 80,
csr1d kicks); off (CSR off, the reference for the energy change).

Each run records, at every kick, the longitudinal and transverse wake as the PARTICLES receive it
(the mesh interpolated onto each particle, as Beam.apply_wakes does), averaged in 60 z bins over
+-3 sigma_z: comparable across mesh types. At the end: the final beam's moments, and slice
statistics in 40 final-z slices over +-3 sigma_z as in slice_espread_dipole.py. Writes
benchmark_results/quantile_frame/dipole_<variant>.npz; --compare prints the table and writes
dipole_gauss_check.txt and .png there.
"""
import contextlib
import io
import os
import sys
import time

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', '..'))
OUT = os.path.join(HERE, '..', 'benchmark_results', 'quantile_frame')
BASE = 'input/dipole_chirp0_config.yaml'
NSLICE, ZR, NZB = 40, 3.0, 60

Q_DEP = dict(frame='quantile', frame_quantile=0.005, frame_margin=0.1)
VARIANTS = {
    'default': {},
    'off': dict(csr=dict(compute_CSR=0, apply_CSR=0)),
    'quantile': dict(dep=Q_DEP, csr=dict(mesh_mode='charge')),
    'quantile_adapt': dict(dep=dict(Q_DEP, smoothing_adaptive=True), csr=dict(mesh_mode='charge')),
    'quantile_adapt_kp': dict(dep=dict(Q_DEP, smoothing_adaptive=True), csr=dict(mesh_mode='charge'),
                              sc=dict(kick_placement='csr1d', kick_tol=0.005, kick_h_max=0.5)),
    # convergence: both at 2x the deposition and mesh resolution; and fewer tails dropped
    'default_fine': dict(dep=dict(xbins=200, zbins=200), csr=dict(xbins=20, zbins=60)),
    'quantile_fine': dict(dep=dict(Q_DEP, xbins=200, zbins=200), csr=dict(mesh_mode='charge', xbins=20, zbins=60)),
    'quantile_q0005': dict(dep=dict(Q_DEP, frame_quantile=0.0005), csr=dict(mesh_mode='charge')),
    'peak': dict(dep=dict(Q_DEP, frame='peak'), csr=dict(mesh_mode='charge')),
    # transverse force from W1 only (CSR_integration.transverse_terms); the run also computes the
    # three-term force at every kick at the same beam state and stores both mesh maps (decomp)
    'W1': dict(integ=dict(transverse_terms='W1'), decomp=True),
    'adapt': dict(integ=dict(near_adapt=True)),
    'allterms_decomp': dict(decomp=True),
    'bc20set': dict(dep=dict(Q_DEP, xbins=150, zbins=1200, smoothing_sigma=3.0, smoothing_sigma_z=1.0,
                             smoothing_adaptive=True),
                    csr=dict(mesh_mode='charge', xbins=12, zbins=80),
                    sc=dict(kick_placement='csr1d', kick_tol=0.005, kick_h_max=0.5)),
}


def config(name):
    v = VARIANTS[name]
    c = yaml.safe_load(open(BASE))
    c.setdefault('particle_deposition', {}).update(v.get('dep', {}))
    c['CSR_computation'].update(dict(write_beam=None, write_wakes=False, write_name=f'dgc_{name}'),
                                **v.get('csr', {}))
    if 'sc' in v:
        c['step_control'] = dict(dict(mode='auto', kick_interval='midpoint'), **v['sc'])
    c.setdefault('CSR_integration', {}).update(v.get('integ', {}))
    f = os.path.join('input', f'dgc_{name}.yaml')
    yaml.safe_dump(c, open(f, 'w'), sort_keys=False)
    return f


def run(name):
    from pyDFCSR_2D import CSR2D
    from scipy.interpolate import RegularGridInterpolator
    with contextlib.redirect_stdout(io.StringIO()):
        sim = CSR2D(input_file=config(name))
    kicks = {'s': [], 'zc': [], 'W': [], 'F': []}
    apply0 = sim.beam.apply_wakes

    def apply_wakes(dE_dct, x_kick, xrange, zrange, L, transverse_on, x_shift=None, **kw):
        b = sim.beam
        xt = b.x_transform if x_shift is None else b.x_transform - x_shift
        pts = np.array([xt, b.z]).T
        W = RegularGridInterpolator((xrange, zrange), dE_dct, fill_value=0.0, bounds_error=False)(pts)
        F = RegularGridInterpolator((xrange, zrange), x_kick, fill_value=0.0, bounds_error=False)(pts)
        z = b.z - b.z.mean()
        e = np.linspace(-ZR * z.std(), ZR * z.std(), NZB + 1)
        k = np.digitize(z, e) - 1
        ok = (k >= 0) & (k < NZB)
        n = np.maximum(np.bincount(k[ok], minlength=NZB), 1)
        kicks['s'].append(b.position)
        kicks['zc'].append(0.5 * (e[1:] + e[:-1]))
        kicks['W'].append(np.bincount(k[ok], W[ok], NZB) / n)
        kicks['F'].append(np.bincount(k[ok], F[ok], NZB) / n)
        return apply0(dE_dct, x_kick, xrange, zrange, L, transverse_on, x_shift=x_shift, **kw)
    sim.beam.apply_wakes = apply_wakes
    maps = {'s': [], 'z': [], 'xi': [], 'W': [], 'F_all': [], 'F_W1': []}
    if VARIANTS[name].get('decomp'):
        calc0 = sim.calculate_2D_CSR
        ip = sim.integration_params
        want = ip.transverse_terms

        def calc():
            out = {}
            for terms in ('all', 'W1'):
                ip.transverse_terms = terms
                calc0()
                out[terms] = (sim.dE_dct.copy(), sim.x_kick.copy())
            ip.transverse_terms = want
            sim.dE_dct, sim.x_kick = out[want]
            maps['s'].append(sim.beam.position)
            maps['z'].append(np.asarray(sim.CSR_zrange).copy())
            maps['xi'].append(np.asarray(sim.CSR_xrange_transformed).copy())
            maps['W'].append(out['all'][0])
            maps['F_all'].append(out['all'][1])
            maps['F_W1'].append(out['W1'][1])
        sim.calculate_2D_CSR = calc
    t0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        sim.run()
    p = sim.beam.particle
    E0 = p.p0c
    z = p.z - p.z.mean()
    sz = z.std()
    edges = np.linspace(-ZR * sz, ZR * sz, NSLICE + 1)
    idx = np.digitize(z, edges) - 1
    sE = np.array([(p.pz[idx == i] * E0).std() for i in range(NSLICE)])
    mE = np.array([(p.pz[idx == i] * E0).mean() for i in range(NSLICE)])
    from pyDFCSR_2D.twiss import twiss_dispersion_calc
    tw = twiss_dispersion_calc(np.cov([p.x, p.px, p.pz]))
    np.savez(os.path.join(OUT, f'dipole_{name}.npz'), runtime=time.time() - t0,
             n_kicks=sim.lattice.schedule.n_kick, mean_E=p.pz.mean() * E0, sigma_E=p.pz.std() * E0,
             slice_sE=sE, slice_mE=mE, slice_zc=0.5 * (edges[1:] + edges[:-1]), sigma_z=sz,
             emit_x=tw['emit'] * E0 / p.mc2, sigma_x=p.x.std(),
             **{f'kick_{k}': np.array(v) for k, v in kicks.items()},
             **{f'map_{k}': np.array(v) for k, v in maps.items()})
    print(f'{name}: {sim.lattice.schedule.n_kick} kicks, {time.time() - t0:.0f} s', flush=True)


def compare():
    R = {n: np.load(os.path.join(OUT, f'dipole_{n}.npz')) for n in VARIANTS
         if os.path.isfile(os.path.join(OUT, f'dipole_{n}.npz'))}
    off = R['off']
    core = slice(NSLICE // 2 - 2, NSLICE // 2 + 2)      # the 4 central slices, |z| < 0.3 sigma_z
    L = ['Standard Gaussian dipole case (dipole_chirp0_config.yaml), the new 2D options against the default.',
         'dE = mean energy change vs CSR off; core slice sigma_E = mean over the 4 central final-z slices;',
         'wake rel. diff = L2 over z of the particle-averaged wake vs default at the nearest kick, median over kicks.',
         f'{"variant":>20s} {"kicks":>5s} {"time s":>7s} {"dE keV":>8s} {"sigma_E keV":>11s} '
         f'{"core sE keV":>11s} {"eps_n um":>8s} {"sigma_z um":>10s} {"W rel":>6s} {"F rel":>6s}']
    d = R['default']

    def wdiff(r, key):
        out = []
        for s, w in zip(r['kick_s'], r[f'kick_{key}']):
            j = int(np.argmin(np.abs(d['kick_s'] - s)))
            ref = d[f'kick_{key}'][j]
            if abs(d['kick_s'][j] - s) < 0.03 and np.linalg.norm(ref) > 0:
                out.append(np.linalg.norm(w - ref) / np.linalg.norm(ref))
        return np.median(out) if out else np.nan
    for n, r in R.items():
        L.append(f'{n:>20s} {int(r["n_kicks"]):5d} {float(r["runtime"]):7.0f} '
                 f'{(float(r["mean_E"]) - float(off["mean_E"]))/1e3:8.1f} {float(r["sigma_E"])/1e3:11.1f} '
                 f'{r["slice_sE"][core].mean()/1e3:11.1f} {float(r["emit_x"])*1e6:8.4f} '
                 f'{float(r["sigma_z"])*1e6:10.4f} '
                 f'{(wdiff(r, "W") if n not in ("off", "default") else 0):6.3f} '
                 f'{(wdiff(r, "F") if n not in ("off", "default") else 0):6.3f}')
    txt = '\n'.join(L)
    print(txt)
    open(os.path.join(OUT, 'dipole_gauss_check.txt'), 'w').write(txt + '\n')
    plot(R)


def plot(R):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    SURF, INK, GRID = '#fcfcfb', '#0b0b0b', '#e4e3df'
    col = {'default': '#52514e', 'quantile': '#1baf7a', 'quantile_adapt': '#2a78d6',
           'quantile_adapt_kp': '#eb6834', 'bc20set': '#4a3aa7', 'default_fine': '#898781',
           'quantile_fine': '#008300', 'quantile_q0005': '#eda100', 'peak': '#e34948'}
    ls = {'default': '-', 'quantile': '--', 'quantile_adapt': '-', 'quantile_adapt_kp': ':', 'bc20set': '-.',
          'default_fine': ':', 'quantile_fine': ':', 'quantile_q0005': '--', 'peak': '--'}
    d = R['default']
    picks = [0.3, 0.6, 1.0, 1.1, 1.4]
    fig, axs = plt.subplots(3, len(picks), figsize=(3.2 * len(picks), 8), facecolor=SURF, sharex=True)
    for j, sp in enumerate(picks):
        for n, r in R.items():
            if n == 'off':
                continue
            i = int(np.argmin(np.abs(r['kick_s'] - sp)))
            zc = r['kick_zc'][i] * 1e6
            axs[0, j].plot(zc, r['kick_W'][i], color=col[n], ls=ls[n], lw=1.5, label=n)
            axs[1, j].plot(zc, r['kick_F'][i], color=col[n], ls=ls[n], lw=1.5)
        axs[0, j].set_title(f's = {sp:.1f} m (nearest kick)', loc='left', fontsize=9)
    for n, r in R.items():
        if n == 'off':
            continue
        axs[2, 0].plot(r['slice_zc'] * 1e6, r['slice_sE'] / 1e3, color=col[n], ls=ls[n], lw=1.5)
        axs[2, 1].plot(r['slice_zc'] * 1e6, (r['slice_mE'] - R['off']['slice_mE']) / 1e3, color=col[n],
                       ls=ls[n], lw=1.5)
    axs[2, 0].plot(R['off']['slice_zc'] * 1e6, R['off']['slice_sE'] / 1e3, color='#898781', lw=1, label='CSR off')
    axs[0, 0].set_ylabel('W, particle-averaged [MeV/m]')
    axs[1, 0].set_ylabel('transverse force [MeV/m]')
    axs[2, 0].set_ylabel('final slice σE [keV]')
    axs[2, 1].set_ylabel('final slice ΔE vs CSR off [keV]')
    for ax in axs[2, 2:]:
        ax.axis('off')
    for ax in axs.flat:
        ax.set_facecolor(SURF)
        ax.grid(color=GRID, lw=0.5)
    for ax in axs[2, :2]:
        ax.set_xlabel('z [µm]  (head at +z)')
    axs[0, 0].legend(fontsize=7.5, frameon=False)
    axs[2, 0].legend(fontsize=7.5, frameon=False)
    fig.suptitle('Standard Gaussian beam in a 1 m / 1 rad dipole: new 2D options against the default',
                 x=0.01, ha='left', fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(os.path.join(OUT, 'dipole_gauss_check.png'), dpi=130, facecolor=SURF)
    plt.close(fig)


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    if sys.argv[1] == '--compare':
        compare()
    else:
        run(sys.argv[1])
