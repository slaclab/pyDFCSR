"""
Near-field quadrature check at the first kick (s = 0.42 m, B1LE) of a 2D BC20 run: are the
isolated single-point dips of the wake mesh quadrature error, or are they in the density data?

    cd pyDFCSR_2D/example
    conda run -n pydfcsr python ../test/facet_bc20/near_field_check.py [run]   (default dfcsr_2d_peak_1M)

Serial. Tracks the run's own input (runs/inputs/<run>.yaml) to its first kick, then, instead of
the kick:
  1. recomputes the whole wake mesh with the run's settings and checks it against the stored
     wakes.h5 (same beam, same history -> must agree to round-off);
  2. takes the mesh rows holding the worst outliers (deviation from the 3x3 median) and
     recomputes every point of those rows with refined quadratures, one factor at a time:
       patch4   polar near patch 4x nodes in r and in phi (nr 400, nphi 720)
       xp4      4x the x' nodes in every band (Integration xbins 800)
       sp4      4x finer near-region s' cells (near_cell / 4, near_grade / 4)
       far4     4x the far-region s' nodes (far_zbins x4, far_fine_max x4)
       far_dense  the far region on 8000 uniform s' nodes instead of its graded layout
       margin3  the x' ribbons 3x wider (xi_band_margin x3)
       vx_src   observer vx = the windowed source's (no blend with the beam fit at the window edge)
       all      patch4 + sp4 + far_dense + margin3
     and the integral split by region (far, near Cartesian, patch).
Writes near_field_check_<run>.npz and benchmark_results/facet_bc20/near_field_check_<run>.png.
"""
import glob
import os
import sys
import time

import h5py
import numpy as np
from scipy.ndimage import median_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', '..'))
RES = os.path.join(HERE, '..', 'benchmark_results', 'facet_bc20')
RUNS = os.path.join(RES, 'runs')

# parameter factors (multiply; near_cell/near_grade and xi_band_margin are divided/multiplied as
# noted), plus '_far_dense' (far region on uniform s' nodes instead of the graded layout) and
# '_vx_src' (observer vx = the windowed source's, no blend with the beam-wide fit at the edge)
VARIANTS = {
    'base': {},
    'patch4': dict(near_patch_nr=4, near_patch_nphi=4),
    'xp4': dict(xbins=4),
    'sp4': dict(near_cell=4, near_grade=4),
    'far4': dict(far_zbins=4, far_fine_max=4),
    'far_dense': dict(_far_dense=8000),
    'margin3': dict(xi_band_margin=3),
    'vx_src': dict(_vx_src=True),
}
VARIANTS['all'] = dict(near_patch_nr=4, near_patch_nphi=4, near_cell=4, near_grade=4, _far_dense=8000,
                       xi_band_margin=3)
DIVIDE = ('near_cell', 'near_grade')
N_ROWS = 3


class Stop(Exception):
    pass


def wake_parts(sim, s, x):
    """get_CSR_wake's xi_bands branch, returned by region: {'far', 'near', 'prev', 'patch'} -> (dE, xk)"""
    t = sim.beam.position
    bnds = sim._layout_bounds(s, x)
    (s1, s3), (_, s4) = bnds
    sim._region_node_counts(bnds)
    sps = [sim._far_region_nodes(s, x, t, s1, s3), sim._near_region_nodes(s, s3, s4, x=x, t=t)]
    radii = sim._near_patch_radii(s, s3, s4)
    tapers = [None, None if radii is None else (*radii, False)]
    out = {}
    for k, (name, spk, tp) in enumerate(zip(('far', 'near'), sps, tapers)):
        out[name] = sim._integrate_xi_region(s, x, t, spk, False, tp, scale_bands=(k == 0))[:2]
    pb = sim._prev_bend_reach_start()
    out['prev'] = (sim._integrate_xi_region(s, x, t, sim._prev_region_nodes(s, pb, s1), False, None,
                                            scale_bands=True)[:2] if (pb is not None and pb < s1) else (0.0, 0.0))
    out['patch'] = sim._integrate_near_patch(s, x, t, *radii, False)[:2] if radii is not None else (0.0, 0.0)
    out['n_sp'] = (len(sps[0]), len(sps[1]))
    return out


def stored_first_kick(run):
    f = h5py.File(sorted(glob.glob(os.path.join(RUNS, f'{run}-2*-wakes.h5')), key=os.path.getmtime)[-1], 'r')
    k = min(f, key=lambda k: f[k].attrs['position'])
    g = f[k]
    return (float(g.attrs['position']), g['longitudinal/dE_dct'][()] if 'longitudinal/dE_dct' in g
            else g['longitudinal/wake'][()], g['transverse/xkicks'][()])


def analyse(sim, run):
    ip = sim.integration_params
    pos = sim.beam.position
    nx, nz = sim.CSR_params.xbins, sim.CSR_params.zbins
    S = pos + sim.CSR_zmesh
    X = sim.CSR_xmesh
    t0 = time.time()
    base = np.array([sim.get_CSR_wake(s, x) for s, x in zip(S, X)])
    t_base = (time.time() - t0) / S.size
    W = base[:, 0].reshape(nx, nz)
    F = base[:, 1].reshape(nx, nz)
    s_st, W_st, F_st = stored_first_kick(run)
    print(f'first kick at s = {pos:.6f} m (stored {s_st:.6f}); base {t_base*1e3:.1f} ms/point; '
          f'|W - stored| max {np.abs(W - W_st).max():.3e} of max|W| {np.abs(W_st).max():.3e}', flush=True)

    m = np.abs(W).max()
    dev = (W - median_filter(W, size=3, mode='nearest')) / m
    rows = list(np.argsort(-np.abs(dev).max(1))[:N_ROWS])
    print('rows with the worst outliers (ix, worst dev/max|W|, iz):',
          [(int(r), round(float(dev[r].max() if dev[r].max() > -dev[r].min() else dev[r].min()), 3),
            int(np.argmax(np.abs(dev[r])))) for r in rows], flush=True)
    tr = sim.DF_tracker
    print(f'window lo/hi {tr.window[0]*1e6:+.1f}/{tr.window[1]*1e6:+.1f} um, margins '
          f'{tr.window[2]*1e6:.1f}/{tr.window[3]*1e6:.1f} um; beam mean z {sim.beam.mean_z*1e6:+.1f} um, '
          f'sigma_z {sim.beam.sigma_z*1e6:.1f} um, sigma_xi {sim.beam._sigma_x_transform*1e6:.2f} um', flush=True)

    keep = {k: getattr(ip, k) for k in ('near_patch_nr', 'near_patch_nphi', 'xbins', 'near_cell', 'near_grade',
                                         'far_zbins', 'far_fine_max', 'xi_band_margin')}
    far_nodes, obs_vx = sim._far_region_nodes, sim._observer_vx
    res = {}
    for name, fac in VARIANTS.items():
        for k, v in keep.items():
            setattr(ip, k, v)
        sim._far_region_nodes, sim._observer_vx = far_nodes, obs_vx
        for k, f in fac.items():
            if k == '_far_dense':
                sim._far_region_nodes = lambda s, x, t, a, b, transient=None, n=f: np.linspace(a, b, n)
            elif k == '_vx_src':
                sim._observer_vx = lambda x, zobs, vx_src: vx_src
            elif k in DIVIDE:
                setattr(ip, k, keep[k] / f)
            elif isinstance(keep[k], float):
                setattr(ip, k, keep[k] * f)
            else:
                setattr(ip, k, int(keep[k] * f))
        t0 = time.time()
        out = np.zeros((len(rows), nz, 4, 2))
        nsp = np.zeros((len(rows), nz, 2), int)
        for a, r in enumerate(rows):
            for iz in range(nz):
                i = r * nz + iz
                p = wake_parts(sim, S[i], X[i])
                out[a, iz] = [p['far'], p['near'], p['prev'], p['patch']]
                nsp[a, iz] = p['n_sp']
        tot = out.sum(2)
        if name == 'base':
            err = np.abs(tot[..., 0] - W[rows]).max()
            print(f'  base by parts reproduces get_CSR_wake to {err:.2e}', flush=True)
        res[name] = dict(parts=out, nsp=nsp)
        print(f'  {name:7s} {(time.time() - t0) / (len(rows) * nz) * 1e3:7.1f} ms/point', flush=True)
    for k, v in keep.items():
        setattr(ip, k, v)
    sim._far_region_nodes, sim._observer_vx = far_nodes, obs_vx

    zr = sim.CSR_zrange
    # fine observer-z sweep along the worst row through its worst dip (0.5 um steps, +-8 mesh nodes)
    r0 = rows[0]
    i0 = int(np.argmax(np.abs(dev[r0])))
    zf = np.arange(zr[max(i0 - 8, 0)], zr[min(i0 + 8, nz - 1)], 0.5e-6)
    xt = sim.CSR_xrange_transformed[r0]
    xf = xt + np.polyval(sim.beam.slope, zf) + sim._mesh_ridge(zf)
    sweep = np.array([[*wake_parts(sim, pos + z, x)[k]] for z, x in zip(zf, xf) for k in ('far', 'near', 'patch')])
    sweep = sweep.reshape(zf.size, 3, 2)
    print(f'sweep along row {r0}: {zf.size} observers, z {zf[0]*1e6:+.1f} .. {zf[-1]*1e6:+.1f} um', flush=True)
    np.savez(os.path.join(HERE, f'near_field_check_{run}.npz'), s=pos, z=zr, xi=sim.CSR_xrange_transformed,
             W=W, F=F, W_stored=W_st, rows=np.array(rows), dev=dev,
             sweep_z=zf, sweep_x=xf, sweep=sweep,
             window=np.array(sim.DF_tracker.window),
             **{f'{n}_parts': r['parts'] for n, r in res.items()}, **{f'{n}_nsp': r['nsp'] for n, r in res.items()})
    report(run, pos, zr, W, F, rows, dev, res)


def report(run, pos, zr, W, F, rows, dev, res):
    m = np.abs(W).max()
    mF = np.abs(F).max()
    names = list(res)
    print('\nat the outliers (|dev| > 0.1 max|W|) of these rows, W / max|W| per variant, and its 3-point '
          'neighbour median under the same variant:')
    print('  ix  iz   z[um] ' + ''.join(f'{n:>15s}' for n in names))
    for a, r in enumerate(rows):
        for iz in np.flatnonzero(np.abs(dev[r]) > 0.1):
            cells = []
            for n in names:
                w = res[n]['parts'][a, :, :, 0].sum(1) / m
                nb = np.median(w[max(iz - 1, 0):iz + 2])
                cells.append(f'{w[iz]:+.3f} ({nb:+.3f})')
            print(f'  {r:2d} {iz:3d} {zr[iz]*1e6:+7.1f} ' + ''.join(f'{c:>15s}' for c in cells))
    print('\nrow roughness: rms of W - 3-point median along z, / max|W|  (longitudinal; transverse)')
    for a, r in enumerate(rows):
        line = []
        for n in names:
            tot = res[n]['parts'][a].sum(1)
            rl = np.sqrt(np.mean((tot[:, 0] - median_filter(tot[:, 0], 3, mode='nearest')) ** 2)) / m
            rt = np.sqrt(np.mean((tot[:, 1] - median_filter(tot[:, 1], 3, mode='nearest')) ** 2)) / mF
            line.append(f'{n} {rl:.4f}; {rt:.4f}')
        print(f'  ix {r:2d}: ' + '   '.join(line))
    a, r = 0, rows[0]
    iz = int(np.argmax(np.abs(dev[r])))
    print(f'\nworst point ix {r} iz {iz} (z = {zr[iz]*1e6:+.1f} um), by region [MeV/m]:')
    print('          ' + ''.join(f'{k:>11s}' for k in ('far', 'near', 'prev', 'patch', 'total')) + '   n_sp far/near')
    for n in names:
        p = res[n]['parts'][a, iz, :, 0]
        print(f'  {n:7s} ' + ''.join(f'{v:+11.4e}' for v in list(p) + [p.sum()]) + f'   {tuple(res[n]["nsp"][a, iz])}')
    for nb in (iz - 1, iz + 1):
        p = res['base']['parts'][a, nb, :, 0]
        print(f'  base at iz {nb}: ' + ''.join(f'{v:+11.4e}' for v in list(p) + [p.sum()]))

    col = {'base': '#0b0b0b', 'patch4': '#2a78d6', 'xp4': '#1baf7a', 'sp4': '#eda100', 'far4': '#898781',
           'far_dense': '#4a3aa7', 'margin3': '#eb6834', 'vx_src': '#9ec5f4', 'all': '#e34948'}
    fig, axs = plt.subplots(2, len(rows), figsize=(4.4 * len(rows), 6.2), squeeze=False)
    for a, r in enumerate(rows):
        for j, (lab, q, mm) in enumerate((('longitudinal W', 0, m), ('transverse F', 1, mF))):
            ax = axs[j, a]
            for n in names:
                tot = res[n]['parts'][a, :, :, q].sum(1)
                ax.plot(zr * 1e6, tot / mm, color=col[n], lw=2.2 if n == 'base' else 1.2,
                        ls='-' if n in ('base', 'all') else (0, (3, 1.5)), marker='o' if n == 'base' else None,
                        ms=2.5, label=n)
            ax.axhline(0, color='#e1e0d9', lw=0.8)
            ax.set_title(f'{lab}, mesh row ix = {r}', loc='left', fontsize=9)
            ax.set_xlabel('z [µm]')
            ax.set_ylabel('/ max over the mesh')
    axs[0, 0].legend(fontsize=7, frameon=False)
    fig.suptitle(f'{run}: first kick, s = {pos:.3f} m (B1LE), the mesh rows with the worst outliers, '
                 'recomputed with refined near-field quadratures', x=0.01, ha='left', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(RES, f'near_field_check_{run}.png'), dpi=130)
    print(f'\nwrote near_field_check_{run}.png')


def main(run='dfcsr_2d_peak_1M'):
    from pyDFCSR_2D import CSR2D
    sim = CSR2D(input_file=os.path.join(RUNS, 'inputs', f'{run}.yaml'), parallel=False)
    sim.CSR_params.write_wakes = False

    def first_kick():
        analyse(sim, run)
        raise Stop
    sim.calculate_2D_CSR = first_kick
    try:
        sim.run()
    except Stop:
        pass


if __name__ == '__main__':
    main(*sys.argv[1:])
