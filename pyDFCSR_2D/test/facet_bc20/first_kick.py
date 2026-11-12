"""
One BC20 2D run's first kick only (s = 0.42 m, B1LE), with the wake also resolved along one mesh row.

    cd pyDFCSR_2D/example
    conda run -n pydfcsr python ../test/facet_bc20/first_kick.py <run> [row]      (row default 2)
    conda run -n pydfcsr python ../test/facet_bc20/first_kick.py --compare <run> <run> ...
    conda run -n pydfcsr python ../test/facet_bc20/first_kick.py --maps <run> <run> ...

Serial. Writes the run's input with run_dfcsr_bc20.config, tracks to the first kick, and in place of
the kick saves:
  * the wake mesh W, F (exactly what the run would apply), its axes;
  * the wake along mesh row `row` at 1 um observer spacing over the whole mesh z span (far, near,
    patch parts, near_field_check.wake_parts);
  * the kick on every particle, interpolated from the mesh as Beam.apply_wakes does (dE/ds and the
    transverse force, per particle), and the same from the resolved row for the particles in it.
to first_kick_<run>.npz. --compare prints the numbers below and plots first_kick_compare.png:
  mesh error  max and rms over the row of |linear interpolation of the mesh nodes - resolved wake|,
              / max|W|: what the mesh misses of the field it samples
  roughness   rms of (W - 3-point median along z) over the mesh, / max|W|
  <dE/ds>     particle mean of the interpolated kick [MeV/m], and its rms over the particles
--maps plots first_kick_maps.png: the 2D wake maps W(z, xi) and F(z, xi) of each run, on a colour scale
common to each row, with the old rms-grid run (dfcsr_2d, 20k) at its nearest kick (s = 0.396 m) first.
"""
import os
import sys

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import median_filter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, HERE)
RES = os.path.join(HERE, '..', 'benchmark_results', 'facet_bc20')

import run_dfcsr_bc20 as RD  # noqa: E402
from near_field_check import wake_parts, Stop  # noqa: E402


def analyse(sim, run, row):
    pos = sim.beam.position
    nx, nz = sim.CSR_params.xbins, sim.CSR_params.zbins
    S = pos + sim.CSR_zmesh
    X = sim.CSR_xmesh
    base = np.array([sim.get_CSR_wake(s, x) for s, x in zip(S, X)])
    W, F = base[:, 0].reshape(nx, nz), base[:, 1].reshape(nx, nz)
    zr, xr = sim.CSR_zrange, sim.CSR_xrange_transformed
    zf = np.arange(zr[0], zr[-1], 1e-6)
    xf = xr[row] + np.polyval(sim.beam.slope, zf) + sim._mesh_ridge(zf)
    sweep = np.array([[wake_parts(sim, pos + z, x)[k] for k in ('far', 'near', 'patch')] for z, x in zip(zf, xf)])
    b = sim.beam
    xt = b.x_transform - sim._mesh_ridge(b.z)
    pts = np.array([xt, b.z]).T
    kW = RegularGridInterpolator((xr, zr), W, fill_value=0.0, bounds_error=False)(pts)
    kF = RegularGridInterpolator((xr, zr), F, fill_value=0.0, bounds_error=False)(pts)
    tr = sim.DF_tracker
    res_z, res_x = sim._deposition_resolution()
    np.savez(os.path.join(HERE, f'first_kick_{run}.npz'), s=pos, z=zr, xi=xr, W=W, F=F, row=row,
             sweep_z=zf, sweep=sweep, kick_W=kW, kick_F=kF, pz=b.particle.pz, z_part=b.z,
             window=np.array(tr.window), res_z=res_z, res_x=res_x,
             smoothing=np.array([tr.smoothing_sigma, tr.smoothing_sigma_z or tr.smoothing_sigma]))
    print(f'{run}: first kick s = {pos:.6f} m, mesh {nx} x {nz}, deposition resolution z {res_z*1e6:.2f} um, '
          f'x {res_x*1e6:.2f} um; <dE/ds> {kW.mean():+.4e} MeV/m', flush=True)


def run_one(run, row=2):
    from pyDFCSR_2D import CSR2D
    f = RD.config(run)
    sim = CSR2D(input_file=f, parallel=False)
    sim.CSR_params.write_wakes = False

    def first_kick():
        analyse(sim, run, int(row))
        raise Stop
    sim.calculate_2D_CSR = first_kick
    try:
        sim.run()
    except Stop:
        pass


def compare(runs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    D = {r: np.load(os.path.join(HERE, f'first_kick_{r}.npz')) for r in runs}
    print(f'{"run":34s} {"sz [bins]":>9s} {"res_z [um]":>10s} {"mesh":>7s} {"mesh err max/rms":>17s} '
          f'{"roughness":>9s} {"<dE/ds> [MeV/m]":>16s} {"rms over particles":>18s}')
    for r, d in D.items():
        zf, tot = d['sweep_z'], d['sweep'][:, :, 0].sum(1)
        W, zr, row = d['W'], d['z'], int(d['row'])
        m = np.abs(W).max()
        e = np.abs(np.interp(zf, zr, W[row]) - tot) / m
        rough = np.sqrt(np.mean((W - median_filter(W, size=(1, 3), mode='nearest')) ** 2)) / m
        print(f'{r:34s} {d["smoothing"][1]:9.3g} {float(d["res_z"])*1e6:10.2f} {W.shape[0]:3d}x{W.shape[1]:<3d} '
              f'{e.max():8.3f} / {np.sqrt(np.mean(e**2)):.3f} {rough:9.4f} {d["kick_W"].mean():+16.4e} '
              f'{d["kick_W"].std():18.4e}')
    cols = ['#0b0b0b', '#2a78d6', '#1baf7a', '#eda100', '#e34948', '#4a3aa7']
    fig, axs = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for c, (r, d) in zip(cols, D.items()):
        zf, tot, row = d['sweep_z'], d['sweep'][:, :, 0].sum(1), int(d['row'])
        lab = f'{r}  (σz smoothing {d["smoothing"][1]:g} bins = {float(d["res_z"])*1e6:.1f} µm eff.)'
        axs[0].plot(zf * 1e6, tot, color=c, lw=1.3, label=lab)
        axs[0].plot(d['z'] * 1e6, d['W'][row], 'o', color=c, ms=3.5)
        h, e = np.histogram(d['z_part'], bins=200, weights=d['kick_W'])
        n, _ = np.histogram(d['z_part'], bins=e)
        axs[1].plot(0.5 * (e[1:] + e[:-1]) * 1e6, h / np.maximum(n, 1), color=c, lw=1.3)
    axs[0].axhline(0, color='#e1e0d9', lw=0.8)
    axs[1].axhline(0, color='#e1e0d9', lw=0.8)
    axs[0].set_ylabel('dE/ds [MeV/m]')
    axs[0].set_title(f'first kick, s = {float(d["s"]):.3f} m (B1LE): wake along mesh row {row}, resolved at 1 µm '
                     '(lines) and the mesh nodes (dots)', loc='left', fontsize=9)
    axs[0].legend(fontsize=7, frameon=False, loc='lower right')
    axs[1].set_ylabel('kick on particles, mean per z bin [MeV/m]')
    axs[1].set_xlabel('z [µm]  (head at +z)')
    axs[1].set_title('the kick the run applies: mesh interpolated onto every particle (Beam.apply_wakes), '
                     'averaged in 200 z bins', loc='left', fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, 'first_kick_compare.png'), dpi=140)
    print('wrote first_kick_compare.png')


def maps(runs, zoom=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import plot_bc20_wakes as P
    old = P.load2d('dfcsr_2d')
    o = old[P.nearest(old, 0.42)]
    cols = [(f"old rms grid, 20k (dfcsr_2d)\ns = {o['s']:.3f} m, 3 bins = 13.7 µm, 12 × 40 uniform",
             o['z'], o['xi'], o['W'], o['F'])]
    for r in runs:
        d = np.load(os.path.join(HERE, f'first_kick_{r}.npz'))
        cols.append((f"{r}\ns = {float(d['s']):.3f} m, {d['smoothing'][1]:g} bins = {float(d['res_z'])*1e6:.1f} µm, "
                     f"{d['W'].shape[0]} × {d['W'].shape[1]} charge", d['z'], d['xi'], d['W'], d['F']))
    fig, axs = plt.subplots(2, len(cols), figsize=(3.3 * len(cols), 6.0), sharex=True, squeeze=False)
    for j, key in enumerate((3, 4)):
        a = max(np.abs(c[key]).max() for c in cols[1:])
        for i, c in enumerate(cols):
            ax = axs[j, i]
            pc = ax.pcolormesh(c[1] * 1e6, (c[2] - c[2].mean()) * 1e6, c[key], cmap=P.CMAP, vmin=-a, vmax=a,
                               shading='nearest', rasterized=True)
            zz, xx = np.meshgrid(c[1] * 1e6, (c[2] - c[2].mean()) * 1e6)
            ax.plot(zz.ravel(), xx.ravel(), '.', color='#52514e', ms=0.8)
            ax.text(0.03, 0.04, f'{c[key].min():+.3g} … {c[key].max():+.3g}', transform=ax.transAxes, fontsize=6.5,
                    bbox=dict(boxstyle='round,pad=0.1', fc='#fcfcfb', ec='none', alpha=0.85))
            if zoom:
                ax.set_ylim(cols[0][2].min() * 1e6 - cols[0][2].mean() * 1e6 - 15,
                            cols[0][2].max() * 1e6 - cols[0][2].mean() * 1e6 + 15)
            if j == 0:
                ax.set_title(c[0], loc='left', fontsize=7)
            else:
                ax.set_xlabel('z [µm]  (head at +z)')
            if i == 0:
                ax.set_ylabel(('longitudinal dE/ds' if key == 3 else 'transverse force') + '\nξ − <ξ> [µm]')
        fig.colorbar(pc, ax=axs[j, :].tolist(), fraction=0.012, pad=0.01,
                     label='[MeV/m]  (scale common to the row)')
    fig.suptitle('BC20 first kick (B1LE, s = 0.42 m): 2D wake maps; grey dots are the wake-mesh nodes'
                 + ('; ξ zoomed to the old grid\'s ±4σ' if zoom else ''), x=0.01, ha='left', fontsize=10)
    name = 'first_kick_maps_zoom.png' if zoom else 'first_kick_maps.png'
    fig.savefig(os.path.join(RES, name), dpi=140, bbox_inches='tight')
    print(f'wrote {name}')


if __name__ == '__main__':
    if sys.argv[1] == '--compare':
        compare(sys.argv[2:])
    elif sys.argv[1] == '--maps':
        maps(sys.argv[2:])
        maps(sys.argv[2:], zoom=True)
    else:
        run_one(*sys.argv[1:])
