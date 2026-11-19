"""
Fast test bed for the near-field transverse force on a pedestal behind a spike (the BC20 B1RE
problem, progress_facet_bc20.md 9j-9l), in minutes instead of a 70-minute BC20 replay.

    conda run -n pydfcsr python pyDFCSR_2D/test/quantile_frame/jitter_spike_pedestal.py [variant ...]

Beam: 1e6 particles, a 1.2 um rms spike (45% of the charge) on a 200 um flat pedestal, sigma_x =
100 um (so the fixed near-region cell 0.25 sigma_xi = 25 um is far coarser than the 0.6 um over
which the spike is crossed ahead of a pedestal observer), frozen (no energy spread), in a dipole
R = 5 m, 1.5 m; peak-frame deposition 100 x 300. One run builds the history; then the transverse
force F and the wake W are evaluated at the END of the dipole along a 0.5 um observer sweep through
the pedestal (spike - 120 .. spike - 5 um) on the axis and at xi = +0.5 sigma_x, for every variant
of the quadrature (the history is the same, only Integration_params change between sweeps):
  base      the fixed layout        adapt     near_adapt       adapt_x   near_adapt + near_adapt_x
  *_e05 / *_dz01   near_adapt_eps 0.05 / near_adapt_scan_dz 0.1 on top of adapt_x
  sp64_xp4  near cells 64x finer and 4x x' nodes: the brute-force reference (slow, cached)
Prints per variant: noise (rms of F minus its 5 um smoothing), rms(F - reference), both / the
spike's max|F|, and ms per observer. Writes jitter_spike_pedestal.npz / .png in
benchmark_results/quantile_frame/.
"""
import contextlib
import io
import os
import sys
import tempfile
import time

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', '..'))
from pyDFCSR_2D.CSR import CSR2D  # noqa: E402
from validate_thin_beam import profile_z, P0C, Q, R, LB  # noqa: E402

OUT = os.path.join(HERE, '..', 'benchmark_results', 'quantile_frame')
NPZ = os.path.join(OUT, 'jitter_spike_pedestal.npz')
N = 1_000_000
SIG_X = 100e-6
DEP = dict(method='bspline_comoving', xbins=100, zbins=300, xlim=5, zlim=5, smoothing_sigma=2.0,
           velocity_threhold=1000, frame='peak', frame_quantile=0.005, frame_margin=0.1,
           peak_halfwidth=3.0, peak_taper=2.0)
VARIANTS = {
    'base': {},
    'adapt': dict(near_adapt=True),
    'adapt_x': dict(near_adapt=True, near_adapt_x=True),
    'adapt_x_e05': dict(near_adapt=True, near_adapt_x=True, near_adapt_eps=0.05),
    'adapt_x_dz01': dict(near_adapt=True, near_adapt_x=True, near_adapt_scan_dz=0.1),
    'sp64_xp4': dict(near_cell=0.25 / 64, near_grade=0.05 / 64, xbins=800),
}
REF = 'sp64_xp4'


def build():
    """Run the dipole once (history only) and return the simulation at its end."""
    rng = np.random.default_rng(11)
    z = profile_z('spike', rng, N)
    z = z - z.mean()
    tmp = tempfile.mkdtemp()
    coords = np.column_stack([rng.normal(0, SIG_X, N), rng.normal(0, 1e-9, N),
                              rng.normal(0, 1e-6, N), rng.normal(0, 1e-9, N), z, np.zeros(N)])
    np.savetxt(os.path.join(tmp, 'beam.txt'), coords)
    lat = {'step_size': 0.025, 'B1': {'type': 'dipole', 'L': LB, 'angle': LB / R, 'E1': 0.0, 'E2': 0.0,
                                      'nsep': 1, 'steps': 60}}
    yaml.safe_dump(lat, open(os.path.join(tmp, 'lat.yaml'), 'w'), sort_keys=False)
    cfg = {'input_beam': {'style': 'from_file', 'beamfile': os.path.join(tmp, 'beam.txt'), 'charge': Q, 'energy': P0C},
           'input_lattice': {'lattice_input_file': os.path.join(tmp, 'lat.yaml')},
           'step_control': {'mode': 'manual', 'kick_interval': 'midpoint'},
           'particle_deposition': DEP,
           'CSR_integration': {'n_formation_length': 1},
           'CSR_computation': {'compute_CSR': 1, 'apply_CSR': 0, 'transverse_on': 1, 'xbins': 3, 'zbins': 5,
                               'write_beam': None, 'write_wakes': False, 'workdir': tmp, 'write_name': 'jit'}}
    f = os.path.join(tmp, 'cfg.yaml')
    yaml.safe_dump(cfg, open(f, 'w'), sort_keys=False)
    with contextlib.redirect_stdout(io.StringIO()):
        sim = CSR2D(input_file=f)

        def no_wake():
            sim.dE_dct = np.zeros((sim.CSR_params.xbins, sim.CSR_params.zbins))
            sim.x_kick = np.zeros_like(sim.dE_dct)
        sim.calculate_2D_CSR = no_wake
        sim.run()
    sim.get_CSR_mesh()
    return sim


def sweep(sim, zobs, xobs):
    t0 = time.time()
    out = np.array([sim.get_CSR_wake(sim.beam.position + zz, xobs) for zz in zobs])
    return out[:, 1], out[:, 0], (time.time() - t0) / zobs.size * 1e3


def main(variants):
    os.makedirs(OUT, exist_ok=True)
    res = dict(np.load(NPZ)) if os.path.exists(NPZ) else {}
    t0 = time.time()
    sim = build()
    print(f'history built in {time.time() - t0:.0f} s; s = {sim.beam.position:.3f} m, sigma_xi {sim.beam._sigma_x_transform*1e6:.1f} um, '
          f'sigma_z {sim.beam.sigma_z*1e6:.1f} um, res_z {sim._deposition_resolution()[0]*1e6:.2f} um', flush=True)
    z = sim.beam.z
    h, e = np.histogram(z, bins=2000)
    zs = 0.5 * (e[h.argmax()] + e[h.argmax() + 1])
    zobs = np.arange(zs - 120e-6, zs - 5e-6, 0.5e-6)
    res['zobs'] = zobs
    res['z_spike'] = zs
    xi0 = sim.beam._mean_x
    ip = sim.integration_params
    keep = {k: getattr(ip, k) for k in ('near_adapt', 'near_adapt_x', 'near_adapt_eps', 'near_adapt_scan_dz',
                                         'near_cell', 'near_grade', 'xbins')}
    for v in variants:
        if v == REF and f'F_{REF}_0' in res:
            print(f'{REF}: cached', flush=True)
            continue
        for k, val in keep.items():
            setattr(ip, k, val)
        for k, val in VARIANTS[v].items():
            setattr(ip, k, val)
        for j, xo in enumerate((xi0, xi0 + 0.5 * SIG_X)):
            F, W, ms = sweep(sim, zobs, xo)
            res[f'F_{v}_{j}'], res[f'W_{v}_{j}'], res[f'ms_{v}'] = F, W, ms
        # the spike's own force, for the scale
        res[f'Fspike_{v}'] = sim.get_CSR_wake(sim.beam.position + zs, xi0)[1]
        print(f'{v:14s}: {res[f"ms_{v}"]:7.1f} ms/observer, F at the spike {res[f"Fspike_{v}"]:+.4e} MeV/m', flush=True)
        np.savez(NPZ, **res)
    for k, val in keep.items():
        setattr(ip, k, val)
    report(res, variants)


def report(res, variants):
    from scipy.ndimage import gaussian_filter1d
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    zobs = res['zobs']
    have = [v for v in variants if f'F_{v}_0' in res]
    ref = res.get(f'F_{REF}_0')
    spk = abs(float(res.get(f'Fspike_{REF}', res[f'Fspike_{have[0]}'])))
    print(f'\nspike max|F| (reference) {spk:.4e} MeV/m. Pedestal sweep, row 0 (axis) / row 1 (+0.5 sigma_x): '
          'noise = rms(F - 5 um smoothing), diff = rms(F - sp64_xp4); both / spike max|F|')
    print(f'{"variant":14s} {"noise0":>7s} {"diff0":>7s} {"noise1":>7s} {"diff1":>7s} {"ms/obs":>7s}')
    for v in have:
        row = []
        for j in (0, 1):
            F = res[f'F_{v}_{j}']
            sm = gaussian_filter1d(F, 10.0, mode='nearest')
            noise = np.sqrt(np.mean((F - sm) ** 2)) / spk
            d = np.sqrt(np.mean((F - res[f'F_{REF}_{j}']) ** 2)) / spk if f'F_{REF}_{j}' in res else np.nan
            row += [noise, d]
        print(f'{v:14s} ' + ' '.join(f'{x:7.4f}' for x in row) + f' {float(res[f"ms_{v}"]):7.1f}')
    fig, axs = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    col = {'base': '#0b0b0b', 'adapt': '#2a78d6', 'adapt_x': '#1baf7a', 'adapt_x_e05': '#eda100',
           'adapt_x_dz01': '#9ec5f4', 'sp64_xp4': '#7a1f1e'}
    for j, ax in enumerate(axs):
        for v in have:
            ax.plot((zobs - res['z_spike']) * 1e6, res[f'F_{v}_{j}'], color=col.get(v, '#e34948'),
                    lw=2.0 if v == REF else 1.1, label=v)
        ax.axhline(0, color='#e1e0d9', lw=0.8)
        ax.set_ylabel(f'F [MeV/m], {"axis" if j == 0 else "+0.5 sigma_x"}')
    axs[0].legend(fontsize=8, frameon=False, ncol=3)
    axs[1].set_xlabel('z − z_spike [µm]')
    fig.suptitle('Spike on a pedestal, sigma_x = 100 um, end of a 5 m / 1.5 m dipole: transverse force on the pedestal, '
                 '0.5 um observers', x=0.01, ha='left', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(OUT, 'jitter_spike_pedestal.png'), dpi=140)
    print('wrote jitter_spike_pedestal.png')


if __name__ == '__main__':
    main(sys.argv[1:] or list(VARIANTS))
