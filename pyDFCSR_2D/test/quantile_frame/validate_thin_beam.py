"""
2D must reduce to 1D on a thin beam: the quantile frame against the converged 1D wake.

A beam with sigma_x = 1 um (far below (sigma_spike^2 R)^(1/3) ~ 20 um, so the 1D model is
exact) and no energy spread or chirp, so it stays frozen, enters a dipole (R = 5 m, 1.5 m).
At the end of the dipole, where both the spike and the pedestal are in steady state, the 2D
longitudinal wake is evaluated on a fine z line through the beam axis and compared with
csr1d.wake_in_bend on the same particles (0.1 um bins).

    conda run -n pydfcsr python pyDFCSR_2D/test/quantile_frame/validate_thin_beam.py [profile] [variant ...]

profile: spike (1.2 um rms spike, 45% of the charge, on a 200 um flat pedestal) or horns
(flat top +-80 um, 50%, and two 1.5 um horns at +-75 um, 25% each).
variant: a name from VARIANTS (default: rms quantile). Writes
test/benchmark_results/quantile_frame/thin_<profile>.npz and thin_<profile>.png.
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
from pyDFCSR_2D.csr1d import line_density, wake_in_bend  # noqa: E402
from pyDFCSR_2D.physical_constants import e as E_CHARGE  # noqa: E402

OUT = os.path.join(HERE, '..', 'benchmark_results', 'quantile_frame')
P0C = 1e9
Q = 1e-9
R, LB = 5.0, 1.5
N = 1_000_000

BASE_DEP = dict(xbins=32, zbins=300, xlim=5, zlim=5, smoothing_sigma=2.0, velocity_threhold=1000)
VARIANTS = {
    # the original frame at the original resolution (what BC20 used)
    'rms': dict(dep=dict(BASE_DEP)),
    # quantile frame, 1200 z bins (~0.2 um), z smoothed over 1 bin
    'quantile': dict(dep=dict(BASE_DEP, zbins=1200, frame='quantile', smoothing_sigma_z=1.0)),
    'quantile_z2400': dict(dep=dict(BASE_DEP, zbins=2400, frame='quantile', smoothing_sigma_z=1.0)),
    'quantile_z600': dict(dep=dict(BASE_DEP, zbins=600, frame='quantile', smoothing_sigma_z=1.0)),
    # integration refinement checks (on top of 'quantile')
    'quantile_adapt': dict(dep=dict(BASE_DEP, zbins=1200, frame='quantile', smoothing_sigma_z=1.0,
                                    smoothing_adaptive=True)),
    # 20 000 particles, as BC20: shot noise
    'quantile_20k': dict(n=20000, dep=dict(BASE_DEP, zbins=1200, frame='quantile', smoothing_sigma_z=1.0)),
    'quantile_adapt_20k': dict(n=20000, dep=dict(BASE_DEP, zbins=1200, frame='quantile',
                                                 smoothing_sigma_z=1.0, smoothing_adaptive=True)),
    # peak frame (window around the highest current peak), 300 z bins
    'peak': dict(dep=dict(BASE_DEP, zbins=300, frame='peak')),
    'peak_20k': dict(n=20000, dep=dict(BASE_DEP, zbins=300, frame='peak')),
    'peak_adapt_20k': dict(n=20000, dep=dict(BASE_DEP, zbins=300, frame='peak', smoothing_adaptive=True)),
    'quantile_int2x': dict(dep=dict(BASE_DEP, zbins=1200, frame='quantile', smoothing_sigma_z=1.0),
                           integ=dict(xbins=400, zbins=400, far_zbins=400, near_patch_nr=200,
                                      near_patch_nphi=360)),
}


def profile_z(kind, rng, N=N):
    if kind == 'spike':
        n_s = int(0.45 * N)
        return np.concatenate([rng.uniform(-150e-6, 50e-6, N - n_s), rng.normal(40e-6, 1.2e-6, n_s)])
    k = N // 4
    return np.concatenate([rng.uniform(-80e-6, 80e-6, N - 2 * k),
                           rng.normal(-75e-6, 1.5e-6, k), rng.normal(75e-6, 1.5e-6, k)])


def write_inputs(tmp, kind, variant):
    rng = np.random.default_rng(11)
    N = VARIANTS[variant].get('n', globals()['N'])
    z = profile_z(kind, rng, N)
    z = z - z.mean()
    coords = np.column_stack([rng.normal(0, 1e-6, N), rng.normal(0, 1e-9, N),
                              rng.normal(0, 1e-6, N), rng.normal(0, 1e-9, N), z, np.zeros(N)])
    np.savetxt(os.path.join(tmp, 'beam.txt'), coords)
    lat = {'step_size': 0.025,
           'B1': {'type': 'dipole', 'L': LB, 'angle': LB / R, 'E1': 0.0, 'E2': 0.0, 'nsep': 1,
                  'steps': 60}}
    yaml.safe_dump(lat, open(os.path.join(tmp, 'lat.yaml'), 'w'), sort_keys=False)
    v = VARIANTS[variant]
    cfg = {'input_beam': {'style': 'from_file', 'beamfile': os.path.join(tmp, 'beam.txt'),
                          'charge': Q, 'energy': P0C},
           'input_lattice': {'lattice_input_file': os.path.join(tmp, 'lat.yaml')},
           'step_control': {'mode': 'manual', 'kick_interval': 'midpoint'},
           'particle_deposition': v['dep'],
           'CSR_integration': dict({'n_formation_length': 1}, **v.get('integ', {})),
           'CSR_computation': {'compute_CSR': 1, 'apply_CSR': 0, 'transverse_on': 0,
                               'xbins': 3, 'zbins': 5, 'write_beam': None, 'write_wakes': False,
                               'workdir': tmp, 'write_name': 'thin'}}
    f = os.path.join(tmp, 'cfg.yaml')
    yaml.safe_dump(cfg, open(f, 'w'), sort_keys=False)
    return f


def run_2d(kind, variant, zobs):
    with tempfile.TemporaryDirectory() as tmp:
        f = write_inputs(tmp, kind, variant)
        with contextlib.redirect_stdout(io.StringIO()):
            sim = CSR2D(input_file=f)

            # the history is all that is needed during the run: skip the per-kick mesh
            def no_wake():
                sim.dE_dct = np.zeros((sim.CSR_params.xbins, sim.CSR_params.zbins))
                sim.x_kick = np.zeros_like(sim.dE_dct)
            sim.calculate_2D_CSR = no_wake
            sim.run()
        if sim._quantile_frame():
            sim.get_CSR_mesh()               # beam vx fit for the observer
        t0 = time.time()
        W = np.array([sim.get_CSR_wake(sim.beam.position + zz, 0.0)[0] for zz in zobs])
        return W, time.time() - t0, sim.beam.z.copy(), sim.beam.position


def wake_1d(z, ds=0.1e-6, phi=LB / R):
    S, ds, lam, dlam = line_density(z, 0, Q / E_CHARGE, nsmooth=2, ds=ds)
    return S, wake_in_bend(S, ds, lam, dlam, R, phi) / 1e6        # eV/m -> MeV/m


def main(kind='spike', variants=('rms', 'quantile')):
    os.makedirs(OUT, exist_ok=True)
    zc = (40e-6,) if kind == 'spike' else (-75e-6, 75e-6)
    rng_z = profile_z(kind, np.random.default_rng(11))
    shift = rng_z.mean()
    zobs = np.unique(np.concatenate(
        [np.linspace(-160e-6, 90e-6, 60) if kind == 'spike' else np.linspace(-95e-6, 95e-6, 50)]
        + [np.linspace(c - shift - 8e-6, c - shift + 8e-6, 41) for c in zc]))
    res = {'zobs': zobs}
    zfull = None
    for v in variants:
        W, dt, z, s = run_2d(kind, v, zobs)
        if 'n' not in VARIANTS[v] and zfull is None:
            zfull = z
        res[v] = W
        res[f'{v}_time'] = dt
        print(f'{kind} {v}: {len(zobs)} points in {dt:.1f} s', flush=True)
    # the 1D reference from the TRACKED beam of a full-size (1e6) run -- in the dipole every
    # particle's z moves by ~ x * theta, which broadens a 1 um spike -- not from a run's noise
    z = zfull if zfull is not None else z
    S, W1 = wake_1d(z)
    # NaN beyond the 1D grid (np.interp would hold the edge value flat ahead of the bunch)
    res['W1'] = np.interp(zobs, S, W1, left=np.nan, right=np.nan)
    # summary: mean energy loss rate weighted by the particles, and the spike minimum
    Wp1 = np.interp(z, S, W1)
    for v in variants:
        Wp = np.interp(z, zobs, res[v])
        print(f'  {v:16s} <W> {Wp.mean():.4g} MeV/m (1D {Wp1.mean():.4g}, ratio {Wp.mean()/Wp1.mean():.3f});'
              f' min W {res[v].min():.4g} (1D {np.nanmin(res["W1"]):.4g})')
    tag = '' if variants == ('rms', 'quantile') else '_' + '_'.join(variants)
    np.savez(os.path.join(OUT, f'thin_{kind}{tag}.npz'), **res)
    plot(kind, res, variants, tag)


def plot(kind, res, variants, tag=''):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
    col = {'rms': '#eb6834', 'quantile': '#1baf7a', 'quantile_z2400': '#4a3aa7',
           'quantile_z600': '#eda100', 'quantile_int2x': '#e87ba4', 'quantile_adapt': '#008300',
           'quantile_20k': '#eda100', 'quantile_adapt_20k': '#4a3aa7', 'peak': '#e34948',
           'peak_20k': '#e87ba4', 'peak_adapt_20k': '#008300'}
    z = res['zobs'] * 1e6
    fig, ax = plt.subplots(1, 1, figsize=(9, 4), facecolor=SURF)
    ax.set_facecolor(SURF)
    ax.plot(z, res['W1'], color='#2a78d6', lw=2.5, label='1D (0.1 um bins), exact for this thin beam')
    for v in variants:
        ax.plot(z, res[v], color=col.get(v, INK2), lw=1.6, ls='--' if v == 'rms' else '-',
                marker='.', ms=3, label=f'2D {v}')
    ax.set_xlabel('z [µm]  (head at +z)')
    ax.set_ylabel('longitudinal wake [MeV/m]')
    ax.set_title(f'thin beam in a dipole (R = {R:g} m), {kind}', loc='left', fontsize=10, color=INK)
    ax.grid(color=GRID, lw=0.6)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f'thin_{kind}{tag}.png'), dpi=140, facecolor=SURF)
    plt.close(fig)


if __name__ == '__main__':
    kind = sys.argv[1] if len(sys.argv) > 1 else 'spike'
    main(kind, tuple(sys.argv[2:]) or ('rms', 'quantile'))
