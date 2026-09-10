"""
Step 2: does the wake noise track the ghosting parameter G?

The tilt sweep in test_tilt_sweep.py varies the tilt, which changes G *and* the
transverse resolution *and* the integration-band geometry all at once. It cannot
separate them.

This test varies only the *history step size*. At a fixed physical position the
beam has a fixed tilt p' and a fixed sigma, but

    G = |p'(t_{k+1}) - p'(t_k)| * sigma_z / sigma_xi   ~   step_size

so halving step_size halves G with everything else held fixed. Since the
ghosting error grows as G^2 (measured in test_ghosting.py), the prediction is:

    noise from ghosting        ->  falls ~4x per halving of step_size
    noise from OOB / kernel /
    registration / poly_degree ->  roughly flat (these do not involve dt at all)

CSR is computed but NOT applied (apply_CSR = 0) so that every run sees the same
beam evolution and the wakes are compared at identical physical positions.

Noise metric: the relative L2 norm of the second difference of the wake along z.
The physical wake is smooth on the wake-mesh scale, so this isolates
point-to-point jitter from the wake's overall shape and amplitude.
"""
import sys
import os
import time
import numpy as np
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from pyDFCSR_2D.CSR import CSR2D

RESULT_DIR = os.path.join(os.path.dirname(__file__), 'benchmark_results', 'G_scaling')
EXAMPLE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'example'))

SHEAR = 20.0          # z:x shear -> strongly tilted beam
STOP_S = 0.65         # stop just past mid-bend
TARGET_S = 0.60       # compare wakes at this physical position
STEP_SIZES = [0.1, 0.05, 0.025]

WAKE_XBINS = 5
WAKE_ZBINS = 40
INT_BINS = 100
DEP_BINS = 200
N_PARTICLE = 200000


# ---------------------------------------------------------------------------
# Input generation
# ---------------------------------------------------------------------------
def write_beam(path):
    beam = {
        'n_particle': N_PARTICLE,
        'species': 'electron',
        'px_dist': {'sigma_px': {'units': 'keV/c', 'value': 25.0}, 'type': 'gaussian'},
        'py_dist': {'sigma_py': {'units': 'keV/c', 'value': 25.0}, 'type': 'gaussian'},
        'pz_dist': {'avg_pz': {'units': 'GeV/c', 'value': 5},
                    'sigma_pz': {'units': 'MeV/c', 'value': 0.5}, 'type': 'gaussian'},
        'x_dist': {'sigma_x': {'units': 'um', 'value': 50}, 'type': 'gaussian'},
        'y_dist': {'sigma_y': {'units': 'um', 'value': 5}, 'type': 'gaussian'},
        'random': {'type': 'hammersley'},
        'start': {'tstart': {'units': 'sec', 'value': 0}, 'type': 'time'},
        'total_charge': {'units': 'nC', 'value': 1},
        'z_dist': {'avg_z': {'units': 'mm', 'value': 0},
                   'sigma_z': {'units': 'um', 'value': 50}, 'type': 'gaussian'},
        'transforms': {'s1': {'shear_coefficient': {'units': 'dimensionless',
                                                    'value': float(SHEAR)},
                              'type': 'shear z:x'}},
    }
    with open(path, 'w') as f:
        yaml.dump(beam, f, default_flow_style=False)


def write_lattice(path, step_size):
    lat = {
        'step_size': float(step_size),
        'element_1': {'type': 'drift', 'L': 0.1, 'nsep': 1},
        'element_2': {'type': 'dipole', 'L': 1.0, 'angle': 1.0,
                      'E1': 0, 'E2': 0, 'FINT': 0.0, 'FINTX': 0.5,
                      'HGAP': 0.1, 'HGAPX': 0.0, 'FRINGE_AT': 'both_ends',
                      'FRINGE_TYPE': 'linear_edge', 'TILT': 0.0, 'nsep': 1},
        'element_3': {'type': 'drift', 'L': 0.5, 'nsep': 1},
    }
    with open(path, 'w') as f:
        # get_referece_traj() treats the first key as step_size and the rest as
        # elements, so key order is load-bearing here.
        yaml.dump(lat, f, default_flow_style=False, sort_keys=False)


def write_config(path, beam_rel, lattice_rel, method, write_name):
    if method == 'bspline_fft':
        dep = {'method': 'bspline_fft', 'xbins': DEP_BINS, 'zbins': DEP_BINS,
               'xlim': 5, 'zlim': 5, 'smoothing_sigma': 3.0, 'poly_degree': 3,
               'velocity_threhold': 1000}
    else:
        dep = {'xbins': 100, 'zbins': 100, 'xlim': 5, 'zlim': 5,
               'filter_order': 2, 'filter_window': 5, 'velocity_threhold': 1000}

    config = {
        'input_beam': {'style': 'distgen', 'distgen_input_file': beam_rel},
        'input_lattice': {'lattice_input_file': lattice_rel},
        'particle_deposition': dep,
        'CSR_integration': {'n_formation_length': 1.5,
                            'zbins': INT_BINS, 'xbins': INT_BINS},
        'CSR_computation': {'compute_CSR': 1, 'apply_CSR': 0, 'transverse_on': 1,
                            'xbins': WAKE_XBINS, 'zbins': WAKE_ZBINS,
                            'xlim': 3, 'zlim': 3,
                            'write_beam': [], 'write_wakes': False,
                            'write_name': write_name, 'workdir': './output'},
    }
    with open(path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def roughness(wake):
    """
    Relative L2 norm of the second difference along z (axis 1).

    A smooth wake sampled on this mesh has second differences of order
    (d2W/dz2) * dz^2, tiny compared with W. Point-to-point jitter does not.
    """
    if wake.shape[1] < 3:
        return np.nan
    d2 = wake[:, 2:] - 2 * wake[:, 1:-1] + wake[:, :-2]
    n = np.linalg.norm(wake)
    return np.linalg.norm(d2) / n if n > 0 else np.nan


def beam_frame_stats(csr):
    """(slope, sigma_xi, sigma_z) of the current beam, deg-1 tilt removed."""
    x = csr.beam.x
    z = csr.beam.z
    p = np.polyfit(z, x, 1)
    xi = x - np.polyval(p, z)
    return p[0], np.std(xi), np.std(z)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
def run_case(config_rel, step_size):
    """Run one config, capturing per-step frame stats and wakes."""
    os.chdir(EXAMPLE_DIR)
    csr = CSR2D(input_file=config_rel)

    records = []
    original = csr.calculate_2D_CSR

    def wrapped():
        slope, s_xi, s_z = beam_frame_stats(csr)
        original()
        records.append({
            'position': csr.beam.position,
            'slope': slope,
            'sigma_xi': s_xi,
            'sigma_z': s_z,
            'dE_dct': csr.dE_dct.copy(),
            'x_kick': csr.x_kick.copy(),
            'zrange': csr.CSR_zrange.copy(),
        })

    csr.calculate_2D_CSR = wrapped
    csr.run(stop_time=STOP_S)

    # G between consecutive history snapshots
    for i, r in enumerate(records):
        if i == 0:
            r['G'] = 0.0
        else:
            prev = records[i - 1]
            dp = abs(r['slope'] - prev['slope'])
            r['G'] = dp * r['sigma_z'] / r['sigma_xi']
    return records


def pick(records, target_s):
    return min(records, key=lambda r: abs(r['position'] - target_s))


def main():
    os.makedirs(RESULT_DIR, exist_ok=True)

    beam_rel = 'input/Gscale_beam.yaml'
    write_beam(os.path.join(EXAMPLE_DIR, beam_rel))

    results = {}
    for method in ('bspline_fft', 'legacy'):
        results[method] = {}
        for ss in STEP_SIZES:
            tag = f'{method}_ss{ss}'
            lat_rel = f'input/Gscale_lattice_ss{ss}.yaml'
            cfg_rel = f'input/Gscale_config_{tag}.yaml'
            write_lattice(os.path.join(EXAMPLE_DIR, lat_rel), ss)
            write_config(os.path.join(EXAMPLE_DIR, cfg_rel), beam_rel, lat_rel,
                         method, f'Gscale_{tag}')

            print(f'\n{"="*70}\n  {method}, step_size = {ss}\n{"="*70}')
            t0 = time.time()
            rec = run_case(cfg_rel, ss)
            print(f'  ({time.time()-t0:.0f} s, {len(rec)} wake steps)')
            results[method][ss] = rec

    # ---------------- report ----------------
    lines = []

    def emit(s=''):
        print(s)
        lines.append(s)

    emit('=' * 78)
    emit('Step 2: wake roughness vs history step size at FIXED tilt')
    emit('=' * 78)
    emit(f'shear z:x = {SHEAR}, wake compared at s ~ {TARGET_S} m, '
         f'apply_CSR = 0 (identical beam evolution)')
    emit(f'wake mesh {WAKE_XBINS}x{WAKE_ZBINS}, integration {INT_BINS}^2, '
         f'deposition {DEP_BINS}^2')
    emit('')
    emit('G is measured between the two history snapshots bracketing the query,')
    emit('at the compared position. It should scale linearly with step_size.')
    emit('')

    hdr = (f"{'method':<12} {'step':>7} {'G':>8} {'|slope|':>9} "
           f"{'s_xi/um':>9} {'rough(dE)':>11} {'rough(xk)':>11} {'s_cmp':>7}")
    emit(hdr)
    emit('-' * len(hdr))

    summary = {m: {'G': [], 'rdE': [], 'rxk': []} for m in results}
    for method in results:
        for ss in STEP_SIZES:
            r = pick(results[method][ss], TARGET_S)
            rdE = roughness(r['dE_dct'])
            rxk = roughness(r['x_kick'])
            emit(f"{method:<12} {ss:>7.3f} {r['G']:>8.3f} {abs(r['slope']):>9.3f} "
                 f"{r['sigma_xi']*1e6:>9.2f} {rdE:>11.5f} {rxk:>11.5f} "
                 f"{r['position']:>7.3f}")
            summary[method]['G'].append(r['G'])
            summary[method]['rdE'].append(rdE)
            summary[method]['rxk'].append(rxk)

    emit('')
    emit('Scaling of roughness with G (ghosting predicts ~G^2, i.e. 4x per halving):')
    for method in results:
        G = summary[method]['G']
        r = summary[method]['rdE']
        emit(f'  {method}:')
        for i in range(1, len(G)):
            if G[i] > 0 and r[i] > 0:
                gr = G[i - 1] / G[i]
                rr = r[i - 1] / r[i]
                p = np.log(rr) / np.log(gr) if gr > 1 else np.nan
                emit(f'    G {G[i-1]:.3f} -> {G[i]:.3f} ({gr:.2f}x) : '
                     f'roughness {r[i-1]:.5f} -> {r[i]:.5f} ({rr:.2f}x)  '
                     f'=> effective power {p:.2f}')

    # ---------------- plots ----------------
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    ax = axes[0]
    for method, mk in (('bspline_fft', 'ro-'), ('legacy', 'bs--')):
        ax.loglog(summary[method]['G'], summary[method]['rdE'], mk, label=method)
    Gs = np.array(summary['bspline_fft']['G'])
    Gs = Gs[Gs > 0]
    if len(Gs) >= 2:
        ref = summary['bspline_fft']['rdE'][-1]
        ax.loglog(Gs, ref * (Gs / Gs.min()) ** 2, 'k:', alpha=0.6,
                  label=r'$\propto G^2$ (ghosting)')
        ax.loglog(Gs, [ref] * len(Gs), 'g:', alpha=0.6,
                  label='flat (kernel/OOB)')
    ax.set_xlabel('G at the compared position')
    ax.set_ylabel('wake roughness (rel. 2nd difference along z)')
    ax.set_title('Roughness vs G\n(tilt fixed, only step_size varied)')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which='both')

    ax = axes[1]
    for method, mk in (('bspline_fft', 'ro-'), ('legacy', 'bs--')):
        for ss, ls in zip(STEP_SIZES, ('-', '--', ':')):
            r = pick(results[method][ss], TARGET_S)
            w = r['dE_dct'][r['dE_dct'].shape[0] // 2]
            ax.plot(r['zrange'] * 1e6, w, ls,
                    color='r' if method == 'bspline_fft' else 'b',
                    alpha=0.8, label=f'{method} ss={ss}')
    ax.set_xlabel(r'z ($\mu$m)')
    ax.set_ylabel('dE/dct (MeV/m)')
    ax.set_title(f'Wake cut at mid-x, s = {TARGET_S} m')
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[2]
    for method, c in (('bspline_fft', 'r'), ('legacy', 'b')):
        for ss, ls in zip(STEP_SIZES, ('-', '--', ':')):
            rec = results[method][ss]
            ax.plot([r['position'] for r in rec], [r['G'] for r in rec],
                    ls, color=c, alpha=0.8, label=f'{method} ss={ss}')
    ax.axhline(1.0, color='k', ls=':', alpha=0.6)
    ax.text(0.02, 1.05, 'G = 1', fontsize=8)
    ax.set_xlabel('s (m)')
    ax.set_ylabel('G per history step')
    ax.set_yscale('log')
    ax.set_title('Ghosting parameter along the lattice')
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, which='both')

    fig.suptitle('Step 2: is the wake noise controlled by G?', fontsize=13)
    plt.tight_layout()
    p = os.path.join(RESULT_DIR, 'roughness_vs_G.png')
    plt.savefig(p, dpi=130)
    plt.close()
    emit('')
    emit(f'Plot saved: {p}')

    with open(os.path.join(RESULT_DIR, 'G_scaling_log.txt'), 'w') as f:
        f.write('\n'.join(lines))


if __name__ == '__main__':
    main()
