"""
Phase 5.2: does the wake converge once step_size actually RESOLVES the waist?

§6r refined step_size at a waist and found nothing converged -- successive differences of
28-140% down to step_size = 0.00625 m -- and concluded the waist could not be bridged by
sampling. §6w then predicted, from linear optics, that this waist is 2.39 mm wide and needs
step_size <= 0.0006 m for four steps across it.

So §6r stopped 10x short of resolving the thing it was refining. Its conclusion was drawn
entirely from steps that never resolved the waist, which makes it untested rather than
established.

This walks the ladder past that point, to the step_size §6w asks for, and reports the scan's
verdict alongside the wake at every rung. The question is whether "converged" turns on at
the same place "resolved" does.

Configuration notes:
  * drift shortened to 0.35 m. The integration reaches back n_formation_length * L_f =
    1.5 * 0.182 = 0.27 m, so 0.35 m keeps the whole reach inside the history while cutting
    tracking cost 3x against the 1.0 m drift used elsewhere.
  * deposition 128^2 rather than 200^2. At step_size 0.0006 the history holds 251 snapshots,
    and build_interpolant re-stacks the whole deque twice per step; 200^2 pushed peak RSS to
    8.75 GB. 128^2 is held FIXED down the ladder, so it cannot affect the convergence trend.
  * observation at 0.10 m into the dipole, the §6r waist point (waist itself at 0.05 m).
"""
import sys
import os
import gc
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from pyDFCSR_2D.CSR import CSR2D
from pyDFCSR_2D.waist import scan_waists, sigma_from_coords

EXAMPLE_DIR = os.path.join(os.path.dirname(__file__), '..', 'example')
RESULT_DIR = os.path.join(os.path.dirname(__file__), 'benchmark_results', 'waist_step')

SHEAR = 20.0
DRIFT = 0.35
S_DIP = 0.10
DEP_BINS = 128
MESH_XBINS, MESH_ZBINS = 21, 51
# coarse end matches the shipped default; fine end is what 6w asks for
STEPS = (0.05, 0.0125, 0.003, 0.001, 0.0006)


def write_inputs(step, tag):
    beam = {
        'n_particle': 200000, 'species': 'electron',
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
        'transforms': {'s1': {'shear_coefficient':
                              {'units': 'dimensionless', 'value': SHEAR},
                              'type': 'shear z:x'}},
    }
    with open(os.path.join(EXAMPLE_DIR, f'input/ws_beam_{tag}.yaml'), 'w') as f:
        yaml.dump(beam, f, default_flow_style=False, sort_keys=False)

    # step_size MUST be the first key: lattice.py reads it positionally, and a
    # sort_keys=True dump silently moves it last and crashes get_referece_traj.
    lat = {'step_size': step,
           'element_1': {'type': 'drift', 'L': DRIFT, 'nsep': 1},
           'element_2': {'type': 'dipole', 'L': 1.0, 'angle': 1.0, 'nsep': 1},
           'element_3': {'type': 'drift', 'L': 0.5, 'nsep': 1}}
    lp = f'input/ws_lat_{tag}.yaml'
    with open(os.path.join(EXAMPLE_DIR, lp), 'w') as f:
        yaml.dump(lat, f, default_flow_style=False, sort_keys=False)

    cfg = {
        'input_beam': {'style': 'distgen',
                       'distgen_input_file': f'input/ws_beam_{tag}.yaml'},
        'input_lattice': {'lattice_input_file': lp},
        'particle_deposition': {'method': 'bspline_comoving',
                                'xbins': DEP_BINS, 'zbins': DEP_BINS,
                                'xlim': 5, 'zlim': 5, 'smoothing_sigma': 3.0,
                                'poly_degree': 1, 'velocity_threhold': 1000},
        'CSR_integration': {'n_formation_length': 1.5, 'zbins': 200, 'xbins': 200},
        'CSR_computation': {'compute_CSR': 0, 'apply_CSR': 0, 'transverse_on': 1,
                            'xbins': MESH_XBINS, 'zbins': MESH_ZBINS,
                            'xlim': 3, 'zlim': 3,
                            'write_beam': [], 'write_wakes': False,
                            'write_name': f'ws_{tag}', 'workdir': './output'},
    }
    p = f'input/ws_config_{tag}.yaml'
    with open(os.path.join(EXAMPLE_DIR, p), 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    return p


def reldiff(a, b):
    n = np.linalg.norm(b)
    return np.linalg.norm(a - b) / n if n > 0 else np.nan


def abs_rough(w):
    d2 = w[2:] - 2 * w[1:-1] + w[:-2]
    return np.linalg.norm(d2)


def launch_waist(step_ref=STEPS[0]):
    """
    The waist, scanned from the LAUNCH distribution.

    Must not be scanned from the beam as tracked to the observation point: that beam is
    already 0.05 m PAST the waist, so the scan correctly finds nothing ahead of it and
    every width came back nan. The width is a property of the beam and lattice, not of
    step_size, so it is computed once here and steps_across is derived per rung.
    """
    cfg = write_inputs(step_ref, 'launchscan')
    csr = CSR2D(input_file=cfg)          # constructed, never run: this IS the launch state
    b = csr.beam
    rep = scan_waists(csr.lattice.lattice_config,
                      sigma_from_coords(b.x, b.px, b.particle.y, b.particle.py,
                                        b.z, b.pz),
                      step_ref)
    inside = [w for w in rep['waists'] if DRIFT <= w['s'] <= DRIFT + 1.0]
    assert inside, 'no waist found in the dipole from the launch distribution'
    w = inside[0]
    del csr
    gc.collect()
    return w['s'], w['width'], rep['min_steps']


def run_rung(step):
    """One rung. Cached to .npy so a killed run does not lose the expensive rungs."""
    tag = str(step).replace('.', 'p')
    cache = os.path.join(RESULT_DIR, f'cut_{tag}.npy')
    meta = os.path.join(RESULT_DIR, f'meta_{tag}.npy')
    if os.path.exists(cache) and os.path.exists(meta):
        return np.load(cache), np.load(meta, allow_pickle=True).item()

    cfg = write_inputs(step, tag)
    t0 = time.time()
    csr = CSR2D(input_file=cfg)
    target = DRIFT + S_DIP
    csr.run(stop_time=target - 0.5 * step, debug=True)
    csr.get_CSR_mesh()
    assert abs(csr.beam.position - target) < 1e-6, csr.beam.position

    b = csr.beam
    nz, ix = csr.CSR_params.zbins, csr.CSR_params.xbins // 2
    s0 = b.position
    dE = np.zeros(nz)
    for j in range(nz):
        k = ix * nz + j
        dE[j], _ = csr.get_CSR_wake(s0 + csr.CSR_zmesh[k], csr.CSR_xmesh[k])

    m = dict(step=step, snapshots=len(csr.DF_tracker.time_log),
             seconds=time.time() - t0, sigma_z=float(b._sigma_z),
             sigma_xi=float(b._sigma_x_transform))
    os.makedirs(RESULT_DIR, exist_ok=True)
    np.save(cache, dE)
    np.save(meta, np.array(m, dtype=object))
    del csr
    gc.collect()
    return dE, m


def main():
    os.makedirs(RESULT_DIR, exist_ok=True)
    os.chdir(EXAMPLE_DIR)
    lines = []

    def emit(t=''):
        print(t, flush=True)
        lines.append(t)

    emit('=' * 100)
    emit('Phase 5.2: refine step_size PAST the point where the waist is resolved')
    emit('=' * 100)
    emit(f'shear {SHEAR:g}, drift {DRIFT} m, observing {S_DIP} m into the dipole, '
         f'deposition {DEP_BINS}^2 held fixed')
    emit('6r refined only to 0.00625 m; 6w predicts this waist needs <= 0.0006 m')
    emit('')

    s_w, width, min_steps = launch_waist()
    emit(f'  waist (from the launch distribution): s = {s_w:.4f} m '
         f'({s_w-DRIFT:.4f} m into the dipole), width {width*1e3:.4f} mm')
    emit(f'  "resolved" means >= {min_steps:g} steps across that width')
    emit('')

    cuts, metas = {}, {}
    for st in STEPS:
        dE, m = run_rung(st)
        m['steps_across'] = width / st
        m['resolved'] = m['steps_across'] >= min_steps
        cuts[st], metas[st] = dE, m
        emit(f'  ran step_size={st:<8g} {m["snapshots"]:>4} snapshots  '
             f'{m["seconds"]:>6.1f} s  steps across {m["steps_across"]:7.2f}  '
             f'resolved {m["resolved"]}')
    emit('')

    ref = cuts[STEPS[-1]]
    emit(f"  {'step_size':>10} {'snapshots':>10} {'steps across':>13} {'resolved':>9} "
         f"{'abs rough':>10} {'rel L2 vs finest':>17} {'dE range':>24}")
    for st in STEPS:
        m = metas[st]
        emit(f'  {st:>10g} {m["snapshots"]:>10} {m["steps_across"]:>13.2f} '
             f'{str(m["resolved"]):>9} {abs_rough(cuts[st]):>10.5f} '
             f'{reldiff(cuts[st], ref):>17.5f} '
             f'[{cuts[st].min():+9.4f},{cuts[st].max():+9.4f}]')
    emit('')
    emit('  successive differences (the convergence question)')
    for i in range(len(STEPS) - 1):
        emit(f'    {STEPS[i]:>8g} -> {STEPS[i+1]:<8g} : '
             f'{reldiff(cuts[STEPS[i]], cuts[STEPS[i+1]]):.5f}')
    emit('')

    fig, axs = plt.subplots(1, 3, figsize=(17.0, 4.8))
    for st in STEPS:
        axs[0].plot(cuts[st], lw=1.4,
                    label=f'{st:g} m ({metas[st]["steps_across"]:.2f} across)')
    axs[0].set_xlabel('z mesh index')
    axs[0].set_ylabel('dE/dct (MeV/m)')
    axs[0].set_title(f'wake cut, shear {SHEAR:g}, {S_DIP} m into dipole', fontsize=10)
    axs[0].legend(fontsize=8)
    axs[0].grid(alpha=0.25)

    sd = [reldiff(cuts[STEPS[i]], cuts[STEPS[i + 1]]) for i in range(len(STEPS) - 1)]
    axs[1].loglog(STEPS[:-1], sd, 'o-', lw=1.7)
    axs[1].axvline(0.00625, color='r', ls=':', label="6r's finest step")
    axs[1].set_xlabel('step_size (m)')
    axs[1].set_ylabel('rel L2 vs next finer')
    axs[1].set_title('successive differences', fontsize=10)
    axs[1].legend(fontsize=8)
    axs[1].grid(alpha=0.25, which='both')

    sa = [metas[st]['steps_across'] for st in STEPS]
    axs[2].loglog(sa, [abs_rough(cuts[st]) for st in STEPS], 'o-', lw=1.7)
    axs[2].axvline(min_steps, color='r', ls=':',
                   label=f'resolved threshold ({min_steps:g} steps)')
    axs[2].set_xlabel('steps across the waist')
    axs[2].set_ylabel('absolute roughness')
    axs[2].set_title('does roughness fall once the waist is resolved?', fontsize=10)
    axs[2].legend(fontsize=8)
    axs[2].grid(alpha=0.25, which='both')

    fig.tight_layout()
    p = os.path.join(RESULT_DIR, 'waist_step_refine.png')
    fig.savefig(p, dpi=130, bbox_inches='tight')
    emit(f'Plot saved: {p}')
    with open(os.path.join(RESULT_DIR, 'waist_step_log.txt'), 'w') as f:
        f.write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
