"""
Calibrate `edge_steps`, the last uncalibrated constant that changes the schedule.

§11h fixed the transient SCALE at a bend face -- entrance = overtaking length
(24 R^2 * 5 sigma_z)^(1/3), exit = R*phi_m/2 from Stupakov & Emma Eq. 10 -- but not the divisor:

    h_2 = max(L_f_edge, d_edge) / edge_steps

`edge_steps = 20` was a guess. It controls how finely the history is sampled where the wake is
genuinely transient, so the observable is the WAKE near a face, measured the way §6x and §11l
measured theirs: against a fine uniform reference, with the observation point pinned by
`force_nodes` so no rung is offset from another.

Run for BOTH a weak and a strong bend: `python test_edge_steps_calibrate.py [weak|strong]`.
Which one constrains the constant is itself a finding -- see the BEND_CASES comment below.

Two observation points, because the two faces have different physics:
    just inside the ENTRANCE  -- the transient is building, Case A
    just after the EXIT       -- the transient is decaying, Case C

`compute_CSR = 0`: this measures the wake the schedule can RECONSTRUCT, not the integrated kick
(that is `kappa`'s business, §11m). So no kicks are applied and the runs are cheap.
"""
import gc
import os
import sys
import time

import matplotlib
import numpy as np
import yaml

matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from pyDFCSR_2D.CSR import CSR2D

EXAMPLE_DIR = os.path.join(os.path.dirname(__file__), '..', 'example')
RESULT_DIR = os.path.join(os.path.dirname(__file__), 'benchmark_results', 'edge_steps')
# set after CASE is known, below

#
# BEND STRENGTH IS THE WHOLE EXPERIMENT, and the first attempt got it backwards.
#
# I chose a WEAK bend (0.1 rad) on §11h's reasoning that the entrance and exit scales differ most
# there. But `edge_steps` divides the transient scale, so what matters is whether the transient is a
# LOCALIZED feature the schedule has to resolve:
#
#   angle 0.1  R=10.0  L_entrance = 0.8434 m  -> 0.84 of the 1 m bend: the whole bend is transient,
#                                               there is no sharp feature, and refining the face
#                                               resolves nothing. Measured: the wake is unchanged to
#                                               1e-7 over a 16x range of edge_steps.
#   angle 1.0  R= 1.0  L_entrance = 0.1817 m  -> 0.18 of the bend: a genuinely localized entrance
#                                               transient exists.
#
# So both are run. The weak bend is kept because its null result is informative -- it says the edge
# term costs nothing to leave loose in the chicane regime -- and the strong bend is what can actually
# constrain the constant.
BEND_CASES = {'weak': 0.1, 'strong': 1.0}
CASE = sys.argv[1] if len(sys.argv) > 1 else 'weak'
assert CASE in BEND_CASES, f'usage: {sys.argv[0]} [weak|strong]'
DRIFT, BEND_L, POST = 0.35, 1.0, 0.5
BEND_ANGLE = BEND_CASES[CASE]
S_ENTRANCE, S_EXIT = DRIFT, DRIFT + BEND_L
# just inside the entrance, and just after the exit
S_OBS_LIST = (round(S_ENTRANCE + 0.02, 6), round(S_EXIT + 0.02, 6))
DEP_BINS = 128
MESH_XBINS, MESH_ZBINS = 21, 51

RESULT_DIR = os.path.join(RESULT_DIR, CASE)

EDGE_STEPS = (5.0, 10.0, 20.0, 40.0, 80.0)
REF_STEP = 0.0008        # uniform reference, finer than the finest rung's min h_eff


def write_inputs(tag, step_size, step_control=None):
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
    }
    tag = f'{CASE}_{tag}'
    bp = f'input/es_beam_{tag}.yaml'
    with open(os.path.join(EXAMPLE_DIR, bp), 'w') as f:
        yaml.dump(beam, f, default_flow_style=False, sort_keys=False)

    lat = {'step_size': step_size,
           'element_1': {'type': 'drift', 'L': DRIFT, 'nsep': 1},
           'element_2': {'type': 'dipole', 'L': BEND_L, 'angle': BEND_ANGLE, 'nsep': 1},
           'element_3': {'type': 'drift', 'L': POST, 'nsep': 1}}
    lp = f'input/es_lat_{tag}.yaml'
    with open(os.path.join(EXAMPLE_DIR, lp), 'w') as f:
        yaml.dump(lat, f, default_flow_style=False, sort_keys=False)

    cfg = {
        'input_beam': {'style': 'distgen', 'distgen_input_file': bp},
        'input_lattice': {'lattice_input_file': lp},
        'particle_deposition': {'method': 'bspline_comoving',
                                'xbins': DEP_BINS, 'zbins': DEP_BINS,
                                'xlim': 5, 'zlim': 5, 'smoothing_sigma': 3.0,
                                'poly_degree': 1, 'velocity_threhold': 1000},
        'CSR_integration': {'n_formation_length': 1.5, 'zbins': 100, 'xbins': 100},
        'CSR_computation': {'compute_CSR': 0, 'apply_CSR': 0, 'transverse_on': 1,
                            'xbins': MESH_XBINS, 'zbins': MESH_ZBINS, 'xlim': 3, 'zlim': 3,
                            'write_beam': [], 'write_wakes': False,
                            'write_name': f'es_{tag}', 'workdir': './output'},
    }
    if step_control is not None:
        cfg['step_control'] = step_control
    p = f'input/es_config_{tag}.yaml'
    with open(os.path.join(EXAMPLE_DIR, p), 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    return p


def reldiff(a, b):
    n = np.linalg.norm(b)
    return np.linalg.norm(a - b) / n if n > 0 else np.nan


def cut_at_obs(csr):
    csr.get_CSR_mesh()
    nz, ix = csr.CSR_params.zbins, csr.CSR_params.xbins // 2
    s0 = csr.beam.position
    dE = np.zeros(nz)
    xk = np.zeros(nz)
    for j in range(nz):
        k = ix * nz + j
        dE[j], xk[j] = csr.get_CSR_wake(s0 + csr.CSR_zmesh[k], csr.CSR_xmesh[k])
    return dE, xk


def run_case(tag, step_size, step_control, offsets):
    """One fresh run per observation point -- run() is not resumable (§11m)."""
    cache = os.path.join(RESULT_DIR, f'cut_{tag}.npz')
    if os.path.exists(cache):
        d = np.load(cache, allow_pickle=True)
        return d['cuts'].item(), d['meta'].item()

    cuts, m = {}, dict(tag=tag)
    t0 = time.time()
    for s_obs, off in zip(S_OBS_LIST, offsets):
        cfg = write_inputs(tag, step_size, step_control)
        csr = CSR2D(input_file=cfg)
        sch = csr.lattice.schedule
        m['nodes'] = int(sch.n_nodes)
        csr.run(stop_time=s_obs - off, debug=True)
        m[f'landed_{s_obs}'] = float(csr.beam.position)
        m[f'snaps_{s_obs}'] = len(csr.DF_tracker.time_log)
        cuts[s_obs] = cut_at_obs(csr)
        if getattr(sch, 'auto_h_eff', None) is not None:
            j = int(np.argmin(np.abs(sch.auto_s_scan - s_obs)))
            m[f'h_eff_{s_obs}'] = float(sch.auto_h_eff[j])
            m[f'h_2_{s_obs}'] = float(sch.auto_parts['h_2'][j])
            m[f'Lfe_{s_obs}'] = float(sch.auto_parts['L_f_edge'][j])
        del csr
        gc.collect()
    m['seconds'] = time.time() - t0
    os.makedirs(RESULT_DIR, exist_ok=True)
    np.savez(cache, cuts=np.array(cuts, dtype=object), meta=np.array(m, dtype=object))
    return cuts, m


def main():
    os.makedirs(RESULT_DIR, exist_ok=True)
    os.chdir(EXAMPLE_DIR)
    lines = []

    def emit(t=''):
        print(t, flush=True)
        lines.append(t)

    emit('=' * 100)
    emit('Calibrating edge_steps: the wake near a bend FACE against schedule cost')
    emit('=' * 100)
    emit(f'WEAK bend {BEND_ANGLE} rad (R = {BEND_L / BEND_ANGLE:g} m), unsheared beam, '
         f'deposition {DEP_BINS}^2 fixed')
    emit(f'entrance at s = {S_ENTRANCE}, exit at s = {S_EXIT}')
    emit(f'observing just inside the entrance (s = {S_OBS_LIST[0]}) and just after the '
         f'exit (s = {S_OBS_LIST[1]})')
    emit('compute_CSR = 0: this measures the wake the schedule can RECONSTRUCT, not the kick')
    emit('')

    # The reference must land on the SAME s as the rungs. A uniform grid cannot: at step 0.0008
    # its nodes near 0.37 are 0.3688 / 0.3696 / 0.3704, so `stop_time` lands 0.4 mm short -- and
    # the entrance wake varies 62 % over 5.6 mm here (dE peak 0.2348 at 0.3696 vs 0.3798 at
    # 0.3752), so even sub-mm offsets show up as a several-percent amplitude difference. That is
    # what produced a spurious constant ~3.7 % error floor in the first version of this test and
    # made every rung look equally wrong. The reference therefore uses `auto` with a very fine
    # h_max and the same force_nodes, so it is fine AND co-located.
    emit(f'  reference: auto, h_max = {REF_STEP}, force_nodes pinned (uniform cannot land on the'
         f' observation points)')
    ref_sc = {'mode': 'auto', 'kick_interval': 'midpoint', 'n_sub': 400,
              'edge_steps': 20.0, 'h_max': REF_STEP,
              'force_nodes': [float(x) for x in S_OBS_LIST]}
    cuts_ref, m_ref = run_case('ref', 0.05, ref_sc, [0.0] * len(S_OBS_LIST))
    for s_obs in S_OBS_LIST:
        assert abs(m_ref[f'landed_{s_obs}'] - s_obs) < 1e-9, \
            f"reference landed {m_ref[f'landed_{s_obs}']} not {s_obs}"
    for s_obs in S_OBS_LIST:
        emit(f"    s={s_obs}: landed {m_ref[f'landed_{s_obs}']:.6f}, "
             f"{m_ref[f'snaps_{s_obs}']} snapshots, "
             f"dE range [{cuts_ref[s_obs][0].min():+.4f}, {cuts_ref[s_obs][0].max():+.4f}]")
    emit(f"    total {m_ref['seconds']:.0f} s")
    emit('')

    rows = []
    for es in EDGE_STEPS:
        tag = 'e' + str(es).replace('.', 'p')
        sc = {'mode': 'auto', 'kick_interval': 'midpoint', 'n_sub': 400,
              'edge_steps': float(es), 'force_nodes': [float(x) for x in S_OBS_LIST]}
        cuts, m = run_case(tag, 0.05, sc, [0.0] * len(S_OBS_LIST))
        for s_obs in S_OBS_LIST:
            assert abs(m[f'landed_{s_obs}'] - s_obs) < 1e-9, \
                f"edge_steps={es} landed {m[f'landed_{s_obs}']} not {s_obs}"
        m['edge_steps'] = es
        rows.append((m, cuts))
        emit(f"  ran edge_steps={es:<6g} {m['nodes']:>5} nodes, {m['seconds']:>5.0f} s")
    emit('')

    labels = {S_OBS_LIST[0]: 'ENTRANCE (Case A, transient building)',
              S_OBS_LIST[1]: 'EXIT (Case C, transient decaying)'}
    for s_obs in S_OBS_LIST:
        dE_ref, xk_ref = cuts_ref[s_obs]
        emit(f'  s = {s_obs} -- {labels[s_obs]}')
        emit(f"    {'edge_steps':>10} {'snaps':>7} {'h_eff':>10} {'h_2':>10} {'L_f_edge':>9} "
             f"{'dE relL2':>10} {'xk relL2':>10} {'dE range':>22}")
        emit(f"    {'REFERENCE':>10} {m_ref[f'snaps_{s_obs}']:>7} {REF_STEP:>10.6f} {'-':>10} "
             f"{'-':>9} {'reference':>10} {'reference':>10} "
             f"[{dE_ref.min():+8.4f},{dE_ref.max():+8.4f}]")
        for m, cuts in rows:
            dE, xk = cuts[s_obs]
            emit(f"    {m['edge_steps']:>10g} {m[f'snaps_{s_obs}']:>7} "
                 f"{m.get(f'h_eff_{s_obs}', np.nan):>10.6f} "
                 f"{m.get(f'h_2_{s_obs}', np.nan):>10.6f} "
                 f"{m.get(f'Lfe_{s_obs}', np.nan):>9.4f} "
                 f"{reldiff(dE, dE_ref):>10.5f} {reldiff(xk, xk_ref):>10.5f} "
                 f"[{dE.min():+8.4f},{dE.max():+8.4f}]")
        emit('    successive differences')
        for i in range(len(rows) - 1):
            emit(f"      {rows[i][0]['edge_steps']:>6g} -> {rows[i+1][0]['edge_steps']:<6g} : "
                 f"{reldiff(rows[i][1][s_obs][0], rows[i + 1][1][s_obs][0]):.5f}")
        emit('')

    emit('  cost')
    for m, _ in rows:
        emit(f"    edge_steps={m['edge_steps']:<6g} {m['nodes'] / m_ref['nodes']:.3f}x the nodes, "
             f"{m['seconds'] / m_ref['seconds']:.3f}x the time")

    # --- plot -------------------------------------------------------------------------------
    n = len(S_OBS_LIST)
    fig, axes = plt.subplots(2, n, figsize=(7 * n, 9))
    for c, s_obs in enumerate(S_OBS_LIST):
        dE_ref = cuts_ref[s_obs][0]
        zc = np.linspace(-1, 1, dE_ref.size)
        ax = axes[0, c]
        ax.plot(zc, dE_ref, 'k-', lw=2.4, label=f'reference (h_max={REF_STEP})')
        for m, cuts in rows:
            ax.plot(zc, cuts[s_obs][0], lw=1.0, label=f"edge_steps={m['edge_steps']:g}")
        ax.set_xlabel('z / (zlim sigma_z)')
        ax.set_ylabel('dE/ds  [MeV/m]')
        ax.set_title(f's = {s_obs} -- {labels[s_obs]}', fontsize=10)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

        ax = axes[1, c]
        es = np.array([m['edge_steps'] for m, _ in rows])
        err = np.array([reldiff(cuts[s_obs][0], dE_ref) for _, cuts in rows])
        ax.loglog(es, np.maximum(err, 1e-12), 'o-')
        ax.set_xlabel('edge_steps  (finer ->)')
        ax.set_ylabel('rel L2 vs reference')
        ax.set_title(f'accuracy against edge_steps, s = {s_obs}')
        ax.grid(alpha=0.3, which='both')
    fig.tight_layout()
    fig.savefig(os.path.join(RESULT_DIR, 'edge_steps_calibrate.png'), dpi=130)

    with open(os.path.join(RESULT_DIR, 'edge_steps_log.txt'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"\nwrote {RESULT_DIR}/edge_steps_calibrate.png")


if __name__ == '__main__':
    main()
