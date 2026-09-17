"""
Calibrate `tau_frac`, the one uncalibrated constant in `auto` that decides its cost.

§11i found that h_tau BINDS almost everywhere inside a bend -- 0.0075 m against h_1's 0.108 m,
14x tighter -- so tau_frac alone accounts for the whole 3.9x snapshot rise it measured. It was a
guess. §6x calibrated m_steps by measuring wake error against steps across a waist and taking the
knee; this does the same for tau_frac, on the same case, so the two constants are calibrated by one
method.

The measurement, exactly:
  * one lattice, one beam (the §6x shear-20 waist case), deposition held FIXED at 128^2 so only
    the schedule varies;
  * `auto` with every other constant at its default, tau_frac swept over a dyadic ladder;
  * `force_nodes=[s_obs]` so every rung observes at EXACTLY the same s. Without it each rung lands
    up to half a step from the target -- up to 1.5 mm against a 2.4 mm waist, which would be the
    dominant "error" in the table and would have nothing to do with tau_frac;
  * a uniform reference at step_size 0.0006, which §6x measured to be converged (0.7% at 0.001,
    and its dE range stable to 4 digits).

The reference is RECOMPUTED here rather than reused from §6x's cache: §11k changed the frame_blend
default from 'coeff' to 'orient', so the cached cuts describe a different code.
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
from pyDFCSR_2D.waist import propagate_frame, scan_waists, sigma_from_coords

EXAMPLE_DIR = os.path.join(os.path.dirname(__file__), '..', 'example')
RESULT_DIR = os.path.join(os.path.dirname(__file__), 'benchmark_results', 'tau_frac')

SHEAR = 20.0
DRIFT = 0.35
S_DIP = 0.10
DEP_BINS = 128
MESH_XBINS, MESH_ZBINS = 21, 51

# Two observation points, because tau_frac's consequence is NOT uniform along the lattice. The
# error a tau slip causes scales as dtau*sigma_z/sigma_xi, i.e. with the tilt amplification
# sigma_x/sigma_xi, and §6r established the tolerance at amplification 525. Measured here:
#
#     s = 0.45 (0.10 into the bend)   amplification  20    <- §6x's waist point
#     s = 0.75 (0.40 into the bend)   amplification 109
#
# At 0.45 the longitudinal term h_1 is comparably tight, so relaxing tau cannot show up. Calibrating
# only there would license a tau_frac that is wrong deeper in the bend.
S_OBS_LIST = (DRIFT + S_DIP, DRIFT + 0.40)

TAU_FRACS = (4.0, 2.0, 1.0, 0.5, 0.25, 0.125)
REF_STEP = 0.0006          # §6x's converged rung


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
        'transforms': {'s1': {'shear_coefficient':
                              {'units': 'dimensionless', 'value': SHEAR},
                              'type': 'shear z:x'}},
    }
    bp = f'input/tf_beam_{tag}.yaml'
    with open(os.path.join(EXAMPLE_DIR, bp), 'w') as f:
        yaml.dump(beam, f, default_flow_style=False, sort_keys=False)

    lat = {'step_size': step_size,
           'element_1': {'type': 'drift', 'L': DRIFT, 'nsep': 1},
           'element_2': {'type': 'dipole', 'L': 1.0, 'angle': 1.0, 'nsep': 1},
           'element_3': {'type': 'drift', 'L': 0.5, 'nsep': 1}}
    lp = f'input/tf_lat_{tag}.yaml'
    with open(os.path.join(EXAMPLE_DIR, lp), 'w') as f:
        yaml.dump(lat, f, default_flow_style=False, sort_keys=False)

    cfg = {
        'input_beam': {'style': 'distgen', 'distgen_input_file': bp},
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
                            'write_name': f'tf_{tag}', 'workdir': './output'},
    }
    if step_control is not None:
        cfg['step_control'] = step_control
    p = f'input/tf_config_{tag}.yaml'
    with open(os.path.join(EXAMPLE_DIR, p), 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    return p


def reldiff(a, b):
    n = np.linalg.norm(b)
    return np.linalg.norm(a - b) / n if n > 0 else np.nan


def abs_rough(w):
    d2 = w[2:] - 2 * w[1:-1] + w[:-2]
    return np.linalg.norm(d2)


def cut_at_obs(csr):
    """Mid-x longitudinal cut of the wake at the current position."""
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
    """
    One FRESH run per observation point.

    Reusing a single CSR2D and calling run() twice looks like it should work -- the run passes
    through both points -- but `run()` always restarts its `for ele in lattice_config` loop from
    the first element, so a second call re-tracks an already-advanced beam from the lattice
    entrance. Measured: the second call asked to stop at 0.750 and landed at 0.778, and the
    density history it produced was not the history of a single forward pass. Caught only because
    this test asserts on beam.position; without that assertion the contaminated cut would have
    looked plausible.

    `offsets` gives, per observation point, how far short of it to request the stop, so a uniform
    run (whose nodes cannot be steered) lands on its own last node before the target rather than
    stepping past it.
    """
    cache = os.path.join(RESULT_DIR, f'cut_{tag}.npz')
    if os.path.exists(cache):
        d = np.load(cache, allow_pickle=True)
        return d['cuts'].item(), d['meta'].item()

    cuts = {}
    m = dict(tag=tag)
    t0 = time.time()
    for s_obs, off in zip(S_OBS_LIST, offsets):
        cfg = write_inputs(tag, step_size, step_control)
        csr = CSR2D(input_file=cfg)
        sch = csr.lattice.schedule
        m['nodes_total'] = int(sch.n_nodes)
        csr.run(stop_time=s_obs - off, debug=True)
        landed = csr.beam.position
        cuts[s_obs] = cut_at_obs(csr)
        m[f'landed_{s_obs}'] = float(landed)
        m[f'snaps_{s_obs}'] = len(csr.DF_tracker.time_log)
        m[f'kicks_{s_obs}'] = int(sch.is_kick[sch.s_nodes <= landed + 1e-12].sum())
        if getattr(sch, 'auto_h_eff', None) is not None:
            j = int(np.argmin(np.abs(sch.auto_s_scan - s_obs)))
            m[f'h_eff_{s_obs}'] = float(sch.auto_h_eff[j])
            m[f'h_tau_{s_obs}'] = float(sch.auto_parts['h_tau'][j])
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

    emit('=' * 108)
    emit('Calibrating tau_frac: wake error against schedule cost, at two tilt amplifications')
    emit('=' * 108)
    emit(f'shear {SHEAR:g}, drift {DRIFT} m, deposition {DEP_BINS}^2 FIXED so only the schedule varies')
    emit(f'observing at s = {S_OBS_LIST}, i.e. {[round(s - DRIFT, 3) for s in S_OBS_LIST]} m into the dipole')
    emit('every auto rung pins both points with force_nodes, so no rung is offset from another')
    emit('')

    # frame at the observation points, so the amplification is on the record
    cfg = write_inputs('scan', 0.05, None)
    csr = CSR2D(input_file=cfg)
    b = csr.beam
    sigma0 = sigma_from_coords(b.x, b.px, b.particle.y, b.particle.py, b.z, b.pz)
    latcfg = csr.lattice.lattice_config
    del csr
    gc.collect()
    rep = scan_waists(latcfg, sigma0, 0.05)
    inside = [w for w in rep['waists'] if DRIFT <= w['s'] <= DRIFT + 1.0]
    assert inside, 'no waist in the dipole'
    emit(f'  waist at s = {inside[0]["s"]:.4f} m, width {inside[0]["width"] * 1e3:.4f} mm')

    s_sc, sz_sc, sx_sc, tau_sc, sxi_sc, rho_sc = propagate_frame(latcfg, sigma0, n_sub=400)
    emit(f'  the frame at each observation point (linear optics)')
    emit(f"    {'s':>7} {'sigma_z um':>11} {'sigma_xi um':>12} {'tau':>9} {'amplification':>14}")
    amp = {}
    for s_obs in S_OBS_LIST:
        j = int(np.argmin(np.abs(s_sc - s_obs)))
        amp[s_obs] = sx_sc[j] / max(sxi_sc[j], 1e-30)
        emit(f'    {s_obs:>7.3f} {sz_sc[j] * 1e6:>11.2f} {sxi_sc[j] * 1e6:>12.4f} '
             f'{tau_sc[j]:>9.3f} {amp[s_obs]:>13.1f}x')
    emit('')

    # reference: uniform, at the step size 6x measured converged
    emit(f'  reference: uniform step_size = {REF_STEP} (6x: converged, 0.7% at 0.001)')
    ref_off = [0.5 * REF_STEP] * len(S_OBS_LIST)
    cuts_ref, m_ref = run_case('ref', REF_STEP, None, ref_off)
    for s_obs in S_OBS_LIST:
        emit(f"    s={s_obs:.3f}: landed {m_ref[f'landed_{s_obs}']:.6f}, "
             f"{m_ref[f'snaps_{s_obs}']} snapshots, "
             f"dE range [{cuts_ref[s_obs][0].min():+.4f}, {cuts_ref[s_obs][0].max():+.4f}]")
    emit(f"    total {m_ref['seconds']:.1f} s")
    emit('')

    rows = []
    for tf in TAU_FRACS:
        tag = 'tf' + str(tf).replace('.', 'p')
        sc = {'mode': 'auto', 'kick_interval': 'midpoint', 'n_sub': 400,
              'tau_frac': float(tf), 'force_nodes': [float(s) for s in S_OBS_LIST]}
        cuts, m = run_case(tag, 0.05, sc, [0.0] * len(S_OBS_LIST))
        for s_obs in S_OBS_LIST:
            assert abs(m[f'landed_{s_obs}'] - s_obs) < 1e-9, \
                f"tau_frac={tf} landed {m[f'landed_{s_obs}']} not {s_obs}"
        m['tau_frac'] = tf
        rows.append((m, cuts))
        emit(f"  ran tau_frac={tf:<7g} {m['nodes_total']:>5} nodes, {m['seconds']:>6.1f} s")
    emit('')

    for s_obs in S_OBS_LIST:
        dE_ref, xk_ref = cuts_ref[s_obs]
        emit(f'  s = {s_obs:.3f} m ({s_obs - DRIFT:.2f} m into the bend), '
             f'tilt amplification {amp[s_obs]:.0f}x')
        emit(f"    {'tau_frac':>9} {'snaps':>7} {'kicks':>7} {'h_eff':>10} {'h_tau':>10} "
             f"{'rough':>9} {'dE relL2':>10} {'xk relL2':>10} {'dE range':>23}")
        emit(f"    {'uniform':>9} {m_ref[f'snaps_{s_obs}']:>7} {m_ref[f'kicks_{s_obs}']:>7} "
             f"{REF_STEP:>10.6f} {'-':>10} {abs_rough(dE_ref):>9.5f} "
             f"{'reference':>10} {'reference':>10} "
             f"[{dE_ref.min():+8.4f},{dE_ref.max():+8.4f}]")
        for m, cuts in rows:
            dE, xk = cuts[s_obs]
            emit(f"    {m['tau_frac']:>9g} {m[f'snaps_{s_obs}']:>7} {m[f'kicks_{s_obs}']:>7} "
                 f"{m.get(f'h_eff_{s_obs}', np.nan):>10.6f} "
                 f"{m.get(f'h_tau_{s_obs}', np.nan):>10.6f} {abs_rough(dE):>9.5f} "
                 f"{reldiff(dE, dE_ref):>10.5f} {reldiff(xk, xk_ref):>10.5f} "
                 f"[{dE.min():+8.4f},{dE.max():+8.4f}]")
        emit('    successive differences between adjacent rungs')
        for i in range(len(rows) - 1):
            emit(f"      {rows[i][0]['tau_frac']:>7g} -> {rows[i+1][0]['tau_frac']:<7g} : "
                 f"{reldiff(rows[i][1][s_obs][0], rows[i + 1][1][s_obs][0]):.5f}")
        emit('')

    emit('  cost, against the uniform reference (whole run to the far point)')
    far = S_OBS_LIST[-1]
    for m, cuts in rows:
        emit(f"    tau_frac={m['tau_frac']:<7g} "
             f"{m[f'snaps_{far}'] / m_ref[f'snaps_{far}']:.3f}x the snapshots, "
             f"{m['seconds'] / m_ref['seconds']:.3f}x the time")

    # --- plot -----------------------------------------------------------------------------
    n = len(S_OBS_LIST)
    fig, axes = plt.subplots(2, n + 1, figsize=(6 * (n + 1), 9))
    for c, s_obs in enumerate(S_OBS_LIST):
        dE_ref, xk_ref = cuts_ref[s_obs]
        zc = np.linspace(-1, 1, dE_ref.size)
        ax = axes[0, c]
        ax.plot(zc, dE_ref, 'k-', lw=2.4, label=f'uniform {REF_STEP} (ref)')
        for m, cuts in rows:
            ax.plot(zc, cuts[s_obs][0], lw=1.0, label=f"tau_frac={m['tau_frac']:g}")
        ax.set_xlabel('z / (zlim sigma_z)')
        ax.set_ylabel('dE/ds  [MeV/m]')
        ax.set_title(f's = {s_obs:.3f} m, amplification {amp[s_obs]:.0f}x')
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

        ax = axes[1, c]
        tf = np.array([m['tau_frac'] for m, _ in rows])
        err = np.array([reldiff(cuts[s_obs][0], dE_ref) for _, cuts in rows])
        ax.loglog(tf, err, 'o-')
        ax.invert_xaxis()
        ax.set_xlabel('tau_frac  (finer ->)')
        ax.set_ylabel('rel L2 vs uniform reference')
        ax.set_title(f'accuracy against tau_frac, s = {s_obs:.3f}')
        ax.grid(alpha=0.3, which='both')

    ax = axes[0, n]
    for c, s_obs in enumerate(S_OBS_LIST):
        dE_ref = cuts_ref[s_obs][0]
        sn = np.array([m[f'snaps_{s_obs}'] for m, _ in rows], float)
        err = np.array([reldiff(cuts[s_obs][0], dE_ref) for _, cuts in rows])
        ax.loglog(sn, err, 'o-', label=f's = {s_obs:.3f} ({amp[s_obs]:.0f}x)')
        for k, (m, _) in enumerate(rows):
            ax.annotate(f"{m['tau_frac']:g}", (sn[k], err[k]), fontsize=7,
                        textcoords='offset points', xytext=(4, 4))
    ax.set_xlabel('snapshots held at the observation point')
    ax.set_ylabel('rel L2 vs uniform reference')
    ax.set_title('cost against accuracy')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which='both')

    ax = axes[1, n]
    j0 = (s_sc >= DRIFT - 0.05) & (s_sc <= DRIFT + 1.05)
    ax.semilogy(s_sc[j0], sx_sc[j0] / np.maximum(sxi_sc[j0], 1e-30), 'k-')
    for s_obs in S_OBS_LIST:
        ax.axvline(s_obs, ls='--', lw=1, color='C3')
    ax.set_xlabel('s  [m]')
    ax.set_ylabel('tilt amplification sigma_x / sigma_xi')
    ax.set_title('where tau errors are amplified')
    ax.grid(alpha=0.3, which='both')

    fig.tight_layout()
    fig.savefig(os.path.join(RESULT_DIR, 'tau_frac_calibrate.png'), dpi=130)

    with open(os.path.join(RESULT_DIR, 'tau_frac_log.txt'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"\nwrote {RESULT_DIR}/tau_frac_calibrate.png")


if __name__ == '__main__':
    main()
