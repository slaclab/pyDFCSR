"""
The FODO example with the current code, and the one clean test of `m_steps_xi`.

Two purposes.

**1. What the current code does on `example_fodo.ipynb`'s lattice.** Statistics along the lattice and
x-z wake maps at representative positions, the same treatment §11q gave the chicane.

**2. Calibrate `m_steps_xi`, the last `auto` constant never swept.** This lattice is the right place,
and that is not a coincidence -- it is the only case in this work where the transverse term is the
*only* thing asking for refinement:

```
  h_1   (sigma_z)  finite but useless: sigma_z is EXACTLY constant, so L_z ~ 5443 m
  h_xi  (sigma_xi) 0.496 m  <- the only finite constraint
  h_tau (tilt)     inf: tau = 0 identically, there is no x-z correlation to track
  h_2   (bend)     inf: NO DIPOLES, so r = 0 everywhere
```

In the dipole and chicane cases `h_tau` or `h_2` always dominated, so sweeping `m_steps_xi` there
would have measured nothing. Here it is unopposed.

Caveat stated up front: with no dipoles there is **no steady-state CSR source**. The wake comes only
from whatever transient the quads and the sextupole produce, so the amplitudes are tiny and this is a
weak-signal test. It answers "does `m_steps_xi` matter where it is the only active term", not "does
`m_steps_xi` matter in a bend".

Fixed while writing this: `waist.py` read the quad strength as `'k1'`, but the lattice YAMLs and
`get_bmadx_element` (CSR.py:243) use `'K1'`. Every quad therefore looked like a DRIFT to the
linear-optics scan, so `auto` sized its steps from the wrong optics. Small here (2.6 um in sigma_x,
because these quads are weak) but wrong in principle and unbounded for a strong lattice.
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
from mpi4py import MPI
from pyDFCSR_2D.CSR import CSR2D

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
PARALLEL = COMM.Get_size() > 1

EXAMPLE_DIR = os.path.join(os.path.dirname(__file__), '..', 'example')
RESULT_DIR = os.path.join(os.path.dirname(__file__), 'benchmark_results', 'fodo_auto')

# The shipped fodo_init_beam.yaml uses distgen's OLD flat key `random_type: hammersley`; current
# distgen wants `random: {type: hammersley}` and raises
# "Unexpected distgen input parameter: random_type" otherwise, so example_fodo.ipynb cannot run as
# committed. A corrected copy is generated rather than editing the author's input file.
BEAM_YAML = 'input/fodo_init_beam_fixed.yaml'
LATTICE_YAML = 'input/fodo_lattice.yaml'
MESH_XBINS, MESH_ZBINS = 10, 30      # the notebook's wake mesh
DEP_BINS = 200                       # the notebook's deposition

# A first sweep of (4, 2, 1, 0.5) produced four IDENTICAL wakes, and that was a broken test rather
# than a null result: h_xi >= 0.499 m on this lattice while h_max = step_size = 0.07 m, so the
# reciprocal sum 1/h_max + 1/h_xi is dominated by 1/h_max and h_eff sits pinned at h_max/2 = 0.035
# by the dyadic snap. m_steps_xi has to reach ~8 before h_xi drops below h_max at all. Sweeping up
# to 64 puts the transverse term genuinely in control; the loose end is kept so the crossover is
# visible in the table rather than assumed.
M_STEPS_XI = (2.0, 8.0, 16.0, 32.0, 64.0)    # 2.0 is the current default


def write_config(tag, m_steps_xi=None, force_nodes=None):
    p = f'input/fodo_auto_{tag}.yaml'
    if RANK != 0:
        COMM.Barrier()
        return p
    sc = {'mode': 'auto', 'kick_interval': 'midpoint', 'n_sub': 400}
    if m_steps_xi is not None:
        sc['m_steps_xi'] = float(m_steps_xi)
    if force_nodes:
        sc['force_nodes'] = [float(x) for x in force_nodes]
    cfg = {
        'input_beam': {'style': 'distgen', 'distgen_input_file': BEAM_YAML},
        'input_lattice': {'lattice_input_file': LATTICE_YAML},
        'particle_deposition': {'method': 'bspline_comoving',
                                'xbins': DEP_BINS, 'zbins': DEP_BINS,
                                'xlim': 5, 'zlim': 5, 'smoothing_sigma': 3.0,
                                'poly_degree': 1, 'velocity_threhold': 1000},
        'CSR_integration': {'n_formation_length': 1.0, 'zbins': 100, 'xbins': 100},
        'CSR_computation': {'compute_CSR': 1, 'apply_CSR': 1, 'transverse_on': 1,
                            'xbins': MESH_XBINS, 'zbins': MESH_ZBINS, 'xlim': 3, 'zlim': 3,
                            'write_beam': [], 'write_wakes': False,
                            'write_name': f'fodo_{tag}', 'workdir': './output'},
        'step_control': sc,
    }
    with open(os.path.join(EXAMPLE_DIR, p), 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    if PARALLEL:
        COMM.Barrier()
    return p


def roughness(w):
    w = np.asarray(w, float)
    amp = np.abs(w).max()
    if amp <= 0:
        return 0.0
    d2 = w[2:] - 2.0 * w[1:-1] + w[:-2]
    return float(np.linalg.norm(d2) / amp)


def run_full(tag='run'):
    """Full run capturing statistics and every kick's map."""
    cache = os.path.join(RESULT_DIR, f'full_{tag}.npz')
    hit = COMM.bcast(os.path.exists(cache) if RANK == 0 else None, root=0) if PARALLEL \
        else os.path.exists(cache)
    if hit:
        d = np.load(cache, allow_pickle=True)
        # maps is a LIST of dicts, so it round-trips as an object ARRAY: .item() raises
        # "can only convert an array of size 1". Only the dict fields take .item().
        return d['stat'].item(), list(d['maps']), d['meta'].item()

    cfg = write_config(tag)
    csr = CSR2D(input_file=cfg, parallel=PARALLEL)
    sch = csr.lattice.schedule
    lat = csr.lattice
    maps = []
    orig = csr.calculate_2D_CSR_parallel if PARALLEL else csr.calculate_2D_CSR

    def cap():
        orig()
        b = csr.beam
        maps.append(dict(s=float(b.position),
                         dE=csr.dE_dct.reshape(MESH_XBINS, MESH_ZBINS).copy(),
                         xk=csr.x_kick.reshape(MESH_XBINS, MESH_ZBINS).copy(),
                         zz=csr.CSR_zmesh.reshape(MESH_XBINS, MESH_ZBINS).copy(),
                         xx=csr.CSR_xmesh.reshape(MESH_XBINS, MESH_ZBINS).copy(),
                         slope=np.asarray(b.slope, float).copy(),
                         sigma_z=float(b._sigma_z), sigma_x=float(b._sigma_x),
                         sigma_xi=float(b._sigma_x_transform),
                         **velocity_field_slope(b)))

    if PARALLEL:
        csr.calculate_2D_CSR_parallel = cap
    else:
        csr.calculate_2D_CSR = cap

    t0 = time.time()
    csr.run(debug=False)
    wall = time.time() - t0
    st = csr.statistics
    n = len(st['sigma_z'])
    stat = dict(s=np.asarray(sch.s_nodes[:n], float),
                sigma_z=np.array(st['sigma_z']), sigma_x=np.array(st['sigma_x']),
                sigma_energy=np.array(st['sigma_energy']),
                mean_energy=np.array(st['mean_energy']),
                beta_x=np.array(st['twiss']['beta_x']),
                alpha_x=np.array(st['twiss']['alpha_x']),
                norm_emit_x=np.array(st['twiss']['norm_emit_x']))
    meta = dict(nodes=int(sch.n_nodes), kicks=int(sch.n_kick), seconds=wall,
                distance=np.asarray(lat.distance, float),
                ele_types=[lat.lattice_config[k].get('type', 'drift')
                           for k in lat.lattice_config if k != 'step_size'],
                s_nodes=np.asarray(sch.s_nodes, float),
                kick_s=np.asarray(sch.s_nodes[sch.is_kick], float),
                s_scan=np.asarray(sch.auto_s_scan, float),
                h_eff=np.asarray(sch.auto_h_eff, float),
                h_xi=np.asarray(sch.auto_parts['h_xi'], float))
    if RANK == 0:
        os.makedirs(RESULT_DIR, exist_ok=True)
        np.savez(cache, stat=np.array(stat, dtype=object),
                 maps=np.array(maps, dtype=object), meta=np.array(meta, dtype=object))
    if PARALLEL:
        COMM.Barrier()
    del csr
    gc.collect()
    return stat, maps, meta


def run_mxi(m_xi, s_obs):
    """One m_steps_xi rung, wake cut at a pinned position."""
    tag = f'mxi{m_xi:g}'.replace('.', 'p')
    cache = os.path.join(RESULT_DIR, f'cut_{tag}.npz')
    hit = COMM.bcast(os.path.exists(cache) if RANK == 0 else None, root=0) if PARALLEL \
        else os.path.exists(cache)
    if hit:
        d = np.load(cache, allow_pickle=True)
        return d['cut'].item(), d['meta'].item()

    cfg = write_config(tag, m_steps_xi=m_xi, force_nodes=[s_obs])
    csr = CSR2D(input_file=cfg, parallel=PARALLEL)
    sch = csr.lattice.schedule
    t0 = time.time()
    csr.run(stop_time=s_obs, debug=False)
    assert abs(csr.beam.position - s_obs) < 1e-9, \
        f'm_steps_xi {m_xi} landed {csr.beam.position} not {s_obs}'
    csr.get_CSR_mesh()
    if PARALLEL:
        csr.calculate_2D_CSR_parallel()
    else:
        csr.calculate_2D_CSR()
    cut = dict(dE=csr.dE_dct.reshape(MESH_XBINS, MESH_ZBINS).copy(),
               xk=csr.x_kick.reshape(MESH_XBINS, MESH_ZBINS).copy(),
               zz=csr.CSR_zmesh.reshape(MESH_XBINS, MESH_ZBINS).copy())
    meta = dict(m_steps_xi=m_xi, nodes=int(sch.n_nodes), kicks=int(sch.n_kick),
                snaps=len(csr.DF_tracker.time_log),
                h_xi_min=float(np.nanmin(sch.auto_parts['h_xi'])),
                h_eff_min=float(sch.auto_h_eff.min()), seconds=time.time() - t0)
    if RANK == 0:
        os.makedirs(RESULT_DIR, exist_ok=True)
        np.savez(cache, cut=np.array(cut, dtype=object), meta=np.array(meta, dtype=object))
    if PARALLEL:
        COMM.Barrier()
    del csr
    gc.collect()
    return cut, meta


def velocity_field_slope(beam, nbin=41, min_per_bin=200):
    """Measure d<px|x>/dx from the tracked particles, and how non-linear it is.

    div(v) in the wake integrand is the divergence of the MEAN velocity field, which
    is only a well defined single number if <px|x> is linear in x. Rather than assume
    that, bin the particles in x, fit the local mean angle, and also report the
    residual of that fit -- a curved velocity field would show up here.
    """
    x = np.asarray(beam.x, float)
    px = np.asarray(beam.px, float)
    sx = x.std()
    edges = np.linspace(-3.0 * sx, 3.0 * sx, nbin)
    idx = np.digitize(x, edges)
    cx, cv = [], []
    for k in range(1, edges.size):
        m = idx == k
        if m.sum() >= min_per_bin:
            cx.append(x[m].mean())
            cv.append(px[m].mean())
    if len(cx) < 4:
        return dict(dvdx_binned=np.nan, dvdx_moment=np.nan, dvdx_nonlin=np.nan)
    cx = np.asarray(cx)
    cv = np.asarray(cv)
    sl, ic = np.polyfit(cx, cv, 1)
    resid = cv - (sl * cx + ic)
    # Normalise the fit residual by sigma_px, NOT by the range of <px|x>: the latter
    # collapses to zero at a waist, where alpha = 0 makes the linear part vanish, and
    # the ratio then blows up on a denominator that is itself noise. sigma_px is the
    # angle scale of the beam and stays finite everywhere.
    return dict(dvdx_binned=float(sl),
                dvdx_moment=float(np.mean(x * px) / np.mean(x * x)),
                dvdx_nonlin=float(np.max(np.abs(resid)) / max(px.std(), 1e-30)))


def report_div_v_identity(stat, maps, emit, ele_edges=()):
    """With rho = 0 the only surviving wake term is W2 ~ div(v), and for an
    uncorrelated beam div(v) = d v_x/dx = <x x'>/<x^2> = dln(sigma_x)/ds = -alpha/beta.
    Checked three ways: the local mean-angle slope really is x-independent (so div(v)
    is a single number at all), the identity holds against the tracked sigma_x, and
    the wake peak is proportional to it THROUGH the sign change at the waist -- the
    last being what a W2 sign error would break on one side of the waist only.
    """
    s = np.asarray(stat['s'], float)
    sx = np.asarray(stat['sigma_x'], float)
    a = np.asarray(stat['alpha_x'], float)
    b = np.asarray(stat['beta_x'], float)
    ex = np.asarray(stat['norm_emit_x'], float)

    if all('dvdx_binned' in mp for mp in maps):
        emit('  step 1: d<px|x>/dx from the tracked particles, and is it linear in x')
        emit(f"    {'s [m]':>8} {'binned slope':>13} {'<x px>/<x^2>':>13} "
             f"{'-alpha/beta':>12} {'fit nonlin':>11}")
        for mp in maps:
            sm = float(mp['s'])
            q = -float(np.interp(sm, s, a)) / float(np.interp(sm, s, b))
            emit(f"    {sm:>8.4f} {mp['dvdx_binned']:>+13.5f} "
                 f"{mp['dvdx_moment']:>+13.5f} {q:>+12.5f} {mp['dvdx_nonlin']:>11.2e}")
        nl = np.array([mp['dvdx_nonlin'] for mp in maps], float)
        emit(f'    max nonlinearity of <px|x> vs x: {np.nanmax(nl):.2e} of sigma_px '
             f'-> div(v) is a single number, not a function of x')
        emit('')

    emit('  step 2/3: div(v) = dln(sigma_x)/ds = -alpha_x/beta_x, against the tracked run')
    dln = np.gradient(np.log(sx), s)
    emit(f"    {'s [m]':>8} {'dln(sx)/ds':>12} {'-alpha/beta':>12} {'difference':>12}")
    for tgt in np.linspace(s[1], s[-2], 7):
        j = int(np.argmin(np.abs(s - tgt)))
        emit(f'    {s[j]:>8.4f} {dln[j]:>+12.5f} {-a[j] / b[j]:>+12.5f} '
             f'{dln[j] + a[j] / b[j]:>+12.2e}')
    # np.gradient uses a one-sided difference at the two endpoints and a central
    # difference elsewhere; both are first-order wrong wherever dln(sx)/ds jumps, i.e.
    # at every quadrupole edge. Those are properties of the finite difference, not of
    # the identity, so quote the smooth interior separately from the edges.
    res = np.abs(dln + a / b)
    edge = np.zeros(s.size, bool)
    edge[[0, -1]] = True
    for x_e in np.asarray(ele_edges, float):
        edge |= np.abs(s - x_e) <= 1.5 * np.max(np.diff(s))
    emit(f'    max |difference| all s      : {res.max():.2e}  '
         f'(at s = {s[int(np.argmax(res))]:.4f})')
    if (~edge).any():
        ri = res[~edge]
        emit(f'    max away from element edges : {ri.max():.2e}  '
             f'(the edge values are the finite difference straddling a jump in '
             f'dln(sx)/ds, not a failure of the identity)')
    emit(f'    median |difference|         : {np.median(res):.2e}')
    emit(f'    norm_emit_x drift           : {ex.max() / ex.min() - 1:.2e}  '
         f'(constant-eps assumption)')
    emit('')

    emit('  and the wake peak is proportional to it, through the sign change')
    emit(f"    {'s [m]':>8} {'alpha_x':>9} {'beta_x':>8} {'-alpha/beta':>12} "
         f"{'wake peak':>12} {'ratio':>9}")
    ratios = []
    for mp in maps:
        sm = float(mp['s'])
        dE = np.asarray(mp['dE'], float)
        mid = dE[dE.shape[0] // 2, :]
        pk = float(mid[np.argmax(np.abs(mid))])
        av = float(np.interp(sm, s, a))
        bv = float(np.interp(sm, s, b))
        q = -av / bv
        if abs(q) > 1e-3:
            ratios.append(pk / q)
            rs = f'{pk / q:>9.4f}'
        else:
            rs = f"{'--':>9}"
        emit(f'    {sm:>8.4f} {av:>+9.4f} {bv:>8.3f} {q:>+12.5f} {pk:>+12.4e} {rs}')
    ratios = np.asarray(ratios)
    emit(f'    ratio {ratios.min():.4f} .. {ratios.max():.4f}, '
         f'spread {ratios.ptp() / np.mean(ratios) * 100:.1f} % over '
         f'{ratios.size} kicks spanning both signs')
    emit(f"    max |x_kick| over all maps  : "
         f"{max(float(np.abs(np.asarray(mp['xk'], float)).max()) for mp in maps):.2e}  "
         f"(n - n' and n.tau' vanish in a straight line, so W1/W3 are structurally zero)")
    emit('')


def main():
    os.chdir(EXAMPLE_DIR)
    lines = []

    def emit(t=''):
        if RANK == 0:
            print(t, flush=True)
            lines.append(t)

    emit('=' * 100)
    emit('FODO example with the current code, and the m_steps_xi calibration')
    emit('=' * 100)

    stat, maps, meta = run_full()
    dist = meta['distance']
    types = meta['ele_types']
    emit(f"  lattice: {len(types)} elements, {dist[-1]:.4f} m, types {types}")
    emit(f"  NO DIPOLES -> no steady-state CSR source; amplitudes are transient-only")
    emit(f"  auto: {meta['nodes']} nodes, {meta['kicks']} kicks, {meta['seconds']:.0f} s")
    emit(f"  captured {len(maps)} wake maps")
    emit('')

    emit('  statistics along the lattice')
    emit(f"    {'s [m]':>8} {'sigma_z um':>11} {'sigma_x um':>11} {'beta_x m':>10} "
         f"{'alpha_x':>9} {'sigma_E keV':>12}")
    for tgt in np.linspace(0, dist[-1], 8):
        j = int(np.argmin(np.abs(stat['s'] - tgt)))
        emit(f"    {stat['s'][j]:>8.3f} {stat['sigma_z'][j] * 1e6:>11.3f} "
             f"{stat['sigma_x'][j] * 1e6:>11.3f} {stat['beta_x'][j]:>10.4f} "
             f"{stat['alpha_x'][j]:>9.4f} {stat['sigma_energy'][j] / 1e3:>12.4f}")
    e0, e1 = stat['mean_energy'][0], stat['mean_energy'][-1]
    emit(f"    mean energy {e0 / 1e9:.6f} -> {e1 / 1e9:.6f} GeV "
         f"(loss {(e1 - e0) / 1e3:+.4f} keV)")
    emit(f"    nan in statistics: "
         f"{sum(int(np.isnan(np.asarray(v, float)).sum()) for v in stat.values() if np.asarray(v).ndim == 1)}")
    emit('')

    report_div_v_identity(stat, maps, emit, ele_edges=dist)

    # --- m_steps_xi sweep, at the position where h_xi is tightest ------------------------
    hx = meta['h_xi']
    ss = meta['s_scan']
    j = int(np.nanargmin(hx))
    s_obs = float(round(min(max(ss[j], 0.3), dist[-1] - 0.3), 3))
    emit(f'  m_steps_xi sweep, observing at s = {s_obs} m '
         f'(h_xi is tightest near s = {ss[j]:.3f})')
    emit('  h_tau and h_2 are INERT here (tau = 0, no dipoles), so h_xi is the only')
    emit('  active refinement term -- the one lattice where this constant can be tested.')
    emit(f'  h_max = {0.07} m, so h_xi only BINDS once m_steps_xi pushes it below that;')
    emit('  the table reports h_xi_min so the crossover is visible rather than assumed.')
    rows = []
    for mxi in M_STEPS_XI:
        cut, m = run_mxi(mxi, s_obs)
        rows.append((mxi, cut, m))
        emit(f"    ran m_steps_xi={mxi:<5g} {m['nodes']:>4} nodes, {m['snaps']:>4} snaps, "
             f"h_xi_min {m['h_xi_min']:.4f}, {m['seconds']:>4.0f} s")
    emit('')

    ref = rows[-1][1]['dE']
    refn = np.linalg.norm(ref)
    emit(f"    {'m_steps_xi':>11} {'nodes':>6} {'snaps':>6} {'h_eff min':>10} "
         f"{'|dE| peak':>11} {'rough':>8} {'rel L2 vs finest':>17}")
    for mxi, cut, m in rows:
        mid = cut['dE'][MESH_XBINS // 2, :]
        rel = np.linalg.norm(cut['dE'] - ref) / refn if refn > 0 else np.nan
        emit(f"    {mxi:>11g} {m['nodes']:>6} {m['snaps']:>6} {m['h_eff_min']:>10.5f} "
             f"{np.abs(cut['dE']).max():>11.6f} {roughness(mid):>8.4f} {rel:>17.6f}")
    emit('')
    emit('    successive differences')
    for i in range(len(rows) - 1):
        a, b = rows[i][1]['dE'], rows[i + 1][1]['dE']
        nb = np.linalg.norm(b)
        emit(f"      {rows[i][0]:>6g} -> {rows[i+1][0]:<6g} : "
             f"{np.linalg.norm(a - b) / nb if nb > 0 else float('nan'):.6f}")
    emit('')

    if RANK != 0:
        return
    os.makedirs(RESULT_DIR, exist_ok=True)

    starts = np.concatenate([[0.0], dist[:-1]])
    quads = [(starts[i], dist[i]) for i, t in enumerate(types)
             if t in ('quad', 'quadrupole')]
    sexts = [(starts[i], dist[i]) for i, t in enumerate(types) if t == 'sextupole']

    def shade(ax):
        for a, b in quads:
            ax.axvspan(a, b, color='C0', alpha=0.18)
        for a, b in sexts:
            ax.axvspan(a, b, color='C2', alpha=0.18)

    # --- statistics figure ---------------------------------------------------------------
    fig, axes = plt.subplots(4, 1, figsize=(13, 12), sharex=True)
    axes[0].plot(stat['s'], stat['beta_x'], lw=1.8, color='C0')
    axes[0].set_ylabel('beta_x  [m]')
    axes[0].set_title('FODO with the current code (blue = quads, green = sextupole)')
    axes[1].plot(stat['s'], stat['sigma_x'] * 1e6, lw=1.8, color='C2', label='sigma_x')
    axes[1].plot(stat['s'], stat['sigma_z'] * 1e6, lw=1.4, color='C0', label='sigma_z')
    axes[1].set_ylabel('beam size  [um]')
    axes[1].legend(fontsize=8)
    axes[2].plot(stat['s'], stat['sigma_energy'] / 1e3, lw=1.8, color='C3')
    axes[2].set_ylabel('sigma_E  [keV]')
    axes[3].semilogy(meta['s_scan'], meta['h_eff'], '--', color='C4', lw=1.3, label='h_eff')
    axes[3].semilogy(meta['s_scan'], np.minimum(meta['h_xi'], 1e3), ':', color='C1', lw=1.2,
                     label='h_xi (the only active term)')
    lo = meta['h_eff'].min()
    axes[3].plot(meta['s_nodes'], np.full(meta['s_nodes'].size, lo * 0.85), '|',
                 color='k', ms=5, alpha=0.5, label=f"nodes ({meta['nodes']})")
    axes[3].plot(meta['kick_s'], np.full(meta['kick_s'].size, lo * 0.72), '|',
                 color='C3', ms=8, label=f"kicks ({meta['kicks']})")
    axes[3].set_ylabel('step size  [m]')
    axes[3].set_xlabel('s  [m]')
    axes[3].legend(fontsize=7)
    for ax in axes:
        shade(ax)
        ax.grid(alpha=0.3, which='both')
    fig.tight_layout()
    fig.savefig(os.path.join(RESULT_DIR, 'fodo_statistics.png'), dpi=130)
    plt.close(fig)
    emit('  wrote fodo_statistics.png')

    # --- x-z wake maps at representative positions ---------------------------------------
    ms = np.array([m['s'] for m in maps])
    picks = []
    for i, (t, (a, b)) in enumerate(zip(types, zip(starts, dist))):
        j = int(np.argmin(np.abs(ms - 0.5 * (a + b))))
        picks.append((f'{t} {i + 1}', ms[j], j))
    fig, axes = plt.subplots(2, len(picks), figsize=(3.4 * len(picks), 7))
    if len(picks) == 1:
        axes = axes.reshape(2, 1)
    for c, (lab, sval, j) in enumerate(picks):
        for r, (q, ql) in enumerate((('dE', 'dE/ds [MeV/m]'), ('xk', 'x_kick [MeV/m]'))):
            m = maps[j]
            a = m[q]
            lo_, hi_ = float(a.min()), float(a.max())
            if lo_ < 0.0 < hi_:
                v = max(abs(lo_), abs(hi_)) or 1.0
                kw = dict(cmap='RdBu_r', vmin=-v, vmax=v)
            else:
                kw = (dict(cmap='Reds', vmin=0.0, vmax=hi_ if hi_ > 0 else 1.0) if hi_ > 0
                      else dict(cmap='Blues_r', vmin=lo_, vmax=0.0))
            xt = m['xx'] - np.polyval(m['slope'], m['zz'])
            im = axes[r, c].pcolormesh(m['zz'] * 1e3, xt * 1e6, a, shading='auto', **kw)
            if r == 0:
                axes[r, c].set_title(f'{lab}\ns = {sval:.3f} m', fontsize=8.5)
            axes[r, c].set_xlabel('z [mm]', fontsize=7)
            axes[r, c].set_ylabel(ql, fontsize=7)
            axes[r, c].tick_params(labelsize=6)
            axes[r, c].text(0.03, 0.96, f'[{lo_:.2g}, {hi_:.2g}]',
                            transform=axes[r, c].transAxes, fontsize=5.5, va='top',
                            bbox=dict(fc='white', ec='0.6', alpha=0.85, pad=1.2))
            fig.colorbar(im, ax=axes[r, c], fraction=0.046)
    fig.suptitle('FODO x-z wakes, one column per element (each panel self-normalised; '
                 'TOP longitudinal, BOTTOM transverse)', fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(RESULT_DIR, 'fodo_wakes_xz.png'), dpi=130)
    plt.close(fig)
    emit('  wrote fodo_wakes_xz.png')

    # --- m_steps_xi convergence ----------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for mxi, cut, m in rows:
        mid = cut['dE'][MESH_XBINS // 2, :]
        axes[0].plot(cut['zz'][MESH_XBINS // 2, :] * 1e3, mid, lw=1.2,
                     label=f'm_steps_xi={mxi:g}')
    axes[0].set_xlabel('z  [mm]')
    axes[0].set_ylabel('dE/ds  [MeV/m]  (mid-x row)')
    axes[0].set_title(f'wake cut at s = {s_obs} m')
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.3)
    mx = np.array([r[0] for r in rows])
    err = np.array([np.linalg.norm(r[1]['dE'] - ref) / refn if refn > 0 else np.nan
                    for r in rows])
    nn = np.array([r[2]['nodes'] for r in rows], float)
    axes[1].loglog(mx, np.maximum(err, 1e-16), 'o-', label='rel L2 vs finest')
    ax2 = axes[1].twinx()
    ax2.loglog(mx, nn, 's--', color='C1', label='nodes')
    ax2.set_ylabel('nodes', color='C1')
    axes[1].set_xlabel('m_steps_xi  (finer ->)')
    axes[1].set_ylabel('rel L2 vs finest')
    axes[1].set_title('convergence and cost')
    axes[1].grid(alpha=0.3, which='both')
    axes[1].legend(fontsize=8, loc='center left')
    fig.tight_layout()
    fig.savefig(os.path.join(RESULT_DIR, 'fodo_m_steps_xi.png'), dpi=130)
    plt.close(fig)
    emit('  wrote fodo_m_steps_xi.png')

    with open(os.path.join(RESULT_DIR, 'fodo_auto_log.txt'), 'w') as f:
        f.write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
