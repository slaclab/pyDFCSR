"""
Step 4 of progress_cancellation.md (§8): the shipped vs integrated-by-parts check, and the u edge
taper, on the benchmark chicane (the closed-chicane configuration '2dC' of chicane_2d_vs_1d.py).

    conda run -n pydfcsr python cancellation_chicane.py run C1|C2|C3      # one run, ~30-50 min
    conda run -n pydfcsr python cancellation_chicane.py analyze

  C3  2dC as recorded (u_taper 0, the hard edge), ibp_check 3: reproduces closed/2dC.npz
  C1  2dC with u_taper 0.5 (the default since 2026-11-12), ibp_check 3
  C2  C1 with CSR_integration.ibp true

Writes benchmark_results/cancellation_effects/chicane/<name>.npz (final beam, statistics,
ibp_check_log) and the run's wakes/statistics h5 under .../chicane/runs.
chicane/v1_partition/ holds the first set, made with the in-column chi partition that failed at
high tilt (progress_cancellation.md §8a); its C2 beam and all three check logs are invalid.
"""
import os
import sys
import time

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..'))
sys.path.insert(0, HERE)
OUT = os.path.join(HERE, 'benchmark_results', 'cancellation_effects', 'chicane')
CLOSED = os.path.join(HERE, 'benchmark_results', 'chicane_2d_vs_1d', 'closed')

VARIANTS = {'C3': dict(u_taper=0.0, ibp=False), 'C1': dict(u_taper=0.5, ibp=False),
            'C2': dict(u_taper=0.5, ibp=True)}


def config(name):
    import chicane_2d_vs_1d as C      # BASE and RUNS['2dC'], read only (no run, no argv use)
    r = C.RUNS['2dC']
    cfg = yaml.safe_load(yaml.safe_dump(C.BASE))
    ip, dep, cc = cfg['CSR_integration'], cfg['particle_deposition'], cfg['CSR_computation']
    # the 2dC overrides, as chicane_2d_vs_1d.config applies them
    ip.update(tangent_bands=True, drift_reach=True, drift_snap_thin=0.01, far_fine_max=500,
              band_jacobian_min=0.2, prev_bend_reach=True, drift_cutoff=0.0, **r['integ'])
    dep.update(vx_model='affine', **r['dep'])
    v = VARIANTS[name]
    dep['u_taper'] = v['u_taper']
    ip.update(ibp=v['ibp'], ibp_check=3)
    cc.update(workdir=os.path.join(OUT, 'runs'), write_name=name)
    os.makedirs(os.path.join(OUT, 'runs'), exist_ok=True)
    f = os.path.join(OUT, f'{name}.yaml')
    yaml.safe_dump(cfg, open(f, 'w'))
    return f


def run(name):
    from pyDFCSR_2D import CSR2D
    t0 = time.time()
    sim = CSR2D(input_file=config(name))
    sim.run()
    p = sim.beam.particle
    st = {k: v for k, v in sim.statistics.items() if not isinstance(v, dict)}
    log = np.array(getattr(sim, 'ibp_check_log', []), dtype=float)
    np.savez(os.path.join(OUT, f'{name}.npz'), x=p.x, px=p.px, z=p.z, pz=p.pz,
             E0=sim.beam.init_energy, s=np.array(sim.lattice.steps_record),
             runtime=time.time() - t0, prefix=sim.prefix, ibp_check_log=log,
             edge_u_max=sim.DF_tracker.edge_u_max, **st)
    print(f'{name}: done in {(time.time() - t0) / 60:.1f} min', flush=True)


def analyze():
    import chicane_2d_vs_1d as C
    lines = []

    def emit(t=''):
        print(t, flush=True)
        lines.append(t)

    off = dict(np.load(os.path.join(CLOSED, 'off.npz')))
    E0 = float(off['E0'])
    runs = {'2dC (recorded)': dict(np.load(os.path.join(CLOSED, '2dC.npz')))}
    for n in ('C3', 'C1', 'C2'):
        f = os.path.join(OUT, f'{n}.npz')
        if os.path.exists(f):
            runs[n] = dict(np.load(f, allow_pickle=True))
    emit('Chicane (2dC configuration), CSR effects on the final beam against the CSR-off run')
    emit(f'{"run":16s} {"dE keV":>9s} {"dsE_slice keV":>13s} {"rel %":>7s} {"d emit um":>10s} '
         f'{"slice emit um":>13s} {"runtime min":>11s}')
    M = {}
    for n, b in runs.items():
        m = C.metrics(b, off, E0)
        M[n] = m
        emit(f'{n:16s} {m["dE"]:9.3f} {m["dsEs"]:13.3f} {m["rsEs"]:7.2f} {m["demit"]:10.4f} '
             f'{m["semit"]:13.4f} {float(b.get("runtime", np.nan))/60:11.1f}')
    emit('')
    emit('Energy change per element [keV]:')
    emit(f'{"run":16s} ' + ' '.join(f'{e:>8s}' for e in C.ENAMES))
    for n, b in runs.items():
        emit(f'{n:16s} ' + ' '.join(f'{v:8.2f}' for v in C.per_element_loss(b)))
    emit('')
    dpz_csr = runs['2dC (recorded)']['pz'] - off['pz']
    for a, b, what in (('2dC (recorded)', 'C3', 'C3 reproduces the recorded run'),
                       ('C3', 'C1', 'the u taper (C1 vs C3)'),
                       ('C1', 'C2', 'by parts vs shipped (C2 vs C1)')):
        if a in runs and b in runs:
            d = runs[b]['pz'] - runs[a]['pz']
            same = np.array_equal(runs[a]['pz'], runs[b]['pz'])
            emit(f'{what}: rms per-particle d(pz) / rms CSR-induced pz change = '
                 f'{np.std(d)/np.std(dpz_csr):.2e}' + ('  (bit-identical)' if same else ''))
    emit('')
    emit('ibp_check along the chicane (3 centre-row points per kick, both forms at 400 x\' nodes):')
    edges = C.EDGES
    for n in ('C3', 'C1', 'C2'):
        if n not in runs or not len(runs[n]['ibp_check_log']):
            continue
        L = runs[n]['ibp_check_log']
        row = []
        for i, en in enumerate(C.ENAMES):
            m = (L[:, 0] >= edges[i]) & (L[:, 0] < edges[i + 1] + 1e-9)
            if m.any():
                row.append(f'{en} {np.nanmax(L[m, 1]):.1e} (zone {np.nanmax(L[m, 4]):.0e})')
        emit(f'  {n}: max per element: ' + '; '.join(row))
        emit(f'      kicks above 1e-3: {int(np.sum(L[:, 1] > 1e-3))} of {len(L)};  '
             f'u-edge density max {float(runs[n]["edge_u_max"]):.1e}')
    with open(os.path.join(OUT, 'chicane_summary.txt'), 'w') as f:
        f.write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    if sys.argv[1] == 'run':
        run(sys.argv[2])
    else:
        analyze()
