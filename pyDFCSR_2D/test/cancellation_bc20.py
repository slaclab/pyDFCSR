"""
Step 4 of progress_cancellation.md (§8): the shipped vs integrated-by-parts check on FACET-II BC20,
the 1M-particle peak-frame run dfcsr_2d_peak_1M_sz6.5cap_m40_k115 (progress_facet_bc20.md §9j).

    cd pyDFCSR_2D/example
    mpirun -np 3 python ../test/cancellation_bc20.py B0|B1|B2          (pydfcsr-mpi env)

  B0  the recorded run (u_taper 0), ibp_check 3: reproduces it, plus the check log
  B1  u_taper 0.5 (the default since 2026-11-12), ibp_check 3
  B2  B1 with CSR_integration.ibp true

Writes benchmark_results/cancellation_effects/bc20/<name>.npz on rank 0.
"""
import os
import sys
import time

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..'))
OUT = os.path.join(HERE, 'benchmark_results', 'cancellation_effects', 'bc20')
SRC = os.path.join(HERE, 'benchmark_results', 'facet_bc20', 'runs', 'inputs',
                   'dfcsr_2d_peak_1M_sz6.5cap_m40_k115.yaml')
VARIANTS = {'B0': dict(u_taper=0.0, ibp=False), 'B1': dict(u_taper=0.5, ibp=False),
            'B2': dict(u_taper=0.5, ibp=True)}


def main(name):
    from mpi4py import MPI
    from pyDFCSR_2D import CSR2D
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    cfg = yaml.safe_load(open(SRC))
    v = VARIANTS[name]
    cfg['particle_deposition']['u_taper'] = v['u_taper']
    cfg['CSR_integration'].update(ibp=v['ibp'], ibp_check=3)
    cfg['CSR_computation'].update(workdir=os.path.join(OUT, 'runs'), write_name=name)
    f = os.path.join(OUT, f'{name}.yaml')
    if rank == 0:
        os.makedirs(os.path.join(OUT, 'runs'), exist_ok=True)
        yaml.safe_dump(cfg, open(f, 'w'))
    comm.Barrier()
    t0 = time.time()
    sim = CSR2D(input_file=f, parallel=comm.Get_size() > 1)
    sim.run()
    if rank == 0:
        p = sim.beam.particle
        st = {k: v for k, v in sim.statistics.items() if not isinstance(v, dict)}
        np.savez(os.path.join(OUT, f'{name}.npz'), x=p.x, px=p.px, z=p.z, pz=p.pz,
                 E0=sim.beam.init_energy, s=np.array(sim.lattice.steps_record),
                 runtime=time.time() - t0, prefix=sim.prefix,
                 ibp_check_log=np.array(getattr(sim, 'ibp_check_log', []), dtype=float),
                 edge_u_max=sim.DF_tracker.edge_u_max, **st)
        print(f'{name}: done in {(time.time() - t0) / 60:.1f} min', flush=True)


# the energy-change segments of progress_facet_bc20.md 9j [m]
SEGS = [(0.0, 18.5), (18.5, 20.0), (20.0, 36.65), (36.65, 38.67), (38.67, 39.5), (39.5, 40.5),
        (40.5, 43.5), (43.5, 48.0), (48.0, 1e9)]
SEGN = ['0-18.5', '18.5-20', '20-36.65', 'B2RE', '38.67-39.5', '39.5-40.5', '40.5-43.5',
        '43.5-48', 'B1RE']


def analyze():
    rec_f = os.path.join(HERE, 'benchmark_results', 'facet_bc20', 'runs',
                         'dfcsr_2d_peak_1M_sz6.5cap_m40_k115.npz')
    rec = dict(np.load(rec_f, allow_pickle=True))
    runs = {'recorded': dict(s=rec['step_s'], mean_energy=rec['mean_energy'],
                             pz=rec['coords_end'][5], runtime=rec['runtime'])}
    for n in VARIANTS:
        f = os.path.join(OUT, f'{n}.npz')
        if os.path.exists(f):
            runs[n] = dict(np.load(f, allow_pickle=True))
    lines = []

    def emit(t=''):
        print(t, flush=True)
        lines.append(t)

    E0 = float(rec['p0c'])
    emit('BC20, 1M particles, peak frame (dfcsr_2d_peak_1M_sz6.5cap_m40_k115), 3 MPI ranks per run')
    emit('Energy change per segment [MeV], from the per-step mean energy:')
    emit(f'{"run":9s} ' + ' '.join(f'{n:>10s}' for n in SEGN) + f' {"total":>9s} {"min":>6s}')
    for n, b in runs.items():
        ss, me = np.asarray(b['s']), np.asarray(b['mean_energy'])
        at = lambda x: np.interp(min(x, ss[-1]), ss, me)
        seg = [(at(b_) - at(a_)) / 1e6 for a_, b_ in SEGS]
        emit(f'{n:9s} ' + ' '.join(f'{v:10.3f}' for v in seg) + f' {sum(seg):9.3f} '
             f'{float(b["runtime"])/60:6.1f}')
    emit('')
    if 'B0' in runs:
        same = np.array_equal(runs['B0']['pz'], runs['recorded']['pz'])
        d = np.std(runs['B0']['pz'] - runs['recorded']['pz'])
        emit(f'B0 vs recorded: ' + ('bit-identical' if same else f'rms d(pz) = {d:.2e}'))
    for a, b, what in (('B0', 'B1', 'the u taper (B1 vs B0)'), ('B1', 'B2', 'by parts (B2 vs B1)')):
        if a in runs and b in runs:
            d = runs[b]['pz'] - runs[a]['pz']
            ref = runs[a]['pz'] - np.mean(runs[a]['pz'])
            emit(f'{what}: rms per-particle d(pz) = {np.std(d):.2e} (rms pz {np.std(ref):.2e})')
    emit('')
    emit('ibp_check (3 centre-row points per kick, both forms at 400 x\' nodes): max per segment')
    for n in VARIANTS:
        if n not in runs or not len(runs[n]['ibp_check_log']):
            continue
        L = runs[n]['ibp_check_log']
        row = []
        for (a_, b_), nm in zip(SEGS, SEGN):
            m = (L[:, 0] >= a_) & (L[:, 0] < b_)
            if m.any():
                row.append(f'{nm} {np.nanmax(L[m, 1]):.1e}')
        emit(f'  {n}: ' + '; '.join(row))
        emit(f'      kicks above 1e-3: {int(np.sum(L[:, 1] > 1e-3))} of {len(L)};  max zone share '
             f'{np.nanmax(L[:, 4]):.1e};  u-edge density max {float(runs[n]["edge_u_max"]):.1e}')
    with open(os.path.join(OUT, 'bc20_summary.txt'), 'w') as f:
        f.write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    if sys.argv[1] == 'analyze':
        analyze()
    else:
        main(sys.argv[1])
