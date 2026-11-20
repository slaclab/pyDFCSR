"""
Kick-convergence check for the two 2D-only transients the 1D-guided schedule under-samples
(progress_facet_bc20.md 9j): the base run's schedule with a kick at EVERY snapshot inside the
windows [18.0, 20.0] m (drift D5, the sextupole-region feature) and [38.6, 41.0] m (after the
B2RE exit), midpoint intervals recomputed (clipped at the dipole edges as in the auto schedule).

    cd pyDFCSR_2D/example
    mpirun -np 6 python ../test/facet_bc20/rerun_kick_dense.py [base_run]         (pydfcsr-mpi)
    conda run -n pydfcsr python ../test/facet_bc20/rerun_kick_dense.py --report [base_run]

Saved as <base_run>_kdense in the run_dfcsr_bc20 format. --report: energy change per segment from
the per-step mean energy, against the base run (116 kicks) and the 222-kick run.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, HERE)
RES = os.path.join(HERE, '..', 'benchmark_results', 'facet_bc20')
RUNS = os.path.join(RES, 'runs')
BASE = 'dfcsr_2d_peak_1M_sz6.5cap_m40_k115'
WINDOWS = ((18.0, 20.0), (38.6, 41.0))
SEGS = ((0, 18.0), (18.0, 20.0), (20.0, 36.65), (36.65, 38.67), (38.67, 39.5), (39.5, 40.5), (40.5, 43.5),
        (43.5, 48.02), (48.02, 49.1))


def densify(sim):
    """Kicks at every snapshot inside WINDOWS; midpoint intervals rebuilt, clipped at the dipole edges."""
    from pyDFCSR_2D.schedule import midpoint_intervals
    sch = sim.lattice.schedule
    s = np.asarray(sch.s_nodes)
    add = np.zeros(s.size, bool)
    for lo, hi in WINDOWS:
        add |= (s >= lo) & (s <= hi)
    add &= np.asarray(sch.is_snap)
    add[0] = False
    n0 = int(sch.is_kick.sum())
    sch.is_kick = sch.is_kick | add
    cfg = sim.lattice.lattice_config
    edges, s0 = [], 0.0
    for k, v in cfg.items():
        if k == 'step_size':
            continue
        if v['type'] == 'dipole':
            edges += [s0, s0 + v['L']]
        s0 += v['L']
    lo, hi = midpoint_intervals(s, np.flatnonzero(sch.is_kick), edges, float(s0))
    sch.kick_lo, sch.kick_hi = lo, hi
    return n0, int(sch.is_kick.sum())


def main(base=BASE):
    from mpi4py import MPI
    import run_dfcsr_bc20 as RD
    from pyDFCSR_2D.CSR import CSR2D
    rank = MPI.COMM_WORLD.Get_rank()
    name = f'{base}_kdense'
    RD.RUNS[name] = dict(RD.RUNS[base])
    orig_run = CSR2D.run

    def run(self, *a, **k):
        n0, n1 = densify(self)
        if rank == 0:
            print(f'  [kdense] kicks {n0} -> {n1} (every snapshot in {WINDOWS})', flush=True)
        return orig_run(self, *a, **k)
    CSR2D.run = run
    RD.main(name)


def seg_table(name):
    import glob
    import h5py
    f = h5py.File(sorted(glob.glob(os.path.join(RUNS, f'{name}-2*-statistics.h5')), key=os.path.getmtime)[-1], 'r')
    s = f['step_schedule/s_nodes'][()]
    E = f['mean_energy'][()]
    kk = f['step_schedule/is_kick'][()].astype(bool)
    dE = np.diff(E) / 1e6
    sm = s[1:]
    return [(dE[(sm > a) & (sm <= b)].sum(), int(kk[1:][(sm > a) & (sm <= b)].sum())) for a, b in SEGS]


def report(base=BASE):
    rows = [(f'{base}_kdense', 'dense in the windows'), (base, '116 kicks'), ('dfcsr_2d_peak_1M', '222 kicks')]
    print('mean energy change per segment [MeV] (kicks in the segment)')
    print(f'{"run":22s}' + ''.join(f'{a:.1f}-{b:.1f}'.rjust(16) for a, b in SEGS) + f'{"total":>10s}')
    for name, lab in rows:
        try:
            t = seg_table(name)
        except (IndexError, KeyError, OSError):
            continue
        print(f'{lab:22s}' + ''.join(f'{v:+9.3f} ({k:3d})' for v, k in t) + f'{sum(v for v, _ in t):+10.3f}')
    import compare_1M
    compare_1M.main([f'{base}_kdense', base, 'dfcsr_2d_peak_1M', 'dfcsr_1d_1M'])


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--report':
        report(*sys.argv[2:])
    else:
        main(*sys.argv[1:])
