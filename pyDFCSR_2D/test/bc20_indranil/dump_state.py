"""
Pickle the CSR2D state of an Indranil run (run_indranil.py) at one or more kicks, optionally with the
CSR switched off before s_start (the kick is replaced by zeros there, the density history is still
built), so a late state can be reached in a fraction of the run time.

    cd pyDFCSR_2D/example
    [mpirun -np 3] python ../test/bc20_indranil/dump_state.py <run> <s_target>[,<s_target2>...] [s_start]

Writes benchmark_results/bc20_indranil/runs/state_<run>_s<s_target>.pkl (dump_kick_state.state_path with
this directory's RUNS). Load with dump_kick_state.load_state.
"""
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, os.path.join(HERE, '..', 'facet_bc20'))
import dump_kick_state as D  # noqa: E402

RUNS = os.path.join(HERE, '..', 'benchmark_results', 'bc20_indranil', 'runs')
D.RUNS = RUNS


def main(run, targets, s_start=None):
    from mpi4py import MPI
    from pyDFCSR_2D import CSR2D
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    targets = sorted(float(v) for v in str(targets).split(','))
    s_start = None if s_start is None else float(s_start)
    sim = CSR2D(input_file=os.path.join(RUNS, 'inputs', f'{run}.yaml'), parallel=comm.Get_size() > 1)
    sim.CSR_params.write_wakes = False
    orig = sim.calculate_2D_CSR_parallel if sim.parallel else sim.calculate_2D_CSR
    done = []

    def kick():
        pos = sim.beam.position
        due = [t for t in targets if t not in done and pos >= t - 1e-9]
        if due:
            done.extend(due)
            if rank == 0:
                sim.get_CSR_mesh()
                drop = {k: getattr(sim, k) for k in ('calculate_2D_CSR', 'calculate_2D_CSR_parallel') if k in sim.__dict__}
                for k in drop:
                    del sim.__dict__[k]
                p = D.state_path(run, due[0])
                with open(p, 'wb') as f:
                    pickle.dump(sim, f, protocol=pickle.HIGHEST_PROTOCOL)
                print(f'state at s = {pos:.4f} m saved to {p} ({os.path.getsize(p)/1e6:.0f} MB)', flush=True)
                for k, v in drop.items():
                    setattr(sim, k, v)
            if len(done) == len(targets):
                raise D.Stop
        if s_start is not None and pos < s_start - 1e-9:
            # CSR off before s_start (zero kicks; states before s_start are dumped too, with a CSR-off beam)
            n = (sim.CSR_params.xbins, sim.CSR_params.zbins)
            sim.dE_dct, sim.x_kick = np.zeros(n), np.zeros(n)
            return
        return orig()
    if sim.parallel:
        sim.calculate_2D_CSR_parallel = kick
    else:
        sim.calculate_2D_CSR = kick
    try:
        sim.run()
    except D.Stop:
        pass


if __name__ == '__main__':
    main(*sys.argv[1:])
