"""
Replay a BC20 2D run to the first kick at or after s_target and pickle the whole CSR2D object there
(density history, beam, lattice, schedule, mesh), so quadrature experiments on that kick take
seconds instead of a 45-minute replay.

    cd pyDFCSR_2D/example
    mpirun -np 6 python ../test/facet_bc20/dump_kick_state.py <run> <s_target> [--dense]   (pydfcsr-mpi)

--dense applies rerun_kick_dense.densify (a kick at every snapshot in its windows), so a state can be
pickled at any snapshot inside them (e.g. the 19.11 m glitch).

Writes benchmark_results/facet_bc20/runs/state_<run>_s<s_target>.pkl (rank 0; ~200 MB for the 1M
beam). Load with

    sim = load_state(path)        # from this module; sets parallel False, rebuilds the mesh
    dE, xk = sim.get_CSR_wake(sim.beam.position + z, x)
"""
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, HERE)
RES = os.path.join(HERE, '..', 'benchmark_results', 'facet_bc20')
RUNS = os.path.join(RES, 'runs')


class Stop(Exception):
    pass


def state_path(run, s_target):
    return os.path.join(RUNS, f'state_{run}_s{float(s_target):.2f}.pkl')


def load_state(path):
    with open(path, 'rb') as f:
        sim = pickle.load(f)
    sim.parallel = False
    sim.rank = 0
    # parameters added to Integration_params after the pickle was written get their defaults
    from pyDFCSR_2D.params import CSR_params, Integration_params
    for k, v in vars(Integration_params()).items():
        if not hasattr(sim.integration_params, k):
            setattr(sim.integration_params, k, v)
    for k, v in vars(CSR_params()).items():
        if not hasattr(sim.CSR_params, k):
            setattr(sim.CSR_params, k, v)
    sim.get_CSR_mesh()
    return sim


def main(run, s_target, *flags):
    from mpi4py import MPI
    from pyDFCSR_2D import CSR2D
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    s_target = float(s_target)
    sim = CSR2D(input_file=os.path.join(RUNS, 'inputs', f'{run}.yaml'), parallel=comm.Get_size() > 1)
    sim.CSR_params.write_wakes = False
    if '--dense' in flags:
        from rerun_kick_dense import densify
        n0, n1 = densify(sim)
        if rank == 0:
            print(f'  [dense] kicks {n0} -> {n1}', flush=True)
    orig = sim.calculate_2D_CSR_parallel if sim.parallel else sim.calculate_2D_CSR

    def kick():
        if sim.beam.position < s_target - 1e-9:
            return orig()
        if rank == 0:
            sim.get_CSR_mesh()
            drop = {k: getattr(sim, k) for k in ('calculate_2D_CSR', 'calculate_2D_CSR_parallel') if k in sim.__dict__}
            for k in drop:
                del sim.__dict__[k]
            p = state_path(run, s_target)
            with open(p, 'wb') as f:
                pickle.dump(sim, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f'state at s = {sim.beam.position:.4f} m saved to {p} ({os.path.getsize(p)/1e6:.0f} MB)', flush=True)
            sim2 = load_state(p)
            s, x = sim2.beam.position + sim2.CSR_zmesh[0], sim2.CSR_xmesh[0]
            print(f'reload check: get_CSR_wake at mesh point 0 = {sim2.get_CSR_wake(s, x)}', flush=True)
        raise Stop
    if sim.parallel:
        sim.calculate_2D_CSR_parallel = kick
    else:
        sim.calculate_2D_CSR = kick
    try:
        sim.run()
    except Stop:
        pass


if __name__ == '__main__':
    main(*sys.argv[1:])
