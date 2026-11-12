"""
MPI against serial for the quantile frame + charge mesh, on the golden case.

    mpirun -np 3 python pyDFCSR_2D/test/golden/mpi_check_quantile.py     (pydfcsr-mpi env)

Rank 0 runs the serial case too and asserts every wake array is identical.
"""
import os
import sys

import numpy as np
from mpi4py import MPI

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_golden_2d as g  # noqa: E402

DEP = {'frame': 'quantile', 'zbins': 128, 'smoothing_sigma_z': 1.0}
CSR = {'mesh_mode': 'charge'}

par = g.run_case(DEP, CSR, parallel=True)
if MPI.COMM_WORLD.Get_rank() == 0:
    ser = g.run_case(DEP, CSR, parallel=False)
    bad = [k for k in ser if not np.array_equal(ser[k], par[k])]
    print(f'MPI ({MPI.COMM_WORLD.Get_size()} ranks) vs serial: {len(ser)} arrays, '
          f'{"all identical" if not bad else "DIFFER: " + str(bad)}')
    sys.exit(1 if bad else 0)
