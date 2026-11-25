import argparse
import os
import sys

# mpi4py is imported before pyDFCSR_2D.CSR (which defaults this too): on macOS without it the libfabric
# sockets provider spins two kernel threads per rank and each rank spent half its CPU in the kernel
if sys.platform == 'darwin':
    os.environ.setdefault('FI_PROVIDER', 'tcp')

from mpi4py import MPI

from pyDFCSR_2D import CSR2D

# from mpi4py.futures import MPIPoolExecutor


comm = MPI.COMM_WORLD
mpi_rank = comm.Get_rank()
mpi_size = comm.Get_size()

"""
Adopted from https://github.com/ChristopherMayes/Xopt/blob/main/xopt/mpi/run.py

Xopt MPI driver

Basic usage:

mpirun -n 4 python -m mpi4py.futures -m pyDFCSR_mpi_run ./input/dipole_config.yaml


"""


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Configure pyDFCSR")
    parser.add_argument("input_file", help="input_file")

    args = parser.parse_args()
    #print(args)

    infile = args.input_file
    assert os.path.exists(infile), f"Input file does not exist: {infile}"

    CSR = CSR2D(input_file= infile, parallel= True)
    CSR.run()
