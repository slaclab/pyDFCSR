#!/bin/bash
#SBATCH --account=ad:beamphysics
#SBATCH --partition=milano
#SBATCH --job-name=CSR
#SBATCH --output=job_%j.out
#SBATCH --error=job_%j.err
#SBATCH --ntasks=300
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4g
#SBATCH --time=48:00:00
#--exclude=sdfmilan[023-026.120]
#--exclusive

module load mpi/openmpi-x86_64



#mpirun -n 128 python -u /sdf/group/beamphysics/jytang/pyDFCSR/pyDFCSR_2D/debug_file.py


mpirun -n 300 python  -m pyDFCSR_mpi_run ./input/chicane_config.yaml
