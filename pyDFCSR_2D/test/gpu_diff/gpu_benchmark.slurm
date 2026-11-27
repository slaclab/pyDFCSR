#!/bin/bash
# S3DF GPU job for gpu_benchmark.py. Fill in the account (s3df: `sacctmgr show assoc user=$USER`)
# and pick a float64-capable partition (ampere = A100).
#SBATCH --job-name=dfcsr-gpu-bench
#SBATCH --partition=ampere
#SBATCH --account=FILL_ME_IN
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=gpu_benchmark_%j.log

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pydfcsr-gpu
cd "$SLURM_SUBMIT_DIR"          # submit from pyDFCSR_2D/test/gpu_diff
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
python gpu_benchmark.py 20000
