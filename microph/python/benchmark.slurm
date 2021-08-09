#!/bin/bash
#SBATCH --constraint=gpu
#SBATCH --job-name=physics_benchmark
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --output=out.log
#SBATCH --time=01:30:00
#SBATCH --gres=gpu:1
#SBATCH --account=s1053
#SBATCH --partition=cscsci

export VERSION="gtcuda"
export STENCILS="normal"
export RUN_MODE="benchmark"
export REBUILD="False"
export REPS=10
# export CN=24

for i in 256 384
do
    export CN=$i
    srun python3 python/benchmark.py
done