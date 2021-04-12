#!/bin/bash
#SBATCH --constraint=gpu
#SBATCH --job-name=physics_benchmark
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=12
#SBATCH --cpus-per-task=1
#SBATCH --output=out.log
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --account=s1053
#SBATCH --partition=cscsci

mkdir -p out

for i in 64 96 128 192 256 384
do
    OPT="-DREPS=10 -DCNINPUT=$i -DEXACT_SVP -DUSE_MIN_EVAP -DUSE_MPIF90" make WARN=False -s -C .
    srun ./main.x
    make clean -s -C .
done