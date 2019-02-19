#! /bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH --time=24:00:00
#SBATCH --job-name=alpha03_tau4_che
#SBATCH --output=/scratch/oem214/vanilla-rtrl/log/alpha03_tau4_che.o

module purge
SAVEPATH=/scratch/oem214/vanilla-rtrl/library/alpha03_tau4_checkpoints
export SAVEPATH
module load python3/intel/3.6.3
cd /scratch/oem214/vanilla-rtrl/
pwd > log/alpha03_tau4_checkpoints.log
date >> log/alpha03_tau4_checkpoints.log
which python >> log/alpha03_tau4_checkpoints.log
python main.py
