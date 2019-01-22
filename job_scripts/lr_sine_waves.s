#! /bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH --time=24:00:00
#SBATCH --job-name=lr_sine_waves
#SBATCH --output=/scratch/oem214/vanilla-rtrl/log/lr_sine_waves.o

module purge
SAVEPATH=/scratch/oem214/vanilla-rtrl/library/lr_sine_waves
export SAVEPATH
module load python3/intel/3.6.3
cd /scratch/oem214/vanilla-rtrl/
pwd > log/lr_sine_waves.log
date >> log/lr_sine_waves.log
which python >> log/lr_sine_waves.log
python main.py
