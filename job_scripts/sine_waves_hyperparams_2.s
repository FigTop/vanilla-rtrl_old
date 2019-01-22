#! /bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH --time=24:00:00
#SBATCH --job-name=sine_waves_hyper
#SBATCH --output=/scratch/oem214/vanilla-rtrl/log/sine_waves_hyper.o

module purge
SAVEPATH=/scratch/oem214/vanilla-rtrl/library/sine_waves_hyperparams_2
export SAVEPATH
module load python3/intel/3.6.3
cd /scratch/oem214/vanilla-rtrl/
pwd > log/sine_waves_hyperparams_2.log
date >> log/sine_waves_hyperparams_2.log
which python >> log/sine_waves_hyperparams_2.log
python main.py
