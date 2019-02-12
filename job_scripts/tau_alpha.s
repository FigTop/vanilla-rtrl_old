#! /bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH --time=24:00:00
#SBATCH --job-name=tau_alpha
#SBATCH --output=/scratch/oem214/vanilla-rtrl/log/tau_alpha.o

module purge
SAVEPATH=/scratch/oem214/vanilla-rtrl/library/tau_alpha
export SAVEPATH
module load python3/intel/3.6.3
cd /scratch/oem214/vanilla-rtrl/
pwd > log/tau_alpha.log
date >> log/tau_alpha.log
which python >> log/tau_alpha.log
python main.py
