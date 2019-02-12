#! /bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH --time=24:00:00
#SBATCH --job-name=tau_alpha_2
#SBATCH --output=/scratch/oem214/vanilla-rtrl/log/tau_alpha_2.o

module purge
SAVEPATH=/scratch/oem214/vanilla-rtrl/library/tau_alpha_2
export SAVEPATH
module load python3/intel/3.6.3
cd /scratch/oem214/vanilla-rtrl/
pwd > log/tau_alpha_2.log
date >> log/tau_alpha_2.log
which python >> log/tau_alpha_2.log
python main.py
