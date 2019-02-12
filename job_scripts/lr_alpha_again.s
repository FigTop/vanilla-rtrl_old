#! /bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH --time=24:00:00
#SBATCH --job-name=lr_alpha_again
#SBATCH --output=/scratch/oem214/vanilla-rtrl/log/lr_alpha_again.o

module purge
SAVEPATH=/scratch/oem214/vanilla-rtrl/library/lr_alpha_again
export SAVEPATH
module load python3/intel/3.6.3
cd /scratch/oem214/vanilla-rtrl/
pwd > log/lr_alpha_again.log
date >> log/lr_alpha_again.log
which python >> log/lr_alpha_again.log
python main.py
