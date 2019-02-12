#! /bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH --time=24:00:00
#SBATCH --job-name=final_poster
#SBATCH --output=/scratch/oem214/vanilla-rtrl/log/final_poster.o

module purge
SAVEPATH=/scratch/oem214/vanilla-rtrl/library/final_poster
export SAVEPATH
module load python3/intel/3.6.3
cd /scratch/oem214/vanilla-rtrl/
pwd > log/final_poster.log
date >> log/final_poster.log
which python >> log/final_poster.log
python main.py
