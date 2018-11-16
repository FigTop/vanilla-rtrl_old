#! /bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH --time=24:00:00
#SBATCH --job-name=cosyne_results
#SBATCH --output=/scratch/oem214/vanilla-rtrl/log/cosyne_results.o

module purge
SAVEPATH=/scratch/oem214/vanilla-rtrl/library/cosyne_results
export SAVEPATH
module load python3/intel/3.6.3
cd /scratch/oem214/vanilla-rtrl/
pwd > log/cosyne_results.log
date >> log/cosyne_results.log
which python >> log/cosyne_results.log
python main.py
