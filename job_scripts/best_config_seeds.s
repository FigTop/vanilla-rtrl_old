#! /bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH --time=24:00:00
#SBATCH --job-name=best_config_seed
#SBATCH --output=/scratch/oem214/vanilla-rtrl/log/best_config_seed.o

module purge
SAVEPATH=/scratch/oem214/vanilla-rtrl/library/best_config_seeds
export SAVEPATH
module load python3/intel/3.6.3
cd /scratch/oem214/vanilla-rtrl/
pwd > log/best_config_seeds.log
date >> log/best_config_seeds.log
which python >> log/best_config_seeds.log
python main.py
