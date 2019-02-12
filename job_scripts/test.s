#! /bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH --time=24:00:00
#SBATCH --job-name=test
#SBATCH --output=/scratch/oem214/vanilla-rtrl/log/test.o

module purge
SAVEPATH=/scratch/oem214/vanilla-rtrl/library/test
export SAVEPATH
module load python3/intel/3.6.3
cd /scratch/oem214/vanilla-rtrl/
pwd > log/test.log
date >> log/test.log
which python >> log/test.log
python main.py
