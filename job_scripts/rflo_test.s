#! /bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH --time=24:00:00
#SBATCH --job-name=rflo_test
#SBATCH --output=/scratch/oem214/vanilla-rtrl/log/rflo_test.o

module purge
SAVEPATH=/scratch/oem214/vanilla-rtrl/library/rflo_test
export SAVEPATH
module load python3/intel/3.6.3
cd /scratch/oem214/vanilla-rtrl/
pwd > log/rflo_test.log
date >> log/rflo_test.log
which python >> log/rflo_test.log
python main.py
