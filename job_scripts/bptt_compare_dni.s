#! /bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH --time=24:00:00
#SBATCH --job-name=bptt_compare_dni
#SBATCH --output=/scratch/oem214/vanilla-rtrl/log/bptt_compare_dni.o

module purge
SAVEPATH=/scratch/oem214/vanilla-rtrl/library/bptt_compare_dni
export SAVEPATH
module load python3/intel/3.6.3
cd /scratch/oem214/vanilla-rtrl/
pwd > log/bptt_compare_dni.log
date >> log/bptt_compare_dni.log
which python >> log/bptt_compare_dni.log
python main.py
