#! /bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH --time=24:00:00
#SBATCH --job-name=dni_compare_bptt
#SBATCH --output=/scratch/oem214/vanilla-rtrl/log/dni_compare_bptt.o

module purge
SAVEPATH=/scratch/oem214/vanilla-rtrl/library/dni_compare_bptt
export SAVEPATH
module load python3/intel/3.6.3
cd /scratch/oem214/vanilla-rtrl/
pwd > log/dni_compare_bptt.log
date >> log/dni_compare_bptt.log
which python >> log/dni_compare_bptt.log
python main.py
