#! /bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH --time=24:00:00
#SBATCH --job-name=ssa_learning_bpt
#SBATCH --output=/scratch/oem214/vanilla-rtrl/log/ssa_learning_bpt.o

module purge
SAVEPATH=/scratch/oem214/vanilla-rtrl/library/ssa_learning_bptt_sg
export SAVEPATH
module load python3/intel/3.6.3
cd /scratch/oem214/vanilla-rtrl/
pwd > log/ssa_learning_bptt_sg.log
date >> log/ssa_learning_bptt_sg.log
which python >> log/ssa_learning_bptt_sg.log
python main.py
