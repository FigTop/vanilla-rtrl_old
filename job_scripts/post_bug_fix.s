#! /bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH --time=24:00:00
#SBATCH --job-name=post_bug_fix
#SBATCH --output=/scratch/oem214/vanilla-rtrl/log/post_bug_fix.o

module purge
SAVEPATH=/scratch/oem214/vanilla-rtrl/library/post_bug_fix
export SAVEPATH
module load python3/intel/3.6.3
cd /scratch/oem214/vanilla-rtrl/
pwd > log/post_bug_fix.log
date >> log/post_bug_fix.log
which python >> log/post_bug_fix.log
python main.py
