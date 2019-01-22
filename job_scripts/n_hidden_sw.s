#! /bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH --time=24:00:00
#SBATCH --job-name=n_hidden_sw
#SBATCH --output=/scratch/oem214/vanilla-rtrl/log/n_hidden_sw.o

module purge
SAVEPATH=/scratch/oem214/vanilla-rtrl/library/n_hidden_sw
export SAVEPATH
module load python3/intel/3.6.3
cd /scratch/oem214/vanilla-rtrl/
pwd > log/n_hidden_sw.log
date >> log/n_hidden_sw.log
which python >> log/n_hidden_sw.log
python main.py
