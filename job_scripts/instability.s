#! /bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=24:00:00
#SBATCH --job-name=instability
#SBATCH --output=/scratch/oem214/vanilla-rtrl/log/instability.o

module purge
module load python3/intel/3.6.3
cd /scratch/oem214/vanilla-rtrl/
pwd > log/instability.log
date >> log/instability.log
which python >> log/instability.log
python main.py
