#! /bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH --time=24:00:00
#SBATCH --job-name=full_bio_lr_sear
#SBATCH --output=/scratch/oem214/vanilla-rtrl/log/full_bio_lr_sear.o

module purge
SAVEPATH=/scratch/oem214/vanilla-rtrl/library/full_bio_lr_search
export SAVEPATH
module load python3/intel/3.6.3
cd /scratch/oem214/vanilla-rtrl/
pwd > log/full_bio_lr_search.log
date >> log/full_bio_lr_search.log
which python >> log/full_bio_lr_search.log
python main.py
