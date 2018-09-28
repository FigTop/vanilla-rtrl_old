#!/bin/bash
#
##SBATCH --nodes=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=1:00:00
#SBATCH --mem=8GB
#SBATCH --job-name=rtrl
#SBATCH --mail-type=END
##SBATCH --mail-user=oem214@nyu.edu
#SBATCH --output=slurm_%j.out

module purge

python3 cluster_main.py
