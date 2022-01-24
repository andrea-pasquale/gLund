#!/bin/bash
#
#SBATCH --job-name=qgan
#SBATCH --output=first_run.txt
#SBATCH --ntasks=1
#SBATCH --time=7-0:0
#SBATCH --mem-per-cpu=1G

srun python script.py
