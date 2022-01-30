#!/bin/bash
#SBATCH --job-name=optuna
#SBATCH --ntasks=1
#SBATCH --time=5-0:0
#SBATCH --mem-per-cpu=1G
#SBATCH --partition=general,multicore
#SBATCH --exclude=fermi,bose,chandra,bethe,hawking,higgs,salam,schroedinger,maxwell,cooper,haas,maisner
#SBATCH --array=0-5
NAME=simple_gaussian1

DATADIR=/home/andreapasquale/gLund/first_scan

mkdir -p $DATADIR

optuna create-study --study-name "hyperopt2" --storage 'mysql://root:9FsbrzY5FQ@galileo/example' --skip-if-exists

srun -N1 -n1 -c1 --exclusive python script.py > $DATADIR/$NAME.$SLURM_JOB_ID.out 
