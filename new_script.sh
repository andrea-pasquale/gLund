#!/bin/bash
#SBATCH --job-name=training
#SBATCH --ntasks=1
#SBATCH --time=10-0:0
#SBATCH --partition=general
#SBATCH --exclude=fermi,bose,chandra,bethe,hawking,higgs,salam,schroedinger,maxwell,cooper,haas,maisner

python less_pixel_simplified.py --layers 3 --latent_dim 3  --lr 5e-1 --lr_d 1e-2 --dataset 8x8Digits --folder test8x8 --nqubits 6 --pixels 64 --training_samples 150 --batch_samples 64
