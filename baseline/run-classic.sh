#!/usr/bin/env bash
#SBATCH -c 4  # Number of Cores per Task
#SBATCH --mem=40G  # Requested Memory
#SBATCH -p gypsum-titanx  # Partition
#SBATCH -G 1  # Number of GPUs
#SBATCH -t 48:00:00  # Job time limit
#SBATCH -o slurm-%j.out  # %j = job ID

module load cuda

cd ~/learning_based_img_compression/baseline
/home/kfaria_umass_edu/.conda/envs/ds696-project/bin/python baseline_experiment_classic.py $1