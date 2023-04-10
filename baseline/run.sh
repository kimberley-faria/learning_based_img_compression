#!/usr/bin/env bash
#SBATCH -c 4  # Number of Cores per Task
#SBATCH --mem=40G  # Requested Memory
#SBATCH -p gypsum-titanx  # Partition
#SBATCH -G 1  # Number of GPUs
#SBATCH -t 48:00:00  # Job time limit
#SBATCH -o slurm-%j.out  # %j = job ID

module load cuda conda python
conda init bash
conda activate ds696-project

cd ~/texture_recogn_reimpl/baseline
python baseline_experiment.py $1