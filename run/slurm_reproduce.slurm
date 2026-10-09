#!/bin/bash
#SBATCH --job-name=bmas_reproduce
#SBATCH --output=logs/bmas_reproduce_%j.out
#SBATCH --error=logs/bmas_reproduce_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:1
# Adjust account/partition/GPU type to your cluster, e.g.
##SBATCH --account=uit
##SBATCH --partition=normal
##SBATCH --gres=gpu:2g.35gb:1
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs
bash reproduce_all.sh
