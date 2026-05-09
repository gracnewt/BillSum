#!/bin/bash
#SBATCH --job-name=quality
#SBATCH --nodes=1
#SBATCH --gres=gpu:1        # Request a GPU for your BERT model
#SBATCH --mem=32G
#SBATCH --nodelist=student-gpu-002

# 1. Force the correct environment
source /home/gracenewton/miniconda3/etc/profile.d/conda.sh
conda activate legal_nlp

# 2. Set Paths - DOUBLE CHECK THESE!
export BILLSUM_PREFIX="/home/gracenewton/nlp_final/BillSum/billsum/data/"
export PYTHONPATH=$PYTHONPATH:.

echo "Extracting examples..."
python qualitative_examples.py

bash ./organize_slurm_output.sh