#!/bin/bash
#SBATCH --job-name=BSBertFast
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

# Stop script if any command fails
set -e

#echo "Starting Clean Text..."
#python -u billsum/data_prep/clean_text.py

echo "Starting BERT_BSFAST Training..."
python -u bert_full_train_BSFast.py
