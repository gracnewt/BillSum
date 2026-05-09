#!/bin/bash
#SBATCH --job-name=fast2
#SBATCH --nodes=1
#SBATCH --gres=gpu:1        # Request a GPU for your BERT model
#SBATCH --mem=32G
#SBATCH --nodelist=student-gpu-004

# Clear any existing python paths
unset PYTHONPATH

# Initialize Conda for the compute node
source /home/gracenewton/miniconda3/etc/profile.d/conda.sh

# Activate the environment
conda activate legal_nlp

# Verify Torch is actually there before starting
python -c "import torch; print('Successfully loaded Torch version:', torch.__version__)"

# 2. Set Paths - DOUBLE CHECK THESE!
export BILLSUM_PREFIX="/home/gracenewton/nlp_final/BillSum/billsum/data/"
export PYTHONPATH=$PYTHONPATH:.

# Stop script if any command fails
set -e

#echo "Starting Clean Text..."
#python -u billsum/data_prep/clean_text.py

#echo "Starting Sentence Labeling..."
#python -u billsum/data_prep/label_sentences.py

echo "Starting BERT Eval..."
/home/gracenewton/miniconda3/envs/legal_nlp/bin/python -u evaluate.py
