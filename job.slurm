#!/bin/bash
#SBATCH --job-name=billsum_bert
#SBATCH --output=result.out
#SBATCH --nodes=1
#SBATCH --gres=gpu:1        # Request a GPU for your BERT model
#SBATCH --mem=32G

# 1. INITIALIZE CONDA
# This points directly to the activation script in your new install
source $HOME/miniconda3/etc/profile.d/conda.sh

# 2. ACTIVATE YOUR ENVIRONMENT
conda activate legal_nlp

# 3. SET PROJECT VARIABLES
export BILLSUM_PREFIX="/home/gracenewton/BillSum/data"
export BERT_DATA_DIR=$BILLSUM_PREFIX/bert_data
export HF_HOME="/home/gracenewton/huggingface_cache"
export PYTHONPATH=$PYTHONPATH:.

# Create necessary scratch folders
mkdir -p $BILLSUM_PREFIX
mkdir -p $HF_HOME

# 4. DATA PREP (Only needs to run once)
# If you've already run these, you can comment them out to save time
python billsum/data_prep/clean_text.py
python billsum/data_prep/label_sentences.py

# 5. RUN BERT TRAINING
# This is the script we talked about that uses Hugging Face
python bert_training.py