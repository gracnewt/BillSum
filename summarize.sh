#!/bin/bash
#SBATCH --job-name=legal_summ_test    # Job name
#SBATCH --output=slurms/summ_test_%j.out # Standard output log (%j inserts job ID)
#SBATCH --error=slurms/summ_test_%j.err  # Standard error log
#SBATCH --gres=gpu:1                  # Request 1 GPU
#SBATCH --nodes=1                     # Run on a single node
#SBATCH --ntasks=1                    # Run a single task
#SBATCH --cpus-per-task=4             # Request 4 CPU cores
#SBATCH --mem=16G                     # Request 16GB of system memory

# Exit immediately if a command exits with a non-zero status
set -e

# Create a logs directory if it doesn't exist
mkdir -p logs

echo "======================================================"
echo "Starting Legal-BERT Summarization Test Job"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU Allocated: $CUDA_VISIBLE_DEVICES"
echo "Start Time: $(date)"
echo "======================================================"

# 1. Initialize Conda
# Adjust the path below to point to your miniconda3/anaconda3 installation if different
source /home/gracenewton/miniconda3/etc/profile.d/conda.sh

# 2. Activate your legal NLP environment
echo "Activating Conda environment: legal_nlp..."
conda activate legal_nlp

# 3. Define local paths for the test run
# Adjust these paths to match your directory layout
PROJECT_DIR="/home/gracenewton/nlp_final/BillSum"
INPUT_BILL="$PROJECT_DIR/test_input_bill.txt"
OUTPUT_SUMM="$PROJECT_DIR/test_output_summary.txt"
MODEL_CHECKPOINT="$PROJECT_DIR/billsum/data/models/fullBS" # Testing with the BERTScore config

# Create a dummy test bill if one doesn't exist to prevent file-not-found errors
if [ ! -f "$INPUT_BILL" ]; then
    echo "Creating a dummy legislative bill for test validation..."
    cat << 'EOF' > "$INPUT_BILL"
SECTION 1. SHORT TITLE.
This Act may be cited as the "Legal NLP Evaluation and Demonstration Act of 2026".
SEC. 2. FINDINGS AND PURPOSE.
The Congress finds that the complexity of legislative drafting has increased significantly over the past decade. It is the purpose of this Act to establish a standardized framework for the automatic, extractive summarization of congressional bills.
SEC. 3. RESEARCH AND DEVELOPMENT GRANTS.
The Director of the National Science Foundation shall carry out a program to award grants to institutions of higher education to develop specialized language encoders. These encoders shall be fine-tuned specifically on legal databases to identify salient structural information.
SEC. 4. AUTHORIZATION OF APPROPRIATIONS.
There are authorized to be appropriated to carry out this Act $5,000,000 for each of fiscal years 2026 through 2030.
EOF
fi

# 4. Run the applied summarization pipeline script
echo "Executing python pipeline..."
python "$PROJECT_DIR/summarize_bill.py" \
    --input_file "$INPUT_BILL" \
    --output_file "$OUTPUT_SUMM" \
    --model_path "$MODEL_CHECKPOINT" \
    --lambda_val 0.5 \
    --summary_length 2

echo "======================================================"
echo "Summary generated successfully!"
echo "Output saved to: $OUTPUT_SUMM"
echo "End Time: $(date)"
echo "======================================================"