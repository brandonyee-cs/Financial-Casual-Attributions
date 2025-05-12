#!/bin/bash

# Script to run attributions using existing data and models
# This script assumes data and trained models already exist

# Set up directories
BASE_DIR="$(pwd)"
DATA_DIR="${BASE_DIR}/data"
MODELS_DIR="${BASE_DIR}/models"
RESULTS_DIR="${BASE_DIR}/results"
PLOTS_DIR="${BASE_DIR}/plots"
LOGS_DIR="${BASE_DIR}/logs"

# Set parameters
N_SAMPLES=1000
SEED=42

# Create directories if they don't exist
mkdir -p "$RESULTS_DIR" "$PLOTS_DIR" "$LOGS_DIR"

# Log file
LOG_FILE="${LOGS_DIR}/run_attributions_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "===== Starting Attribution Computation ====="
echo "Date: $(date)"
echo "Base directory: $BASE_DIR"

# Verify data and models exist
if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Data directory not found at $DATA_DIR"
    exit 1
fi

if [ ! -d "$MODELS_DIR" ]; then
    echo "Error: Models directory not found at $MODELS_DIR"
    exit 1
fi

# Run attributions
echo -e "\n===== Computing attributions ====="
python3 scripts/run_all_attributions.py \
    --base_dir="$BASE_DIR" \
    --n_samples="$N_SAMPLES" \
    --random_state="$SEED"

# Check if attribution computation was successful
if [ $? -ne 0 ]; then
    echo "Error: Attribution computation failed. Exiting."
    exit 1
fi

# Evaluate faithfulness of attributions
echo -e "\n===== Evaluating causal faithfulness ====="
python3 scripts/evaluate_faithfulness.py --data_dir="$DATA_DIR" --results_dir="$RESULTS_DIR" \
                                       --plots_dir="$PLOTS_DIR"

# Check if evaluation was successful
if [ $? -ne 0 ]; then
    echo "Error: Faithfulness evaluation failed. Exiting."
    exit 1
fi

echo -e "\n===== Attribution pipeline completed successfully! ====="
echo "Results saved to $RESULTS_DIR"
echo "Plots saved to $PLOTS_DIR"
echo "Logs saved to $LOG_FILE"
echo "Date: $(date)" 