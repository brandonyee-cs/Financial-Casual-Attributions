#!/bin/bash

# Script to run the entire pipeline for financial causal attributions
# This script will generate data, train models, compute attributions, and evaluate faithfulness

# Set up directories
BASE_DIR="$(pwd)"
DATA_DIR="${BASE_DIR}/data"
MODELS_DIR="${BASE_DIR}/models"
RESULTS_DIR="${BASE_DIR}/results"
PLOTS_DIR="${BASE_DIR}/plots"
LOGS_DIR="${BASE_DIR}/logs"

# Create directories if they don't exist
mkdir -p "$DATA_DIR" "$MODELS_DIR" "$RESULTS_DIR" "$PLOTS_DIR" "$LOGS_DIR"

# Log file
LOG_FILE="${LOGS_DIR}/run_all_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "===== Starting Financial Causal Attributions Pipeline ====="
echo "Date: $(date)"
echo "Base directory: $BASE_DIR"

# Step 1: Generate data
echo -e "\n===== Step 1: Generating synthetic financial data ====="
python3 scripts/generate_data.py --output_dir="$DATA_DIR" --n_samples=10000

# Check if data generation was successful
if [ $? -ne 0 ]; then
    echo "Error: Data generation failed. Exiting."
    exit 1
fi

# Step 2: Train models for each scenario
echo -e "\n===== Step 2: Training models ====="
MODEL_TYPES=("mlp" "lstm" "xgboost")
SCENARIOS=("asset_pricing" "credit_risk" "fraud_detection")

for scenario in "${SCENARIOS[@]}"; do
    for model_type in "${MODEL_TYPES[@]}"; do
        echo "Training $model_type model for $scenario scenario..."
        python3 scripts/train_models.py --scenario="$scenario" --model_type="$model_type" \
                                      --data_dir="$DATA_DIR" --model_dir="$MODELS_DIR"
    done
done

# Check if model training was successful
if [ $? -ne 0 ]; then
    echo "Error: Model training failed. Exiting."
    exit 1
fi

# Step 3: Run attributions for each model and scenario
echo -e "\n===== Step 3: Computing attributions ====="

# Define attribution methods based on model type
declare -A MODEL_ATTRIBUTION_METHODS
MODEL_ATTRIBUTION_METHODS["mlp"]=("saliency" "gradient_input" "integrated_gradients" "shap")
MODEL_ATTRIBUTION_METHODS["lstm"]=("saliency" "gradient_input" "integrated_gradients" "shap")
MODEL_ATTRIBUTION_METHODS["xgboost"]=("xgboost" "shap")

# Compute attributions for each model and scenario
echo "Computing attributions..."
for model_type in "${MODEL_TYPES[@]}"; do
    for scenario in "${SCENARIOS[@]}"; do
        echo "Computing attributions for $model_type model in $scenario scenario..."
        python3 scripts/run_attributions.py \
            --model_type="$model_type" \
            --scenario="$scenario" \
            --data_dir="$DATA_DIR" \
            --models_dir="$MODELS_DIR" \
            --output_dir="$RESULTS_DIR" \
            --attribution_methods "${MODEL_ATTRIBUTION_METHODS[$model_type]}" \
            --n_samples=1000 \
            --batch_size=32 \
            --seed=42 \
            --log_file="$LOG_FILE"
        
        if [ $? -ne 0 ]; then
            echo "Error: Failed to compute attributions for $model_type model in $scenario scenario"
            exit 1
        fi
    done
done

# Step 4: Evaluate faithfulness of attributions
echo -e "\n===== Step 4: Evaluating causal faithfulness ====="
python3 scripts/evaluate_faithfulness.py --data_dir="$DATA_DIR" --results_dir="$RESULTS_DIR" \
                                       --plots_dir="$PLOTS_DIR"

# Check if evaluation was successful
if [ $? -ne 0 ]; then
    echo "Error: Faithfulness evaluation failed. Exiting."
    exit 1
fi

echo -e "\n===== Pipeline completed successfully! ====="
echo "Results saved to $RESULTS_DIR"
echo "Plots saved to $PLOTS_DIR"
echo "Logs saved to $LOG_FILE"
echo "Date: $(date)"