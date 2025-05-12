#!/bin/bash

# Script to run the entire pipeline for financial causal attributions
# This script will generate data, train models, compute attributions, evaluate faithfulness,
# perform detailed analysis, and generate a comprehensive report for the paper

# Set up directories
BASE_DIR="$(pwd)"
DATA_DIR="${BASE_DIR}/data"
MODELS_DIR="${BASE_DIR}/models"
RESULTS_DIR="${BASE_DIR}/results"
PLOTS_DIR="${BASE_DIR}/plots"
LOGS_DIR="${BASE_DIR}/logs"
REPORT_DIR="${BASE_DIR}/report"

# Set parameters
N_SAMPLES=10000
EVAL_SAMPLES=1000
SEED=42

# Create directories if they don't exist
mkdir -p "$DATA_DIR" "$MODELS_DIR" "$RESULTS_DIR" "$PLOTS_DIR" "$LOGS_DIR" "$REPORT_DIR"

# Log file
LOG_FILE="${LOGS_DIR}/run_all_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "===== Starting Financial Causal Attributions Pipeline ====="
echo "Date: $(date)"
echo "Base directory: $BASE_DIR"

# Step 1: Generate data
echo -e "\n===== Step 1: Generating synthetic financial data ====="
python3 scripts/generate_data.py --output_dir="$DATA_DIR" --n_samples=$N_SAMPLES

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

# Step 3: Compute attributions
echo -e "\n===== Step 3: Computing attributions ====="
echo "Computing attributions..."

python3 scripts/run_all_attributions.py \
    --base_dir="$BASE_DIR" \
    --n_samples="$EVAL_SAMPLES" \
    --random_state="$SEED"

# Check if attribution computation was successful
if [ $? -ne 0 ]; then
    echo "Error: Attribution computation failed. Exiting."
    exit 1
fi

# Step 4: Evaluate faithfulness of attributions
echo -e "\n===== Step 4: Evaluating causal faithfulness ====="
python3 scripts/evaluate_faithfulness.py --data_dir="$DATA_DIR" --results_dir="$RESULTS_DIR" \
                                       --plots_dir="$PLOTS_DIR"

# Check if evaluation was successful
if [ $? -ne 0 ]; then
    echo "Error: Faithfulness evaluation failed. Exiting."
    exit 1
fi

# Step 5: Perform detailed analysis of attribution combinations
echo -e "\n===== Step 5: Analyzing attribution combinations ====="
python3 scripts/analyze_attribution_combinations.py --results_dir="$RESULTS_DIR" \
                                                  --output_dir="${RESULTS_DIR}/analysis" \
                                                  --plots_dir="${PLOTS_DIR}/analysis"

# Check if analysis was successful
if [ $? -ne 0 ]; then
    echo "Warning: Attribution combination analysis failed. Continuing anyway."
fi

# Step 6: Generate comprehensive report for the paper
echo -e "\n===== Step 6: Generating comprehensive report ====="
python3 scripts/generate_report.py --results_dir="$RESULTS_DIR" \
                                 --output_dir="$REPORT_DIR" \
                                 --format="markdown" \
                                 --include_plots

# Also generate LaTeX version if available
python3 scripts/generate_report.py --results_dir="$RESULTS_DIR" \
                                 --output_dir="$REPORT_DIR" \
                                 --format="latex" \
                                 --include_plots

# Check if report generation was successful
if [ $? -ne 0 ]; then
    echo "Warning: Report generation failed. Results are still available in $RESULTS_DIR."
fi

# Step 7: Create a summary of key findings
echo -e "\n===== Step 7: Creating summary of key findings ====="
if [ -f "${REPORT_DIR}/report.md" ]; then
    echo "Key findings from the report:" > "${REPORT_DIR}/key_findings.md"
    echo "" >> "${REPORT_DIR}/key_findings.md"
    
    # Extract key findings section from the markdown report
    sed -n '/^## Key Findings/,/^## /p' "${REPORT_DIR}/report.md" | sed '$d' >> "${REPORT_DIR}/key_findings.md"
    
    echo "Key findings saved to ${REPORT_DIR}/key_findings.md"
else
    echo "Warning: Markdown report not found. Cannot extract key findings."
fi

echo -e "\n===== Pipeline completed successfully! ====="
echo "Results saved to $RESULTS_DIR"
echo "Plots saved to $PLOTS_DIR"
echo "Report saved to $REPORT_DIR"
echo "Logs saved to $LOG_FILE"
echo "Date: $(date)"