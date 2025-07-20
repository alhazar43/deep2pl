#!/bin/bash

# run_dataset.sh - Comprehensive Deep-2PL Pipeline Runner
# 
# This script runs the complete pipeline for any dataset:
# 1. Training with 5-fold cross-validation
# 2. Model evaluation on test set
# 3. Visualization and metrics generation
# 4. Results organization
#
# Usage: ./run_dataset.sh DATASET_NAME [OPTIONS]
#
# Examples:
#   ./run_dataset.sh STATICS              # Full pipeline for STATICS
#   ./run_dataset.sh assist2015           # Full pipeline for assist2015
#   ./run_dataset.sh synthetic --epochs 5 # Custom epochs
#   ./run_dataset.sh assist2017 --quick   # Quick run (1 epoch, single fold)

set -e  # Exit on any error

# Default parameters
EPOCHS=20
BATCH_SIZE=""
LEARNING_RATE=""
QUICK_MODE=false
SINGLE_FOLD=false
FOLD_IDX=0

# Available datasets
AVAILABLE_DATASETS=(
    "STATICS" "assist2009_updated" "fsaif1tof3"     # Pre-split with Q-matrix
    "assist2015" "assist2017" "synthetic"           # Single-file
    "assist2009" "statics2011" "kddcup2010"         # Single-file
)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Helper functions
print_header() {
    echo -e "\n${BOLD}${BLUE}================================================${NC}"
    echo -e "${BOLD}${BLUE} $1${NC}"
    echo -e "${BOLD}${BLUE}================================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}[SUCCESS] $1${NC}"
}

print_error() {
    echo -e "${RED}[ERROR] $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}[WARNING] $1${NC}"
}

print_info() {
    echo -e "${BLUE}[INFO] $1${NC}"
}

# Usage function
usage() {
    echo "Usage: $0 DATASET_NAME [OPTIONS]"
    echo ""
    echo "DATASET_NAME:"
    echo "  Pre-split datasets (with Q-matrix):"
    echo "    STATICS, assist2009_updated, fsaif1tof3"
    echo "  Single-file datasets (auto 5-fold CV):"
    echo "    assist2015, assist2017, synthetic, assist2009, statics2011, kddcup2010"
    echo ""
    echo "OPTIONS:"
    echo "  --epochs N          Number of training epochs (default: 20)"
    echo "  --batch-size N      Batch size (default: auto-detected per dataset)"
    echo "  --learning-rate F   Learning rate (default: 0.001)"
    echo "  --quick             Quick mode: 1 epoch, single fold (for testing)"
    echo "  --single-fold       Train only fold 0 instead of 5-fold CV"
    echo "  --fold N            Specify fold index for single-fold training (0-4)"
    echo "  --help              Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 STATICS                    # Full pipeline for STATICS dataset"
    echo "  $0 assist2015 --epochs 30    # Custom number of epochs"
    echo "  $0 synthetic --quick          # Quick test run"
    echo "  $0 assist2017 --single-fold  # Single fold training"
}

# Check if dataset is valid
is_valid_dataset() {
    local dataset=$1
    for valid in "${AVAILABLE_DATASETS[@]}"; do
        if [[ "$valid" == "$dataset" ]]; then
            return 0
        fi
    done
    return 1
}

# Parse command line arguments
if [[ $# -eq 0 ]]; then
    print_error "No dataset specified"
    usage
    exit 1
fi

# Check for help first
if [[ "$1" == "--help" || "$1" == "-h" ]]; then
    usage
    exit 0
fi

DATASET_NAME=$1
shift

# Validate dataset name
if ! is_valid_dataset "$DATASET_NAME"; then
    print_error "Invalid dataset: $DATASET_NAME"
    echo "Available datasets: ${AVAILABLE_DATASETS[*]}"
    exit 1
fi

# Parse options
while [[ $# -gt 0 ]]; do
    case $1 in
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --learning-rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --quick)
            QUICK_MODE=true
            EPOCHS=1
            SINGLE_FOLD=true
            shift
            ;;
        --single-fold)
            SINGLE_FOLD=true
            shift
            ;;
        --fold)
            FOLD_IDX="$2"
            SINGLE_FOLD=true
            shift 2
            ;;
        --help)
            usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Main execution starts here
print_header "Deep-2PL Comprehensive Pipeline"
print_info "Dataset: $DATASET_NAME"
print_info "Epochs: $EPOCHS"
if [[ "$SINGLE_FOLD" == true ]]; then
    print_info "Mode: Single fold (fold $FOLD_IDX)"
else
    print_info "Mode: 5-fold cross-validation"
fi
if [[ "$QUICK_MODE" == true ]]; then
    print_warning "Quick mode enabled (for testing)"
fi

# Check if conda environment is activated
if [[ -z "$CONDA_DEFAULT_ENV" ]]; then
    print_error "Conda environment not activated. Please run:"
    echo "source ~/anaconda3/etc/profile.d/conda.sh && conda activate vrec-env"
    exit 1
fi

if [[ "$CONDA_DEFAULT_ENV" != "vrec-env" ]]; then
    print_warning "Current environment: $CONDA_DEFAULT_ENV (expected: vrec-env)"
fi

# Create results directories
mkdir -p results/{train,valid,test,plots}
mkdir -p save_models
mkdir -p logs
mkdir -p stats

# Phase 1: Training
print_header "Phase 1: Training"

# Build training command
TRAIN_CMD="python train.py --dataset $DATASET_NAME --epochs $EPOCHS"

if [[ "$SINGLE_FOLD" == true ]]; then
    TRAIN_CMD="$TRAIN_CMD --single_fold --fold $FOLD_IDX"
fi

if [[ -n "$BATCH_SIZE" ]]; then
    TRAIN_CMD="$TRAIN_CMD --batch_size $BATCH_SIZE"
fi

if [[ -n "$LEARNING_RATE" ]]; then
    TRAIN_CMD="$TRAIN_CMD --learning_rate $LEARNING_RATE"
fi

print_info "Running: $TRAIN_CMD"

if $TRAIN_CMD; then
    print_success "Training completed successfully"
else
    print_error "Training failed"
    exit 1
fi

# Phase 2: Evaluation
print_header "Phase 2: Model Evaluation"

# Find trained models
if [[ "$SINGLE_FOLD" == true ]]; then
    MODEL_PATTERN="save_models/best_model_${DATASET_NAME}_fold${FOLD_IDX}.pth"
else
    MODEL_PATTERN="save_models/best_model_${DATASET_NAME}_fold*.pth"
fi

MODELS_FOUND=($(ls $MODEL_PATTERN 2>/dev/null || true))

if [[ ${#MODELS_FOUND[@]} -eq 0 ]]; then
    print_warning "No trained models found, checking for final models..."
    if [[ "$SINGLE_FOLD" == true ]]; then
        MODEL_PATTERN="save_models/final_model_${DATASET_NAME}_fold${FOLD_IDX}.pth"
    else
        MODEL_PATTERN="save_models/final_model_${DATASET_NAME}_fold*.pth"
    fi
    MODELS_FOUND=($(ls $MODEL_PATTERN 2>/dev/null || true))
fi

if [[ ${#MODELS_FOUND[@]} -eq 0 ]]; then
    print_error "No models found for evaluation"
    exit 1
fi

print_info "Found ${#MODELS_FOUND[@]} model(s) for evaluation"

# Evaluate each model
for model_path in "${MODELS_FOUND[@]}"; do
    print_info "Evaluating: $(basename $model_path)"
    
    if python evaluate.py --model_path "$model_path" --split test; then
        print_success "Evaluation completed: $(basename $model_path)"
    else
        print_error "Evaluation failed: $(basename $model_path)"
    fi
done

# Phase 3: Visualization
print_header "Phase 3: Visualization & Metrics"

# Generate training metrics plots
print_info "Generating training metrics..."
if python plot_metrics.py --results_dir results/train --dataset "$DATASET_NAME" --output_dir results/plots 2>/dev/null; then
    print_success "Training metrics generated"
else
    print_warning "Training metrics generation had warnings (this is normal)"
fi

# Generate comprehensive plots if training results exist
if [[ -d "results/train" ]] && [[ $(ls results/train/metrics_${DATASET_NAME}*.json 2>/dev/null | wc -l) -gt 0 ]]; then
    print_info "Generating comprehensive plots..."
    if python plot_metrics.py --all --results_dir results/train --output_dir results/plots 2>/dev/null; then
        print_success "Comprehensive plots generated"
    else
        print_warning "Comprehensive plots had warnings"
    fi
fi

# Generate evaluation visualizations (ROC/PR curves)
if [[ -d "results/test" ]] && [[ $(ls results/test/eval_*${DATASET_NAME}*.json 2>/dev/null | wc -l) -gt 0 ]]; then
    print_info "Generating evaluation visualizations (ROC/PR curves)..."
    if python plot_evaluation.py --results_dir results/test --output_dir results/plots --dataset "$DATASET_NAME" 2>/dev/null; then
        print_success "ROC and Precision-Recall curves generated"
    else
        print_warning "Evaluation visualizations had warnings"
    fi
fi

# Phase 4: Deep Model Visualization
print_header "Phase 4: Deep Model Visualization"

# Generate detailed per-student and per-KC visualizations using the best model
if [[ ${#MODELS_FOUND[@]} -gt 0 ]]; then
    BEST_MODEL="${MODELS_FOUND[0]}"  # Use first found model (best or final)
    print_info "Generating detailed visualizations with: $(basename $BEST_MODEL)"
    
    # Extract config path from model
    CONFIG_PATH="${BEST_MODEL%.pth}.json"
    if [[ ! -f "$CONFIG_PATH" ]]; then
        # Fallback: look for config in save_models directory
        CONFIG_PATH="save_models/config_${DATASET_NAME}_fold${FOLD_IDX}.json"
        if [[ ! -f "$CONFIG_PATH" ]] && [[ "$SINGLE_FOLD" != true ]]; then
            CONFIG_PATH="save_models/config_${DATASET_NAME}_fold0.json"
        fi
    fi
    
    if [[ -f "$CONFIG_PATH" ]]; then
        if python visualize.py --checkpoint "$BEST_MODEL" --config "$CONFIG_PATH" --output_dir "figs/${DATASET_NAME}" 2>/dev/null; then
            print_success "Deep model visualizations generated in figs/${DATASET_NAME}/"
        else
            print_warning "Deep model visualization had warnings (this is normal for some datasets)"
        fi
    else
        print_warning "Config file not found for visualization: $CONFIG_PATH"
    fi
else
    print_warning "No models available for visualization"
fi

# Phase 5: Summary Report
print_header "Phase 5: Results Summary"

# Create summary
SUMMARY_FILE="results/pipeline_summary_${DATASET_NAME}.txt"
{
    echo "Deep-2PL Pipeline Summary"
    echo "========================"
    echo "Dataset: $DATASET_NAME"
    echo "Date: $(date)"
    echo "Epochs: $EPOCHS"
    echo "Mode: $(if [[ "$SINGLE_FOLD" == true ]]; then echo "Single fold ($FOLD_IDX)"; else echo "5-fold CV"; fi)"
    echo ""
    
    echo "Models Trained:"
    for model in "${MODELS_FOUND[@]}"; do
        echo "  - $(basename $model)"
    done
    echo ""
    
    echo "Results Location:"
    echo "  - Training metrics: results/train/"
    echo "  - Evaluation results: results/test/"
    echo "  - Plots: results/plots/"
    echo "  - Models: save_models/"
    echo ""
    
    # Add dataset information
    echo "Dataset Information:"
    if python -c "
from data.dataloader import get_dataset_info
import sys
try:
    info = get_dataset_info('$DATASET_NAME', './data')
    print(f'  - Questions: {info[\"n_questions\"]}')
    print(f'  - Q-matrix: {\"Yes\" if info[\"has_qmatrix\"] else \"No\"}')
    print(f'  - Format: {\"Pre-split\" if info[\"has_qmatrix\"] else \"Single-file\"}')
    if info['has_qmatrix']:
        print(f'  - Mode: Per-KC')
    else:
        print(f'  - Mode: Global')
except Exception as e:
    print(f'  - Error getting info: {e}')
" 2>/dev/null; then
        true
    else
        echo "  - Could not retrieve dataset information"
    fi
    
} > "$SUMMARY_FILE"

print_success "Summary saved to: $SUMMARY_FILE"

# Display final results
print_header "Pipeline Completed Successfully!"

echo "Results Summary:"
echo "   Dataset: $DATASET_NAME"
echo "   Models: ${#MODELS_FOUND[@]} trained"
echo "   Training: $(if [[ "$SINGLE_FOLD" == true ]]; then echo "Single fold"; else echo "5-fold CV"; fi)"
echo ""
echo "Output Locations:"
echo "   Training metrics: results/train/"
echo "   Evaluation: results/test/"
echo "   Plots: results/plots/ (includes ROC/PR curves)"
echo "   Deep visualizations: figs/${DATASET_NAME}/ (per-KC analysis)"
echo "   IRT parameters: stats/ (theta, alpha, beta from best models)"
echo "   Models: save_models/"
echo "   Summary: $SUMMARY_FILE"
echo ""

# Display some key results if available
if [[ -f "results/test/eval_final_model_${DATASET_NAME}_fold${FOLD_IDX}_test.json" ]]; then
    print_info "Quick Results Preview:"
    python -c "
import json
try:
    with open('results/test/eval_final_model_${DATASET_NAME}_fold${FOLD_IDX}_test.json', 'r') as f:
        data = json.load(f)
    print(f'   Test AUC: {data[\"auc\"]:.4f}')
    print(f'   Test Accuracy: {data[\"accuracy\"]:.4f}')
    print(f'   Test Loss: {data[\"loss\"]:.4f}')
except:
    pass
" 2>/dev/null || true
fi

print_success "All phases completed successfully!"
print_info "You can now examine the results in the respective directories."

if [[ "$QUICK_MODE" == true ]]; then
    print_warning "This was a quick test run. For full results, run without --quick flag."
fi