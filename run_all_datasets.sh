#!/bin/bash

# run_all_datasets.sh - Comprehensive Deep-2PL Pipeline Runner for All Datasets
# 
# This script runs the complete pipeline for ALL datasets:
# 1. Training with 5-fold cross-validation
# 2. Model evaluation on test sets
# 3. Visualization and metrics generation
# 4. Comprehensive results comparison
#
# Usage: ./run_all_datasets.sh [OPTIONS]
#
# Examples:
#   ./run_all_datasets.sh                    # Full pipeline for all datasets (30 epochs)
#   ./run_all_datasets.sh --epochs 10       # Custom epochs for all datasets
#   ./run_all_datasets.sh --quick           # Quick test (1 epoch, single fold)
#   ./run_all_datasets.sh --exclude large   # Exclude large datasets
#   ./run_all_datasets.sh --only small      # Only small/medium datasets

set -e  # Exit on any error

# Default parameters
EPOCHS=30
BATCH_SIZE=""
LEARNING_RATE=""
QUICK_MODE=false
SINGLE_FOLD=false
FOLD_IDX=0
EXCLUDE_DATASETS=""
ONLY_DATASETS=""
PARALLEL_MODE=false
MAX_PARALLEL=2

# Dataset categories
SMALL_DATASETS=("synthetic" "assist2009_updated" "assist2015" "assist2017" "assist2009")
MEDIUM_DATASETS=("kddcup2010" "STATICS")
LARGE_DATASETS=("statics2011" "fsaif1tof3")

ALL_DATASETS=("${SMALL_DATASETS[@]}" "${MEDIUM_DATASETS[@]}" "${LARGE_DATASETS[@]}")

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Helper functions
print_header() {
    echo -e "\n${BOLD}${BLUE}================================================================${NC}"
    echo -e "${BOLD}${BLUE} $1${NC}"
    echo -e "${BOLD}${BLUE}================================================================${NC}\n"
}

print_section() {
    echo -e "\n${BOLD}${CYAN}--- $1 ---${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

print_dataset() {
    echo -e "${PURPLE}📊 $1${NC}"
}

# Usage function
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "This script runs the complete Deep-2PL pipeline for all datasets:"
    echo "  • Training with 5-fold cross-validation"
    echo "  • Model evaluation and testing"
    echo "  • Visualization and metrics generation"
    echo "  • Comprehensive results comparison"
    echo ""
    echo "OPTIONS:"
    echo "  --epochs N          Number of training epochs (default: 30)"
    echo "  --batch-size N      Batch size for all datasets"
    echo "  --learning-rate F   Learning rate for all datasets"
    echo "  --quick             Quick mode: 1 epoch, single fold (for testing)"
    echo "  --single-fold       Train only fold 0 instead of 5-fold CV"
    echo "  --fold N            Specify fold index for single-fold training (0-4)"
    echo ""
    echo "DATASET FILTERING:"
    echo "  --exclude TYPE      Exclude dataset type: 'large', 'medium', 'small'"
    echo "  --only TYPE         Include only dataset type: 'large', 'medium', 'small'"
    echo "  --datasets LIST     Comma-separated list of specific datasets"
    echo ""
    echo "EXECUTION:"
    echo "  --parallel          Run datasets in parallel (experimental)"
    echo "  --max-parallel N    Maximum parallel jobs (default: 2)"
    echo "  --help              Show this help message"
    echo ""
    echo "Dataset Categories:"
    echo "  Small:  ${SMALL_DATASETS[*]}"
    echo "  Medium: ${MEDIUM_DATASETS[*]}"
    echo "  Large:  ${LARGE_DATASETS[*]}"
    echo ""
    echo "Examples:"
    echo "  $0                         # Full pipeline, all datasets, 30 epochs"
    echo "  $0 --epochs 10             # All datasets with 10 epochs"
    echo "  $0 --exclude large         # Skip large datasets (faster)"
    echo "  $0 --only small            # Only small datasets (quick test)"
    echo "  $0 --quick                 # Ultra-quick test run"
    echo "  $0 --datasets synthetic,assist2015  # Specific datasets only"
}

# Function to check if dataset is in array
contains_dataset() {
    local dataset=$1
    shift
    local arr=("$@")
    for item in "${arr[@]}"; do
        [[ "$item" == "$dataset" ]] && return 0
    done
    return 1
}

# Function to estimate training time
estimate_time() {
    local datasets=("$@")
    local total_hours=0
    
    print_info "Estimating training time..."
    
    # Time estimates in hours (30 epochs, 5-fold CV)
    declare -A TIME_ESTIMATES=(
        ["synthetic"]=0.075
        ["assist2009_updated"]=0.1
        ["assist2015"]=0.14
        ["assist2017"]=0.15
        ["assist2009"]=0.19
        ["kddcup2010"]=0.98
        ["STATICS"]=1.43
        ["statics2011"]=1.8
        ["fsaif1tof3"]=2.55
    )
    
    for dataset in "${datasets[@]}"; do
        if [[ -n "${TIME_ESTIMATES[$dataset]}" ]]; then
            local dataset_time=${TIME_ESTIMATES[$dataset]}
            # Scale by epochs
            dataset_time=$(echo "scale=2; $dataset_time * $EPOCHS / 30" | bc -l)
            
            # Scale for single fold
            if [[ "$SINGLE_FOLD" == true ]]; then
                dataset_time=$(echo "scale=2; $dataset_time / 5" | bc -l)
            fi
            
            total_hours=$(echo "scale=2; $total_hours + $dataset_time" | bc -l)
            echo "    $dataset: ${dataset_time}h"
        fi
    done
    
    echo "    Total estimated time: ${total_hours}h"
    
    if (( $(echo "$total_hours > 8" | bc -l) )); then
        print_warning "Long training time estimated. Consider using --exclude large or --only small"
    fi
}

# Function to run single dataset pipeline
run_dataset_pipeline() {
    local dataset=$1
    local dataset_start_time=$(date +%s)
    
    print_section "Processing Dataset: $dataset"
    
    # Build command
    local cmd="./run_dataset.sh $dataset --epochs $EPOCHS"
    
    if [[ "$SINGLE_FOLD" == true ]]; then
        cmd="$cmd --single-fold --fold $FOLD_IDX"
    fi
    
    if [[ "$QUICK_MODE" == true ]]; then
        cmd="$cmd --quick"
    fi
    
    if [[ -n "$BATCH_SIZE" ]]; then
        cmd="$cmd --batch-size $BATCH_SIZE"
    fi
    
    if [[ -n "$LEARNING_RATE" ]]; then
        cmd="$cmd --learning-rate $LEARNING_RATE"
    fi
    
    print_info "Command: $cmd"
    
    # Run the pipeline
    local log_file="logs/run_all_${dataset}_$(date +%Y%m%d_%H%M%S).log"
    
    if $cmd 2>&1 | tee "$log_file"; then
        local dataset_end_time=$(date +%s)
        local dataset_duration=$((dataset_end_time - dataset_start_time))
        print_success "$dataset completed in ${dataset_duration}s"
        echo "$dataset:SUCCESS:${dataset_duration}" >> "$PROGRESS_FILE"
        return 0
    else
        local dataset_end_time=$(date +%s)
        local dataset_duration=$((dataset_end_time - dataset_start_time))
        print_error "$dataset failed after ${dataset_duration}s"
        echo "$dataset:FAILED:${dataset_duration}" >> "$PROGRESS_FILE"
        return 1
    fi
}

# Function to generate comprehensive comparison
generate_comparison() {
    print_section "Generating Comprehensive Comparison"
    
    # Create comprehensive comparison script
    cat > "compare_all_results.py" << 'EOF'
#!/usr/bin/env python3
"""
Generate comprehensive comparison of all dataset results.
"""

import json
import os
import glob
import pandas as pd
import numpy as np
from pathlib import Path

def collect_results():
    """Collect results from all datasets."""
    results = []
    
    # Find all test evaluation files
    test_files = glob.glob("results/test/eval_*_test.json")
    
    for file_path in test_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Extract dataset name from filename
            filename = os.path.basename(file_path)
            parts = filename.split('_')
            dataset = '_'.join(parts[2:-2])  # Remove eval_ prefix and _test.json suffix
            
            results.append({
                'dataset': dataset,
                'auc': data.get('auc', 0),
                'accuracy': data.get('accuracy', 0),
                'loss': data.get('loss', float('inf')),
                'precision': data.get('precision', 0),
                'recall': data.get('recall', 0),
                'f1': data.get('f1', 0),
                'file': file_path
            })
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    
    return results

def main():
    results = collect_results()
    
    if not results:
        print("No results found!")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Group by dataset and aggregate
    summary = df.groupby('dataset').agg({
        'auc': ['mean', 'std', 'count'],
        'accuracy': ['mean', 'std'],
        'loss': 'mean',
        'f1': 'mean'
    }).round(4)
    
    print("\n" + "="*80)
    print("COMPREHENSIVE RESULTS COMPARISON")
    print("="*80)
    
    print(f"\nResults Summary ({len(df)} evaluations across {len(df['dataset'].unique())} datasets):")
    print("-" * 80)
    print(f"{'Dataset':<18} {'AUC (mean±std)':<15} {'Accuracy':<10} {'F1':<8} {'Loss':<8} {'Folds':<6}")
    print("-" * 80)
    
    for dataset in summary.index:
        auc_mean = summary.loc[dataset, ('auc', 'mean')]
        auc_std = summary.loc[dataset, ('auc', 'std')]
        acc_mean = summary.loc[dataset, ('accuracy', 'mean')]
        f1_mean = summary.loc[dataset, ('f1', 'mean')]
        loss_mean = summary.loc[dataset, ('loss', 'mean')]
        count = int(summary.loc[dataset, ('auc', 'count')])
        
        auc_str = f"{auc_mean:.4f}±{auc_std:.4f}"
        print(f"{dataset:<18} {auc_str:<15} {acc_mean:.4f}    {f1_mean:.4f}   {loss_mean:.4f}   {count}")
    
    # Best and worst performers
    best_auc = df.loc[df['auc'].idxmax()]
    worst_auc = df.loc[df['auc'].idxmin()]
    
    print(f"\n🏆 Best AUC: {best_auc['dataset']} ({best_auc['auc']:.4f})")
    print(f"📉 Worst AUC: {worst_auc['dataset']} ({worst_auc['auc']:.4f})")
    
    # Save detailed results
    df.to_csv('results/comprehensive_comparison.csv', index=False)
    summary.to_csv('results/dataset_summary.csv')
    
    print(f"\nDetailed results saved to:")
    print(f"  - results/comprehensive_comparison.csv")
    print(f"  - results/dataset_summary.csv")

if __name__ == "__main__":
    main()
EOF
    
    if python compare_all_results.py; then
        print_success "Comprehensive comparison generated"
    else
        print_warning "Comparison generation had issues"
    fi
    
    # Clean up
    rm -f compare_all_results.py
}

# Parse command line arguments
DATASETS_TO_RUN=()

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
        --exclude)
            case $2 in
                large)
                    EXCLUDE_DATASETS="${LARGE_DATASETS[*]}"
                    ;;
                medium)
                    EXCLUDE_DATASETS="${MEDIUM_DATASETS[*]}"
                    ;;
                small)
                    EXCLUDE_DATASETS="${SMALL_DATASETS[*]}"
                    ;;
                *)
                    print_error "Unknown exclude type: $2"
                    usage
                    exit 1
                    ;;
            esac
            shift 2
            ;;
        --only)
            case $2 in
                large)
                    ONLY_DATASETS="${LARGE_DATASETS[*]}"
                    ;;
                medium)
                    ONLY_DATASETS="${MEDIUM_DATASETS[*]}"
                    ;;
                small)
                    ONLY_DATASETS="${SMALL_DATASETS[*]}"
                    ;;
                *)
                    print_error "Unknown only type: $2"
                    usage
                    exit 1
                    ;;
            esac
            shift 2
            ;;
        --datasets)
            IFS=',' read -ra DATASETS_TO_RUN <<< "$2"
            shift 2
            ;;
        --parallel)
            PARALLEL_MODE=true
            shift
            ;;
        --max-parallel)
            MAX_PARALLEL="$2"
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

# Determine final dataset list
if [[ ${#DATASETS_TO_RUN[@]} -gt 0 ]]; then
    FINAL_DATASETS=("${DATASETS_TO_RUN[@]}")
elif [[ -n "$ONLY_DATASETS" ]]; then
    read -ra FINAL_DATASETS <<< "$ONLY_DATASETS"
else
    FINAL_DATASETS=("${ALL_DATASETS[@]}")
    
    # Apply exclusions
    if [[ -n "$EXCLUDE_DATASETS" ]]; then
        read -ra EXCLUDE_LIST <<< "$EXCLUDE_DATASETS"
        TEMP_DATASETS=()
        for dataset in "${FINAL_DATASETS[@]}"; do
            if ! contains_dataset "$dataset" "${EXCLUDE_LIST[@]}"; then
                TEMP_DATASETS+=("$dataset")
            fi
        done
        FINAL_DATASETS=("${TEMP_DATASETS[@]}")
    fi
fi

# Main execution starts here
print_header "Deep-2PL Comprehensive Pipeline for All Datasets"

print_info "Configuration:"
echo "  📊 Datasets: ${#FINAL_DATASETS[@]} (${FINAL_DATASETS[*]})"
echo "  🔄 Epochs: $EPOCHS"
echo "  📈 Mode: $(if [[ "$SINGLE_FOLD" == true ]]; then echo "Single fold ($FOLD_IDX)"; else echo "5-fold CV"; fi)"
if [[ "$QUICK_MODE" == true ]]; then
    echo "  ⚡ Quick mode enabled"
fi
if [[ "$PARALLEL_MODE" == true ]]; then
    echo "  🔀 Parallel execution (max: $MAX_PARALLEL jobs)"
fi

# Check environment
if [[ -z "$CONDA_DEFAULT_ENV" ]]; then
    print_error "Conda environment not activated. Please run:"
    echo "source ~/anaconda3/etc/profile.d/conda.sh && conda activate vrec-env"
    exit 1
fi

if [[ "$CONDA_DEFAULT_ENV" != "vrec-env" ]]; then
    print_warning "Current environment: $CONDA_DEFAULT_ENV (expected: vrec-env)"
fi

# Check dependencies
if ! command -v bc &> /dev/null; then
    print_error "bc calculator not found. Please install: sudo apt-get install bc"
    exit 1
fi

# Create directories
mkdir -p results/{train,valid,test,plots}
mkdir -p save_models
mkdir -p logs

# Estimate training time
estimate_time "${FINAL_DATASETS[@]}"

# Confirm execution for long runs
if [[ "$QUICK_MODE" != true ]] && [[ ${#FINAL_DATASETS[@]} -gt 3 ]]; then
    echo ""
    read -p "Continue with training ${#FINAL_DATASETS[@]} datasets? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_info "Training cancelled by user"
        exit 0
    fi
fi

# Initialize progress tracking
PROGRESS_FILE="logs/run_all_progress_$(date +%Y%m%d_%H%M%S).txt"
START_TIME=$(date +%s)

print_header "Starting Pipeline Execution"

# Track results
SUCCESSFUL_DATASETS=()
FAILED_DATASETS=()

# Process each dataset
for i in "${!FINAL_DATASETS[@]}"; do
    dataset="${FINAL_DATASETS[$i]}"
    current=$((i + 1))
    total=${#FINAL_DATASETS[@]}
    
    print_header "Dataset $current/$total: $dataset"
    
    if run_dataset_pipeline "$dataset"; then
        SUCCESSFUL_DATASETS+=("$dataset")
    else
        FAILED_DATASETS+=("$dataset")
        
        if [[ "$QUICK_MODE" != true ]]; then
            echo ""
            read -p "Continue with remaining datasets? [Y/n] " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Nn]$ ]]; then
                print_info "Stopping execution as requested"
                break
            fi
        fi
    fi
done

# Generate comprehensive comparison
if [[ ${#SUCCESSFUL_DATASETS[@]} -gt 1 ]]; then
    generate_comparison
fi

# Final summary
END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))

print_header "Pipeline Execution Complete!"

echo "📊 Execution Summary:"
echo "   Total time: $((TOTAL_DURATION / 3600))h $((TOTAL_DURATION % 3600 / 60))m $((TOTAL_DURATION % 60))s"
echo "   Successful: ${#SUCCESSFUL_DATASETS[@]}/${#FINAL_DATASETS[@]} datasets"
echo "   Failed: ${#FAILED_DATASETS[@]}/${#FINAL_DATASETS[@]} datasets"
echo ""

if [[ ${#SUCCESSFUL_DATASETS[@]} -gt 0 ]]; then
    echo "✓ Successful datasets:"
    for dataset in "${SUCCESSFUL_DATASETS[@]}"; do
        echo "   - $dataset"
    done
    echo ""
fi

if [[ ${#FAILED_DATASETS[@]} -gt 0 ]]; then
    echo "✗ Failed datasets:"
    for dataset in "${FAILED_DATASETS[@]}"; do
        echo "   - $dataset"
    done
    echo ""
fi

echo "📁 Results Locations:"
echo "   📈 Training metrics: results/train/"
echo "   📊 Evaluations: results/test/"
echo "   📉 Plots: results/plots/"
echo "   🤖 Models: save_models/"
echo "   📄 Progress log: $PROGRESS_FILE"
if [[ ${#SUCCESSFUL_DATASETS[@]} -gt 1 ]]; then
    echo "   📋 Comparison: results/comprehensive_comparison.csv"
fi

print_success "All pipeline phases completed!"

if [[ "$QUICK_MODE" == true ]]; then
    print_warning "This was a quick test run. For full results, run without --quick flag."
fi

if [[ ${#SUCCESSFUL_DATASETS[@]} -eq ${#FINAL_DATASETS[@]} ]]; then
    print_success "🎉 All datasets completed successfully!"
else
    print_warning "Some datasets failed. Check logs for details."
fi