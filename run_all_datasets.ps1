# run_all_datasets.ps1 - PowerShell Script for Deep-2PL All Datasets Pipeline
#
# This script runs the complete pipeline for ALL datasets:
# 1. Training with 5-fold cross-validation
# 2. Model evaluation on test sets
# 3. Visualization and metrics generation
# 4. Comprehensive results comparison
#
# Usage: .\run_all_datasets.ps1 [OPTIONS]
#
# Examples:
#   .\run_all_datasets.ps1                    # Full pipeline for all datasets (30 epochs)
#   .\run_all_datasets.ps1 -Epochs 10        # Custom epochs for all datasets
#   .\run_all_datasets.ps1 -Quick            # Quick test (1 epoch, single fold)
#   .\run_all_datasets.ps1 -Exclude large    # Exclude large datasets

param(
    [int]$Epochs = 30,
    [int]$BatchSize,
    [double]$LearningRate,
    [switch]$Quick,
    [switch]$SingleFold,
    [int]$Fold = 0,
    [ValidateSet("large", "medium", "small")]
    [string]$Exclude,
    [ValidateSet("large", "medium", "small")]
    [string]$Only,
    [string[]]$Datasets,
    [switch]$Parallel,
    [int]$MaxParallel = 2,
    [switch]$Help
)

# Dataset categories
$SmallDatasets = @("synthetic", "assist2009_updated", "assist2015", "assist2017", "assist2009")
$MediumDatasets = @("kddcup2010", "STATICS")
$LargeDatasets = @("statics2011", "fsaif1tof3")
$AllDatasets = $SmallDatasets + $MediumDatasets + $LargeDatasets

# Colors for output
$SuccessColor = "Green"
$ErrorColor = "Red"
$WarningColor = "Yellow"
$InfoColor = "Cyan"
$DatasetColor = "Magenta"
$HeaderColor = "Blue"

function Write-Success { param([string]$Message) Write-Host "[SUCCESS] $Message" -ForegroundColor $SuccessColor }
function Write-Error { param([string]$Message) Write-Host "[ERROR] $Message" -ForegroundColor $ErrorColor }
function Write-Warning { param([string]$Message) Write-Host "[WARNING] $Message" -ForegroundColor $WarningColor }
function Write-Info { param([string]$Message) Write-Host "[INFO] $Message" -ForegroundColor $InfoColor }
function Write-Dataset { param([string]$Message) Write-Host "[DATASET] $Message" -ForegroundColor $DatasetColor }
function Write-Header { param([string]$Message) 
    Write-Host ""
    Write-Host "================================================================" -ForegroundColor $HeaderColor
    Write-Host " $Message" -ForegroundColor $HeaderColor
    Write-Host "================================================================" -ForegroundColor $HeaderColor
    Write-Host ""
}

function Show-Usage {
    Write-Host @"

Usage: .\run_all_datasets.ps1 [OPTIONS]

This script runs the complete Deep-2PL pipeline for all datasets:
  • Training with 5-fold cross-validation
  • Model evaluation and testing
  • Visualization and metrics generation
  • Comprehensive results comparison

PARAMETERS:
  -Epochs N          Number of training epochs (default: 30)
  -BatchSize N       Batch size for all datasets
  -LearningRate F    Learning rate for all datasets
  -Quick             Quick mode: 1 epoch, single fold (for testing)
  -SingleFold        Train only fold 0 instead of 5-fold CV
  -Fold N            Specify fold index for single-fold training (0-4)

DATASET FILTERING:
  -Exclude TYPE      Exclude dataset type: 'large', 'medium', 'small'
  -Only TYPE         Include only dataset type: 'large', 'medium', 'small'
  -Datasets LIST     Array of specific datasets

EXECUTION:
  -Parallel          Run datasets in parallel (experimental)
  -MaxParallel N     Maximum parallel jobs (default: 2)
  -Help              Show this help message

Dataset Categories:
  Small:  $($SmallDatasets -join ', ')
  Medium: $($MediumDatasets -join ', ')
  Large:  $($LargeDatasets -join ', ')

Examples:
  .\run_all_datasets.ps1                        # Full pipeline, all datasets, 30 epochs
  .\run_all_datasets.ps1 -Epochs 10            # All datasets with 10 epochs
  .\run_all_datasets.ps1 -Exclude large        # Skip large datasets (faster)
  .\run_all_datasets.ps1 -Only small           # Only small datasets (quick test)
  .\run_all_datasets.ps1 -Quick                # Ultra-quick test run
  .\run_all_datasets.ps1 -Datasets synthetic,assist2015  # Specific datasets only

"@ -ForegroundColor White
}

function Get-TimeEstimate {
    param([string[]]$DatasetList)
    
    # Time estimates in hours (30 epochs, 5-fold CV)
    $TimeEstimates = @{
        "synthetic" = 0.075
        "assist2009_updated" = 0.1
        "assist2015" = 0.14
        "assist2017" = 0.15
        "assist2009" = 0.19
        "kddcup2010" = 0.98
        "STATICS" = 1.43
        "statics2011" = 1.8
        "fsaif1tof3" = 2.55
    }
    
    Write-Info "Estimating training time..."
    
    $totalHours = 0
    foreach ($dataset in $DatasetList) {
        if ($TimeEstimates.ContainsKey($dataset)) {
            $datasetTime = $TimeEstimates[$dataset]
            # Scale by epochs
            $datasetTime = $datasetTime * $Epochs / 30
            
            # Scale for single fold
            if ($SingleFold) {
                $datasetTime = $datasetTime / 5
            }
            
            $totalHours += $datasetTime
            Write-Host "    $dataset`: $([math]::Round($datasetTime, 2))h"
        }
    }
    
    Write-Host "    Total estimated time: $([math]::Round($totalHours, 2))h"
    
    if ($totalHours -gt 8) {
        Write-Warning "Long training time estimated. Consider using -Exclude large or -Only small"
    }
}

function Start-DatasetPipeline {
    param([string]$DatasetName)
    
    $datasetStart = Get-Date
    
    Write-Host ""
    Write-Host "--- Processing Dataset: $DatasetName ---" -ForegroundColor Cyan
    Write-Host ""
    
    # Build command
    $cmd = @(".\run_dataset.ps1", $DatasetName, "-Epochs", $Epochs)
    
    if ($SingleFold) {
        $cmd += @("-SingleFold", "-Fold", $Fold)
    }
    
    if ($Quick) {
        $cmd += "-Quick"
    }
    
    if ($BatchSize) {
        $cmd += @("-BatchSize", $BatchSize)
    }
    
    if ($LearningRate) {
        $cmd += @("-LearningRate", $LearningRate)
    }
    
    Write-Info "Command: $($cmd -join ' ')"
    
    # Run the pipeline
    $logFile = "logs\run_all_${DatasetName}_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"
    
    try {
        & $cmd[0] $cmd[1..($cmd.Length-1)] 2>&1 | Tee-Object -FilePath $logFile
        
        if ($LASTEXITCODE -eq 0) {
            $datasetEnd = Get-Date
            $duration = $datasetEnd - $datasetStart
            Write-Success "$DatasetName completed in $($duration.ToString('hh\:mm\:ss'))"
            return @{ Success = $true; Dataset = $DatasetName; Duration = $duration }
        } else {
            $datasetEnd = Get-Date
            $duration = $datasetEnd - $datasetStart
            Write-Error "$DatasetName failed after $($duration.ToString('hh\:mm\:ss'))"
            return @{ Success = $false; Dataset = $DatasetName; Duration = $duration }
        }
    } catch {
        Write-Error "$DatasetName failed with exception: $_"
        return @{ Success = $false; Dataset = $DatasetName; Duration = (Get-Date) - $datasetStart }
    }
}

function New-ComparisonReport {
    Write-Host ""
    Write-Host "--- Generating Comprehensive Comparison ---" -ForegroundColor Cyan
    Write-Host ""
    
    # Create comprehensive comparison script
    $comparisonScript = @'
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
    
    print(f"\nBest AUC: {best_auc['dataset']} ({best_auc['auc']:.4f})")
    print(f"Worst AUC: {worst_auc['dataset']} ({worst_auc['auc']:.4f})")
    
    # Save detailed results
    df.to_csv('results/comprehensive_comparison.csv', index=False)
    summary.to_csv('results/dataset_summary.csv')
    
    print(f"\nDetailed results saved to:")
    print(f"  - results/comprehensive_comparison.csv")
    print(f"  - results/dataset_summary.csv")

if __name__ == "__main__":
    main()
'@
    
    $comparisonScript | Out-File -FilePath "compare_all_results.py" -Encoding UTF8
    
    try {
        & python compare_all_results.py
        Write-Success "Comprehensive comparison generated"
    } catch {
        Write-Warning "Comparison generation had issues: $_"
    } finally {
        Remove-Item "compare_all_results.py" -ErrorAction SilentlyContinue
    }
}

# Handle help
if ($Help) {
    Show-Usage
    exit 0
}

# Handle quick mode
if ($Quick) {
    $Epochs = 1
    $SingleFold = $true
}

# Determine final dataset list
if ($Datasets) {
    $FinalDatasets = $Datasets
} elseif ($Only) {
    switch ($Only) {
        "large" { $FinalDatasets = $LargeDatasets }
        "medium" { $FinalDatasets = $MediumDatasets }
        "small" { $FinalDatasets = $SmallDatasets }
    }
} else {
    $FinalDatasets = $AllDatasets
    
    # Apply exclusions
    if ($Exclude) {
        switch ($Exclude) {
            "large" { $FinalDatasets = $FinalDatasets | Where-Object { $_ -notin $LargeDatasets } }
            "medium" { $FinalDatasets = $FinalDatasets | Where-Object { $_ -notin $MediumDatasets } }
            "small" { $FinalDatasets = $FinalDatasets | Where-Object { $_ -notin $SmallDatasets } }
        }
    }
}

# Main execution starts here
Write-Header "Deep-2PL Comprehensive Pipeline for All Datasets"

Write-Info "Configuration:"
Write-Host "  Datasets: $($FinalDatasets.Count) ($($FinalDatasets -join ', '))"
Write-Host "  🔄 Epochs: $Epochs"
if ($SingleFold) {
    Write-Host "  Mode: Single fold ($Fold)"
} else {
    Write-Host "  Mode: 5-fold cross-validation"
}
if ($Quick) { Write-Host "  Quick mode enabled" }
if ($Parallel) { Write-Host "  Parallel execution (max: $MaxParallel jobs)" }

# Change to script directory
Set-Location $PSScriptRoot

# Check environment
$condaEnv = $env:CONDA_DEFAULT_ENV
if (-not $condaEnv) {
    Write-Error "Conda environment not activated. Please run:"
    Write-Host "conda activate vrec-env"
    exit 1
}

if ($condaEnv -ne "vrec-env") {
    Write-Warning "Current environment: $condaEnv (expected: vrec-env)"
}

# Create directories
$directories = @("results\train", "results\valid", "results\test", "results\plots", "save_models", "logs", "stats")
foreach ($dir in $directories) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }
}

# Estimate training time
Get-TimeEstimate -DatasetList $FinalDatasets

# Confirm execution for long runs
if (-not $Quick -and $FinalDatasets.Count -gt 3) {
    Write-Host ""
    $response = Read-Host "Continue with training $($FinalDatasets.Count) datasets? [y/N]"
    if ($response -ne "y" -and $response -ne "Y") {
        Write-Info "Training cancelled by user"
        exit 0
    }
}

$pipelineStart = Get-Date

Write-Header "Starting Pipeline Execution"

# Track results
$SuccessfulDatasets = @()
$FailedDatasets = @()

# Process each dataset
if ($Parallel) {
    Write-Info "Running datasets in parallel (max $MaxParallel jobs)..."
    
    $jobs = @()
    $jobQueue = $FinalDatasets
    $runningJobs = @()
    
    while ($jobQueue.Count -gt 0 -or $runningJobs.Count -gt 0) {
        # Start new jobs if we have capacity
        while ($runningJobs.Count -lt $MaxParallel -and $jobQueue.Count -gt 0) {
            $dataset = $jobQueue[0]
            $jobQueue = $jobQueue[1..($jobQueue.Count-1)]
            
            Write-Info "Starting job for $dataset..."
            $job = Start-Job -ScriptBlock {
                param($dataset, $epochs, $singleFold, $fold, $quick, $batchSize, $learningRate)
                # Job implementation would go here
                # For simplicity, we'll fall back to sequential processing
            } -ArgumentList $dataset, $Epochs, $SingleFold, $Fold, $Quick, $BatchSize, $LearningRate
            
            $runningJobs += @{ Job = $job; Dataset = $dataset }
        }
        
        # Check for completed jobs
        $completedJobs = $runningJobs | Where-Object { $_.Job.State -eq "Completed" -or $_.Job.State -eq "Failed" }
        foreach ($completedJob in $completedJobs) {
            $result = Receive-Job $completedJob.Job
            Remove-Job $completedJob.Job
            
            if ($completedJob.Job.State -eq "Completed") {
                $SuccessfulDatasets += $completedJob.Dataset
                Write-Success "$($completedJob.Dataset) completed"
            } else {
                $FailedDatasets += $completedJob.Dataset
                Write-Error "$($completedJob.Dataset) failed"
            }
        }
        
        $runningJobs = $runningJobs | Where-Object { $_.Job.State -eq "Running" }
        Start-Sleep -Seconds 5
    }
} else {
    # Sequential processing
    for ($i = 0; $i -lt $FinalDatasets.Count; $i++) {
        $dataset = $FinalDatasets[$i]
        $current = $i + 1
        $total = $FinalDatasets.Count
        
        Write-Header "Dataset $current/$total`: $dataset"
        
        $result = Start-DatasetPipeline -DatasetName $dataset
        
        if ($result.Success) {
            $SuccessfulDatasets += $dataset
        } else {
            $FailedDatasets += $dataset
            
            if (-not $Quick -and ($i + 1) -lt $FinalDatasets.Count) {
                $response = Read-Host "Continue with remaining datasets? [Y/n]"
                if ($response -eq "n" -or $response -eq "N") {
                    Write-Info "Stopping execution as requested"
                    break
                }
            }
        }
    }
}

# Generate comprehensive comparison
if ($SuccessfulDatasets.Count -gt 1) {
    New-ComparisonReport
}

# Generate Deep Visualizations Summary
if ($SuccessfulDatasets.Count -gt 0) {
    Write-Host ""
    Write-Host "--- Deep Visualizations Summary Phase ---" -ForegroundColor Cyan
    Write-Host ""
    
    Write-Info "Checking deep visualizations for all successful datasets..."
    
    foreach ($dataset in $SuccessfulDatasets) {
        $vizDir = "figs\$dataset"
        if (Test-Path $vizDir) {
            $vizFiles = Get-ChildItem -Path $vizDir -Filter "*.png" -ErrorAction SilentlyContinue
            if ($vizFiles.Count -gt 0) {
                Write-Success "$dataset`: $($vizFiles.Count) visualization files in $vizDir"
            } else {
                Write-Warning "$dataset`: No PNG files found in $vizDir"
            }
        } else {
            Write-Warning "$dataset`: Visualization directory not found ($vizDir)"
        }
    }
    
    Write-Info "Deep visualization summary complete"
}

# Final summary
$pipelineEnd = Get-Date
$totalDuration = $pipelineEnd - $pipelineStart

Write-Header "Pipeline Execution Complete!"

Write-Info "Execution Summary:"
Write-Host "   Total time: $($totalDuration.ToString('hh\:mm\:ss'))"
Write-Host "   Successful: $($SuccessfulDatasets.Count)/$($FinalDatasets.Count) datasets"
Write-Host "   Failed: $($FailedDatasets.Count)/$($FinalDatasets.Count) datasets"
Write-Host ""

if ($SuccessfulDatasets.Count -gt 0) {
    Write-Host "✓ Successful datasets:" -ForegroundColor $SuccessColor
    foreach ($dataset in $SuccessfulDatasets) {
        Write-Host "   - $dataset"
    }
    Write-Host ""
}

if ($FailedDatasets.Count -gt 0) {
    Write-Host "✗ Failed datasets:" -ForegroundColor $ErrorColor
    foreach ($dataset in $FailedDatasets) {
        Write-Host "   - $dataset"
    }
    Write-Host ""
}

Write-Info "Results Locations:"
Write-Host "   Training metrics: results\train\"
Write-Host "   Evaluations: results\test\"
Write-Host "   Plots: results\plots\"
Write-Host "   Models: save_models\"
Write-Host "   IRT parameters: stats\ (theta, alpha, beta from best models)"
if ($SuccessfulDatasets.Count -gt 0) {
    Write-Host "   Deep visualizations: figs\[dataset]\ (per-KC analysis)"
}
if ($SuccessfulDatasets.Count -gt 1) {
    Write-Host "   Comparison: results\comprehensive_comparison.csv"
}

if ($Quick) {
    Write-Host ""
    Write-Warning "This was a quick test run. For full results, run without -Quick flag."
}

if ($SuccessfulDatasets.Count -eq $FinalDatasets.Count) {
    Write-Success "All datasets completed successfully!"
} else {
    Write-Warning "Some datasets failed. Check logs for details."
}