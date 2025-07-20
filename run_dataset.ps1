# run_dataset.ps1 - PowerShell Script for Deep-2PL Single Dataset Pipeline
#
# This script runs the complete pipeline for a single dataset:
# 1. Training with 5-fold cross-validation (or single fold)
# 2. Model evaluation on test sets
# 3. Visualization and metrics generation
#
# Usage: .\run_dataset.ps1 DATASET_NAME [OPTIONS]
#
# Examples:
#   .\run_dataset.ps1 synthetic
#   .\run_dataset.ps1 STATICS -epochs 20
#   .\run_dataset.ps1 assist2015 -quick

param(
    [Parameter(Mandatory=$true, Position=0)]
    [string]$DatasetName,
    
    [int]$Epochs = 30,
    [int]$BatchSize,
    [double]$LearningRate,
    [switch]$Quick,
    [switch]$SingleFold,
    [int]$Fold = 0,
    [switch]$SkipTraining,
    [switch]$SkipEvaluation,
    [switch]$SkipVisualization,
    [switch]$Help
)

# Colors for output
$SuccessColor = "Green"
$ErrorColor = "Red"
$WarningColor = "Yellow"
$InfoColor = "Cyan"
$DatasetColor = "Magenta"

function Write-Success { param([string]$Message) Write-Host "[SUCCESS] $Message" -ForegroundColor $SuccessColor }
function Write-Error { param([string]$Message) Write-Host "[ERROR] $Message" -ForegroundColor $ErrorColor }
function Write-Warning { param([string]$Message) Write-Host "[WARNING] $Message" -ForegroundColor $WarningColor }
function Write-Info { param([string]$Message) Write-Host "[INFO] $Message" -ForegroundColor $InfoColor }
function Write-Dataset { param([string]$Message) Write-Host "[DATASET] $Message" -ForegroundColor $DatasetColor }

function Show-Usage {
    Write-Host @"

Usage: .\run_dataset.ps1 DATASET_NAME [OPTIONS]

This script runs the complete Deep-2PL pipeline for a single dataset:
  • Training with 5-fold cross-validation
  • Model evaluation and testing
  • Visualization and metrics generation

Available datasets:
  Small:  synthetic, assist2009_updated, assist2015, assist2017, assist2009
  Medium: kddcup2010, STATICS
  Large:  statics2011, fsaif1tof3

PARAMETERS:
  -DatasetName        Dataset to process (required)
  -Epochs N          Number of training epochs (default: 30)
  -BatchSize N       Batch size for training
  -LearningRate F    Learning rate for training
  -Quick             Quick mode: 1 epoch, single fold (for testing)
  -SingleFold        Train only fold 0 instead of 5-fold CV
  -Fold N            Specify fold index for single-fold training (0-4)
  -SkipTraining      Skip training phase
  -SkipEvaluation    Skip evaluation phase
  -SkipVisualization Skip visualization phase
  -Help              Show this help message

Examples:
  .\run_dataset.ps1 synthetic                    # Quick test dataset
  .\run_dataset.ps1 STATICS -Epochs 20         # Large dataset with custom epochs
  .\run_dataset.ps1 assist2015 -Quick          # Ultra-quick test run
  .\run_dataset.ps1 synthetic -SingleFold      # Single fold training only

"@ -ForegroundColor White
}

if ($Help) {
    Show-Usage
    exit 0
}

# Handle quick mode
if ($Quick) {
    $Epochs = 1
    $SingleFold = $true
}

# Header
Write-Host ""
Write-Host "================================================================" -ForegroundColor Blue
Write-Host " Deep-2PL Pipeline for $DatasetName" -ForegroundColor Blue
Write-Host "================================================================" -ForegroundColor Blue
Write-Host ""

Write-Info "Configuration:"
Write-Host "  Dataset: $DatasetName"
Write-Host "  Epochs: $Epochs"
if ($SingleFold) {
    Write-Host "  Mode: Single fold ($Fold)"
} else {
    Write-Host "  Mode: 5-fold cross-validation"
}
if ($Quick) { Write-Host "  Quick mode enabled" }
if ($BatchSize) { Write-Host "  Batch size: $BatchSize" }
if ($LearningRate) { Write-Host "  Learning rate: $LearningRate" }
Write-Host ""

# Change to script directory
Set-Location $PSScriptRoot

# Check if conda is available
try {
    $condaPath = Get-Command conda -ErrorAction Stop
    Write-Info "Conda found: $($condaPath.Source)"
} catch {
    Write-Error "Conda not found. Please install Anaconda/Miniconda and add to PATH."
    exit 1
}

# Check environment
Write-Info "Checking environment..."
$condaEnv = $env:CONDA_DEFAULT_ENV
if (-not $condaEnv) {
    Write-Warning "No conda environment detected. Attempting to activate vrec-env..."
    try {
        & conda activate vrec-env
        if ($LASTEXITCODE -ne 0) { throw "Failed to activate" }
    } catch {
        Write-Error "Failed to activate vrec-env environment."
        Write-Host "Please run: conda activate vrec-env"
        exit 1
    }
} elseif ($condaEnv -ne "vrec-env") {
    Write-Warning "Current environment: $condaEnv (expected: vrec-env)"
}

# Create directories
$directories = @("results\train", "results\valid", "results\test", "results\plots", "save_models", "logs", "stats")
foreach ($dir in $directories) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }
}

$pipelineStart = Get-Date

# Training Phase
if (-not $SkipTraining) {
    Write-Host ""
    Write-Host "--- Training Phase ---" -ForegroundColor Cyan
    Write-Host ""
    
    $cmd = @("python", "train.py", "--dataset", $DatasetName, "--epochs", $Epochs)
    
    if ($SingleFold) {
        $cmd += @("--single_fold", "--fold", $Fold)
    }
    
    if ($BatchSize) { $cmd += @("--batch_size", $BatchSize) }
    if ($LearningRate) { $cmd += @("--learning_rate", $LearningRate) }
    
    Write-Info "Running: $($cmd -join ' ')"
    
    & $cmd[0] $cmd[1..($cmd.Length-1)]
    
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Training failed!"
        exit 1
    }
    
    Write-Success "Training completed"
} else {
    Write-Info "Skipping training phase"
}

# Evaluation Phase
if (-not $SkipEvaluation) {
    Write-Host ""
    Write-Host "--- Evaluation Phase ---" -ForegroundColor Cyan
    Write-Host ""
    
    if ($SingleFold) {
        $modelFile = "save_models\best_model_${DatasetName}_fold${Fold}.pth"
        if (Test-Path $modelFile) {
            Write-Info "Evaluating model: $modelFile"
            & python evaluate.py --model_path $modelFile --split test
            if ($LASTEXITCODE -ne 0) {
                Write-Warning "Evaluation failed for $modelFile"
            } else {
                Write-Success "Evaluation completed for fold $Fold"
            }
        } else {
            Write-Warning "Model file not found: $modelFile"
        }
    } else {
        Write-Info "Evaluating all 5 folds..."
        for ($i = 0; $i -lt 5; $i++) {
            $modelFile = "save_models\best_model_${DatasetName}_fold${i}.pth"
            if (Test-Path $modelFile) {
                Write-Info "Evaluating fold $i..."
                & python evaluate.py --model_path $modelFile --split test
                if ($LASTEXITCODE -ne 0) {
                    Write-Warning "Evaluation failed for fold $i"
                } else {
                    Write-Success "Evaluation completed for fold $i"
                }
            } else {
                Write-Warning "Model file not found for fold ${i}: $modelFile"
            }
        }
    }
} else {
    Write-Info "Skipping evaluation phase"
}

# Visualization Phase
if (-not $SkipVisualization) {
    Write-Host ""
    Write-Host "--- Visualization Phase ---" -ForegroundColor Cyan
    Write-Host ""
    
    # Generate training metrics plots
    Write-Info "Generating training metrics plots..."
    & python plot_metrics.py --dataset $DatasetName --results_dir "results\train" --output_dir "results\plots"
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "Training metrics plotting had issues"
    } else {
        Write-Success "Training metrics plots generated"
    }
    
    # Generate evaluation plots
    Write-Info "Generating evaluation plots..."
    & python plot_evaluation.py --dataset $DatasetName --results_dir "results\test" --output_dir "results\plots"
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "Evaluation plotting had issues"
    } else {
        Write-Success "Evaluation plots generated"
    }
    
    # Run main visualization (theta/beta plots)
    $vizModelFile = "save_models\best_model_${DatasetName}_fold0.pth"
    if (Test-Path $vizModelFile) {
        Write-Info "Generating visualization plots..."
        & python visualize.py --model_path $vizModelFile --dataset $DatasetName
        if ($LASTEXITCODE -ne 0) {
            Write-Warning "Visualization had issues"
        } else {
            Write-Success "Visualization plots generated"
        }
    } else {
        Write-Warning "No model found for visualization"
    }
} else {
    Write-Info "Skipping visualization phase"
}

# Deep Model Visualization Phase
if (-not $SkipVisualization) {
    Write-Host ""
    Write-Host "--- Deep Model Visualization Phase ---" -ForegroundColor Cyan
    Write-Host ""
    
    # Find the best trained model
    $BestModel = $null
    $ConfigPath = $null
    
    # Check for best_model first, then final_model
    $bestModelPattern = "save_models\best_model_${DatasetName}_fold0.pth"
    $finalModelPattern = "save_models\final_model_${DatasetName}_fold0.pth"
    
    if (Test-Path $bestModelPattern) {
        $BestModel = $bestModelPattern
        Write-Info "Found best model: $BestModel"
    } elseif (Test-Path $finalModelPattern) {
        $BestModel = $finalModelPattern
        Write-Info "Found final model: $BestModel"
    } else {
        Write-Warning "No suitable model found for deep visualization"
    }
    
    if ($BestModel) {
        # Find corresponding config file
        $configPattern = "save_models\config_${DatasetName}_fold0.json"
        if (Test-Path $configPattern) {
            $ConfigPath = $configPattern
            Write-Info "Found config: $ConfigPath"
        } else {
            Write-Warning "Config file not found, using default configuration"
        }
        
        # Create output directory
        $outputDir = "figs\$DatasetName"
        if (-not (Test-Path $outputDir)) {
            New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
            Write-Info "Created visualization directory: $outputDir"
        }
        
        # Run deep visualization
        Write-Info "Generating deep model visualizations..."
        
        $vizCmd = @("python", "visualize.py", "--checkpoint", $BestModel, "--output_dir", $outputDir)
        if ($ConfigPath) {
            $vizCmd += @("--config", $ConfigPath)
        }
        
        Write-Info "Running: $($vizCmd -join ' ')"
        
        try {
            & $vizCmd[0] $vizCmd[1..($vizCmd.Length-1)]
            if ($LASTEXITCODE -eq 0) {
                Write-Success "Deep visualizations generated successfully"
            } else {
                Write-Warning "Deep visualization completed with warnings"
            }
        } catch {
            Write-Warning "Deep visualization had issues: $_"
        }
    }
} else {
    Write-Info "Skipping deep visualization phase"
}

$pipelineEnd = Get-Date
$duration = $pipelineEnd - $pipelineStart

Write-Host ""
Write-Host "================================================================" -ForegroundColor Blue
Write-Host " Pipeline Complete for $DatasetName!" -ForegroundColor Blue
Write-Host "================================================================" -ForegroundColor Blue
Write-Host ""

Write-Info "Results Summary:"
Write-Host "  Training metrics: results\train\"
Write-Host "  Test evaluations: results\test\"
Write-Host "  Plots: results\plots\"
Write-Host "  Models: save_models\"
Write-Host "  Visualizations: figs\$DatasetName\"
Write-Host "  Deep visualizations: figs\$DatasetName\ (per-KC analysis)"
Write-Host "  IRT parameters: stats\ (theta, alpha, beta from best models)"
Write-Host "  Duration: $($duration.ToString('hh\:mm\:ss'))"

if ($Quick) {
    Write-Host ""
    Write-Warning "This was a quick test run. For full results, run without -Quick flag."
}

Write-Host ""
Write-Success "All pipeline phases completed for $DatasetName!"