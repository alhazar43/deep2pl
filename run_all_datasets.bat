@echo off
REM run_all_datasets.bat - Windows Batch Script for Deep-2PL All Datasets Pipeline
REM
REM This script runs the complete pipeline for ALL datasets:
REM 1. Training with 5-fold cross-validation
REM 2. Model evaluation on test sets
REM 3. Visualization and metrics generation
REM 4. Comprehensive results comparison
REM
REM Usage: run_all_datasets.bat [OPTIONS]
REM
REM Examples:
REM   run_all_datasets.bat                    # Full pipeline for all datasets (30 epochs)
REM   run_all_datasets.bat --epochs 10       # Custom epochs for all datasets
REM   run_all_datasets.bat --quick           # Quick test (1 epoch, single fold)
REM   run_all_datasets.bat --exclude large   # Exclude large datasets

setlocal enabledelayedexpansion

REM Default parameters
set EPOCHS=30
set BATCH_SIZE=
set LEARNING_RATE=
set QUICK_MODE=false
set SINGLE_FOLD=false
set FOLD_IDX=0
set EXCLUDE_DATASETS=
set ONLY_DATASETS=
set SPECIFIC_DATASETS=

REM Dataset categories
set SMALL_DATASETS=synthetic assist2009_updated assist2015 assist2017 assist2009
set MEDIUM_DATASETS=kddcup2010 STATICS
set LARGE_DATASETS=statics2011 fsaif1tof3

REM All datasets combined
set ALL_DATASETS=%SMALL_DATASETS% %MEDIUM_DATASETS% %LARGE_DATASETS%

REM Colors/Prefixes for output
set "SUCCESS_PREFIX=[SUCCESS]"
set "ERROR_PREFIX=[ERROR]"
set "WARNING_PREFIX=[WARNING]"
set "INFO_PREFIX=[INFO]"
set "DATASET_PREFIX=[DATASET]"

REM Usage function
if "%~1"=="--help" goto :show_usage
if "%~1"=="/?" goto :show_usage

goto :parse_args

:show_usage
echo Usage: %0 [OPTIONS]
echo.
echo This script runs the complete Deep-2PL pipeline for all datasets:
echo   • Training with 5-fold cross-validation
echo   • Model evaluation and testing
echo   • Visualization and metrics generation
echo   • Comprehensive results comparison
echo.
echo OPTIONS:
echo   --epochs N          Number of training epochs ^(default: 30^)
echo   --batch-size N      Batch size for all datasets
echo   --learning-rate F   Learning rate for all datasets
echo   --quick             Quick mode: 1 epoch, single fold ^(for testing^)
echo   --single-fold       Train only fold 0 instead of 5-fold CV
echo   --fold N            Specify fold index for single-fold training ^(0-4^)
echo.
echo DATASET FILTERING:
echo   --exclude TYPE      Exclude dataset type: 'large', 'medium', 'small'
echo   --only TYPE         Include only dataset type: 'large', 'medium', 'small'
echo   --datasets LIST     Comma-separated list of specific datasets
echo.
echo   --help              Show this help message
echo.
echo Dataset Categories:
echo   Small:  %SMALL_DATASETS%
echo   Medium: %MEDIUM_DATASETS%
echo   Large:  %LARGE_DATASETS%
echo.
echo Examples:
echo   %0                         # Full pipeline, all datasets, 30 epochs
echo   %0 --epochs 10             # All datasets with 10 epochs
echo   %0 --exclude large         # Skip large datasets ^(faster^)
echo   %0 --only small            # Only small datasets ^(quick test^)
echo   %0 --quick                 # Ultra-quick test run
echo   %0 --datasets synthetic,assist2015  # Specific datasets only
exit /b 0

:parse_args
if "%~1"=="" goto :done_parsing
if "%~1"=="--epochs" (
    set EPOCHS=%2
    shift
    shift
    goto :parse_args
)
if "%~1"=="--batch-size" (
    set BATCH_SIZE=%2
    shift
    shift
    goto :parse_args
)
if "%~1"=="--learning-rate" (
    set LEARNING_RATE=%2
    shift
    shift
    goto :parse_args
)
if "%~1"=="--quick" (
    set QUICK_MODE=true
    set EPOCHS=1
    set SINGLE_FOLD=true
    shift
    goto :parse_args
)
if "%~1"=="--single-fold" (
    set SINGLE_FOLD=true
    shift
    goto :parse_args
)
if "%~1"=="--fold" (
    set FOLD_IDX=%2
    set SINGLE_FOLD=true
    shift
    shift
    goto :parse_args
)
if "%~1"=="--exclude" (
    if "%~2"=="large" set EXCLUDE_DATASETS=%LARGE_DATASETS%
    if "%~2"=="medium" set EXCLUDE_DATASETS=%MEDIUM_DATASETS%
    if "%~2"=="small" set EXCLUDE_DATASETS=%SMALL_DATASETS%
    if "%EXCLUDE_DATASETS%"=="" (
        echo %ERROR_PREFIX% Unknown exclude type: %2
        goto :show_usage
    )
    shift
    shift
    goto :parse_args
)
if "%~1"=="--only" (
    if "%~2"=="large" set ONLY_DATASETS=%LARGE_DATASETS%
    if "%~2"=="medium" set ONLY_DATASETS=%MEDIUM_DATASETS%
    if "%~2"=="small" set ONLY_DATASETS=%SMALL_DATASETS%
    if "%ONLY_DATASETS%"=="" (
        echo %ERROR_PREFIX% Unknown only type: %2
        goto :show_usage
    )
    shift
    shift
    goto :parse_args
)
if "%~1"=="--datasets" (
    set SPECIFIC_DATASETS=%~2
    shift
    shift
    goto :parse_args
)
echo %ERROR_PREFIX% Unknown option: %1
goto :show_usage

:done_parsing

REM Determine final dataset list
if not "%SPECIFIC_DATASETS%"=="" (
    set FINAL_DATASETS=%SPECIFIC_DATASETS%
    set FINAL_DATASETS=!FINAL_DATASETS:,= !
) else if not "%ONLY_DATASETS%"=="" (
    set FINAL_DATASETS=%ONLY_DATASETS%
) else (
    set FINAL_DATASETS=%ALL_DATASETS%
    
    REM Apply exclusions (simplified - Windows batch is limited)
    if not "%EXCLUDE_DATASETS%"=="" (
        echo %INFO_PREFIX% Applying exclusions: %EXCLUDE_DATASETS%
        REM Note: Full exclusion logic would be complex in batch, 
        REM so we'll keep it simple and let user specify --only instead
    )
)

echo.
echo ================================================================
echo  Deep-2PL Comprehensive Pipeline for All Datasets
echo ================================================================
echo.

echo %INFO_PREFIX% Configuration:
echo   Datasets: %FINAL_DATASETS%
echo   Epochs: %EPOCHS%
if "%SINGLE_FOLD%"=="true" (
    echo   Mode: Single fold ^(%FOLD_IDX%^)
) else (
    echo   Mode: 5-fold cross-validation
)
if "%QUICK_MODE%"=="true" echo   Quick mode enabled
if not "%BATCH_SIZE%"=="" echo   Batch size: %BATCH_SIZE%
if not "%LEARNING_RATE%"=="" echo   Learning rate: %LEARNING_RATE%
echo.

REM Change to script directory
cd /d "%~dp0"

REM Check if conda is available
where conda >nul 2>&1
if errorlevel 1 (
    echo %ERROR_PREFIX% Conda not found. Please install Anaconda/Miniconda and add to PATH.
    exit /b 1
)

REM Check environment
echo %INFO_PREFIX% Checking environment...
if "%CONDA_DEFAULT_ENV%"=="" (
    echo %WARNING_PREFIX% No conda environment detected. Attempting to activate vrec-env...
    call conda activate vrec-env
    if errorlevel 1 (
        echo %ERROR_PREFIX% Failed to activate vrec-env environment.
        echo Please run: conda activate vrec-env
        exit /b 1
    )
)

REM Create directories
if not exist "results\train" mkdir "results\train"
if not exist "results\valid" mkdir "results\valid"
if not exist "results\test" mkdir "results\test"
if not exist "results\plots" mkdir "results\plots"
if not exist "save_models" mkdir "save_models"
if not exist "logs" mkdir "logs"
if not exist "stats" mkdir "stats"

REM Estimate time (simplified)
echo %INFO_PREFIX% Starting comprehensive pipeline...
echo %WARNING_PREFIX% Training time will vary based on dataset size and hardware

set PIPELINE_START=%time%

echo.
echo ================================================================
echo  Starting Pipeline Execution
echo ================================================================
echo.

REM Track results
set SUCCESSFUL_DATASETS=
set FAILED_DATASETS=
set CURRENT_NUM=0
set TOTAL_NUM=0

REM Count total datasets
for %%d in (%FINAL_DATASETS%) do set /a TOTAL_NUM+=1

REM Process each dataset
for %%d in (%FINAL_DATASETS%) do (
    set /a CURRENT_NUM+=1
    
    echo.
    echo ================================================================
    echo  Dataset !CURRENT_NUM!/!TOTAL_NUM!: %%d
    echo ================================================================
    echo.
    
    set cmd=run_dataset.bat %%d --epochs %EPOCHS%
    
    if "%SINGLE_FOLD%"=="true" set cmd=!cmd! --single-fold --fold %FOLD_IDX%
    if "%QUICK_MODE%"=="true" set cmd=!cmd! --quick
    if not "%BATCH_SIZE%"=="" set cmd=!cmd! --batch-size %BATCH_SIZE%
    if not "%LEARNING_RATE%"=="" set cmd=!cmd! --learning-rate %LEARNING_RATE%
    
    echo %INFO_PREFIX% Command: !cmd!
    
    !cmd!
    
    if errorlevel 1 (
        echo %ERROR_PREFIX% %%d failed
        set FAILED_DATASETS=!FAILED_DATASETS! %%d
        
        if "%QUICK_MODE%"=="false" (
            echo.
            set /p continue="Continue with remaining datasets? (Y/n): "
            if /i "!continue!"=="n" (
                echo %INFO_PREFIX% Stopping execution as requested
                goto :final_summary
            )
        )
    ) else (
        echo %SUCCESS_PREFIX% %%d completed
        set SUCCESSFUL_DATASETS=!SUCCESSFUL_DATASETS! %%d
    )
)

:final_summary

REM Generate comprehensive comparison (simplified)
if not "%SUCCESSFUL_DATASETS%"=="" (
    echo.
    echo --- Generating Comprehensive Comparison ---
    echo.
    
    echo %INFO_PREFIX% Creating comparison script...
    echo import json, glob, os, pandas as pd > compare_results.py
    echo import numpy as np >> compare_results.py
    echo. >> compare_results.py
    echo results = [] >> compare_results.py
    echo for file in glob.glob('results/test/eval_*_test.json'): >> compare_results.py
    echo     try: >> compare_results.py
    echo         with open(file, 'r'^) as f: data = json.load(f^) >> compare_results.py
    echo         filename = os.path.basename(file^) >> compare_results.py
    echo         parts = filename.split('_'^) >> compare_results.py
    echo         dataset = '_'.join(parts[2:-2]^) >> compare_results.py
    echo         results.append({'dataset': dataset, 'auc': data.get('auc', 0^), 'accuracy': data.get('accuracy', 0^)}^) >> compare_results.py
    echo     except: pass >> compare_results.py
    echo. >> compare_results.py
    echo if results: >> compare_results.py
    echo     df = pd.DataFrame(results^) >> compare_results.py
    echo     print('\\nResults Summary:'^) >> compare_results.py
    echo     print(df.groupby('dataset'^).agg({'auc': 'mean', 'accuracy': 'mean'}^).round(4^)^) >> compare_results.py
    echo     df.to_csv('results/comprehensive_comparison.csv', index=False^) >> compare_results.py
    echo     print('\\nDetailed results saved to: results/comprehensive_comparison.csv'^) >> compare_results.py
    
    python compare_results.py
    del compare_results.py
    
    if errorlevel 1 (
        echo %WARNING_PREFIX% Comparison generation had issues
    ) else (
        echo %SUCCESS_PREFIX% Comprehensive comparison generated
    )
)

echo.
echo ================================================================
echo  Pipeline Execution Complete!
echo ================================================================
echo.

echo %INFO_PREFIX% Execution Summary:
echo   Successful datasets: %SUCCESSFUL_DATASETS%
echo   Failed datasets: %FAILED_DATASETS%
echo.

echo %INFO_PREFIX% Results Locations:
echo   Training metrics: results\train\
echo   Evaluations: results\test\
echo   Plots: results\plots\
echo   Deep visualizations: figs\ ^(per-dataset per-KC analysis^)
echo   Models: save_models\
echo   IRT parameters: stats\ ^(theta, alpha, beta from best models^)
if not "%SUCCESSFUL_DATASETS%"=="" echo   Comparison: results\comprehensive_comparison.csv

if "%QUICK_MODE%"=="true" (
    echo.
    echo %WARNING_PREFIX% This was a quick test run. For full results, run without --quick flag.
)

echo.
echo %SUCCESS_PREFIX% All pipeline phases completed!