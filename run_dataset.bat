@echo off
REM run_dataset.bat - Windows Batch Script for Deep-2PL Single Dataset Pipeline
REM
REM This script runs the complete pipeline for a single dataset:
REM 1. Training with 5-fold cross-validation (or single fold)
REM 2. Model evaluation on test sets
REM 3. Visualization and metrics generation
REM
REM Usage: run_dataset.bat DATASET_NAME [OPTIONS]
REM
REM Examples:
REM   run_dataset.bat synthetic
REM   run_dataset.bat STATICS --epochs 20
REM   run_dataset.bat assist2015 --quick

setlocal enabledelayedexpansion

REM Default parameters
set DATASET_NAME=
set EPOCHS=30
set BATCH_SIZE=
set LEARNING_RATE=
set QUICK_MODE=false
set SINGLE_FOLD=false
set FOLD_IDX=0
set SKIP_TRAINING=false
set SKIP_EVALUATION=false
set SKIP_VISUALIZATION=false

REM Colors (Windows CMD doesn't support colors well, but we'll use echo for messages)
set "SUCCESS_PREFIX=[SUCCESS]"
set "ERROR_PREFIX=[ERROR]"
set "WARNING_PREFIX=[WARNING]"
set "INFO_PREFIX=[INFO]"

REM Check if dataset name provided
if "%~1"=="" (
    echo Usage: %0 DATASET_NAME [OPTIONS]
    echo.
    echo This script runs the complete Deep-2PL pipeline for a single dataset:
    echo   • Training with 5-fold cross-validation
    echo   • Model evaluation and testing
    echo   • Visualization and metrics generation
    echo.
    echo Available datasets:
    echo   Small:  synthetic, assist2009_updated, assist2015, assist2017, assist2009
    echo   Medium: kddcup2010, STATICS
    echo   Large:  statics2011, fsaif1tof3
    echo.
    echo OPTIONS:
    echo   --epochs N          Number of training epochs ^(default: 30^)
    echo   --batch-size N      Batch size for training
    echo   --learning-rate F   Learning rate for training
    echo   --quick             Quick mode: 1 epoch, single fold ^(for testing^)
    echo   --single-fold       Train only fold 0 instead of 5-fold CV
    echo   --fold N            Specify fold index for single-fold training ^(0-4^)
    echo   --skip-training     Skip training phase
    echo   --skip-evaluation   Skip evaluation phase
    echo   --skip-visualization Skip visualization phase
    echo.
    echo Examples:
    echo   %0 synthetic                    # Quick test dataset
    echo   %0 STATICS --epochs 20         # Large dataset with custom epochs
    echo   %0 assist2015 --quick          # Ultra-quick test run
    echo   %0 synthetic --single-fold     # Single fold training only
    exit /b 1
)

set DATASET_NAME=%1
shift

REM Parse command line arguments
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
if "%~1"=="--skip-training" (
    set SKIP_TRAINING=true
    shift
    goto :parse_args
)
if "%~1"=="--skip-evaluation" (
    set SKIP_EVALUATION=true
    shift
    goto :parse_args
)
if "%~1"=="--skip-visualization" (
    set SKIP_VISUALIZATION=true
    shift
    goto :parse_args
)
echo %ERROR_PREFIX% Unknown option: %1
exit /b 1

:done_parsing

echo.
echo ================================================================
echo  Deep-2PL Pipeline for %DATASET_NAME%
echo ================================================================
echo.

echo %INFO_PREFIX% Configuration:
echo   Dataset: %DATASET_NAME%
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

REM Check environment (simplified check)
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

set PIPELINE_START=%time%

REM Training Phase
if "%SKIP_TRAINING%"=="false" (
    echo.
    echo --- Training Phase ---
    echo.
    
    set cmd=python train.py --dataset %DATASET_NAME% --epochs %EPOCHS%
    
    if "%SINGLE_FOLD%"=="true" (
        set cmd=!cmd! --single_fold --fold %FOLD_IDX%
    )
    
    if not "%BATCH_SIZE%"=="" set cmd=!cmd! --batch_size %BATCH_SIZE%
    if not "%LEARNING_RATE%"=="" set cmd=!cmd! --learning_rate %LEARNING_RATE%
    
    echo %INFO_PREFIX% Running: !cmd!
    !cmd!
    
    if errorlevel 1 (
        echo %ERROR_PREFIX% Training failed!
        exit /b 1
    )
    
    echo %SUCCESS_PREFIX% Training completed
) else (
    echo %INFO_PREFIX% Skipping training phase
)

REM Evaluation Phase
if "%SKIP_EVALUATION%"=="false" (
    echo.
    echo --- Evaluation Phase ---
    echo.
    
    if "%SINGLE_FOLD%"=="true" (
        set model_file=save_models\best_model_%DATASET_NAME%_fold%FOLD_IDX%.pth
        if exist "!model_file!" (
            echo %INFO_PREFIX% Evaluating model: !model_file!
            python evaluate.py --model_path "!model_file!" --split test
            if errorlevel 1 (
                echo %WARNING_PREFIX% Evaluation failed for !model_file!
            ) else (
                echo %SUCCESS_PREFIX% Evaluation completed for fold %FOLD_IDX%
            )
        ) else (
            echo %WARNING_PREFIX% Model file not found: !model_file!
        )
    ) else (
        echo %INFO_PREFIX% Evaluating all 5 folds...
        for /L %%i in (0,1,4) do (
            set model_file=save_models\best_model_%DATASET_NAME%_fold%%i.pth
            if exist "!model_file!" (
                echo %INFO_PREFIX% Evaluating fold %%i...
                python evaluate.py --model_path "!model_file!" --split test
                if errorlevel 1 (
                    echo %WARNING_PREFIX% Evaluation failed for fold %%i
                ) else (
                    echo %SUCCESS_PREFIX% Evaluation completed for fold %%i
                )
            ) else (
                echo %WARNING_PREFIX% Model file not found for fold %%i: !model_file!
            )
        )
    )
) else (
    echo %INFO_PREFIX% Skipping evaluation phase
)

REM Visualization Phase
if "%SKIP_VISUALIZATION%"=="false" (
    echo.
    echo --- Visualization Phase ---
    echo.
    
    REM Generate training metrics plots
    echo %INFO_PREFIX% Generating training metrics plots...
    python plot_metrics.py --dataset %DATASET_NAME% --results_dir results\train --output_dir results\plots
    if errorlevel 1 (
        echo %WARNING_PREFIX% Training metrics plotting had issues
    ) else (
        echo %SUCCESS_PREFIX% Training metrics plots generated
    )
    
    REM Generate evaluation plots
    echo %INFO_PREFIX% Generating evaluation plots...
    python plot_evaluation.py --dataset %DATASET_NAME% --results_dir results\test --output_dir results\plots
    if errorlevel 1 (
        echo %WARNING_PREFIX% Evaluation plotting had issues
    ) else (
        echo %SUCCESS_PREFIX% Evaluation plots generated
    )
    
    REM Run main visualization (theta/beta plots)
    if exist "save_models\best_model_%DATASET_NAME%_fold0.pth" (
        echo %INFO_PREFIX% Generating visualization plots...
        python visualize.py --model_path "save_models\best_model_%DATASET_NAME%_fold0.pth" --dataset %DATASET_NAME%
        if errorlevel 1 (
            echo %WARNING_PREFIX% Visualization had issues
        ) else (
            echo %SUCCESS_PREFIX% Visualization plots generated
        )
    ) else (
        echo %WARNING_PREFIX% No model found for visualization
    )
) else (
    echo %INFO_PREFIX% Skipping visualization phase
)

REM Phase 4: Deep Model Visualization
echo.
echo --- Deep Model Visualization Phase ---
echo.

REM Find the best trained model
set BEST_MODEL=
if "%SINGLE_FOLD%"=="true" (
    if exist "save_models\best_model_%DATASET_NAME%_fold%FOLD_IDX%.pth" (
        set BEST_MODEL=save_models\best_model_%DATASET_NAME%_fold%FOLD_IDX%.pth
    ) else if exist "save_models\final_model_%DATASET_NAME%_fold%FOLD_IDX%.pth" (
        set BEST_MODEL=save_models\final_model_%DATASET_NAME%_fold%FOLD_IDX%.pth
    )
) else (
    REM Look for best model from any fold (prioritize fold 0)
    if exist "save_models\best_model_%DATASET_NAME%_fold0.pth" (
        set BEST_MODEL=save_models\best_model_%DATASET_NAME%_fold0.pth
    ) else (
        for /L %%i in (1,1,4) do (
            if exist "save_models\best_model_%DATASET_NAME%_fold%%i.pth" if "!BEST_MODEL!"=="" (
                set BEST_MODEL=save_models\best_model_%DATASET_NAME%_fold%%i.pth
            )
        )
        REM If no best models, look for final models
        if "!BEST_MODEL!"=="" (
            if exist "save_models\final_model_%DATASET_NAME%_fold0.pth" (
                set BEST_MODEL=save_models\final_model_%DATASET_NAME%_fold0.pth
            ) else (
                for /L %%i in (1,1,4) do (
                    if exist "save_models\final_model_%DATASET_NAME%_fold%%i.pth" if "!BEST_MODEL!"=="" (
                        set BEST_MODEL=save_models\final_model_%DATASET_NAME%_fold%%i.pth
                    )
                )
            )
        )
    )
)

if not "!BEST_MODEL!"=="" (
    echo %INFO_PREFIX% Generating detailed visualizations with: !BEST_MODEL!
    
    REM Find corresponding config file
    set CONFIG_PATH=!BEST_MODEL:.pth=.json!
    if not exist "!CONFIG_PATH!" (
        REM Fallback: look for config in save_models directory
        if "%SINGLE_FOLD%"=="true" (
            set CONFIG_PATH=save_models\config_%DATASET_NAME%_fold%FOLD_IDX%.json
        ) else (
            set CONFIG_PATH=save_models\config_%DATASET_NAME%_fold0.json
        )
    )
    
    if exist "!CONFIG_PATH!" (
        REM Create output directory
        if not exist "figs\%DATASET_NAME%" mkdir "figs\%DATASET_NAME%"
        
        echo %INFO_PREFIX% Running deep model visualization...
        python visualize.py --checkpoint "!BEST_MODEL!" --config "!CONFIG_PATH!" --output_dir "figs\%DATASET_NAME%" >nul 2>&1
        if errorlevel 1 (
            echo %WARNING_PREFIX% Deep model visualization had warnings ^(this is normal for some datasets^)
        ) else (
            echo %SUCCESS_PREFIX% Deep model visualizations generated in figs\%DATASET_NAME%\
        )
    ) else (
        echo %WARNING_PREFIX% Config file not found for visualization: !CONFIG_PATH!
    )
) else (
    echo %WARNING_PREFIX% No models available for deep visualization
)

echo.
echo ================================================================
echo  Pipeline Complete for %DATASET_NAME%!
echo ================================================================
echo.

echo %INFO_PREFIX% Results Summary:
echo   Training metrics: results\train\
echo   Test evaluations: results\test\
echo   Plots: results\plots\
echo   Deep visualizations: figs\%DATASET_NAME%\ ^(per-KC analysis^)
echo   Models: save_models\
echo   IRT parameters: stats\ ^(theta, alpha, beta from best models^)

if "%QUICK_MODE%"=="true" (
    echo.
    echo %WARNING_PREFIX% This was a quick test run. For full results, run without --quick flag.
)

echo.
echo %SUCCESS_PREFIX% All pipeline phases completed for %DATASET_NAME%!