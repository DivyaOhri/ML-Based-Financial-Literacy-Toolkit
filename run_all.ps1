# Runs the full pipeline, stopping at the first error.
# Usage (PowerShell, venv active, from the project folder):  .\run_all.ps1
$ErrorActionPreference = "Stop"

function Run($script) {
    Write-Host "`n>>> python $script" -ForegroundColor Cyan
    python $script
    if ($LASTEXITCODE -ne 0) { throw "Failed: $script" }
}

Run "src/utils/data_loader.py"          # read 500 random stock files
Run "src/utils/data_cleaner.py"         # keep stocks with >= 500 trading days
Run "src/utils/feature_engineering.py"  # compute technical indicators
Run "src/models/pattern_labeling.py"    # label each day by the next 5 days
Run "src/models/train_eval_v2.py"       # train, evaluate (time-based split), SHAP, risk tiers
Run "notebooks/01_EDA.py"               # exploratory charts

Write-Host "`nDone. See results/ and results/visualizations/" -ForegroundColor Green
