#!/bin/bash

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

echo "=================================================="
echo "ML FINANCIAL LITERACY - WEEK 1 PIPELINE"
echo "=================================================="
echo ""

echo "[STEP 1] Setting up Python environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate 2>/dev/null || . venv/Scripts/activate 2>/dev/null
echo "✓ Environment ready"
echo ""

echo "[STEP 2] Installing dependencies..."
pip install -q -r requirements.txt
echo "✓ Dependencies installed"
echo ""

echo "[STEP 3] Loading raw data..."
python src/utils/data_loader.py
echo ""

echo "[STEP 4] Cleaning data..."
python src/utils/data_cleaner.py
echo ""

echo "[STEP 5] Running EDA..."
python notebooks/01_EDA.py
echo ""

echo "=================================================="
echo "✓ WEEK 1 COMPLETE"
echo "=================================================="
echo "Visualizations: results/visualizations/"
echo "Cleaned data: data/processed/cleaned_stocks_sample.pkl"
echo ""