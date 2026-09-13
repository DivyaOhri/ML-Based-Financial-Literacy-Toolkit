# ML-Based Financial Literacy Toolkit (Stock Market Core)

## Project Overview

A classical machine learning system that teaches beginners to recognize stock market patterns and risk profiles using interpretable models and SHAP explainability.

**Timeline:** 1 month | **Tech Stack:** Random Forest, XGBoost, K-Means + SHAP

### Phase 1 (CORE): Stock Market Pattern & Risk Literacy
- **1a. Pattern Classification**: Classify market patterns (bullish/bearish/neutral/consolidation)
- **1b. Risk Clustering**: Segment stocks into risk tiers (stable/moderate/high-volatility)
- **1c. Similar-Stock Finder** (optional): k-NN based recommendation

## Quick Start

```bash
cd ml-financial-literacy-toolkit

# Setup (5 min)
python -m venv venv
source venv/bin/activate  # Mac/Linux
pip install -r requirements.txt

# Run Week 1 pipeline (35-60 min)
python src/utils/data_loader.py      # Load Kaggle dataset
python src/utils/data_cleaner.py     # Clean data
python notebooks/01_EDA.py           # Generate visualizations
```

OR run everything at once:
```bash
bash week1_pipeline.sh
```

## Documentation

1. **[START_HERE.md](./START_HERE.md)** ← **Read this first!** (5 min)
2. **[EXECUTION_GUIDE.md](./EXECUTION_GUIDE.md)** — Step-by-step guide
3. **[WEEK1_CHECKLIST.md](./WEEK1_CHECKLIST.md)** — Detailed walkthrough

## Directory Structure
ml-financial-literacy-toolkit/
├── data/
│ ├── raw/ # Kaggle dataset (to download)
│ └── processed/ # Cleaned data (output)
├── src/
│ └── utils/
│ ├── data_loader.py
│ └── data_cleaner.py
├── notebooks/
│ └── 01_EDA.py # Exploratory Data Analysis
├── results/
│ └── visualizations/ # Output: 4 PNG files
└── requirements.txt


## Dataset

**Source:** Kaggle "Huge Stock Market Dataset"  
**Size:** ~7000 stocks/ETFs, ~2GB  
**After cleaning:** 2000+ stocks with 500+ trading days each

## Next Steps

1. Read **START_HERE.md** (2 min)
2. Set up Kaggle credentials (see WEEK1_CHECKLIST.md)
3. Install dependencies and run pipeline
4. Review visualizations in `results/visualizations/`

**Week 2:** Feature engineering → Model training → SHAP analysis  
**Week 3:** Evaluation & Phase 1 report