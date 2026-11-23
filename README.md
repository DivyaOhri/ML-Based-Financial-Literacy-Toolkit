# ML-Based Financial Literacy Toolkit

An educational toolkit that uses **classical machine learning** on historical US stock prices to teach beginners how market patterns and risk work, and uses **SHAP** to explain what the models learned. It is a learning tool, not a trading system or investment advice.

## What it does

| Component | Method | Output |
|---|---|---|
| Pattern classifier | Random Forest, Gradient Boosting | Labels each day as **Bullish / Bearish / Neutral / Consolidation** based on the next 5 trading days |
| Risk tiers | K-Means (k = 3) | Groups stocks into **Low / Medium / High risk** by volatility, momentum, RSI and volume |
| Explainability | SHAP (TreeExplainer) | Ranks which indicators drive the predictions |

## Data

[Kaggle: Huge Stock Market Dataset](https://www.kaggle.com/datasets/borismarjanovic/price-volume-data-for-all-us-stocks-etfs) (daily OHLCV, US stocks and ETFs, up to Nov 2017).

500 stock files are sampled at random (seed 42); 498 load, and 394 stocks with at least 500 trading days remain, giving **1,001,927 labeled trading days** (1982 to 2017).

## Method in brief

- **Features (15, all scale-free):** RSI, Bollinger position and width, 10-day momentum, 20-day volatility, volume ratio, close vs 5/10/20/50-day averages, 5 vs 20 and 20 vs 50-day averages, MACD / signal / histogram as a share of price.
- **Labels:** forward 5-day return above +3% is Bullish, below -3% is Bearish; otherwise Consolidation if the next 5 days are calm (volatility below 0.8x the current 20-day volatility), else Neutral.
- **Evaluation:** time-based split (train 1982 to Oct 2015, test Oct 2015 to Nov 2017) with a 10-day gap, compared against a most-common-class baseline.

## Results (test set: 200,555 rows)

| Model | Accuracy | Macro-F1 |
|---|---|---|
| Baseline (most common class) | 30.8% | 0.118 |
| Random Forest | 40.2% | 0.367 |
| Gradient Boosting | 40.3% | 0.368 |

- **SHAP:** volatility (0.045) and Bollinger band width (0.024) dominate; every other feature is 0.006 or less. Indicators say more about how volatile a period is than about its direction.
- **Risk tiers:** 286 Low, 83 Medium, 25 High (silhouette 0.41). Average daily volatility 1.81%, 4.13% and 5.07%.
- **Lesson:** an early run with a random split and raw price features reported 88.9% accuracy. This was data leakage; fixing it (time-based split, scale-free features) gives the honest figures above.

## Project structure

```
ML project/
├── src/
│   ├── utils/
│   │   ├── data_loader.py          # load Kaggle .txt files
│   │   ├── data_cleaner.py         # filter and validate stocks
│   │   └── feature_engineering.py  # technical indicators
│   └── models/
│       ├── pattern_labeling.py     # forward-looking labels
│       └── train_eval_v2.py        # training, evaluation, SHAP, risk tiers
├── notebooks/01_EDA.py             # exploratory charts
├── results/                        # metrics (CSV) and visualizations/
├── docs/                           # project report (Word) and presentation (PowerPoint)
├── data/                           # raw and processed data (not tracked by git)
├── requirements.txt
└── run_all.ps1                     # runs the whole pipeline
```

## Setup (Windows, PowerShell)

```powershell
python -m venv C:\venvs\mlproject            # keep the venv outside OneDrive
C:\venvs\mlproject\Scripts\Activate.ps1
pip install -r requirements.txt
```

Download the dataset from Kaggle and extract it so the stock files are in `data/raw/stocks/` (files like `aapl.us.txt`), for example:

```powershell
pip install kaggle
kaggle datasets download -d borismarjanovic/price-volume-data-for-all-us-stocks-etfs -p data/raw --unzip
```

(This needs a Kaggle API token in `~/.kaggle/kaggle.json`.)

## Run

```powershell
.\run_all.ps1
```

or step by step:

```powershell
python src/utils/data_loader.py
python src/utils/data_cleaner.py
python src/utils/feature_engineering.py
python src/models/pattern_labeling.py
python src/models/train_eval_v2.py
```

Outputs: `results/v2_classification_summary.csv`, `results/v2_feature_importance.csv`, `results/v2_risk_tiers.csv`, and charts in `results/visualizations/`. Intermediate `.pkl` files are written to `data/processed/` and are not tracked by git.

## Limitations

394 of several thousand stocks; a single test period (2015 to 2017); untuned label thresholds; volatility is partly built into the labels; risk tiers not validated against external measures; no trading costs or strategy.

## Author

Divya Ohri, B.Tech CSE, Machine Learning course project.
