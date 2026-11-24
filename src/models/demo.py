"""
Live demo: pick a stock and a date from the unseen test period, then show
what the model predicts, what really happened, the stock's risk tier, and why (SHAP).

  python src/models/demo.py                  random stock and date
  python src/models/demo.py aapl             a specific stock (if it is in the sample)
  python src/models/demo.py aapl 2016-03-15  a specific stock and date
"""
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from train_eval_v2 import build_dataset   # same folder

warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parent.parent.parent
DATA = ROOT / 'data' / 'processed'
RESULTS = ROOT / 'results'
CACHE = DATA / 'demo_cache.pkl'


def load_test_rows():
    if CACHE.exists():
        return pd.read_pickle(CACHE)
    print("First run: preparing data (about a minute). Later runs start in seconds...")
    with open(DATA / 'labeled_stocks_sample.pkl', 'rb') as f:
        labeled = pickle.load(f)
    data = build_dataset(labeled)
    test = data[data['Date'] >= data['Date'].quantile(0.8)].reset_index(drop=True)
    test.to_pickle(CACHE)
    return test


def main():
    args = sys.argv[1:]
    test = load_test_rows()
    tickers = sorted(test['Ticker'].unique())

    if args:
        q = args[0].lower()
        match = [t for t in tickers if str(t).lower() == q or str(t).lower().startswith(q + '.')]
        if not match:
            print(f"'{args[0]}' is not in the sample. Examples you can try: {', '.join(map(str, tickers[:8]))}")
            sys.exit(1)
        ticker = match[0]
    else:
        ticker = tickers[np.random.randint(len(tickers))]

    rows = test[test['Ticker'] == ticker].sort_values('Date')
    if len(args) > 1:
        earlier = rows[rows['Date'] <= pd.Timestamp(args[1])]
        row = earlier.iloc[-1] if len(earlier) else rows.iloc[0]
    else:
        row = rows.iloc[np.random.randint(len(rows))]

    with open(RESULTS / 'models' / 'v2_random_forest.pkl', 'rb') as f:
        saved = pickle.load(f)
    model, feats = saved['model'], saved['features']
    x = pd.DataFrame([row[feats].astype(float)])

    proba = model.predict_proba(x)[0]
    pred = model.classes_[int(np.argmax(proba))]
    tiers = pd.read_csv(RESULTS / 'v2_risk_tiers.csv').set_index('ticker')['Risk_Tier']
    tier = tiers.get(ticker, 'n/a')

    print("\n" + "=" * 64)
    print(f" STOCK: {ticker}    DATE: {pd.Timestamp(row['Date']).date()}   (unseen test period)")
    print("=" * 64)
    print("\nTODAY'S CLUES")
    print(f"  Volatility (20-day) : {row['Volatility']:.2%}   <- how bumpy the price has been")
    print(f"  RSI                 : {row['RSI']:.1f}      <- above ~70 'overbought', below ~30 'oversold'")
    print(f"  Price vs 20-day avg : {row['Close_vs_SMA20']:+.2%}")
    print(f"  10-day momentum     : {row['Momentum']:+.2%}")

    print("\nMODEL PREDICTION FOR THE NEXT 5 TRADING DAYS")
    for c, p in sorted(zip(model.classes_, proba), key=lambda z: -z[1]):
        print(f"  {c:14s} {p:6.1%}  {'#' * int(p * 40)}")
    print(f"\n  Predicted: {pred}")
    print(f"  What really happened: {row['Pattern']}   -> {'CORRECT' if pred == row['Pattern'] else 'WRONG'}")
    print("  (The model is right about 40% of the time overall, so wrong answers are expected.)")
    print(f"\nRISK TIER OF THIS STOCK: {tier}")

    try:
        import shap
        sv = shap.TreeExplainer(model).shap_values(x, check_additivity=False)
        k = list(model.classes_).index(pred)
        if isinstance(sv, list):
            contrib = np.asarray(sv[k])[0]
        else:
            sv = np.asarray(sv)
            contrib = sv[0, :, k] if sv.ndim == 3 else sv[0]
        print(f"\nWHY? TOP CLUES BEHIND '{pred}' (SHAP)")
        for i in np.argsort(-np.abs(contrib))[:5]:
            direction = f"pushes toward {pred}" if contrib[i] > 0 else f"pushes away from {pred}"
            print(f"  {feats[i]:18s} value {float(x.iloc[0, i]):+.4f}   {contrib[i]:+.3f}  {direction}")
    except Exception as e:
        print(f"\n(SHAP explanation unavailable: {e})")
    print()


if __name__ == '__main__':
    main()
