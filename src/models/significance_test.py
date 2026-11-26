"""
Is the model's edge over the baseline real, or could it be noise?

Method: month-block bootstrap on the test period. Whole calendar months are resampled with
replacement (2,000 times), which keeps the dependence between neighbouring days and between
stocks on the same day. For each resample we compute  accuracy - baseline accuracy.
Reported: the observed gain and a 95% confidence interval.

Run:  python src/models/significance_test.py     (about 5-8 minutes)
"""
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from extension_checks import prepare, make_dataset, GAP, MAX_TRAIN   # same folder

warnings.filterwarnings('ignore')
ROOT = Path(__file__).resolve().parent.parent.parent
DATA = ROOT / 'data' / 'processed'
RESULTS = ROOT / 'results'


def gain_with_ci(data, cutoff, n_boot=2000, seed=0):
    cols = [c for c in data.columns if c not in ('Pattern', 'Date', 'Ticker')]
    train = data[data['Date'] < cutoff - GAP]
    test = data[data['Date'] >= cutoff].copy()
    tr = train.sample(n=min(len(train), MAX_TRAIN), random_state=42)
    model = RandomForestClassifier(n_estimators=100, max_depth=12, min_samples_leaf=50,
                                   n_jobs=-1, random_state=42).fit(tr[cols], tr['Pattern'])
    pred = model.predict(test[cols])
    majority = tr['Pattern'].value_counts().idxmax()
    test['correct'] = (pred == test['Pattern'].values).astype(int)
    test['base'] = (test['Pattern'] == majority).astype(int)
    test['month'] = test['Date'].dt.to_period('M')
    g = test.groupby('month').agg(n=('correct', 'size'), c=('correct', 'sum'), b=('base', 'sum'))
    n, c, b = g['n'].values, g['c'].values, g['b'].values
    point = c.sum() / n.sum() - b.sum() / n.sum()
    rng = np.random.default_rng(seed)
    k = len(g)
    gains = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, k, k)
        gains[i] = c[idx].sum() / n[idx].sum() - b[idx].sum() / n[idx].sum()
    lo, hi = np.percentile(gains, [2.5, 97.5])
    return {'accuracy': c.sum() / n.sum(), 'baseline': b.sum() / n.sum(), 'gain': point,
            'ci_low': lo, 'ci_high': hi, 'share_gain_leq_0': float((gains <= 0).mean()), 'months': k}


if __name__ == '__main__':
    with open(DATA / 'labeled_stocks_sample.pkl', 'rb') as f:
        labeled = pickle.load(f)
    base = prepare(labeled)
    original = make_dataset(base, h=5, mode='fixed', thr=0.03)
    cutoff = original['Date'].quantile(0.8)
    scaled = make_dataset(base, h=5, mode='scaled', k=1.0)

    rows = []
    for name, data in (('Original labels (fixed 3%, 5 days)', original),
                       ('Volatility-scaled labels', scaled)):
        r = gain_with_ci(data, cutoff)
        r['labels'] = name
        rows.append(r)
        print("\n" + "=" * 72)
        print(name)
        print("=" * 72)
        print(f"  Accuracy {r['accuracy']:.3f} | baseline {r['baseline']:.3f} | gain {100 * r['gain']:+.1f} points")
        print(f"  95% confidence interval for the gain: {100 * r['ci_low']:+.1f} to {100 * r['ci_high']:+.1f} points "
              f"({r['months']} test months resampled)")
        print(f"  Share of resamples where the gain is zero or negative: {r['share_gain_leq_0']:.1%}")
        print("  Reading: " + ("the interval excludes zero, so the edge is unlikely to be noise within this period."
                               if r['ci_low'] > 0 else "the interval includes zero, so the edge could be noise."))
    pd.DataFrame(rows)[['labels', 'accuracy', 'baseline', 'gain', 'ci_low', 'ci_high', 'share_gain_leq_0', 'months']].to_csv(
        RESULTS / 'v2_significance.csv', index=False)
    print("\nSaved: results/v2_significance.csv")
    print("Note: this covers sampling variation inside the test period only, not changes in market regime.")
