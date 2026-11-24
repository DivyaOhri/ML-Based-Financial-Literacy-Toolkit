"""
Robustness checks (v2) - reduces four limitations:
  1. Only one test period        -> walk-forward validation (4 test windows)
  2. Volatility built into labels -> ablation (remove / keep only volatility features)
  3. Risk tiers not validated     -> do tiers built on the past still hold later?
  4. SHAP on only 500 rows        -> repeat SHAP on a larger sample and compare ranking

Run:  python src/models/robustness_checks.py      (about 10 minutes)
"""
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

from train_eval_v2 import build_dataset   # same folder

warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parent.parent.parent
DATA = ROOT / 'data' / 'processed'
MODELS = ROOT / 'results' / 'models'
RESULTS = ROOT / 'results'

VOL = ['Volatility', 'BB_Width']
GAP = pd.Timedelta(days=10)
MAX_TRAIN = 300_000          # subsample for speed


def make_rf():
    return RandomForestClassifier(n_estimators=100, max_depth=12, min_samples_leaf=50,
                                  n_jobs=-1, random_state=42)


def fit_eval(train, test, cols):
    tr = train.sample(n=min(len(train), MAX_TRAIN), random_state=42)
    model = make_rf().fit(tr[cols], tr['Pattern'])
    pred = model.predict(test[cols])
    majority = tr['Pattern'].value_counts().idxmax()
    return {'accuracy': accuracy_score(test['Pattern'], pred),
            'macro_f1': f1_score(test['Pattern'], pred, average='macro'),
            'baseline_accuracy': float((test['Pattern'] == majority).mean()),
            'n_train': len(train), 'n_test': len(test)}


def main_split(data):
    cutoff = data['Date'].quantile(0.8)
    return data[data['Date'] < cutoff - GAP], data[data['Date'] >= cutoff], cutoff


def walk_forward(data, cols):
    print("\n" + "=" * 70)
    print("CHECK 1: WALK-FORWARD VALIDATION (several test periods)")
    print("=" * 70)
    rows = []
    for a, b in [(2009, 2010), (2011, 2012), (2013, 2014), (2015, 2017)]:
        start, end = pd.Timestamp(f'{a}-01-01'), pd.Timestamp(f'{b}-12-31')
        train = data[data['Date'] < start - GAP]
        test = data[(data['Date'] >= start) & (data['Date'] <= end)]
        if len(train) < 10_000 or len(test) < 1_000:
            print(f"  {a}-{b}: skipped (not enough data)")
            continue
        r = fit_eval(train, test, cols)
        r['test_period'] = f'{a}-{b}'
        rows.append(r)
        print(f"  Test {a}-{b}: accuracy {r['accuracy']:.3f} | macro-F1 {r['macro_f1']:.3f} "
              f"| baseline {r['baseline_accuracy']:.3f} | train rows {r['n_train']:,} | test rows {r['n_test']:,}")
    df = pd.DataFrame(rows)[['test_period', 'accuracy', 'macro_f1', 'baseline_accuracy', 'n_train', 'n_test']]
    df.to_csv(RESULTS / 'v2_walk_forward.csv', index=False)
    if len(df):
        print(f"\n  Average accuracy {df['accuracy'].mean():.3f} (range {df['accuracy'].min():.3f} to {df['accuracy'].max():.3f}); "
              f"average baseline {df['baseline_accuracy'].mean():.3f}")


def ablation(data, cols):
    print("\n" + "=" * 70)
    print("CHECK 2: ABLATION (how much comes from volatility?)")
    print("=" * 70)
    train, test, cutoff = main_split(data)
    variants = {'All 15 features': cols,
                'Without volatility features': [c for c in cols if c not in VOL],
                'Only volatility features': VOL}
    rows = []
    for name, use in variants.items():
        r = fit_eval(train, test, use)
        r['variant'] = name
        rows.append(r)
        print(f"  {name:30s}: accuracy {r['accuracy']:.3f} | macro-F1 {r['macro_f1']:.3f} | baseline {r['baseline_accuracy']:.3f}")
    pd.DataFrame(rows)[['variant', 'accuracy', 'macro_f1', 'baseline_accuracy']].to_csv(RESULTS / 'v2_ablation.csv', index=False)


def tiers_from(feats):
    X = StandardScaler().fit_transform(feats)
    labels = pd.Series(KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(X), index=feats.index)
    order = feats['Volatility'].groupby(labels).mean().sort_values().index.tolist()
    return labels.map({order[0]: 'Low', order[1]: 'Medium', order[2]: 'High'})


def tier_stability(data):
    print("\n" + "=" * 70)
    print("CHECK 3: DO RISK TIERS HOLD OVER TIME?")
    print("=" * 70)
    train, test, cutoff = main_split(data)
    cols = ['Volatility', 'Momentum', 'RSI', 'Volume_Ratio']
    e, l = train.groupby('Ticker')[cols].mean(), test.groupby('Ticker')[cols].mean()
    ne, nl = train.groupby('Ticker').size(), test.groupby('Ticker').size()
    keep = [t for t in e.index.intersection(l.index) if ne[t] >= 250 and nl[t] >= 100]
    e, l = e.loc[keep], l.loc[keep]
    tier_early, tier_late = tiers_from(e), tiers_from(l)

    summary = l['Volatility'].groupby(tier_early).agg(['count', 'mean']).reindex(['Low', 'Medium', 'High'])
    summary.columns = ['stocks', 'avg_volatility_in_later_period']
    print(f"  Stocks present in both periods: {len(keep)}")
    print("  Tiers built ONLY from the early period; their average volatility in the later (test) period:")
    print(summary.round(4).to_string())
    agree = float((tier_early == tier_late).mean())
    print(f"\n  Same tier when re-clustered on the later period: {agree:.1%} of stocks")
    summary.to_csv(RESULTS / 'v2_tier_stability.csv')


def shap_check(data):
    print("\n" + "=" * 70)
    print("CHECK 4: SHAP ON A LARGER SAMPLE (1,500 rows)")
    print("=" * 70)
    import shap
    with open(MODELS / 'v2_random_forest.pkl', 'rb') as f:
        saved = pickle.load(f)
    model, feats = saved['model'], saved['features']
    _, test, _ = main_split(data)
    X = test[feats].sample(n=min(1500, len(test)), random_state=7)
    sv = shap.TreeExplainer(model).shap_values(X, check_additivity=False)
    if isinstance(sv, list):
        m = np.mean([np.abs(s).mean(0) for s in sv], axis=0)
    else:
        sv = np.asarray(sv)
        m = np.abs(sv).mean(axis=(0, 2)) if sv.ndim == 3 else np.abs(sv).mean(0)
    new = pd.Series(m, index=feats).sort_values(ascending=False)
    old = pd.read_csv(RESULTS / 'v2_feature_importance.csv').set_index('Feature')['Mean_Abs_SHAP']
    rho = new.rank().corr(old.reindex(new.index).rank())
    print("  Top 5 with 1,500 rows :", ', '.join(new.index[:5]))
    print("  Top 5 with 500 rows   :", ', '.join(old.sort_values(ascending=False).index[:5]))
    print(f"  Rank agreement between the two runs (Spearman): {rho:.3f}  (1.0 = identical ranking)")
    print(f"  Volatility importance: {new['Volatility']:.4f}; next feature: {new.iloc[1]:.4f}")
    new.rename('Mean_Abs_SHAP').to_csv(RESULTS / 'v2_shap_check.csv')


if __name__ == '__main__':
    with open(DATA / 'labeled_stocks_sample.pkl', 'rb') as f:
        labeled = pickle.load(f)
    data = build_dataset(labeled)
    cols = [c for c in data.columns if c not in ('Pattern', 'Date', 'Ticker')]
    print(f"Loaded {data['Ticker'].nunique()} stocks, {len(data):,} rows, {len(cols)} features")
    for step in (lambda: walk_forward(data, cols), lambda: ablation(data, cols),
                 lambda: tier_stability(data), lambda: shap_check(data)):
        try:
            step()
        except Exception as e:
            print(f"  This check failed: {e}")
    print("\n" + "=" * 70)
    print("DONE. CSV files saved in results/ (v2_walk_forward, v2_ablation, v2_tier_stability, v2_shap_check)")
    print("=" * 70)
