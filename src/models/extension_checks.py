"""
Extension checks (v2) - tries to reduce three more limitations
  A. Volatility is built into the labels -> volatility-scaled labels (threshold = k x stock volatility)
  B. Untuned 3% / 5-day label settings    -> sensitivity to the threshold and the horizon
  C. Unstable risk tiers                  -> compare three tier methods on stability over time

Run:  python src/models/extension_checks.py      (about 10-15 minutes)
"""
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

from train_eval_v2 import scale_free_features, get_dates   # same folder

warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parent.parent.parent
DATA = ROOT / 'data' / 'processed'
RESULTS = ROOT / 'results'
GAP = pd.Timedelta(days=10)
MAX_TRAIN = 300_000
VOL = ['Volatility', 'BB_Width']
CLASSES = ['Bullish', 'Bearish', 'Consolidation', 'Neutral']


def prepare(labeled):
    base = {}
    for t, df in labeled.items():
        try:
            src = df.reset_index(drop=True)
            X = scale_free_features(src)
            X['Date'] = get_dates(df).values
            X['Ticker'] = t
            base[t] = (X, src)
        except Exception as e:
            print(f"  skipped {t}: {e}")
    return base


def relabel(src, h=5, mode='fixed', thr=0.03, k=1.0):
    fwd = src['Close'].pct_change(h).shift(-h)
    limit = thr if mode == 'fixed' else k * src['Volatility'] * np.sqrt(h)
    lab = pd.Series('Neutral', index=src.index, dtype=object)
    lab[fwd > limit] = 'Bullish'
    lab[fwd < -limit] = 'Bearish'
    fv = src['Daily_Return'].rolling(h).std().shift(-h)
    bv = src['Daily_Return'].rolling(20).std()
    lab[(fv < bv * 0.8) & (lab == 'Neutral')] = 'Consolidation'
    lab[fwd.isna()] = np.nan
    return lab


def make_dataset(base, **kw):
    parts = []
    for t, (X, src) in base.items():
        d = X.copy()
        d['Pattern'] = relabel(src, **kw).values
        parts.append(d)
    data = pd.concat(parts, ignore_index=True)
    return data.replace([np.inf, -np.inf], np.nan).dropna()


def fit_eval(train, test, cols):
    tr = train.sample(n=min(len(train), MAX_TRAIN), random_state=42)
    model = RandomForestClassifier(n_estimators=100, max_depth=12, min_samples_leaf=50,
                                   n_jobs=-1, random_state=42).fit(tr[cols], tr['Pattern'])
    pred = model.predict(test[cols])
    majority = tr['Pattern'].value_counts().idxmax()
    return (accuracy_score(test['Pattern'], pred), f1_score(test['Pattern'], pred, average='macro'),
            float((test['Pattern'] == majority).mean()))


def label_experiments(base):
    print("\n" + "=" * 78)
    print("CHECKS A + B: LABEL DESIGN (volatility-scaled labels and sensitivity to settings)")
    print("=" * 78)
    variants = [('Original: fixed 3%, 5 days', dict(h=5, mode='fixed', thr=0.03)),
                ('Fixed 2%, 5 days', dict(h=5, mode='fixed', thr=0.02)),
                ('Fixed 4%, 5 days', dict(h=5, mode='fixed', thr=0.04)),
                ('Fixed 3%, 10 days', dict(h=10, mode='fixed', thr=0.03)),
                ('Volatility-scaled (1 x vol x sqrt(5)), 5 days', dict(h=5, mode='scaled', k=1.0))]
    rows, cutoff = [], None
    for name, kw in variants:
        data = make_dataset(base, **kw)
        cols = [c for c in data.columns if c not in ('Pattern', 'Date', 'Ticker')]
        if cutoff is None:
            cutoff = data['Date'].quantile(0.8)
        train, test = data[data['Date'] < cutoff - GAP], data[data['Date'] >= cutoff]
        mix = data['Pattern'].value_counts(normalize=True) * 100
        acc, f1, bl = fit_eval(train, test, cols)
        rows.append({'variant': name, 'features': 'all 15', 'accuracy': acc, 'macro_f1': f1, 'baseline': bl})
        print(f"\n  {name}")
        print("    class mix %: " + ', '.join(f"{c} {mix.get(c, 0):.1f}" for c in CLASSES))
        print(f"    RF accuracy {acc:.3f} | macro-F1 {f1:.3f} | baseline {bl:.3f} | gain {100 * (acc - bl):+.1f} pts")
        if kw.get('mode') == 'scaled':
            for label, use in (('without volatility features', [c for c in cols if c not in VOL]), ('only volatility features', VOL)):
                a2, f2, b2 = fit_eval(train, test, use)
                rows.append({'variant': name, 'features': label, 'accuracy': a2, 'macro_f1': f2, 'baseline': b2})
                print(f"    {label:28s}: accuracy {a2:.3f} | macro-F1 {f2:.3f} | baseline {b2:.3f}")
    pd.DataFrame(rows).to_csv(RESULTS / 'v2_label_sensitivity.csv', index=False)
    return make_dataset(base, h=5, mode='fixed', thr=0.03), cutoff


def name_by_vol(labels, vol):
    order = vol.groupby(labels).mean().sort_values().index.tolist()
    return labels.map({order[0]: 'Low', order[1]: 'Medium', order[2]: 'High'})


def method_a(f):      # current method: K-Means on 4 averaged features
    X = StandardScaler().fit_transform(f[['Volatility', 'Momentum', 'RSI', 'Volume_Ratio']])
    lab = pd.Series(KMeans(3, random_state=42, n_init=10).fit_predict(X), index=f.index)
    return name_by_vol(lab, f['Volatility'])


def method_b(f):      # K-Means on log volatility only
    X = np.log(f[['Volatility']].clip(lower=1e-6)).values
    lab = pd.Series(KMeans(3, random_state=42, n_init=10).fit_predict(X), index=f.index)
    return name_by_vol(lab, f['Volatility'])


def method_c(f):      # volatility terciles
    q = pd.qcut(f['Volatility'].rank(method='first'), 3, labels=['Low', 'Medium', 'High'])
    return pd.Series(q.astype(str).values, index=f.index)


def tier_experiments(data, cutoff):
    print("\n" + "=" * 78)
    print("CHECK C: WHICH RISK-TIER METHOD IS MOST STABLE OVER TIME?")
    print("=" * 78)
    cols = ['Volatility', 'Momentum', 'RSI', 'Volume_Ratio']
    train, test = data[data['Date'] < cutoff - GAP], data[data['Date'] >= cutoff]
    e, l = train.groupby('Ticker')[cols].mean(), test.groupby('Ticker')[cols].mean()
    ne, nl = train.groupby('Ticker').size(), test.groupby('Ticker').size()
    keep = [t for t in e.index.intersection(l.index) if ne[t] >= 250 and nl[t] >= 100]
    e, l = e.loc[keep], l.loc[keep]
    rho = spearmanr(e['Volatility'], l['Volatility'])[0]
    print(f"  Stocks in both periods: {len(keep)}")
    print(f"  Do volatile stocks stay volatile? Spearman correlation of early vs later volatility: {rho:.3f}")
    allf = data.groupby('Ticker')[cols].mean()
    rows = []
    for name, fn in (('A: K-Means on 4 features (current)', method_a),
                     ('B: K-Means on log volatility only', method_b),
                     ('C: volatility terciles', method_c)):
        te, tl = fn(e), fn(l)
        summ = l['Volatility'].groupby(te).agg(['count', 'mean']).reindex(['Low', 'Medium', 'High'])
        ordered = bool(summ['mean'].is_monotonic_increasing)
        agree = float((te == tl).mean())
        sizes = fn(allf).value_counts().reindex(['Low', 'Medium', 'High'])
        print(f"\n  {name}")
        print("    Early-period tier -> later-period average volatility (stocks):  " +
              ' | '.join(f"{t} {summ.loc[t, 'mean']:.4f} ({int(summ.loc[t, 'count'])})" for t in summ.index))
        print(f"    Tiers ordered correctly later (Low < Medium < High): {'YES' if ordered else 'NO'}")
        print(f"    Same tier when recomputed on the later period: {agree:.1%}")
        print(f"    Tier sizes using all data: Low {sizes['Low']}, Medium {sizes['Medium']}, High {sizes['High']}")
        rows.append({'method': name, 'ordered_correctly': ordered, 'same_tier_pct': agree,
                     'low_later_vol': summ.loc['Low', 'mean'], 'medium_later_vol': summ.loc['Medium', 'mean'],
                     'high_later_vol': summ.loc['High', 'mean'], 'sizes_low': sizes['Low'],
                     'sizes_medium': sizes['Medium'], 'sizes_high': sizes['High']})
    pd.DataFrame(rows).to_csv(RESULTS / 'v2_tier_methods.csv', index=False)


if __name__ == '__main__':
    with open(DATA / 'labeled_stocks_sample.pkl', 'rb') as f:
        labeled = pickle.load(f)
    base = prepare(labeled)
    print(f"Prepared {len(base)} stocks")
    data, cutoff = None, None
    try:
        data, cutoff = label_experiments(base)
    except Exception as e:
        print(f"  label experiments failed: {e}")
    try:
        if data is None:
            data = make_dataset(base, h=5, mode='fixed', thr=0.03)
            cutoff = data['Date'].quantile(0.8)
        tier_experiments(data, cutoff)
    except Exception as e:
        print(f"  tier experiments failed: {e}")
    print("\n" + "=" * 78)
    print("DONE. Saved: results/v2_label_sensitivity.csv and results/v2_tier_methods.csv")
    print("=" * 78)
