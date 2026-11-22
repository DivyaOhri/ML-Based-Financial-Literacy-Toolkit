"""
Leakage-free training & evaluation (v2)
- Scale-free features only
- Time-based split with a purge gap
- Baseline comparison
- Risk tiers ordered by volatility
- SHAP on the Random Forest
"""

import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, silhouette_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parent.parent.parent
DATA = ROOT / 'data' / 'processed'
MODELS = ROOT / 'results' / 'models'
VIZ = ROOT / 'results' / 'visualizations'
RESULTS = ROOT / 'results'
MODELS.mkdir(parents=True, exist_ok=True)
VIZ.mkdir(parents=True, exist_ok=True)


def get_dates(df):
    if 'Date' in df.columns:
        return pd.to_datetime(df['Date'])
    return pd.to_datetime(df.index)


def scale_free_features(df):
    c = df['Close']
    out = pd.DataFrame(index=df.index)
    out['RSI'] = df['RSI']
    out['BB_Position'] = df['BB_Position']
    out['Momentum'] = df['Momentum']
    out['Volatility'] = df['Volatility']
    out['Volume_Ratio'] = df['Volume_Ratio']
    for w in (5, 10, 20, 50):
        out[f'Close_vs_SMA{w}'] = c / df[f'SMA_{w}'] - 1
    out['SMA5_vs_SMA20'] = df['SMA_5'] / df['SMA_20'] - 1
    out['SMA20_vs_SMA50'] = df['SMA_20'] / df['SMA_50'] - 1
    out['MACD_pct'] = df['MACD'] / c
    out['MACD_Signal_pct'] = df['MACD_Signal'] / c
    out['MACD_Hist_pct'] = df['MACD_Histogram'] / c
    out['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['SMA_20']
    return out


def build_dataset(labeled):
    parts = []
    for ticker, df in labeled.items():
        try:
            X = scale_free_features(df)
            X['Pattern'] = df['Pattern'].values
            X['Date'] = get_dates(df).values
            X['Ticker'] = ticker
            parts.append(X.reset_index(drop=True))
        except Exception as e:
            print(f"  skipped {ticker}: {e}")
    data = pd.concat(parts, ignore_index=True)
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    return data


def plot_cm(y_true, y_pred, classes, name):
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(f'Confusion Matrix (time-based test): {name}')
    plt.ylabel('True'); plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(VIZ / f'v2_confusion_{name.lower().replace(" ", "_")}.png', dpi=100)
    plt.close()


def run_classification(labeled):
    print("\n" + "=" * 70)
    print("CLASSIFICATION (time-based split, scale-free features)")
    print("=" * 70)

    data = build_dataset(labeled)
    feature_cols = [c for c in data.columns if c not in ('Pattern', 'Date', 'Ticker')]

    cutoff = data['Date'].quantile(0.8)
    gap = pd.Timedelta(days=10)
    train = data[data['Date'] < cutoff - gap]
    test = data[data['Date'] >= cutoff]

    print(f"\nTotal rows: {len(data):,}")
    print(f"Train: {len(train):,} rows ({train['Date'].min().date()} to {train['Date'].max().date()})")
    print(f"Test:  {len(test):,} rows ({test['Date'].min().date()} to {test['Date'].max().date()})")
    print(f"Features ({len(feature_cols)}): {feature_cols}")
    print("\nTest label distribution:")
    print((test['Pattern'].value_counts(normalize=True) * 100).round(1).to_string())

    X_tr, y_tr = train[feature_cols], train['Pattern']
    X_te, y_te = test[feature_cols], test['Pattern']

    models = {
        'Baseline (most common)': DummyClassifier(strategy='most_frequent'),
        'Random Forest': RandomForestClassifier(
            n_estimators=200, max_depth=12, min_samples_leaf=50, n_jobs=-1, random_state=42),
        'Gradient Boosting': HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.05, max_depth=6, random_state=42),
    }

    summary = []
    fitted = {}
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)
        acc = accuracy_score(y_te, pred)
        f1 = f1_score(y_te, pred, average='macro')
        print(f"  Accuracy: {acc:.4f} | Macro-F1: {f1:.4f}")
        summary.append((name, acc, f1))
        fitted[name] = model
        if name != 'Baseline (most common)':
            print(classification_report(y_te, pred))
            plot_cm(y_te, pred, sorted(y_te.unique()), name)
            with open(MODELS / f'v2_{name.lower().replace(" ", "_")}.pkl', 'wb') as f:
                pickle.dump({'model': model, 'features': feature_cols}, f)

    pd.DataFrame(summary, columns=['Model', 'Accuracy', 'Macro_F1']).to_csv(
        RESULTS / 'v2_classification_summary.csv', index=False)

    return fitted['Random Forest'], X_te, feature_cols


def run_shap(rf, X_te, feature_cols):
    print("\n" + "=" * 70)
    print("SHAP (Random Forest)")
    print("=" * 70)
    try:
        import shap
        X_s = X_te.sample(n=min(500, len(X_te)), random_state=42)
        explainer = shap.TreeExplainer(rf)
        sv = explainer.shap_values(X_s, check_additivity=False)

        if isinstance(sv, list):
            mean_abs = np.mean([np.abs(s).mean(0) for s in sv], axis=0)
        else:
            sv = np.asarray(sv)
            mean_abs = np.abs(sv).mean(axis=(0, 2)) if sv.ndim == 3 else np.abs(sv).mean(0)

        imp = pd.DataFrame({'Feature': feature_cols, 'Mean_Abs_SHAP': mean_abs}) \
                .sort_values('Mean_Abs_SHAP', ascending=False)
        imp.to_csv(RESULTS / 'v2_feature_importance.csv', index=False)
        print(imp.to_string(index=False))

        plt.figure(figsize=(9, 6))
        top = imp.head(12).iloc[::-1]
        plt.barh(top['Feature'], top['Mean_Abs_SHAP'])
        plt.xlabel('Mean |SHAP value| (average over classes)')
        plt.title('Which indicators drive the Random Forest predictions')
        plt.tight_layout()
        plt.savefig(VIZ / 'v2_shap_importance.png', dpi=100)
        plt.close()
        print("\nSaved: v2_shap_importance.png")
    except Exception as e:
        print(f"SHAP failed: {e}")


def run_clustering(labeled):
    print("\n" + "=" * 70)
    print("RISK TIERS (K-Means, ordered by volatility)")
    print("=" * 70)

    rows = []
    for t, df in labeled.items():
        rows.append({'ticker': t,
                     'Volatility': df['Volatility'].mean(),
                     'Momentum': df['Momentum'].mean(),
                     'RSI': df['RSI'].mean(),
                     'Volume_Ratio': df['Volume_Ratio'].mean()})
    sf = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan).dropna()
    cols = ['Volatility', 'Momentum', 'RSI', 'Volume_Ratio']

    Xs = StandardScaler().fit_transform(sf[cols])
    km = KMeans(n_clusters=3, random_state=42, n_init=10).fit(Xs)
    sf['cluster'] = km.labels_

    order = sf.groupby('cluster')['Volatility'].mean().sort_values().index.tolist()
    names = {order[0]: 'Low Risk', order[1]: 'Medium Risk', order[2]: 'High Risk'}
    sf['Risk_Tier'] = sf['cluster'].map(names)

    print(f"\nStocks clustered: {len(sf)}")
    print(f"Silhouette score: {silhouette_score(Xs, km.labels_):.4f}")
    print("\nTier sizes:")
    print(sf['Risk_Tier'].value_counts().to_string())
    print("\nAverage features per tier:")
    print(sf.groupby('Risk_Tier')[cols].mean().round(4).to_string())

    print("\nSanity check on well-known tickers:")
    low = sf['ticker'].astype(str).str.lower()
    for name in ['aapl', 'msft', 'jnj', 'ko', 'tsla']:
        m = sf[(low == name) | low.str.startswith(name + '.')]
        for _, r in m.iterrows():
            print(f"  {r['ticker']}: {r['Risk_Tier']} (volatility {r['Volatility']:.4f})")

    sf.to_csv(RESULTS / 'v2_risk_tiers.csv', index=False)


if __name__ == '__main__':
    with open(DATA / 'labeled_stocks_sample.pkl', 'rb') as f:
        labeled = pickle.load(f)
    print(f"Loaded {len(labeled)} stocks")

    rf, X_te, feature_cols = run_classification(labeled)
    run_shap(rf, X_te, feature_cols)
    run_clustering(labeled)

    print("\n" + "=" * 70)
    print("DONE. Outputs in results/ and results/visualizations/")
    print("=" * 70)