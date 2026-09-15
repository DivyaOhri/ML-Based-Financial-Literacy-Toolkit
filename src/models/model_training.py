"""
Model Training: Train classifiers and clusterers
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import warnings

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_PROCESSED = PROJECT_ROOT / 'data' / 'processed'
RESULTS_MODELS = PROJECT_ROOT / 'results' / 'models'
RESULTS_VIZ = PROJECT_ROOT / 'results' / 'visualizations'
RESULTS_MODELS.mkdir(parents=True, exist_ok=True)


def prepare_classification_data(labeled_stocks):
    """
    Prepare data for pattern classification
    
    Returns:
        X (features), y (labels), feature_names
    """
    
    print("\nPreparing classification data...\n")
    
    # Feature columns (exclude Date, OHLCV, and Pattern)
    exclude_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close', 
                   'Daily_Return', 'Pattern']
    
    all_features = []
    all_labels = []
    
    for ticker, df in labeled_stocks.items():
        feature_cols = [c for c in df.columns if c not in exclude_cols]
        
        X = df[feature_cols].values
        y = df['Pattern'].values
        
        all_features.append(X)
        all_labels.extend(y)
    
    X = np.vstack(all_features)
    y = np.array(all_labels)
    
    print(f"Total samples: {len(X):,}")
    print(f"Total features: {X.shape[1]}")
    print(f"Feature columns: {feature_cols}")
    
    return X, y, feature_cols


def train_pattern_classifier(X, y, feature_names):
    """
    Train Random Forest and Gradient Boosting classifiers
    
    Returns:
        dict: {model_name: (model, X_test, y_test)}
    """
    
    print("\n" + "="*70)
    print("TRAINING PATTERN CLASSIFIERS")
    print("="*70)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTrain set: {len(X_train):,} samples")
    print(f"Test set:  {len(X_test):,} samples")
    
    models = {}
    
    # Random Forest
    print("\n[1/2] Training Random Forest Classifier...")
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    rf_score = rf_model.score(X_test, y_test)
    print(f"  ✓ Accuracy: {rf_score:.4f}")
    models['RandomForest'] = (rf_model, X_test, y_test)
    
    # Gradient Boosting
    print("\n[2/2] Training Gradient Boosting Classifier...")
    gb_model = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
    gb_model.fit(X_train, y_train)
    gb_score = gb_model.score(X_test, y_test)
    print(f"  ✓ Accuracy: {gb_score:.4f}")
    models['GradientBoosting'] = (gb_model, X_test, y_test)
    
    # Save models
    for name, (model, _, _) in models.items():
        model_path = RESULTS_MODELS / f'{name.lower()}_pattern_classifier.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"  ✓ Saved: {model_path}")
    
    return models


def evaluate_classifier(model, X_test, y_test, model_name):
    """Evaluate classifier and print metrics"""
    
    print(f"\n{model_name} - EVALUATION")
    print("-" * 70)
    
    y_pred = model.predict(X_test)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Feature importance (for tree-based models)
    if hasattr(model, 'feature_importances_'):
        top_features_idx = np.argsort(model.feature_importances_)[-10:]
        print("\nTop 10 Important Features:")
        for idx in reversed(top_features_idx):
            print(f"  Feature {idx}: {model.feature_importances_[idx]:.4f}")


def train_risk_clusterer(labeled_stocks):
    """
    Train K-Means clusterer for risk tier segmentation
    
    Features: volatility, momentum, RSI, volume ratio
    """
    
    print("\n" + "="*70)
    print("TRAINING RISK TIER CLUSTERER")
    print("="*70)
    
    # Aggregate features per stock
    stock_features = []
    tickers = []
    
    for ticker, df in labeled_stocks.items():
        try:
            features = {
                'ticker': ticker,
                'Volatility_mean': df['Volatility'].mean(),
                'Momentum_mean': df['Momentum'].mean(),
                'RSI_mean': df['RSI'].mean(),
                'Volume_Ratio_mean': df['Volume_Ratio'].mean() if 'Volume_Ratio' in df.columns else 1.0,
            }
            stock_features.append(features)
            tickers.append(ticker)
        except:
            continue
    
    features_df = pd.DataFrame(stock_features)
    X = features_df[[c for c in features_df.columns if c != 'ticker']].values
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"\nClustering {len(X)} stocks...")
    
    # K-Means with 3 clusters (low/medium/high risk)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    # Silhouette score
    sil_score = silhouette_score(X_scaled, clusters)
    print(f"Silhouette Score: {sil_score:.4f}")
    
    # Cluster distribution
    print("\nCluster Distribution:")
    unique, counts = np.unique(clusters, return_counts=True)
    for cluster_id, count in zip(unique, counts):
        print(f"  Cluster {cluster_id}: {count} stocks")
    
    # Save model
    model_path = RESULTS_MODELS / 'kmeans_risk_clusterer.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump({'model': kmeans, 'scaler': scaler, 'tickers': tickers}, f)
    print(f"\n✓ Saved: {model_path}")
    
    return kmeans, scaler


if __name__ == '__main__':
    print("\n" + "="*70)
    print("WEEK 2: MODEL TRAINING")
    print("="*70)
    
    # Load labeled data
    pkl_file = DATA_PROCESSED / 'labeled_stocks_sample.pkl'
    
    if not pkl_file.exists():
        print(f"File not found: {pkl_file}")
        print("Run pattern_labeling.py first!")
    else:
        with open(pkl_file, 'rb') as f:
            labeled_stocks = pickle.load(f)
        
        # Prepare data
        X, y, feature_names = prepare_classification_data(labeled_stocks)
        
        # Train classifiers
        models = train_pattern_classifier(X, y, feature_names)
        
        # Evaluate models
        for model_name, (model, X_test, y_test) in models.items():
            evaluate_classifier(model, X_test, y_test, model_name)
        
        # Train risk clusterer
        kmeans, scaler = train_risk_clusterer(labeled_stocks)
        
        print("\n" + "="*70)
        print("MODEL TRAINING COMPLETE")
        print("="*70)
        print("\nModels saved to: results/models/")