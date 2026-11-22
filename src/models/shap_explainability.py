"""
SHAP Explainability: Explain model predictions using SHAP values
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import warnings

import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_PROCESSED = PROJECT_ROOT / 'data' / 'processed'
RESULTS_MODELS = PROJECT_ROOT / 'results' / 'models'
RESULTS_VIZ = PROJECT_ROOT / 'results' / 'visualizations'


def load_models_and_data():
    """Load trained models and test data"""
    
    print("\nLoading models and data...\n")
    
    # Load labeled data
    with open(DATA_PROCESSED / 'labeled_stocks_sample.pkl', 'rb') as f:
        labeled_stocks = pickle.load(f)
    
    # Load Random Forest model
    with open(RESULTS_MODELS / 'randomforest_pattern_classifier.pkl', 'rb') as f:
        rf_model = pickle.load(f)
    
    # Prepare data (same as in model_training.py)
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
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Use subset for SHAP (it's computationally expensive)
    subset_size = min(500, len(X_test) // 10)
    X_subset = X_test[:subset_size]
    
    return rf_model, X_subset, feature_cols


def analyze_feature_importance(model, X, feature_cols, model_name):
    """Analyze feature importance using SHAP"""
    
    print(f"\n{'='*70}")
    print(f"SHAP ANALYSIS: {model_name}")
    print(f"{'='*70}\n")
    
    try:
        # Create SHAP explainer
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        
        print(f"✓ Computed SHAP values for {len(X)} samples")
        
        # Feature importance (mean absolute SHAP values)
        if isinstance(shap_values, list):
            # Multi-class: average across classes
            mean_abs_shap = np.mean([np.abs(sv).mean(0) for sv in shap_values], axis=0)
        else:
            mean_abs_shap = np.abs(shap_values).mean(0)
        
        # Sort by importance
        feature_importance = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': mean_abs_shap
        }).sort_values('Importance', ascending=False)
        
        print("\nTop 15 Most Important Features:")
        print("-" * 70)
        for idx, row in feature_importance.head(15).iterrows():
            print(f"  {row['Feature']:20s}: {row['Importance']:.6f}")
        
        # Save importance to CSV
        importance_path = PROJECT_ROOT / 'results' / 'feature_importance.csv'
        feature_importance.to_csv(importance_path, index=False)
        print(f"\n✓ Saved: feature_importance.csv")
        
        return explainer, shap_values, feature_importance
        
    except Exception as e:
        print(f"Error in SHAP analysis: {e}")
        return None, None, None


def plot_shap_summary(shap_values, X, feature_cols, model_name):
    """Plot SHAP summary visualization"""
    
    try:
        if isinstance(shap_values, list):
            # Multi-class: use first class
            sv = shap_values[0]
        else:
            sv = shap_values
        
        # Summary plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(sv, X, feature_names=feature_cols, show=False)
        plt.title(f'SHAP Summary Plot: {model_name}')
        plt.tight_layout()
        plt.savefig(RESULTS_VIZ / f'shap_summary_{model_name.lower().replace(" ", "_")}.png', 
                   dpi=100, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved: shap_summary_{model_name.lower().replace(' ', '_')}.png")
        
    except Exception as e:
        print(f"Could not create SHAP plot: {e}")


def analyze_feature_distribution(rf_model, feature_cols):
    """Analyze feature importances from Random Forest"""
    
    print(f"\n{'='*70}")
    print(f"FEATURE IMPORTANCE (Random Forest)")
    print(f"{'='*70}\n")
    
    importances = rf_model.feature_importances_
    feature_imp = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    print("Top 15 Features by Importance:")
    print("-" * 70)
    for idx, row in feature_imp.head(15).iterrows():
        print(f"  {row['Feature']:20s}: {row['Importance']:.6f}")
    
    # Plot
    plt.figure(figsize=(10, 6))
    top_features = feature_imp.head(15)
    plt.barh(range(len(top_features)), top_features['Importance'].values)
    plt.yticks(range(len(top_features)), top_features['Feature'].values)
    plt.xlabel('Importance')
    plt.title('Top 15 Features by Importance (Random Forest)')
    plt.tight_layout()
    plt.savefig(RESULTS_VIZ / 'feature_importance_barplot.png', dpi=100, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Saved: feature_importance_barplot.png")


if __name__ == '__main__':
    print("\n" + "="*70)
    print("WEEK 3: SHAP EXPLAINABILITY ANALYSIS")
    print("="*70)
    
    # Load models and data
    rf_model, X_subset, feature_cols = load_models_and_data()
    
    # Analyze Random Forest
    rf_explainer, rf_shap, rf_importance = analyze_feature_importance(
        rf_model, X_subset, feature_cols, "Random Forest"
    )
    
    # Plot Random Forest SHAP
    if rf_shap is not None:
        plot_shap_summary(rf_shap, X_subset, feature_cols, "Random Forest")
    
    # Plot feature importance
    analyze_feature_distribution(rf_model, feature_cols)
    
    print("\n" + "="*70)
    print("SHAP ANALYSIS COMPLETE")
    print("="*70)
    print("\nVisualizations saved to: results/visualizations/")