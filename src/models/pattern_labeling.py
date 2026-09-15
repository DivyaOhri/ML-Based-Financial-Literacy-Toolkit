"""
Pattern Labeling: Create ground-truth labels for supervised learning
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import warnings

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_PROCESSED = PROJECT_ROOT / 'data' / 'processed'


def create_pattern_labels(df, close_col='Close', forward_days=5, threshold=0.03):
    """
    Create pattern labels based on forward returns
    
    Labels:
    - Bullish: forward return > +3%
    - Bearish: forward return < -3%
    - Neutral: forward return between -3% and +3%
    - Consolidation: low forward volatility
    """
    
    # Calculate forward returns
    forward_return = df[close_col].pct_change(periods=forward_days).shift(-forward_days)
    
    # Initialize labels
    labels = pd.Series('Neutral', index=df.index)
    
    # Bullish: forward return > threshold
    labels[forward_return > threshold] = 'Bullish'
    
    # Bearish: forward return < -threshold
    labels[forward_return < -threshold] = 'Bearish'
    
    # Consolidation: low forward volatility
    if 'Daily_Return' in df.columns:
        forward_volatility = df['Daily_Return'].rolling(window=forward_days).std().shift(-forward_days)
        baseline_volatility = df['Daily_Return'].rolling(window=20).std()
        consolidation_mask = (forward_volatility < baseline_volatility * 0.8) & (labels == 'Neutral')
        labels[consolidation_mask] = 'Consolidation'
    
    return labels


def label_all_stocks(featured_stocks):
    """
    Create labels for all stocks
    
    Returns:
        dict: {ticker: DataFrame with labels}
    """
    
    print(f"\nCreating pattern labels for {len(featured_stocks)} stocks...\n")
    
    labeled_stocks = {}
    
    for i, (ticker, df) in enumerate(featured_stocks.items()):
        try:
            # Create labels
            df_copy = df.copy()
            df_copy['Pattern'] = create_pattern_labels(df_copy)
            
            # Remove rows with NaN labels (forward-looking rows at end)
            df_copy = df_copy.dropna(subset=['Pattern'])
            
            if len(df_copy) > 0:
                labeled_stocks[ticker] = df_copy
            
            if (i + 1) % 500 == 0:
                print(f"  Labeled {i + 1} stocks...")
                
        except Exception as e:
            print(f"  Error labeling {ticker}: {e}")
            continue
    
    print(f"\n✓ Successfully labeled {len(labeled_stocks)} stocks")
    return labeled_stocks


def save_labeled_data(labeled_stocks, filename='labeled_stocks_sample.pkl'):
    """Save labeled stocks to pickle"""
    output_path = DATA_PROCESSED / filename
    
    with open(output_path, 'wb') as f:
        pickle.dump(labeled_stocks, f)
    
    print(f"✓ Saved to {output_path}")


def analyze_label_distribution(labeled_stocks):
    """Analyze class distribution"""
    print("\n" + "="*70)
    print("LABEL DISTRIBUTION ANALYSIS")
    print("="*70)
    
    all_labels = []
    
    for ticker, df in labeled_stocks.items():
        all_labels.extend(df['Pattern'].values)
    
    label_counts = pd.Series(all_labels).value_counts()
    label_pct = (label_counts / len(all_labels) * 100).round(2)
    
    print(f"\nTotal samples: {len(all_labels):,}")
    print(f"\nPattern distribution:")
    for label in ['Bullish', 'Neutral', 'Bearish', 'Consolidation']:
        if label in label_counts.index:
            count = label_counts[label]
            pct = label_pct[label]
            print(f"  {label:15s}: {count:7,} ({pct:5.2f}%)")


if __name__ == '__main__':
    print("\n" + "="*70)
    print("WEEK 2: PATTERN LABELING")
    print("="*70)
    
    # Load featured data
    pkl_file = DATA_PROCESSED / 'featured_stocks_sample.pkl'
    
    if not pkl_file.exists():
        print(f"File not found: {pkl_file}")
        print("Run feature_engineering.py first!")
    else:
        with open(pkl_file, 'rb') as f:
            featured_stocks = pickle.load(f)
        
        # Create labels
        labeled_stocks = label_all_stocks(featured_stocks)
        
        # Analyze distribution
        analyze_label_distribution(labeled_stocks)
        
        # Save
        save_labeled_data(labeled_stocks, 'labeled_stocks_sample.pkl')
        
        print("\n" + "="*70)
        print("PATTERN LABELING COMPLETE")
        print("="*70)