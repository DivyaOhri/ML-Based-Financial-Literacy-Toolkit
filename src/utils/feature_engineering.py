"""
Feature Engineering: Compute technical indicators from cleaned stock data
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import warnings

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_PROCESSED = PROJECT_ROOT / 'data' / 'processed'


def compute_technical_indicators(df, close_col='Close'):
    """
    Compute technical indicators for a stock
    
    Indicators:
    - SMA: Simple Moving Averages (5, 10, 20, 50 days)
    - RSI: Relative Strength Index (14 days)
    - MACD: Moving Average Convergence Divergence
    - Bollinger Bands: Upper and lower bands (20 days, 2 std)
    - Momentum: Price momentum (10 days)
    - Volatility: Rolling standard deviation (20 days)
    - Volume Ratio: Current volume vs 20-day MA
    """
    
    df = df.copy()
    
    # Daily returns
    df['Daily_Return'] = df[close_col].pct_change()
    
    # Simple Moving Averages
    df['SMA_5'] = df[close_col].rolling(window=5).mean()
    df['SMA_10'] = df[close_col].rolling(window=10).mean()
    df['SMA_20'] = df[close_col].rolling(window=20).mean()
    df['SMA_50'] = df[close_col].rolling(window=50).mean()
    
    # RSI (Relative Strength Index)
    delta = df[close_col].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema_12 = df[close_col].ewm(span=12, adjust=False).mean()
    ema_26 = df[close_col].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
    
    # Bollinger Bands
    bb_sma = df[close_col].rolling(window=20).mean()
    bb_std = df[close_col].rolling(window=20).std()
    df['BB_Upper'] = bb_sma + (bb_std * 2)
    df['BB_Lower'] = bb_sma - (bb_std * 2)
    df['BB_Position'] = (df[close_col] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
    
    # Momentum
    df['Momentum'] = df[close_col].pct_change(periods=10)
    
    # Volatility
    df['Volatility'] = df['Daily_Return'].rolling(window=20).std()
    
    # Volume Ratio
    if 'Volume' in df.columns:
        volume_ma = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / volume_ma
    
    return df


def engineer_features(cleaned_stocks):
    """
    Engineer features for all stocks
    
    Returns:
        dict: {ticker: DataFrame with engineered features}
    """
    
    print(f"\nEngineering features for {len(cleaned_stocks)} stocks...\n")
    
    featured_stocks = {}
    
    for i, (ticker, df) in enumerate(cleaned_stocks.items()):
        try:
            # Compute indicators
            featured_df = compute_technical_indicators(df)
            
            # Remove rows with NaN values (from rolling calculations)
            featured_df = featured_df.dropna()
            
            if len(featured_df) > 0:
                featured_stocks[ticker] = featured_df
            
            if (i + 1) % 500 == 0:
                print(f"  Engineered features for {i + 1} stocks...")
                
        except Exception as e:
            print(f"  Error processing {ticker}: {e}")
            continue
    
    print(f"\n✓ Successfully engineered features for {len(featured_stocks)} stocks")
    return featured_stocks


def save_featured_data(featured_stocks, filename='featured_stocks_sample.pkl'):
    """Save featured stocks to pickle"""
    output_path = DATA_PROCESSED / filename
    
    with open(output_path, 'wb') as f:
        pickle.dump(featured_stocks, f)
    
    print(f"✓ Saved to {output_path}")


if __name__ == '__main__':
    print("\n" + "="*70)
    print("WEEK 2: FEATURE ENGINEERING")
    print("="*70)
    
    # Load cleaned data
    pkl_file = DATA_PROCESSED / 'cleaned_stocks_sample.pkl'
    
    if not pkl_file.exists():
        print(f"File not found: {pkl_file}")
        print("Run Week 1 pipeline first!")
    else:
        with open(pkl_file, 'rb') as f:
            cleaned_stocks = pickle.load(f)
        
        # Engineer features
        featured_stocks = engineer_features(cleaned_stocks)
        
        # Save
        save_featured_data(featured_stocks, 'featured_stocks_sample.pkl')
        
        print("\n" + "="*70)
        print("FEATURE ENGINEERING COMPLETE")
        print("="*70)
        
        # Sample statistics
        sample_ticker = list(featured_stocks.keys())[0]
        sample_df = featured_stocks[sample_ticker]
        
        print(f"\nSample stock: {sample_ticker}")
        print(f"Shape: {sample_df.shape}")
        print(f"Features created: {list(sample_df.columns)}")