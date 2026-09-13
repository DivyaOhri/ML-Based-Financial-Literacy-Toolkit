"""
Data Cleaner: Clean stock market data
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import pickle

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_RAW = PROJECT_ROOT / 'data' / 'raw'
DATA_PROCESSED = PROJECT_ROOT / 'data' / 'processed'


class StockDataCleaner:
    """Data cleaning pipeline"""
    
    def __init__(self, min_trading_days=500):
        self.min_trading_days = min_trading_days
    
    def clean_single_stock(self, ticker, df):
        """Clean one stock"""
        status = {'ticker': ticker, 'initial_rows': len(df), 'issues': []}
        
        try:
            # Parse date
            if 'Date' in df.columns:
                df = df.copy()
                df['Date'] = pd.to_datetime(df['Date'])
                df = df.set_index('Date')
            else:
                df.index = pd.to_datetime(df.index)
            
            # Sort & remove duplicates
            df = df.sort_index()
            initial_len = len(df)
            df = df[~df.index.duplicated(keep='first')]
            if len(df) < initial_len:
                status['issues'].append(f"Removed {initial_len - len(df)} duplicates")
            
            # Check columns
            ohlcv_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            ohlcv_present = [c for c in ohlcv_cols if c in df.columns]
            
            if len(ohlcv_present) < 4:
                return None, status
            
            # Remove missing values
            initial_len = len(df)
            df = df.dropna(subset=ohlcv_present)
            if len(df) < initial_len:
                status['issues'].append(f"Removed {initial_len - len(df)} rows with NaN")
            
            # Flag extreme returns
            close_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
            df['Daily_Return'] = df[close_col].pct_change()
            extreme = (df['Daily_Return'].abs() > 0.5).sum()
            if extreme > 0:
                status['issues'].append(f"{extreme} extreme returns (>50%)")
            
            # Check minimum trading days
            if len(df) < self.min_trading_days:
                status['issues'].append(f"Only {len(df)} trading days (min: {self.min_trading_days})")
                return None, status
            
            status['final_rows'] = len(df)
            status['status'] = 'OK'
            return df, status
            
        except Exception as e:
            status['status'] = 'ERROR'
            status['issues'].append(str(e))
            return None, status
    
    def clean_all_stocks(self, stock_data):
        """Clean all stocks"""
        print(f"Cleaning {len(stock_data)} stocks (min {self.min_trading_days} trading days)...\n")
        
        cleaned = {}
        statuses = []
        
        for i, (ticker, df) in enumerate(stock_data.items()):
            cleaned_df, status = self.clean_single_stock(ticker, df)
            statuses.append(status)
            
            if cleaned_df is not None:
                cleaned[ticker] = cleaned_df
            
            if (i + 1) % 50 == 0:
                print(f"Processed {i + 1} stocks...")
        
        summary_df = pd.DataFrame(statuses)
        
        print(f"\n" + "="*70)
        print(f"CLEANING SUMMARY")
        print("="*70)
        print(f"Input:  {len(stock_data)} stocks")
        print(f"Output: {len(cleaned)} stocks")
        print(f"Dropped: {len(stock_data) - len(cleaned)} stocks")
        
        ok_count = (summary_df['status'] == 'OK').sum()
        error_count = (summary_df['status'] == 'ERROR').sum()
        print(f"\n✓ OK:    {ok_count}")
        print(f"✗ ERROR: {error_count}")
        
        return cleaned, summary_df


def save_cleaned_data(cleaned_stocks, filename='cleaned_stocks.pkl'):
    """Save cleaned data"""
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    output_path = DATA_PROCESSED / filename
    
    with open(output_path, 'wb') as f:
        pickle.dump(cleaned_stocks, f)
    
    print(f"\n✓ Saved {len(cleaned_stocks)} stocks to {output_path}")


if __name__ == '__main__':
    pkl_file = DATA_RAW / 'stock_data_sample.pkl'
    
    if not pkl_file.exists():
        print(f"File not found: {pkl_file}")
        print("Run data_loader.py first.")
    else:
        with open(pkl_file, 'rb') as f:
            stock_data = pickle.load(f)
        
        cleaner = StockDataCleaner(min_trading_days=500)
        cleaned_stocks, summary_df = cleaner.clean_all_stocks(stock_data)
        
        save_cleaned_data(cleaned_stocks, 'cleaned_stocks_sample.pkl')
        
        # Save cleaning report
        report_path = DATA_PROCESSED / 'cleaning_report.csv'
        summary_df.to_csv(report_path, index=False)
        print(f"✓ Saved report to {report_path}")