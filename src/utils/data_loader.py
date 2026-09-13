"""
Data Loader: Download and load Kaggle stock market dataset
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_RAW = PROJECT_ROOT / 'data' / 'raw'
DATA_PROCESSED = PROJECT_ROOT / 'data' / 'processed'


def download_kaggle_dataset():
    """Download Kaggle 'Huge Stock Market Dataset'"""
    import subprocess
    
    print("Downloading Kaggle dataset...")
    print("Prerequisites: pip install kaggle, API token in ~/.kaggle/kaggle.json\n")
    
    try:
        os.makedirs(DATA_RAW, exist_ok=True)
        
        cmd = f"kaggle datasets download -d borismarjanovic/price-volume-data-for-all-us-stocks-etfs -p {DATA_RAW}"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✓ Dataset downloaded successfully!")
            
            import zipfile
            zip_file = DATA_RAW / 'price-volume-data-for-all-us-stocks-etfs.zip'
            if zip_file.exists():
                print("Extracting files...")
                with zipfile.ZipFile(zip_file, 'r') as z:
                    z.extractall(DATA_RAW)
                print("✓ Files extracted!")
        else:
            print("✗ Download failed. Check API credentials.")
            print(result.stderr)
            
    except Exception as e:
        print(f"Error: {e}")


def load_csv_files(limit=None):
    """Load all CSV files from data/raw/stocks"""
    stocks_dir = DATA_RAW / 'stocks'
    
    if not stocks_dir.exists():
        print(f"Directory not found: {stocks_dir}")
        return {}
    
    csv_files = sorted(stocks_dir.glob('*.txt'))
    if limit:
        csv_files = csv_files[:limit]
    
    print(f"Loading {len(csv_files)} CSV files...")
    
    stock_data = {}
    failed = []
    
    for i, csv_file in enumerate(csv_files):
        try:
            ticker = csv_file.stem
            df = pd.read_csv(csv_file)
            stock_data[ticker] = df
            
            if (i + 1) % 50 == 0:
                print(f"  Loaded {i + 1} files...")
                
        except Exception as e:
            failed.append((csv_file.name, str(e)))
    
    print(f"✓ Loaded {len(stock_data)} stocks successfully")
    
    if failed:
        print(f"⚠ Failed to load {len(failed)} files")
        for fname, err in failed[:3]:
            print(f"    {fname}: {err}")
    
    return stock_data


def inspect_raw_data(stock_data):
    """Inspect raw data structure"""
    print("\n" + "="*60)
    print("RAW DATA INSPECTION")
    print("="*60)
    
    if not stock_data:
        print("No data loaded!")
        return
    
    sample_ticker = list(stock_data.keys())[0]
    sample_df = stock_data[sample_ticker]
    
    print(f"\nSample ticker: {sample_ticker}")
    print(f"Shape: {sample_df.shape}")
    print(f"Columns: {list(sample_df.columns)}")
    print(f"\nFirst 3 rows:\n{sample_df.head(3)}")
    print(f"\nMissing values:\n{sample_df.isnull().sum()}")
    
    lengths = [len(df) for df in stock_data.values()]
    print(f"\nPortfolio stats ({len(stock_data)} stocks):")
    print(f"  Trading days: {min(lengths)}-{max(lengths)} (median: {np.median(lengths):.0f})")


if __name__ == '__main__':
    print("Loading stock data...")
    stock_data = load_csv_files(limit=50)  # Start with 50 for testing
    
    inspect_raw_data(stock_data)
    
    # Save pickle for fast reload
    import pickle
    pickle_file = DATA_RAW / 'stock_data_sample.pkl'
    with open(pickle_file, 'wb') as f:
        pickle.dump(stock_data, f)
    print(f"\n✓ Data saved to {pickle_file}")
