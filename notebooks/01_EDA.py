"""
Exploratory Data Analysis for Stock Market Data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
import warnings

warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 6)

PROJECT_ROOT = Path(__file__).parent.parent
DATA_PROCESSED = PROJECT_ROOT / 'data' / 'processed'
RESULTS_VIZ = PROJECT_ROOT / 'results' / 'visualizations'
RESULTS_VIZ.mkdir(parents=True, exist_ok=True)


def load_data():
    """Load cleaned stocks"""
    pkl_file = DATA_PROCESSED / 'cleaned_stocks_sample.pkl'
    
    if not pkl_file.exists():
        print(f"File not found: {pkl_file}")
        return None
    
    with open(pkl_file, 'rb') as f:
        cleaned_stocks = pickle.load(f)
    
    print(f"✓ Loaded {len(cleaned_stocks)} stocks")
    return cleaned_stocks


def portfolio_stats(stock_data):
    """Portfolio statistics"""
    print("\n" + "="*70)
    print("PORTFOLIO STATISTICS")
    print("="*70)
    
    lengths = [len(df) for df in stock_data.values()]
    
    print(f"\nNumber of stocks: {len(stock_data)}")
    print(f"Trading days per stock:")
    print(f"  Min:    {min(lengths):,}")
    print(f"  Max:    {max(lengths):,}")
    print(f"  Mean:   {np.mean(lengths):,.0f}")
    print(f"  Median: {np.median(lengths):,.0f}")


def return_analysis(stock_data):
    """Analyze returns"""
    all_returns = []
    close_col = 'Adj Close' if 'Adj Close' in list(stock_data.values())[0].columns else 'Close'
    
    for ticker, df in stock_data.items():
        daily_returns = df[close_col].pct_change().dropna()
        all_returns.extend(daily_returns.values)
    
    all_returns = np.array(all_returns)
    
    print("\n" + "="*70)
    print("DAILY RETURN STATISTICS")
    print("="*70)
    
    print(f"\nTotal returns: {len(all_returns):,}")
    print(f"Mean:    {np.mean(all_returns)*100:+.4f}%")
    print(f"Median:  {np.median(all_returns)*100:+.4f}%")
    print(f"Std Dev: {np.std(all_returns)*100:.4f}%")
    print(f"Min:     {np.min(all_returns)*100:+.2f}%")
    print(f"Max:     {np.max(all_returns)*100:+.2f}%")
    
    return all_returns, close_col


def plot_distributions(stock_data, all_returns, close_col):
    """Create visualizations"""
    lengths = [len(df) for df in stock_data.values()]
    
    # Plot 1: Trading days
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.hist(lengths, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    ax.axvline(np.median(lengths), color='red', linestyle='--', linewidth=2, label=f'Median: {np.median(lengths):.0f}')
    ax.set_xlabel('Trading Days per Stock')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Trading Days per Stock')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_VIZ / '01_trading_days_distribution.png', dpi=100, bbox_inches='tight')
    print("✓ Saved: 01_trading_days_distribution.png")
    plt.close()
    
    # Plot 2: Return distribution
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Histogram
    axes[0, 0].hist(all_returns * 100, bins=100, edgecolor='black', alpha=0.7, color='steelblue')
    axes[0, 0].set_xlabel('Daily Return (%)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of Daily Returns (All)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Zoomed histogram
    q1, q99 = np.percentile(all_returns, [1, 99])
    zoomed = all_returns[(all_returns >= q1) & (all_returns <= q99)]
    axes[0, 1].hist(zoomed * 100, bins=100, edgecolor='black', alpha=0.7, color='darkgreen')
    axes[0, 1].set_xlabel('Daily Return (%)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Distribution of Daily Returns (1-99 Percentile)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Q-Q plot
    from scipy import stats
    stats.probplot(zoomed, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('Q-Q Plot')
    axes[1, 0].grid(True, alpha=0.3)
    
    # CDF
    sorted_ret = np.sort(zoomed)
    axes[1, 1].plot(sorted_ret * 100, np.arange(len(sorted_ret)) / len(sorted_ret), linewidth=2)
    axes[1, 1].set_xlabel('Daily Return (%)')
    axes[1, 1].set_ylabel('Cumulative Probability')
    axes[1, 1].set_title('Cumulative Distribution')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(RESULTS_VIZ / '02_return_distribution.png', dpi=100, bbox_inches='tight')
    print("✓ Saved: 02_return_distribution.png")
    plt.close()
    
    # Plot 3: Sample price series
    tickers = np.random.choice(list(stock_data.keys()), size=4, replace=False)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for i, ticker in enumerate(tickers):
        df = stock_data[ticker]
        axes[i].plot(df.index, df[close_col], linewidth=1, color='steelblue')
        axes[i].set_title(f'{ticker} - {len(df)} trading days')
        axes[i].set_xlabel('Date')
        axes[i].set_ylabel('Close Price ($)')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(RESULTS_VIZ / '03_sample_price_series.png', dpi=100, bbox_inches='tight')
    print("✓ Saved: 03_sample_price_series.png")
    plt.close()
    
    # Plot 4: Volume analysis
    volume_ratios = []
    for ticker, df in stock_data.items():
        if 'Volume' in df.columns:
            vol_ma = df['Volume'].rolling(window=20).mean()
            ratio = df['Volume'] / vol_ma
            volume_ratios.extend(ratio.dropna().values)
    
    volume_ratios = np.array(volume_ratios)
    volume_ratios = volume_ratios[~np.isinf(volume_ratios) & ~np.isnan(volume_ratios)]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].hist(volume_ratios, bins=50, edgecolor='black', alpha=0.7, color='coral')
    axes[0].axvline(1.0, color='red', linestyle='--', linewidth=2, label='Baseline')
    axes[0].set_xlabel('Volume Ratio (Current / 20-day MA)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Distribution of Volume Ratios')
    axes[0].set_xlim([0, 5])
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    stats_text = f"""Volume Stats:
Mean: {np.mean(volume_ratios):.2f}
Median: {np.median(volume_ratios):.2f}
Std: {np.std(volume_ratios):.2f}"""
    axes[1].text(0.5, 0.5, stats_text, fontsize=12, family='monospace')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(RESULTS_VIZ / '04_volume_analysis.png', dpi=100, bbox_inches='tight')
    print("✓ Saved: 04_volume_analysis.png")
    plt.close()


if __name__ == '__main__':
    print("\n" + "="*70)
    print("WEEK 1: EXPLORATORY DATA ANALYSIS")
    print("="*70)
    
    stock_data = load_data()
    if stock_data is None:
        print("\nCould not load data. Run data_cleaner.py first.")
    else:
        portfolio_stats(stock_data)
        all_returns, close_col = return_analysis(stock_data)
        
        print("\n" + "="*70)
        print("GENERATING VISUALIZATIONS")
        print("="*70)
        
        plot_distributions(stock_data, all_returns, close_col)
        
        print(f"\n✓ All visualizations saved to {RESULTS_VIZ}")
        print("\n" + "="*70)
        print("EDA COMPLETE")
        print("="*70)