"""
Asset Class Performance Breakdown
=================================
Runs separate backtests for each asset class to see true performance by segment.
"""

import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quant_backtest import (
    Config, DataLoader, FeatureEngine, RegimeEngine, 
    AlphaEngine, EnsembleSignal, CrisisAlphaEngine, Backtester
)

def run_asset_class_test(symbols, asset_class_name, start_date, end_date):
    """Run backtest for specific asset class."""
    print(f"\n{'='*60}")
    print(f"TESTING: {asset_class_name}")
    print(f"Symbols: {symbols}")
    print(f"{'='*60}")
    
    config = Config()
    config.symbols = symbols
    
    loader = DataLoader(config)
    try:
        data = loader.load_data(start_date, end_date)
    except Exception as e:
        print(f"  Error loading data: {e}")
        return None
    
    if not data:
        print("  No data found.")
        return None
    
    # Process
    fe = FeatureEngine(config)
    re = RegimeEngine(config)
    ae = AlphaEngine(config)
    es = EnsembleSignal(config)
    ce = CrisisAlphaEngine(config)
    
    data = fe.add_features_all(data)
    data = re.add_regimes_all(data)
    ae.train_model(data)
    data = ae.add_signals_all(data)
    data = es.add_ensemble_all(data)
    data = ce.add_crisis_signals(data)
    
    bt = Backtester(config)
    equity_curve = bt.run_backtest(data)
    trades = pd.DataFrame(bt.account.trade_history)
    
    # Calculate metrics
    if trades.empty:
        return {
            'Asset Class': asset_class_name,
            'Symbols': len(symbols),
            'Trades': 0,
            'Total PnL': 0,
            'Return %': 0,
            'Win Rate': 0,
            'Mean R': 0,
            'Sharpe': 0
        }
    
    total_pnl = trades['PnL'].sum()
    win_rate = (trades['PnL'] > 0).mean()
    mean_r = trades['R_Multiple'].mean()
    
    # Calculate Sharpe
    trades['Entry Time'] = pd.to_datetime(trades['Entry Time'])
    trades['Exit Time'] = pd.to_datetime(trades['Exit Time'])
    trading_days = (trades['Exit Time'].max() - trades['Entry Time'].min()).days
    trades_per_year = len(trades) / max(trading_days, 1) * 252
    
    if trades['PnL'].std() > 0:
        sharpe = trades['PnL'].mean() / trades['PnL'].std() * np.sqrt(trades_per_year)
    else:
        sharpe = 0
    
    return {
        'Asset Class': asset_class_name,
        'Symbols': len(symbols),
        'Trades': len(trades),
        'Total PnL': total_pnl,
        'Return %': total_pnl / 100000 * 100,  # Assuming $100k starting
        'Win Rate': win_rate * 100,
        'Mean R': mean_r,
        'Sharpe': sharpe
    }

def main():
    print("="*60)
    print("ASSET CLASS PERFORMANCE BREAKDOWN")
    print("With Realistic Transaction Costs")
    print("="*60)
    
    # Define asset classes
    fx_majors = ["EURUSD=X", "USDJPY=X", "GBPUSD=X", "USDCHF=X", "USDCAD=X", "AUDUSD=X", "NZDUSD=X"]
    fx_crosses = ["EURGBP=X", "EURJPY=X", "GBPJPY=X", "AUDJPY=X", "EURAUD=X", "EURCHF=X"]
    
    indices = ["ES=F", "NQ=F", "YM=F", "RTY=F"]  # S&P, Nasdaq, Dow, Russell
    
    commodities = ["GC=F", "CL=F", "NG=F"]  # Gold, Oil, NatGas
    
    crypto = ["BTC-USD", "ETH-USD"]
    
    # Date range (use available 730-day window)
    start_date = "2024-01-01"
    end_date = "2024-12-01"
    
    results = []
    
    # Run each asset class
    for symbols, name in [
        (fx_majors, "FX Majors"),
        (fx_crosses, "FX Crosses"),
        (indices, "Equity Indices"),
        (commodities, "Commodities"),
        (crypto, "Crypto"),
    ]:
        result = run_asset_class_test(symbols, name, start_date, end_date)
        if result:
            results.append(result)
    
    # Summary table
    print("\n" + "="*80)
    print("PERFORMANCE BY ASSET CLASS (2024 YTD with Realistic Costs)")
    print("="*80)
    
    df = pd.DataFrame(results)
    
    print(f"\n{'Asset Class':<15} | {'Symbols':<8} | {'Trades':<7} | {'PnL':>10} | {'Return %':>9} | {'Win Rate':>9} | {'Mean R':>8} | {'Sharpe':>7}")
    print("-" * 90)
    
    for _, row in df.iterrows():
        print(f"{row['Asset Class']:<15} | {row['Symbols']:<8} | {row['Trades']:<7} | ${row['Total PnL']:>9,.0f} | {row['Return %']:>8.2f}% | {row['Win Rate']:>8.1f}% | {row['Mean R']:>8.3f} | {row['Sharpe']:>7.2f}")
    
    # Total
    print("-" * 90)
    total_pnl = df['Total PnL'].sum()
    total_trades = df['Trades'].sum()
    avg_win_rate = df['Win Rate'].mean()
    avg_sharpe = df['Sharpe'].mean()
    print(f"{'COMBINED':<15} | {df['Symbols'].sum():<8} | {total_trades:<7} | ${total_pnl:>9,.0f} | {total_pnl/100000*100:>8.2f}% | {avg_win_rate:>8.1f}% | {'N/A':>8} | {avg_sharpe:>7.2f}")
    
    # Best/Worst
    print("\n" + "="*60)
    print("ANALYSIS")
    print("="*60)
    
    if not df.empty and df['Trades'].sum() > 0:
        best_class = df.loc[df['Return %'].idxmax()]
        worst_class = df.loc[df['Return %'].idxmin()]
        
        print(f"\n✅ BEST: {best_class['Asset Class']} ({best_class['Return %']:.2f}%)")
        print(f"❌ WORST: {worst_class['Asset Class']} ({worst_class['Return %']:.2f}%)")
        
        if total_pnl > 0:
            print(f"\n📊 Combined strategy is PROFITABLE with ${total_pnl:,.0f} total PnL")
        else:
            print(f"\n⚠️ Combined strategy is UNPROFITABLE with ${total_pnl:,.0f} total PnL")

if __name__ == "__main__":
    main()
