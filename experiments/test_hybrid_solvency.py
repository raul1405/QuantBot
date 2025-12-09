
"""
HYBRID SOLVENCY TEST (FX + INDICES)
===================================
Objective: Determine if FX profits can "carry" the Index Strategy's losing months.
Hypothesis: If FX and Indices are uncorrelated, the combined portfolio should have a 
higher Monthly Win Rate than Indices alone (45%).

Metrics:
1. Combined Monthly Win Rate (Target > 70%).
2. Correlation of Daily PnL.
3. Worst Month Drawdown.
"""

import sys
import os
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quant_backtest import (
    Config, DataLoader, FeatureEngine, RegimeEngine, 
    AlphaEngine, EnsembleSignal, Backtester
)

def run_backtest_for_group(config, symbols, label):
    print(f"\n[Running {label}]...")
    config.symbols = symbols
    loader = DataLoader(config)
    try: data = loader.load_data("2024-01-01", "2024-12-01")
    except: return pd.DataFrame()

    fe = FeatureEngine(config)
    re = RegimeEngine(config)
    ae = AlphaEngine(config)
    es = EnsembleSignal(config)
    
    data = fe.add_features_all(data)
    data = re.add_regimes_all(data)
    ae.train_model(data)
    data = ae.add_signals_all(data)
    data = es.add_ensemble_all(data)
    
    bt = Backtester(config)
    bt.run_backtest(data)
    trades = pd.DataFrame(bt.account.trade_history)
    return trades

def main():
    print("="*60)
    print("HYBRID SOLVENCY CHECK")
    print("="*60)
    
    config = Config()
    config.risk_per_trade = 0.015 # 1.5% Base Risk
    
    # 1. Run FX
    fx_syms = ["EURUSD=X", "USDJPY=X", "GBPUSD=X", 
               "USDCHF=X", "USDCAD=X", "AUDUSD=X", "NZDUSD=X", 
               "EURGBP=X", "EURJPY=X", "GBPJPY=X", 
               "AUDJPY=X", "EURAUD=X", "EURCHF=X"]
    fx_trades = run_backtest_for_group(config, fx_syms, "FX Core 13")
    
    # 2. Run Indices
    idx_syms = ["ES=F", "NQ=F", "YM=F", "RTY=F"]
    idx_trades = run_backtest_for_group(config, idx_syms, "Indices Basket")
    
    if fx_trades.empty or idx_trades.empty:
        print("Data Error.")
        return
        
    # 3. Align PnL by Month
    fx_trades['Entry Time'] = pd.to_datetime(fx_trades['Entry Time'])
    idx_trades['Entry Time'] = pd.to_datetime(idx_trades['Entry Time'])
    
    fx_trades['Month'] = fx_trades['Entry Time'].dt.to_period('M')
    idx_trades['Month'] = idx_trades['Entry Time'].dt.to_period('M')
    
    fx_monthly = fx_trades.groupby('Month')['PnL'].sum()
    idx_monthly = idx_trades.groupby('Month')['PnL'].sum()
    
    # Combine
    combined = pd.DataFrame({'FX': fx_monthly, 'Indices': idx_monthly}).fillna(0)
    combined['Total'] = combined['FX'] + combined['Indices']
    
    print("\n" + "="*60)
    print("MONTHLY SYNERGY REPORT")
    print("="*60)
    print(f"{'Month':<10} | {'FX PnL':>10} | {'Idx PnL':>10} | {'TOTAL':>10} | {'Status'}")
    print("-" * 60)
    
    wins = 0
    total = 0
    fx_saves = 0
    
    for m, row in combined.iterrows():
        status = "✅ WIN" if row['Total'] > 0 else "❌ LOSS"
        print(f"{str(m):<10} | ${row['FX']:>9,.0f} | ${row['Indices']:>9,.0f} | ${row['Total']:>9,.0f} | {status}")
        
        if row['Total'] > 0: 
            wins += 1
            if row['Indices'] < 0: fx_saves += 1 # FX saved the month
        
        total += 1
        
    print("-" * 60)
    win_rate = wins / total * 100
    print(f"Hybrid Win Rate: {win_rate:.1f}% ({wins}/{total})")
    print(f"FX Saves: {fx_saves} months (FX profit covered Index loss)")
    
    # Check Correlation
    # Need Daily PnL for Correlation
    fx_trades['Date'] = fx_trades['Entry Time'].dt.date
    idx_trades['Date'] = idx_trades['Entry Time'].dt.date
    
    fx_daily = fx_trades.groupby('Date')['PnL'].sum()
    idx_daily = idx_trades.groupby('Date')['PnL'].sum()
    
    daily_df = pd.DataFrame({'FX': fx_daily, 'Indices': idx_daily}).fillna(0)
    corr = daily_df['FX'].corr(daily_df['Indices'])
    
    print(f"\nDaily Correlation: {corr:.2f}")
    if abs(corr) < 0.3:
        print("✅ UNCORRELATED. Great diversification.")
    else:
        print("⚠️ CORRELATED. Risks might stack.")

    print("\nVERDICT:")
    if win_rate >= 70 and abs(corr) < 0.3:
        print("🚀 SYNERGY CONFIRMED. Combine them.")
    elif win_rate >= 60:
        print("⚠️ MARGINAL. Better than Indices alone, but worse than FX alone?")
    else:
        print("❌ DANGEROUS. Do not combine.")

if __name__ == "__main__":
    main()
