
"""
PER SYMBOL BREAKDOWN (Core 13)
==============================
Analyzes the performance of each asset in the Core 13 universe.
"""

import sys
import os
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quant_backtest import (
    Config, DataLoader, FeatureEngine, RegimeEngine, 
    AlphaEngine, EnsembleSignal, CrisisAlphaEngine, Backtester
)

def print_symbol_metrics(trades):
    if trades.empty:
        print("No trades found.")
        return

    print("\nPERFORMANCE BY SYMBOL")
    print("="*80)
    print(f"{'Symbol':<10} | {'Count':<5} | {'Win Rate':<8} | {'PnL ($)':<10} | {'Avg ($)':<8} | {'PF':<5}")
    print("-" * 80)
    
    # Ensure PnL is numeric
    trades['PnL'] = pd.to_numeric(trades['PnL'])
    
    grouped = trades.groupby('Symbol')
    
    stats = []
    
    for sym, group in grouped:
        count = len(group)
        wins = group[group['PnL'] > 0]
        losses = group[group['PnL'] <= 0]
        
        win_rate = len(wins) / count if count > 0 else 0
        total_pnl = group['PnL'].sum()
        avg_pnl = group['PnL'].mean()
        
        gross_profit = wins['PnL'].sum()
        gross_loss = abs(losses['PnL'].sum())
        pf = gross_profit / gross_loss if gross_loss > 0 else 99.9
        
        stats.append({
            'Symbol': sym,
            'Count': count,
            'Win Rate': win_rate,
            'Total PnL': total_pnl,
            'Avg PnL': avg_pnl,
            'PF': pf
        })
        
    # Sort by Total PnL Descending
    stats.sort(key=lambda x: x['Total PnL'], reverse=True)
    
    for s in stats:
        print(f"{s['Symbol']:<10} | {s['Count']:<5} | {s['Win Rate']:<8.1%} | ${s['Total PnL']:<10.2f} | ${s['Avg PnL']:<8.2f} | {s['PF']:<5.2f}")

    print("-" * 80)


def main():
    print("Running Core 13 Breakdown...")
    config = Config()
    
    # Load Data
    loader = DataLoader(config)
    try:
        data = loader.load_data("2024-01-01", "2024-12-01") 
    except Exception as e:
        print(e)
        return

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
    bt.run_backtest(data)
    trades = pd.DataFrame(bt.account.trade_history)
    
    print_symbol_metrics(trades)

if __name__ == "__main__":
    main()
