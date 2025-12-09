
"""
INDICES MONTHLY BREAKDOWN
=========================
Analyzes consistency of Indices Strategy (US100, US500, US30, US2000).
If Monthly Win Rate is > 60%, it's worth trading heavily.
If chop, avoid.
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

def main():
    print("="*60)
    print("INDICES: MONTHLY CONSISTENCY CHECK")
    print("="*60)
    
    config = Config()
    config.symbols = ["ES=F", "NQ=F", "YM=F", "RTY=F"]
    config.risk_per_trade = 0.015 # 1.5% Conservative
    
    loader = DataLoader(config)
    try: data = loader.load_data("2024-01-01", "2024-12-01")
    except: return

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
    
    if trades.empty:
        print("No trades.")
        return
        
    trades['Entry Time'] = pd.to_datetime(trades['Entry Time'])
    trades['Month'] = trades['Entry Time'].dt.to_period('M')
    
    monthly = trades.groupby('Month')['PnL'].sum()
    monthly_count = trades.groupby('Month')['PnL'].count()
    
    print("\nMONTHLY PERFORMANCE:")
    print("-" * 40)
    print(f"{'Month':<10} | {'PnL ($)':>10} | {'Trades':>6} | {'Status'}")
    print("-" * 40)
    
    wins = 0
    total_months = 0
    
    for month, pnl in monthly.items():
        status = "✅ WIN" if pnl > 0 else "❌ LOSS"
        count = monthly_count[month]
        print(f"{month}    | ${pnl:,.0f}   | {count:>6} | {status}")
        
        if pnl > 0: wins += 1
        total_months += 1
        
    print("-" * 40)
    win_rate = wins / total_months * 100
    print(f"Monthly Win Rate: {win_rate:.1f}% ({wins}/{total_months})")
    
    print("\nVERDICT:")
    if win_rate >= 70:
        print("🔥 HIGH CONSISTENCY. Definitely trade this.")
    elif win_rate >= 50:
        print("⚠️ CHOPPY. Profitable but stressful.")
    else:
        print("❌ TOXIC. Loses more months than it wins.")

if __name__ == "__main__":
    main()
