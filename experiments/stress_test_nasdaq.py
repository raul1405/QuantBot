
"""
NASDAQ STRESS TEST (Bear Market Simulation)
===========================================
Objective: Determine if the strategy has 'Short Alpha' or is just riding the Bull Market.
Target: NQ=F (Nasdaq 100 Futures)

Methodology:
1. Run Backtest.
2. Isolate Short Trades.
3. Analyze performance during known 2024 Drawdowns:
   - Window A: April Correction (Apr 1 - Apr 30).
   - Window B: August Crash (Jul 15 - Aug 10).
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

def analyze_window(equity_curve, trades, start_date, end_date, label):
    print(f"\n[ANALYSIS: {label}] ({start_date} -> {end_date})")
    
    # Filter Trades
    mask = (trades['Entry Time'] >= start_date) & (trades['Entry Time'] <= end_date)
    window_trades = trades[mask]
    
    if window_trades.empty:
        print("  No trades in this window.")
        return
        
    pnl = window_trades['PnL'].sum()
    count = len(window_trades)
    shorts = window_trades[window_trades['Direction'] == 'SHORT']
    short_pnl = shorts['PnL'].sum()
    
    print(f"  Total PnL:   ${pnl:,.0f}")
    print(f"  Trades:      {count}")
    print(f"  Short PnL:   ${short_pnl:,.0f} ({len(shorts)} trades)")
    
    if short_pnl > 0:
        print("  ✅ BEAR ALPHA: Profited from Shorts during decline.")
    elif len(shorts) == 0:
        print("  ⚠️ NO SHORTS: Strategy sat out (Neutral).")
    else:
        print("  ❌ LOSS: Failed to short effectively.")

def main():
    print("="*60)
    print("STRESS TEST: NASDAQ (NQ=F)")
    print("="*60)
    
    config = Config()
    config.symbols = ["NQ=F"]
    config.risk_per_trade = 0.02 # 2% risk
    
    # 1. Load Data
    loader = DataLoader(config)
    try:
        data = loader.load_data("2024-01-01", "2024-12-01")
    except:
        return

    # Benchmark Info
    df = data['NQ=F']
    bench_ret = (df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0] * 100
    print(f"Benchmark (Buy & Hold) Return: {bench_ret:.2f}%")

    # 2. Run Algo
    print("Running Algo...")
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
    strat_ret = (bt.account.balance - config.initial_balance) / config.initial_balance * 100
    
    print(f"Strategy Return: {strat_ret:.2f}%")
    
    # 3. Short Leg Analysis
    if not trades.empty:
        trades['Entry Time'] = pd.to_datetime(trades['Entry Time'])
        
        shorts = trades[trades['Direction'] == 'SHORT']
        short_pnl = shorts['PnL'].sum()
        short_count = len(shorts)
        
        print("\n" + "-"*40)
        print("SHORT LEG PERFORMANCE (Global)")
        print("-" * 40)
        print(f"Short PnL:   ${short_pnl:,.0f}")
        print(f"Short Trades: {short_count}")
        
        # 4. Window Analysis (Corrections)
        analyze_window(None, trades, "2024-04-01", "2024-04-30", "APRIL CORRECTION")
        analyze_window(None, trades, "2024-07-15", "2024-08-10", "AUGUST CRASH")
        
    print("\n" + "="*60)
    print("VERDICT")
    print("="*60)
    
    if strat_ret > bench_ret:
        print("✅ Strategy Beats Market.")
    else:
        print("❌ Strategy Lags Market.")
        
    if not trades.empty:
        shorts = trades[trades['Direction'] == 'SHORT']
        if shorts['PnL'].sum() > 0:
            print("✅ HEDGE CAPABILITY: Short trades are profitable overall.")
        else:
            print("⚠️ BULL ONLY: Short trades lose money. This is a Bull Strategy.")

if __name__ == "__main__":
    main()
