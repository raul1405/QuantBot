
"""
AUDIT STOCK SHARPE (NVDA 2024)
==============================
Validates the '3.69 Sharpe' claim.
Compares Strategy Performance vs Simple Buy & Hold.

Hypothesis: 
The high Sharpe is due to:
1. Massive Bull Market (Beta).
2. Daily vs Trade Sharpe calculation differences.
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

def calc_daily_sharpe(equity_curve):
    if isinstance(equity_curve, list):
        equity_curve = pd.Series(equity_curve)
    
    returns = equity_curve.pct_change().dropna()
    if returns.std() == 0: return 0
    
    # Sortinor-ish or Standard? Standard.
    return returns.mean() / returns.std() * np.sqrt(252)

def main():
    print("="*60)
    print("AUDIT: NVDA 2024 PERFORMANCE")
    print("="*60)
    
    config = Config()
    config.symbols = ["NVDA"] # Single stock focus
    # Stocks need different cost model? 
    # quant_backtest uses 'indices' logic or default?
    # It uses default 0.1% if not matched.
    # NVDA is not '=X' or '=F'.
    # It will fall to 'else' block: 0.1% of notional.
    # This is conservative (10 bps). Real is ~1-5 bps.
    
    # Load Data
    loader = DataLoader(config)
    try:
        data = loader.load_data("2024-01-01", "2024-12-01")
    except:
        return

    # 1. Buy & Hold Benchmark
    df = data['NVDA'].copy()
    initial_price = df['Close'].iloc[0]
    final_price = df['Close'].iloc[-1]
    bh_return = (final_price - initial_price) / initial_price * 100
    
    # BH Sharpe
    bh_equity = df['Close'] / initial_price * 100000
    bh_sharpe = calc_daily_sharpe(bh_equity)
    
    print(f"\n[BENCHMARK] Buy & Hold")
    print(f"  Return: {bh_return:.2f}%")
    print(f"  Sharpe: {bh_sharpe:.2f}")

    # 2. Strategy Run
    print(f"\n[STRATEGY] Running Algo...")
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
    equity_curve = bt.run_backtest(data)
    
    strat_return = (bt.account.balance - config.initial_balance) / config.initial_balance * 100
    strat_sharpe = calc_daily_sharpe(bt.account.equity_curve)
    
    # Trade Sharpe (Original Metric)
    trades = pd.DataFrame(bt.account.trade_history)
    trade_sharpe = 0
    if not trades.empty:
        trades['Entry Time'] = pd.to_datetime(trades['Entry Time'])
        trades['Exit Time'] = pd.to_datetime(trades['Exit Time'])
        days = (trades['Exit Time'].max() - trades['Entry Time'].min()).days
        tpy = len(trades) / max(days, 1) * 252
        if trades['PnL'].std() > 0:
            trade_sharpe = trades['PnL'].mean() / trades['PnL'].std() * np.sqrt(tpy)
            
    print(f"\n[STRATEGY] Algo Results")
    print(f"  Return:       {strat_return:.2f}%")
    print(f"  Daily Sharpe: {strat_sharpe:.2f}")
    print(f"  Trade Sharpe: {trade_sharpe:.2f} (Metric used previously)")
    
    print("\n" + "="*60)
    print("VERDICT")
    print("="*60)
    
    if strat_sharpe > bh_sharpe:
        print("✅ ALPHA CONFIRMED: Strategy beat Buy & Hold risk-adjusted.")
    else:
        print("⚠️ BETA RIDE: Strategy performs worse/same as just holding.")
        
    diff = abs(strat_sharpe - trade_sharpe)
    if diff > 1.0:
        print(f"⚠️ METRIC WARNING: Trade Sharpe ({trade_sharpe:.2f}) >> Daily Sharpe ({strat_sharpe:.2f}).")
        print("   The 3.69 figure likely inflated by Trade-Based calculation.")

if __name__ == "__main__":
    main()
