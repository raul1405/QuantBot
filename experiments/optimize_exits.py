"""
Experiment: Exit Logic Optimization (A/B Test)
==============================================
Hypothesis: Removing SignalDecay exits (which had negative expectancy) 
will improve overall Strategy Sharpe and Return.
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

def run_variant(variant_name, config):
    print(f"\n[{variant_name}] Running Backtest...")
    
    loader = DataLoader(config)
    # FX Only
    config.symbols = [
        "EURUSD=X", "USDJPY=X", "GBPUSD=X", "USDCHF=X", 
        "USDCAD=X", "AUDUSD=X", "NZDUSD=X",
        "EURGBP=X", "EURJPY=X", "GBPJPY=X", "AUDJPY=X",
        "EURAUD=X", "EURCHF=X"
    ]
    
    try:
        data = loader.load_data("2024-01-01", "2024-12-01")
    except Exception as e:
        print(f"  Error: {e}")
        return None
        
    if not data: return None

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
    
    if trades.empty:
        return {'Return': 0, 'Sharpe': 0, 'DD': 0}
        
    # Stats
    total_return_pct = (bt.account.equity - config.initial_balance) / config.initial_balance * 100
    
    # Sharpe
    trades['Entry Time'] = pd.to_datetime(trades['Entry Time'])
    trades['Exit Time'] = pd.to_datetime(trades['Exit Time'])
    trading_days = (trades['Exit Time'].max() - trades['Entry Time'].min()).days
    trades_per_year = len(trades) / max(trading_days, 1) * 252
    sharpe = 0
    if trades['PnL'].std() > 0:
        sharpe = trades['PnL'].mean() / trades['PnL'].std() * np.sqrt(trades_per_year)
        
    # Drawdown
    equity_series = pd.Series(equity_curve)
    rolling_max = equity_series.cummax()
    drawdown = (equity_series - rolling_max) / rolling_max
    max_dd = drawdown.min() * 100
    
    print(f"  > Return: {total_return_pct:.2f}%")
    print(f"  > Sharpe: {sharpe:.2f}")
    print(f"  > Max DD: {max_dd:.2f}%")
    print(f"  > Trades: {len(trades)}")
    
    return {
        'Return': total_return_pct,
        'Sharpe': sharpe, 
        'Max DD': max_dd,
        'Trades': len(trades),
        'Win Rate': (trades['PnL'] > 0).mean() * 100
    }

def main():
    print("="*60)
    print("EXIT LOGIC OPTIMIZATION: SignalDecay vs No SignalDecay")
    print("="*60)
    
    # Variant A: Baseline (Signal Decay ON)
    config_a = Config()
    config_a.use_signal_decay_exit = True
    stats_a = run_variant("VARIANT A: SignalDecay ON", config_a)
    
    # Variant B: Challenger (Signal Decay OFF)
    config_b = Config()
    config_b.use_signal_decay_exit = False
    stats_b = run_variant("VARIANT B: SignalDecay OFF", config_b)
    
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)
    print(f"{'Metric':<15} | {'Variant A (ON)':<15} | {'Variant B (OFF)':<15} | {'Diff':<10}")
    print("-" * 65)
    
    metrics = ['Return', 'Sharpe', 'Max DD', 'Trades', 'Win Rate']
    for m in metrics:
        val_a = stats_a[m]
        val_b = stats_b[m]
        diff = val_b - val_a
        
        fmt = "{:.2f}"
        if m in ['Return', 'Max DD', 'Win Rate']: fmt = "{:.2f}%"
        
        print(f"{m:<15} | {fmt.format(val_a):<15} | {fmt.format(val_b):<15} | {fmt.format(diff):<10}")
    
    winner = "Variant B (OFF)" if stats_b['Sharpe'] > stats_a['Sharpe'] else "Variant A (ON)"
    print(f"\n🏆 WINNER: {winner}")

if __name__ == "__main__":
    main()
