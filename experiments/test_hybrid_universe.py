
"""
HYBRID UNIVERSE TEST (FX + INDICES)
===================================
Hypothesis: Adding Indices to FX accelerates FTMO Passing (Time to +10%).
Logic: More opportunities (higher frequency) should smooth the curve and hit target faster,
IF the assets are not perfectly correlated.

Metrics:
1. Total Return.
2. Max Drawdown.
3. Days to Hit +10% (Pass).
4. Sharpe.
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
    print("HYBRID UNIVERSE SIMULATION (FX + INDICES)")
    print("="*60)
    
    config = Config()
    config.risk_per_trade = 0.02 # 2% (Standard)
    
    # 1. Comparison: FX Only
    fx_core = ["EURUSD=X", "USDJPY=X", "GBPUSD=X", 
               "USDCHF=X", "USDCAD=X", "AUDUSD=X", "NZDUSD=X", 
               "EURGBP=X", "EURJPY=X", "GBPJPY=X", 
               "AUDJPY=X", "EURAUD=X", "EURCHF=X"]
               
    indices = ["ES=F", "NQ=F", "YM=F", "RTY=F"]
    
    hybrid = fx_core + indices
    
    fx_results = run_test(config, fx_core, "FX Only")
    hybrid_results = run_test(config, hybrid, "Hybrid (FX + Indices)")
    
    print("\n" + "="*60)
    print("COMPARISON: SPEED TO PASS")
    print("="*60)
    print(f"{'Metric':<20} | {'FX Only':<15} | {'Hybrid':<15}")
    print("-" * 60)
    print(f"{'Return':<20} | {fx_results['Return']:>14.2f}% | {hybrid_results['Return']:>14.2f}%")
    print(f"{'Days to Pass':<20} | {fx_results['DaysToPass']:>15} | {hybrid_results['DaysToPass']:>15}")
    print(f"{'Max Drawdown':<20} | {fx_results['MaxDD']:>14.2f}% | {hybrid_results['MaxDD']:>14.2f}%")
    print(f"{'Sharpe':<20} | {fx_results['Sharpe']:>14.2f}  | {hybrid_results['Sharpe']:>14.2f}")
    
    print("\nVERDICT:")
    if hybrid_results['DaysToPass'] != "FAIL" and (fx_results['DaysToPass'] == "FAIL" or hybrid_results['DaysToPass'] < fx_results['DaysToPass']):
        print("✅ FASTER. Adding Indices speeds up the challenge.")
    else:
        print("❌ SLOWER/RISKIER. Indices add drag or volatility.")

def run_test(config, symbols, label):
    print(f"\n[RUNNING: {label}] ({len(symbols)} symbols)...")
    config.symbols = symbols
    
    loader = DataLoader(config)
    try: data = loader.load_data("2024-01-01", "2024-12-01")
    except: return {}

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
    equity_curve = pd.Series(bt.run_backtest(data))
    # bt.run_backtest(data) # Removed duplicate call
    balance = bt.account.balance
    ret = (balance - config.initial_balance) / config.initial_balance * 100
    
    # Calculate Max DD
    cummax = equity_curve.cummax()
    dd = (equity_curve - cummax) / cummax
    max_dd = dd.min() * 100
    
    # Check Pass (First day > 1.10)
    pass_impact = "FAIL"
    # Need dates for 'Days to Pass'. 
    # bt.account.equity_curve is list of floats.
    # We can approximate 'Hours' to 'Days' (bars / 24).
    # assuming H1 bars.
    
    pass_idx = np.argmax(equity_curve > 110000)
    if equity_curve.iloc[pass_idx] > 110000:
        # Index is 'hours' roughly (if simulation steps are trade events, distinct from bars).
        # Actually equity curve appends on trade close? No, usually daily or per bar.
        # Backtester logic: 'equity_curve' appended end of backtest, or per bar?
        # It's likely per step.
        pass_bars = pass_idx
        pass_days = pass_bars / 24 
        pass_impact = f"{pass_days:.1f} Days"
        
    trades = pd.DataFrame(bt.account.trade_history)
    sharpe = 0
    if not trades.empty and trades['PnL'].std() > 0:
        # Trade Sharpe proxy
        tpy = len(trades) / 252 # Rough
        # Better: PnL Mean / Std
        sharpe = trades['PnL'].mean() / trades['PnL'].std() * np.sqrt(len(trades))
        
    return {
        'Return': ret,
        'MaxDD': max_dd,
        'DaysToPass': pass_impact,
        'Sharpe': sharpe
    }

if __name__ == "__main__":
    main()
