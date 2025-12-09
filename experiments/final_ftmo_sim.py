"""
FINAL FTMO SIMULATION (Corrected)
=================================
Configuration:
- Core 13 FX Universe (Inherited from Config)
- Risk Per Trade: 1.5% (Inherited)
- SignalDecay Exit: ENABLED (Inherited)
- Full 2024 Year
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

def get_metrics(trades):
    if trades.empty: return {}
    
    trades['Entry Time'] = pd.to_datetime(trades['Entry Time'])
    trades['Exit Time'] = pd.to_datetime(trades['Exit Time'])
    
    # Win Rate
    win_rate = (trades['PnL'] > 0).mean()
    
    # Avg PnL
    avg_pnl = trades['PnL'].mean()
    total_pnl = trades['PnL'].sum()
    
    # Sharpe
    trading_days = (trades['Exit Time'].max() - trades['Entry Time'].min()).days
    if trading_days < 1: trading_days = 1
    trades_per_year = len(trades) / trading_days * 252
    
    sharpe = 0
    if trades['PnL'].std() > 0:
        sharpe = trades['PnL'].mean() / trades['PnL'].std() * np.sqrt(trades_per_year)
        
    return {
        'Win_Rate': win_rate,
        'Count': len(trades),
        'Total_PnL': total_pnl,
        'Sharpe_Trade': sharpe,
        'Avg_PnL': avg_pnl
    }

def print_exit_analysis(trades):
    print("\nEXIT REASON ATTRIBUTION")
    print("="*40)
    print(f"{'Exit Reason':<15} | {'Count':<6} | {'%':<5} | {'Win Rate':<8} | {'Avg PnL':<8}")
    print("-"*55)
    
    if 'Exit Reason' not in trades.columns:
        print("No Exit Reason data.")
        return

    total = len(trades)
    grouped = trades.groupby('Exit Reason')
    
    for reason, group in grouped:
        count = len(group)
        pct = count / total * 100
        wr = (group['PnL'] > 0).mean() * 100
        avg_pnl = group['PnL'].mean()
        
        print(f"{reason:<15} | {count:<6} | {pct:<5.1f} | {wr:<8.1f}% | ${avg_pnl:<8.0f}")

def main():
    print("="*60)
    print("FINAL FTMO CHALLENGE SIMULATION")
    print("="*60)
    
    config = Config()
    
    # Verify Settings (Do not override, just verify)
    print("SETTINGS (From quant_backtest.py):")
    print(f"  > Signal Decay Exit: {config.use_signal_decay_exit}")
    print(f"  > Risk Per Trade:    {config.risk_per_trade:.2%}")
    print(f"  > Universe Size:     {len(config.symbols)} pairs")
    
    loader = DataLoader(config)
    
    print("\n[LOADING DATA] 2024-01-01 to 2024-12-01...")
    try:
        data = loader.load_data("2024-01-01", "2024-12-01")
    except Exception as e:
        print(f"Error: {e}")
        return
        
    fe = FeatureEngine(config)
    re = RegimeEngine(config)
    ae = AlphaEngine(config)
    es = EnsembleSignal(config)
    ce = CrisisAlphaEngine(config)
    
    print("[PROCESSING PIPELINE]...")
    data = fe.add_features_all(data)
    data = re.add_regimes_all(data)
    ae.train_model(data)
    data = ae.add_signals_all(data)
    data = es.add_ensemble_all(data)
    data = ce.add_crisis_signals(data)
    
    print("[RUNNING BACKTEST]...")
    bt = Backtester(config)
    equity_curve = bt.run_backtest(data)
    trades = pd.DataFrame(bt.account.trade_history)
    
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    
    final_balance = bt.account.balance
    total_return = (final_balance - config.initial_balance) / config.initial_balance * 100
    
    # Drawdown (Fix for List vs Series)
    if isinstance(equity_curve, list):
        equity_series = pd.Series(equity_curve)
    else:
        # Check if it's a dict or DataFrame
        if hasattr(equity_curve, 'columns') and 'Equity' in equity_curve.columns:
            equity_series = equity_curve['Equity']
        elif isinstance(equity_curve, pd.DataFrame):
             equity_series = equity_curve.iloc[:, 0] # Assume first col
        else:
             equity_series = pd.Series(equity_curve)

    rolling_max = equity_series.cummax()
    drawdown = (equity_series - rolling_max) / rolling_max
    max_dd = drawdown.min() * 100
    
    print(f"  > Final Balance: ${final_balance:,.2f}")
    print(f"  > Total Return:  {total_return:.2f}%")
    print(f"  > Max Drawdown:  {max_dd:.2f}%")
    
    metrics = get_metrics(trades)
    print(f"  > Sharpe Ratio:  {metrics.get('Sharpe_Trade', 0):.2f}")
    print(f"  > Trades:        {metrics.get('Count', 0)}")
    print(f"  > Win Rate:      {metrics.get('Win_Rate', 0):.1%}")
    
    if not trades.empty:
        print_exit_analysis(trades)
        
    # Validation Check
    success = True
    if total_return < 8.0:
        print(f"\n⚠️ CLOSE BUT UNDER: Return {total_return:.2f}% < 10%")
        success = False
    if max_dd < -10:
        print(f"\n❌ FAILED: Max DD {max_dd:.2f}% < -10%")
        success = False
    
    if success:
        print("\n✅ PASSED FTMO CRITERIA (or acceptable range)!")
    elif total_return > 3.0:
        print("\n⚠️ SOLID PROFITABLE. Just needs more leverage or time.")

if __name__ == "__main__":
    main()
