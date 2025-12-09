"""
Validation: In-Sample vs Out-of-Sample Split Test
Goal: Confirm that the high Sharpe/Skewness was due to In-Sample contamination.
Method: Explicitly split data into Train (80%) and Test (20%) and report metrics separately.
"""
import sys
sys.path.insert(0, '/Users/raulschalkhammer/Desktop/Costum Portfolio Backtest/FTMO Challenge')

from quant_backtest import (
    Config, DataLoader, FeatureEngine, AlphaEngine, 
    RegimeEngine, EnsembleSignal, CrisisAlphaEngine, Backtester
)
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# FX Majors
UNIVERSE = ["EURUSD=X", "USDJPY=X", "GBPUSD=X", "USDCHF=X", "USDCAD=X", "AUDUSD=X"]

def run_split_validation():
    print("="*60)
    print(" IN-SAMPLE vs OUT-OF-SAMPLE VALIDATION")
    print("="*60)
    
    config = Config()
    config.symbols = UNIVERSE
    config.transaction_cost = 0.0001
    config.initial_balance = 100000.0
    # Ensure split is consistent
    config.ml_train_split_pct = 0.80
    
    # Load Data
    loader = DataLoader(config)
    end = datetime.now()
    start = end - timedelta(days=500)
    data = loader.load_data(start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'))
    
    if not data:
        print("No data.")
        return

    # Determine Split Timestamp (approximate based on first symbol)
    first_df = list(data.values())[0]
    split_idx = int(len(first_df) * config.ml_train_split_pct)
    split_date = first_df.index[split_idx]
    
    print(f"\nData Range: {first_df.index[0]} to {first_df.index[-1]}")
    print(f"Split Date: {split_date}")
    print(f"Train Size: {split_idx} bars | Test Size: {len(first_df) - split_idx} bars")
    
    # Pipeline
    fe = FeatureEngine(config)
    re = RegimeEngine(config)
    data = fe.add_features_all(data)
    data = re.add_regimes_all(data)
    
    alpha = AlphaEngine(config)
    alpha.train_model(data) # Trains on first 80%
    
    data = alpha.add_signals_all(data) # Predicts on ALL data
    ens = EnsembleSignal(config)
    data = ens.add_ensemble_all(data)
    crisis = CrisisAlphaEngine(config)
    final_data = crisis.add_crisis_signals(data)
    
    # Create IS and OOS datasets
    data_is = {}
    data_oos = {}
    
    for sym, df in final_data.items():
        data_is[sym] = df.loc[:split_date].copy()
        data_oos[sym] = df.loc[split_date:].copy()
    
    # Run Backtest on In-Sample (Should be amazing)
    print("\n" + "-"*40)
    print(" RUNNING IN-SAMPLE BACKTEST (Training Data)")
    print("-" * 40)
    bt_is = Backtester(config)
    equity_is = bt_is.run_backtest(data_is)
    print_stats("In-Sample", bt_is, equity_is)
    
    # Run Backtest on Out-of-Sample (The Truth)
    print("\n" + "-"*40)
    print(" RUNNING OUT-OF-SAMPLE BACKTEST (Test Data)")
    print("-" * 40)
    bt_oos = Backtester(config)
    equity_oos = bt_oos.run_backtest(data_oos)
    print_stats("Out-of-Sample", bt_oos, equity_oos)

def print_stats(name, bt, equity):
    trades = len(bt.account.trade_history)
    balance = bt.account.balance
    pnl_pct = (balance - 100000) / 100000 * 100
    
    if len(equity) > 1:
        peak = equity.cummax()
        dd = (equity - peak) / peak
        max_dd = dd.min() * 100
        
        returns = equity.pct_change().dropna()
        sharpe = (returns.mean() * 252 * 7) / (returns.std() * np.sqrt(252 * 7)) if returns.std() > 0 else 0
    else:
        max_dd = 0
        sharpe = 0
        
    wins = [t for t in bt.account.trade_history if t['PnL'] > 0]
    wr = len(wins) / trades * 100 if trades > 0 else 0
    
    print(f"{name} Results:")
    print(f"  Return:   {pnl_pct:+.2f}%")
    print(f"  Max DD:   {max_dd:.2f}%")
    print(f"  Sharpe:   {sharpe:.2f}")
    print(f"  Trades:   {trades}")
    print(f"  Win Rate: {wr:.1f}%")

if __name__ == "__main__":
    run_split_validation()
