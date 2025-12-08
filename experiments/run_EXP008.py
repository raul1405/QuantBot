import sys
import os
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quant_backtest import (
    Config, DataLoader, FeatureEngine, RegimeEngine, 
    AlphaEngine, EnsembleSignal, CrisisAlphaEngine, Backtester
)

RESULTS_FILE = "experiments/exp_008_margin_constraints.md"

def run_scenario(account_size, leverage=30.0):
    print(f"\n[EXP 008] Running {account_size/1000}k Setup (Leverage 1:{leverage})...")
    
    config = Config()
    config.transaction_cost = 0.0001
    config.initial_balance = account_size
    config.account_leverage = leverage
    
    # Load Data (Full OOS Window)
    loader = DataLoader(config)
    data = loader.load_data("2024-01-01", "2025-12-01") # 2 Years
    
    # Pipeline
    fe = FeatureEngine(config)
    re = RegimeEngine(config)
    data = fe.add_features_all(data)
    data = re.add_regimes_all(data)
    
    alpha = AlphaEngine(config)
    alpha.train_model(data) 
    
    data = alpha.add_signals_all(data)
    ens = EnsembleSignal(config)
    data = ens.add_ensemble_all(data)
    
    crisis = CrisisAlphaEngine(config)
    final_data = crisis.add_crisis_signals(data)
           
    # Backtest
    bt = Backtester(config)
    bt.run_backtest(final_data)
    
    # Stats
    trades = len(bt.account.trade_history)
    balance = bt.account.balance
    pnl = balance - account_size
    roi = pnl / account_size
    
    # Estimate Blocked Trades?
    # Backtester logic skips silently in loop, difficult to count exactly without instrumentation.
    # We can infer from Total Trades relative to 100k baseline.
    
    return {
        "Size": account_size,
        "Trades": trades,
        "ROI": roi,
        "PnL": pnl,
        "Balance": balance
    }

def main():
    sizes = [10000.0, 25000.0, 100000.0]
    results = []
    
    for s in sizes:
        res = run_scenario(s)
        results.append(res)
        
    with open(RESULTS_FILE, "w") as f:
        f.write("# EXP_008: Margin Constrained Backtest (1:30)\n\n")
        f.write("| Account Size | Trades | ROI | Net PnL |\n")
        f.write("|---|---|---|---|\n")
        
        baseline_trades = 0
        for r in results:
            if r['Size'] == 100000.0: baseline_trades = r['Trades']
            
        for r in results:
            diff_trades = r['Trades'] - baseline_trades
            note = ""
            if diff_trades < 0: note = f"({diff_trades} blocked)"
            elif diff_trades == 0: note = "(Baseline)"
            
            f.write(f"| ${r['Size']:,.0f} | {r['Trades']} {note} | {r['ROI']*100:.2f}% | ${r['PnL']:,.0f} |\n")

    print("\nExperiment Complete. Results in", RESULTS_FILE)

if __name__ == "__main__":
    main()
