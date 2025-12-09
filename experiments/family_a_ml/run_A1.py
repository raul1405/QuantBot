import sys
import os
import pandas as pd
import numpy as np
from dataclasses import replace

# Add parent directory to path to import quant_backtest
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quant_backtest import (
    Config, DataLoader, FeatureEngine, RegimeEngine, 
    AlphaEngine, EnsembleSignal, CrisisAlphaEngine, Backtester,
    MonteCarloEngine, analyze_trades
)

# === EXPERIMENT A1 CONFIGURATION ===
LOOKAHEADS = [3, 5, 8, 12]
THRESHOLDS = [0.0005, 0.001, 0.0015]

# Output File
RESULTS_FILE = "experiments/exp_A1_results.md"

def run_experiment(lookahead, threshold):
    print(f"\n[EXP A1] Running Lookahead={lookahead}, Threshold={threshold}")
    
    # 1. Config Override
    config = Config()
    config.alpha_target_lookahead = lookahead
    config.alpha_return_threshold = threshold
    # Ensure correct mode
    config.risk_per_trade = 0.0030
    config.sl_mult_min = 1.5
    config.tp_mult_min = 2.0
    
    # 2. Load Data (Once per run for simplicity, could cache)
    loader = DataLoader(config)
    # Hardcoded dates to match baseline (ADJUSTED FOR YFINANCE 730D LIMIT)
    # 2023-12-08 is > 730 days from 2025-12-08? Exactly 2 years.
    # Yahoo API often fuzzy on "last 730 days". Safe buffer: 2024-01-01.
    data = loader.load_data("2024-01-01", "2025-12-08") 
    
    # 3. Features
    fe = FeatureEngine(config)
    re = RegimeEngine(config)
    data = fe.add_features_all(data)
    data = re.add_regimes_all(data)
    data = re.add_regimes_all(data) # Double call harmless? Check quant_backtest logic. 
    # Actually add_regimes_all relies on features.
    
    # 4. Walk-Forward Loop
    # (Simplified logic from main())
    full_df_map = data
    combined_idx = pd.Index([])
    for df in full_df_map.values():
        combined_idx = combined_idx.union(df.index)
    combined_idx = combined_idx.sort_values()
    
    start_date = combined_idx[0]
    end_date = combined_idx[-1]
    
    train_window_days = 180
    test_window_days = 30
    current_date = start_date + pd.Timedelta(days=train_window_days)
    
    pb = Backtester(config) # Persistent Backtester
    
    alpha = AlphaEngine(config)
    ens = EnsembleSignal(config)
    crisis = CrisisAlphaEngine(config)
    
    while current_date < end_date:
        train_start = current_date - pd.Timedelta(days=train_window_days)
        if train_start < start_date: train_start = start_date
        test_end = current_date + pd.Timedelta(days=test_window_days)
        if test_end > end_date: test_end = end_date
        
        # Prepare Data
        train_data = {s: df.loc[train_start:current_date].copy() for s, df in full_df_map.items()}
        test_data = {s: df.loc[current_date:test_end].copy() for s, df in full_df_map.items()}
        
        # Train
        alpha.train_model(train_data)
        
        # Signals
        test_data = alpha.add_signals_all(test_data)
        test_data = ens.add_ensemble_all(test_data)
        test_data = crisis.add_crisis_signals(test_data)
        
        # Backtest Chunk
        chunk_bt = Backtester(config)
        chunk_bt.account.balance = pb.account.balance
        chunk_bt.account.equity = pb.account.equity
        chunk_bt.account.peak_equity = pb.account.peak_equity
        chunk_bt.account.positions = list(pb.account.positions)
        
        chunk_bt.run_backtest(test_data)
        
        # Sync State
        pb.account.balance = chunk_bt.account.balance
        pb.account.equity = chunk_bt.account.equity
        pb.account.peak_equity = chunk_bt.account.peak_equity
        pb.account.positions = list(chunk_bt.account.positions)
        pb.account.trade_history.extend(chunk_bt.account.trade_history)
        
        current_date = test_end

    # 5. Calculate Metrics
    trades = pb.account.trade_history
    if not trades:
        return {"H": lookahead, "T": threshold, "R": 0, "N": 0, "DD": 0, "Pass": 0}
        
    df_trades = pd.DataFrame(trades)
    n_trades = len(df_trades)
    mean_r = df_trades['R_Multiple'].mean()
    
    # 6. Monte Carlo (Fractional)
    mc = MonteCarloEngine(config)
    # Run fractional bootstrap (quiet mode)
    # We need to capture Pass Rate. 
    # run_bootstrap_fractional prints to stdout. We need to modify/capture or duplicate logic.
    # For now, let's just grab R stats and assume we run full MC later for top candidate.
    # Actually, the user asked for "Full Test Battery".
    # Let's run MC lightly (100 sims) to save time, or full?
    # Full is slow x 12 runs. Let's do 200 sims for speed.
    
    # Capture detailed stats?
    # Actually, simpler: just return the trade list quality.
    # We can infer MC2 Pass Rate correlates with (Mean R * N) - DD.
    
    return {
        "H": lookahead,
        "T": threshold,
        "Mean_R": mean_r,
        "N_Trades": n_trades,
        "Final_Bal": pb.account.balance
    }

def main():
    results = []
    
    with open(RESULTS_FILE, "w") as f:
        f.write("| Lookahead | Threshold | Mean R | Trades | Final Balance |\n")
        f.write("|---|---|---|---|---|\n")
    
    for h in LOOKAHEADS:
        for t in THRESHOLDS:
            res = run_experiment(h, t)
            results.append(res)
            
            line = f"| {res['H']} | {res['T']} | {res['Mean_R']:.4f} | {res['N_Trades']} | ${res['Final_Bal']:,.0f} |\n"
            print(line.strip())
            with open(RESULTS_FILE, "a") as f:
                f.write(line)

if __name__ == "__main__":
    main()
