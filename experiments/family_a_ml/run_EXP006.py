import sys
import os
import pandas as pd
import numpy as np
from dataclasses import replace

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quant_backtest import (
    Config, DataLoader, FeatureEngine, RegimeEngine, 
    AlphaEngine, EnsembleSignal, CrisisAlphaEngine, Backtester,
    MonteCarloEngine
)

RESULTS_FILE = "experiments/exp_006_stress_test.md"

def run_stress_test(variant_name, cost_basis):
    print(f"\n[EXP 006] Running {variant_name} (Cost={cost_basis:.4f})...")
    
    config = Config()
    # v2.1 Defaults (H=5, T=0.001)
    config.transaction_cost = cost_basis # Override cost
    
    # Load Data 
    loader = DataLoader(config)
    data = loader.load_data("2024-01-01", "2025-12-08")
    
    fe = FeatureEngine(config)
    re = RegimeEngine(config)
    data = fe.add_features_all(data)
    data = re.add_regimes_all(data)
    
    # Walk-Forward Setup
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
    
    pb = Backtester(config)
    alpha = AlphaEngine(config)
    ens = EnsembleSignal(config)
    crisis = CrisisAlphaEngine(config)
    
    while current_date < end_date:
        test_end = current_date + pd.Timedelta(days=test_window_days)
        if test_end > end_date: test_end = end_date
        train_start = current_date - pd.Timedelta(days=train_window_days)
        if train_start < start_date: train_start = start_date
        
        train_data = {s: df.loc[train_start:current_date].copy() for s, df in full_df_map.items()}
        test_data = {s: df.loc[current_date:test_end].copy() for s, df in full_df_map.items()}
        
        alpha.train_model(train_data)
        test_data = alpha.add_signals_all(test_data)
        test_data = ens.add_ensemble_all(test_data)
        test_data = crisis.add_crisis_signals(test_data)
        
        chunk_bt = Backtester(config)
        chunk_bt.account.balance = pb.account.balance
        chunk_bt.account.positions = list(pb.account.positions)
        chunk_bt.run_backtest(test_data)
        
        pb.account.balance = chunk_bt.account.balance
        pb.account.positions = list(chunk_bt.account.positions)
        pb.account.trade_history.extend(chunk_bt.account.trade_history)
        
        current_date = test_end
        
    trades = pb.account.trade_history
    if not trades: return None
    
    df_trades = pd.DataFrame(trades)
    
    return {
        "Variant": variant_name,
        "Mean_R": df_trades['R_Multiple'].mean(),
        "Trades": len(df_trades),
        "Profit": pb.account.balance - 100000,
        "Max_DD_Pct": (pb.account.equity - pb.account.peak_equity) / pb.account.peak_equity * 100
    }

def main():
    with open(RESULTS_FILE, "w") as f:
        f.write("# EXP_006: Stress Test (Costs & Slippage)\n\n")
        f.write("| Config | Mean R | Trades | Profit | Max DD |\n")
        f.write("|---|---|---|---|---|\n")
    
    # 1. Baseline (1bp)
    res_a = run_stress_test("A: Baseline (1bp)", 0.0001)
    line_a = f"| {res_a['Variant']} | {res_a['Mean_R']:.4f} | {res_a['Trades']} | ${res_a['Profit']:,.0f} | {res_a['Max_DD_Pct']:.2f}% |\n"
    print(line_a.strip())
    with open(RESULTS_FILE, "a") as f: f.write(line_a)
    
    # 2. Stress (2bp)
    res_b = run_stress_test("B: Stress (2bp)", 0.0002)
    line_b = f"| {res_b['Variant']} | {res_b['Mean_R']:.4f} | {res_b['Trades']} | ${res_b['Profit']:,.0f} | {res_b['Max_DD_Pct']:.2f}% |\n"
    print(line_b.strip())
    with open(RESULTS_FILE, "a") as f: f.write(line_b)
    
    # 3. Extreme (3bp) - Optional
    # res_c = run_stress_test("C: Extreme (3bp)", 0.0003)
    # line_c = f"| {res_c['Variant']} | {res_c['Mean_R']:.4f} | {res_c['Trades']} | ${res_c['Profit']:,.0f} | {res_c['Max_DD_Pct']:.2f}% |\n"
    # with open(RESULTS_FILE, "a") as f: f.write(line_c)

if __name__ == "__main__":
    main()
