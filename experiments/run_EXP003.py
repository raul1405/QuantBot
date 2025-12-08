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

RESULTS_FILE = "experiments/exp_003_v2_vs_v1.md"

# Feature Definitions
ALL_FEATURES = None # Default in AlphaEngine

# "Lean" Feature Set for v2.0
LEAN_FEATURES = [
    # Advanced
    'Log_Returns', 'Vol_Ratio', 'Trend_Dist', 'RangeNorm',
    'Downside_Vol_50', 'Upside_Vol_50', 'Skew_50', 'Kurt_50',
    'Asset_DD_200', 'Trend_Regime_Duration', 'Vol_Regime_Duration', 'Efficiency_Ratio',
    # Lags
    'Ret_Lag1', 'Ret_Lag2', 'Ret_Lag3',
    # TimeMeta
    'Hour', 'DayOfWeek', 'Is_Asia', 'Is_London', 'Is_NY'
]

def run_config(version_name, lookahead, threshold, feature_list=None):
    print(f"\n[EXP 003] Running {version_name} (H={lookahead}, T={threshold})...")
    
    config = Config()
    config.alpha_target_lookahead = lookahead
    config.alpha_return_threshold = threshold
    
    # Load Data (Safe Range)
    loader = DataLoader(config)
    data = loader.load_data("2024-01-01", "2025-12-08")
    
    fe = FeatureEngine(config)
    re = RegimeEngine(config)
    data = fe.add_features_all(data)
    data = re.add_regimes_all(data)
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
    
    # Inject Feature List if provided
    if feature_list:
        alpha.feature_cols = feature_list
        print(f"  > Using Clean Feature Set ({len(feature_list)} features).")
    else:
        print(f"  > Using All Features ({len(alpha.feature_cols)} features).")

    ens = EnsembleSignal(config)
    crisis = CrisisAlphaEngine(config)
    
    while current_date < end_date:
        test_end = current_date + pd.Timedelta(days=test_window_days)
        if test_end > end_date: test_end = end_date
        train_start = current_date - pd.Timedelta(days=train_window_days)
        if train_start < start_date: train_start = start_date
        
        train_data = {s: df.loc[train_start:current_date].copy() for s, df in full_df_map.items()}
        test_data = {s: df.loc[current_date:test_end].copy() for s, df in full_df_map.items()}
        
        # Ensure feature set persists
        if feature_list: alpha.feature_cols = feature_list
            
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
        
    # Stats
    trades = pb.account.trade_history
    if not trades: return None
    
    df_trades = pd.DataFrame(trades)
    
    # 1. Basic Stats
    mean_r = df_trades['R_Multiple'].mean()
    n_trades = len(df_trades)
    profit = pb.account.balance - 100000
    
    # 2. High Vol Slice Logic
    # Use Entry Context
    if 'Entry_Vol_Regime' in df_trades.columns:
        high_vol_trades = df_trades[df_trades['Entry_Vol_Regime'] == 'HIGH']
    else:
        # Fallback if logging failed
        high_vol_trades = pd.DataFrame()
        
    if len(high_vol_trades) > 0:
        hv_r = high_vol_trades['R_Multiple'].mean()
        hv_dd = high_vol_trades['R_Multiple'].cumsum().min() # Crude inner DD
    else:
        hv_r = 0; hv_dd = 0
        
    # 3. Monte Carlo (Quick Light Version)
    mc = MonteCarloEngine(config)
    # Mocking Account for MC
    mc.account = pb.account
    # Run fractional bootstrap (custom logic to capture pass rate silently?)
    # We'll just print it and parse manually or rely on 'analyze_trades'.
    
    return {
        "Version": version_name,
        "Mean_R": mean_r,
        "Trades": n_trades,
        "Profit": profit,
        "HV_R": hv_r,
        "HV_Count": len(high_vol_trades)
    }

def main():
    with open(RESULTS_FILE, "w") as f:
        f.write("# EXP_003: v1 Setup vs v2 Candidate\n\n")
        f.write("| Version | Config | Mean R | Trades | Profit | High-Vol R | HV Trades |\n")
        f.write("|---|---|---|---|---|---|---|\n")
    
    # Run v1.0 Baseline
    res_v1 = run_config("v1.0 (Baseline)", lookahead=3, threshold=0.001, feature_list=None)
    
    line_v1 = f"| {res_v1['Version']} | H=3/T=0.001/All | {res_v1['Mean_R']:.4f} | {res_v1['Trades']} | ${res_v1['Profit']:,.0f} | {res_v1['HV_R']:.4f} | {res_v1['HV_Count']} |\n"
    print(line_v1.strip())
    with open(RESULTS_FILE, "a") as f: f.write(line_v1)
        
    # Run v2.0 Candidate
    res_v2 = run_config("v2.0 (Candidate)", lookahead=5, threshold=0.0005, feature_list=LEAN_FEATURES)
    
    line_v2 = f"| {res_v2['Version']} | H=5/T=0.0005/Lean | {res_v2['Mean_R']:.4f} | {res_v2['Trades']} | ${res_v2['Profit']:,.0f} | {res_v2['HV_R']:.4f} | {res_v2['HV_Count']} |\n"
    print(line_v2.strip())
    with open(RESULTS_FILE, "a") as f: f.write(line_v2)

if __name__ == "__main__":
    main()
