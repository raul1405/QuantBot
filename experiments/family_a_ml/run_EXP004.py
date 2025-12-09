import sys
import os
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quant_backtest import (
    Config, DataLoader, FeatureEngine, RegimeEngine, 
    AlphaEngine, EnsembleSignal, CrisisAlphaEngine, Backtester,
    MonteCarloEngine
)

RESULTS_FILE = "experiments/exp_004_threshold_tuning.md"

# Validated v2 "Lean" Feature Set
LEAN_FEATURES = [
    'Log_Returns', 'Vol_Ratio', 'Trend_Dist', 'RangeNorm',
    'Downside_Vol_50', 'Upside_Vol_50', 'Skew_50', 'Kurt_50',
    'Asset_DD_200', 'Trend_Regime_Duration', 'Vol_Regime_Duration', 'Efficiency_Ratio',
    'Ret_Lag1', 'Ret_Lag2', 'Ret_Lag3',
    'Hour', 'DayOfWeek', 'Is_Asia', 'Is_London', 'Is_NY'
]

THRESHOLDS = [0.0005, 0.00075, 0.0010]

def run_threshold_test(threshold):
    lookahead = 5 # Fixed for v2
    print(f"\n[EXP 004] Testing Threshold={threshold} (H=5, Lean Features)...")
    
    config = Config()
    config.alpha_target_lookahead = lookahead
    config.alpha_return_threshold = threshold
    
    # Load Data 
    loader = DataLoader(config)
    data = loader.load_data("2024-01-01", "2025-12-08")
    
    fe = FeatureEngine(config)
    re = RegimeEngine(config)
    data = fe.add_features_all(data)
    data = re.add_regimes_all(data)
    data = re.add_regimes_all(data)
    
    # Walk-Forward
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
    alpha.feature_cols = LEAN_FEATURES # Enforce Lean Set
    
    ens = EnsembleSignal(config)
    crisis = CrisisAlphaEngine(config)
    
    while current_date < end_date:
        test_end = current_date + pd.Timedelta(days=test_window_days)
        if test_end > end_date: test_end = end_date
        train_start = current_date - pd.Timedelta(days=train_window_days)
        if train_start < start_date: train_start = start_date
        
        train_data = {s: df.loc[train_start:current_date].copy() for s, df in full_df_map.items()}
        test_data = {s: df.loc[current_date:test_end].copy() for s, df in full_df_map.items()}
        
        alpha.feature_cols = LEAN_FEATURES
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
        "Threshold": threshold,
        "Mean_R": df_trades['R_Multiple'].mean(),
        "Trades": len(df_trades),
        "Profit": pb.account.balance - 100000,
        "Max_DD_Pct": (pb.account.equity - pb.account.peak_equity) / pb.account.peak_equity * 100 # Approx end state, ideally track min
    }

def main():
    with open(RESULTS_FILE, "w") as f:
        f.write("# EXP_004: Threshold Tuning (v2 spec)\n\n")
        f.write("| Threshold | Mean R | Trades | Profit | Comment |\n")
        f.write("|---|---|---|---|---|\n")
    
    for t in THRESHOLDS:
        res = run_threshold_test(t)
        line = f"| {res['Threshold']} | {res['Mean_R']:.4f} | {res['Trades']} | ${res['Profit']:,.0f} | TBD |\n"
        print(line.strip())
        with open(RESULTS_FILE, "a") as f: f.write(line)

if __name__ == "__main__":
    main()
