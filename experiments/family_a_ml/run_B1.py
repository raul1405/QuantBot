import sys
import os
import pandas as pd
import numpy as np

# Add parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quant_backtest import (
    Config, DataLoader, FeatureEngine, RegimeEngine, 
    AlphaEngine, EnsembleSignal, CrisisAlphaEngine, Backtester
)

# === EXPERIMENT B1 CONFIGURATION ===
# Using the Winner from A1? 
# The user didn't explicitly say "Use A1 winner for B1". 
# But usually we want to improve on the best. 
# However, to be scientifically rigorous vs Baseline v1.0, we should stick to v1.0 Config 
# OR use the new candidate.
# The prompt "Baseline is sacred" suggests comparing vs Frozen v1.
# But for feature importance, using the stronger H=5 signal might be clearer.
# Let's stick to Baseline (H=3, T=0.001) for now to keep the "Baseline + 1 Tweak" rule strictly.
# If we used H=5, we'd be changing TWO things (Horizon + Features).
# So: Config = Baseline defaults.

RESULTS_FILE = "experiments/exp_B1_results.md"

# Define Groups
GROUPS = {
    "Basic": [
        'Z_Score', 'ATR', 'Momentum', 'Volatility'
    ],
    "Advanced": [
        'Log_Returns', 'Vol_Ratio', 'Trend_Dist', 'RangeNorm',
        'Downside_Vol_50', 'Upside_Vol_50', 'Skew_50', 'Kurt_50',
        'Asset_DD_200', 'Trend_Regime_Duration', 'Vol_Regime_Duration', 'Efficiency_Ratio'
    ],
    "Lags": [
        'Ret_Lag1', 'Ret_Lag2', 'Ret_Lag3'
    ],
    "CrossSectional": [
        'Mom_24h_rank', 'Mom_5d_rank', 'Vol_rank'
    ],
    "TimeMeta": [
        'Hour', 'DayOfWeek', 'Is_Asia', 'Is_London', 'Is_NY'
    ],
    "ContinuousRegime": [
        'Vol_Intensity', 'Vol_Pct'
    ]
}

def get_feature_list(remove_group=None):
    # Flatten all groups except the removed one
    features = []
    for name, cols in GROUPS.items():
        if name != remove_group:
            features.extend(cols)
    return list(set(features)) # Dedup just in case

def run_ablation(remove_group):
    tag = f"Remove_{remove_group}" if remove_group else "Baseline_All"
    print(f"\n[EXP B1] Running Ablation: {tag}")
    
    config = Config()
    # Baseline Config (H=3, T=0.001) per Frozen Spec v1.0
    config.alpha_target_lookahead = 3
    config.alpha_return_threshold = 0.001
    
    # Load Data (Safe Range)
    loader = DataLoader(config)
    data = loader.load_data("2024-01-01", "2025-12-08")
    
    # Features
    fe = FeatureEngine(config)
    re = RegimeEngine(config)
    data = fe.add_features_all(data)
    data = re.add_regimes_all(data)
    data = re.add_regimes_all(data) # Ensuring
    
    # Setup Walk-Forward
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
    # INJECT SELECTED FEATURES
    target_cols = get_feature_list(remove_group)
    # Verify these cols exist in dataframe
    # Some might be missing if I typo'd.
    # AlphaEngine handles missing cols by crashing/warning or skipping.
    # We should ensure they align with what FeatureEngine produces.
    # For now, trust the GROUPS mapping matches code.
    alpha.feature_cols = target_cols
    
    ens = EnsembleSignal(config)
    crisis = CrisisAlphaEngine(config)
    
    # Backtest Loop
    while current_date < end_date:
        test_end = current_date + pd.Timedelta(days=test_window_days)
        if test_end > end_date: test_end = end_date
        
        train_start = current_date - pd.Timedelta(days=train_window_days)
        if train_start < start_date: train_start = start_date
        
        train_data = {s: df.loc[train_start:current_date].copy() for s, df in full_df_map.items()}
        test_data = {s: df.loc[current_date:test_end].copy() for s, df in full_df_map.items()}
        
        # Override feature cols again just in case train_model resets it? 
        # No, train_model uses self.feature_cols. 
        alpha.feature_cols = target_cols
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
    if not trades:
        return {"Tag": tag, "Mean_R": 0, "N": 0, "Bal": pb.account.balance}
        
    df_trades = pd.DataFrame(trades)
    return {
        "Tag": tag,
        "Mean_R": df_trades['R_Multiple'].mean(),
        "N": len(df_trades),
        "Bal": pb.account.balance
    }

def main():
    experiments = [None] + list(GROUPS.keys()) # None = All Features
    
    with open(RESULTS_FILE, "w") as f:
        f.write("| Experiment | Mean R | Trades | Profit |\n")
        f.write("|---|---|---|---|\n")
        
    for grp in experiments:
        res = run_ablation(grp)
        line = f"| {res['Tag']} | {res['Mean_R']:.4f} | {res['N']} | ${res['Bal'] - 100000:,.0f} |\n"
        print(line.strip())
        with open(RESULTS_FILE, "a") as f:
            f.write(line)

if __name__ == "__main__":
    main()
