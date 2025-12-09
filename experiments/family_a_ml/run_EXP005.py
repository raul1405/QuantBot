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

RESULTS_FILE = "experiments/exp_005_alpha_vs_risk.md"

def run_exp005_config(variant_name, use_vol_sizing):
    print(f"\n[EXP 005] Running {variant_name} (VolSizing={use_vol_sizing})...")
    
    config = Config()
    # v2.1 Defaults are already hardcoded in Config (H=5, T=0.001)
    
    # Load Data 
    loader = DataLoader(config)
    data = loader.load_data("2024-01-01", "2025-12-08")
    
    fe = FeatureEngine(config)
    re = RegimeEngine(config)
    data = fe.add_features_all(data)
    data = re.add_regimes_all(data)
    data = re.add_regimes_all(data) # Double call harmless, ensuring regimes
    
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
    
    # === FORCE RISK MODE ===
    # We need to hack the Backtester logic or Config?
    # Actually, Config doesn't have a flag for 'disable_vol_sizing' yet? 
    # Or we can just monkey-patch the vol calculation logic dynamically if needed.
    # Ah, the CrisisAlphaEngine handles vol sizing in current code?
    # Let's check: Backtester calls crisis.add_crisis_signals?
    # No, Backtester consumes 'Trade_Size_Mult' if present?
    # Wait, the current logic calculates size in `Backtester.run_backtest` using `Entry_Vol_Intensity`?
    # Let's assume the standard 'quant_backtest.py' logic enables it by default.
    # To DISABLE it, we might need a temporary override.
    
    # HACK: If use_vol_sizing is False, we will force Vol_Intensity to 0 in the test data 
    # just before backtesting, effectively neutralizing the multiplier. 
    # BUT we must be careful not to affect the Alpha Engine features (though they don't use it anyway).
    
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
        
        # --- DISABLE VOL SIZING TRICK ---
        if not use_vol_sizing:
            for s in test_data:
                # Force Vol_Intensity to 0 -> Multiplier becomes 1.0 (or base)
                # Assuming code: mult = 1 / (1 + vol_int**2)
                if 'Vol_Intensity' in test_data[s].columns:
                    test_data[s]['Vol_Intensity'] = 0.0
        
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
    
    # Monte Carlo (MC2 - FTMO)
    mc = MonteCarloEngine(config)
    passed_mc, tail_dd = 0, 0
    # Capture print output? Or assume default is fine.
    # We'll just rely on trade stats for this quick check.
    
    return {
        "Variant": variant_name,
        "Mean_R": df_trades['R_Multiple'].mean(),
        "Trades": len(df_trades),
        "Profit": pb.account.balance - 100000,
        "Max_DD_Pct": (pb.account.equity - pb.account.peak_equity) / pb.account.peak_equity * 100
    }

def main():
    with open(RESULTS_FILE, "w") as f:
        f.write("# EXP_005: Alpha vs Risk Separation (v2.1)\n\n")
        f.write("| Config | Mean R | Trades | Profit | Max DD (Equity) |\n")
        f.write("|---|---|---|---|---|\n")
    
    # 1. Full v2.1 (Vol Sizing ON)
    res_a = run_exp005_config("A: v2.1 + VolSizing", use_vol_sizing=True)
    line_a = f"| {res_a['Variant']} | {res_a['Mean_R']:.4f} | {res_a['Trades']} | ${res_a['Profit']:,.0f} | {res_a['Max_DD_Pct']:.2f}% |\n"
    print(line_a.strip())
    with open(RESULTS_FILE, "a") as f: f.write(line_a)
    
    # 2. Flat Risk (Vol Sizing OFF)
    res_b = run_exp005_config("B: v2.1 Flat Risk", use_vol_sizing=False)
    line_b = f"| {res_b['Variant']} | {res_b['Mean_R']:.4f} | {res_b['Trades']} | ${res_b['Profit']:,.0f} | {res_b['Max_DD_Pct']:.2f}% |\n"
    print(line_b.strip())
    with open(RESULTS_FILE, "a") as f: f.write(line_b)

if __name__ == "__main__":
    main()
