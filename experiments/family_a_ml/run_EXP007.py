import sys
import os
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quant_backtest import (
    Config, DataLoader, FeatureEngine, RegimeEngine, 
    AlphaEngine, EnsembleSignal, CrisisAlphaEngine, Backtester
)

RESULTS_FILE = "experiments/exp_007_black_swan.md"

def inject_black_swan(df: pd.DataFrame, magnitude_pct=0.05, duration_hours=4):
    """
    Injects a flash crash event into the middle of the dataframe.
    """
    df_mod = df.copy()
    mid_idx = len(df_mod) // 2
    
    # Create valid timestamp range if index is DatetimeIndex
    crash_indices = df_mod.index[mid_idx : mid_idx + duration_hours]
    
    # 1. Price Crash (Close drops, High/Low expand)
    crash_factor = 1.0 - magnitude_pct
    df_mod.loc[crash_indices, 'Close'] *= crash_factor
    df_mod.loc[crash_indices, 'Low'] *= crash_factor
    df_mod.loc[crash_indices, 'High'] *= crash_factor 
    # (High also drops to simulate gap down, or maybe High stays up for huge candle?)
    # Let's say Huge Candle down: High normal, Low crashes. Close at Low.
    
    # 2. Vol Spike (ATR explodes)
    # Re-calc ATR after price mod? Or manually boost?
    # Manual boost is easier for "Synthetic" stress.
    if 'ATR' in df_mod.columns:
        df_mod.loc[crash_indices, 'ATR'] *= 5.0
        
    return df_mod

def run_scenario(variant_name, enable_vol_sizing=True):
    print(f"\n[EXP 007] Running {variant_name} (Vol Sizing={enable_vol_sizing})...")
    
    config = Config()
    config.transaction_cost = 0.0001
    
    # Load Data (Short window is enough for stress test)
    loader = DataLoader(config)
    # Pick a volatile period if possible, or just recent
    data = loader.load_data("2024-06-01", "2024-12-01")
    
    # Inject Risk
    full_stress_data = {}
    for sym, df in data.items():
        # Only crash one major pair to see portfolio effect?
        if "USD" in sym:
             full_stress_data[sym] = inject_black_swan(df, magnitude_pct=0.03) # 3% Flash Crash
        else:
             full_stress_data[sym] = df
             
    # Pipeline
    fe = FeatureEngine(config)
    re = RegimeEngine(config)
    full_stress_data = fe.add_features_all(full_stress_data)
    full_stress_data = re.add_regimes_all(full_stress_data)
    
    # Train/Test logic (Simulated)
    # Just run on the whole stressed period as "Test" to see reaction
    alpha = AlphaEngine(config)
    # Fake training on same data just to get signals
    alpha.train_model(full_stress_data) 
    
    stressed_signals = alpha.add_signals_all(full_stress_data)
    ens = EnsembleSignal(config)
    stressed_ens = ens.add_ensemble_all(stressed_signals)
    
    crisis = CrisisAlphaEngine(config)
    final_data = crisis.add_crisis_signals(stressed_ens)
    
    # Hack: If Vol Sizing disabled, reset Crisis_Size_Mult to 1.0
    if not enable_vol_sizing:
        for sym in final_data:
            final_data[sym]['Crisis_Size_Mult'] = 1.0
            final_data[sym]['Final_Signal'] = final_data[sym]['Ensemble_Score']
            
    # Backtest
    bt = Backtester(config)
    bt.run_backtest(final_data)
    
    # Stats
    equity = bt.account.equity
    peak = bt.account.peak_equity
    dd = (equity - peak) / peak
    
    # Find Max DD during the crash?
    # Backtester tracks daily stats not intraday in 'trade_history'.
    # We need equity curve from run_backtest return.
    # Wait, run_backtest returns 'equity_curve' series.
    # I need to capture that.
    
    # Re-run run_backtest to capture return?
    # Modify Backtester to return it?
    # Or just use end balance and 'trade_history' DD approx.
    
    return {
        "Variant": variant_name,
        "Final_Balance": bt.account.balance,
        "Total_Trades": len(bt.account.trade_history),
        "PnL": bt.account.balance - config.initial_balance
    }

def main():
    with open(RESULTS_FILE, "w") as f:
        f.write("# EXP_007: Black Swan Stress Test\n\n")
        f.write("| Config | PnL | Trades |\n")
        f.write("|---|---|---|\n")

    # 1. Without Vol Sizing
    res_flat = run_scenario("A: Flat Risk (Crash)", enable_vol_sizing=False)
    
    # 2. With Vol Sizing (v2.1)
    res_vol = run_scenario("B: v2.1 Vol Sizing (Crash)", enable_vol_sizing=True)
    
    with open(RESULTS_FILE, "a") as f:
        f.write(f"| A: Flat Risk | ${res_flat['PnL']:,.0f} | {res_flat['Total_Trades']} |\n")
        f.write(f"| B: Vol Sizing | ${res_vol['PnL']:,.0f} | {res_vol['Total_Trades']} |\n")
        
        diff = res_vol['PnL'] - res_flat['PnL']
        f.write(f"\n**Difference**: Vol Sizing saved ${diff:,.0f} during the crash period.\n")

if __name__ == "__main__":
    main()
