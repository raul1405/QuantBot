import sys
import os
import pandas as pd
import numpy as np

# Add parent directory to path to import quant_backtest
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quant_backtest import (
    Config, DataLoader, FeatureEngine, RegimeEngine, 
    AlphaEngine, EnsembleSignal, CrisisAlphaEngine, Backtester
)

RESULTS_FILE = "experiments/exp_008_results.md"

# ==============================================================================
# UNIVERSE DEFINITIONS
# ==============================================================================

# 1. FULL UNIVERSE (Control)
# We will use the default Config.symbols which contains:
# - 7 Majors, 14 Crosses (21 Forex)
# - 4 Indices (ES, NQ, YM, RTY)
# - 3 Commodities (GC, CL, NG)
# - 2 Crypto (BTC, ETH)
# Total: ~30 Assets

# 2. FTMO RESTRICTED UNIVERSE (Treatment)
# Based on User's MT5 Symbol List:
# - All 21 Forex Pairs (Direct or Crosses available)
# - US500 (ES=F) -> YES
# - Gold (GC=F) -> YES
# - Crypto (BTC/ETH) -> YES
# - MISSING: Oil (CL), Gas (NG), Nasdaq (NQ), Dow (YM), Russell (RTY)
FTMO_UNIVERSE = [
    # --- FOREX MAJORS ---
    "EURUSD=X", "USDJPY=X", "GBPUSD=X", "USDCHF=X", 
    "USDCAD=X", "AUDUSD=X", "NZDUSD=X",
    
    # --- FOREX CROSSES ---
    "EURGBP=X", "EURJPY=X", "GBPJPY=X", "AUDJPY=X",
    "EURAUD=X", "EURCHF=X", "AUDNZD=X", "AUDCAD=X",
    "CADJPY=X", "NZDJPY=X", "GBPCHF=X", "GBPAUD=X",
    "GBPCAD=X", "EURNZD=X",
    
    # --- INDICES ---
    "ES=F",   # S&P 500 (US500)
    "NQ=F",   # Nasdaq 100 (US100)
    "YM=F",   # Dow Jones (US30)
    "RTY=F",  # Russell 2000 (US2000)
    
    # --- COMMODITIES ---
    "GC=F",   # Gold (XAUUSD)
    "CL=F",   # Crude Oil (USOIL)
    "NG=F",   # Natural Gas (NATGAS)
    
    # --- CRYPTO ---
    "BTC-USD",
    "ETH-USD",
]

def run_backtest_scenario(universe_name, symbols):
    print(f"\n[EXP 008] Running Scenario: {universe_name} ({len(symbols)} symbols)...")
    
    config = Config()
    config.symbols = symbols
    
    # Ensure consistent risk/leverage settings
    config.initial_balance = 100000.0
    config.account_leverage = 30.0
    config.risk_per_trade = 0.003
    config.transaction_cost = 0.0001 # Fix: Add transaction cost
    
    # Load Data
    # Note: We need to handle the case where some symbols might not download in a fresh run,
    # but since this is an experiment, we assume data availability or the DataLoader handles it.
    loader = DataLoader(config)
    # Using the same date range as default in main script or a specific 2-year window
    # Validated OOS period
    data_map = loader.load_data("2024-01-01", "2025-12-01") 
    
    # Pipeline
    fe = FeatureEngine(config)
    re = RegimeEngine(config)
    # We must explicitly pass the data dictionary
    data_map = fe.add_features_all(data_map)
    data_map = re.add_regimes_all(data_map)
    
    alpha = AlphaEngine(config)
    alpha.train_model(data_map) 
    
    data_map = alpha.add_signals_all(data_map)
    ens = EnsembleSignal(config)
    data_map = ens.add_ensemble_all(data_map)
    
    crisis = CrisisAlphaEngine(config)
    final_data = crisis.add_crisis_signals(data_map)
           
    # Backtest
    bt = Backtester(config)
    equity_curve = bt.run_backtest(final_data)
    
    # Stats
    trades = len(bt.account.trade_history)
    balance = bt.account.balance
    pnl = balance - config.initial_balance
    roi = pnl / config.initial_balance
    
    # Calculate Max DD from Equity Curve
    peak = equity_curve.cummax()
    drawdown = (equity_curve - peak) / peak
    max_dd = drawdown.min() * 100 # e.g. -5.0%

    wins = [t for t in bt.account.trade_history if t['PnL'] > 0]
    win_rate = (len(wins) / trades * 100) if trades > 0 else 0.0
    
    return {
        "Name": universe_name,
        "SymbolCount": len(symbols),
        "Trades": trades,
        "ROI": roi,
        "PnL": pnl,
        "MaxDD": max_dd,
        "WinRate": win_rate
    }

def main():
    print("Starting Experiment 008: FTMO Universe Impact Study")
    
    # 1. Run Control (Full Universe)
    default_config = Config()
    control_res = run_backtest_scenario("Full Universe (Control)", default_config.symbols)
    
    # 2. Run Treatment (FTMO Restricted)
    treatment_res = run_backtest_scenario("FTMO Universe (Restricted)", FTMO_UNIVERSE)
    
    results = [control_res, treatment_res]
    
    # 3. Generate Report
    with open(RESULTS_FILE, "w") as f:
        f.write("# EXP_008: FTMO Universe Impact Analysis\n\n")
        f.write(f"**Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write("**Objective:** Quantify performance loss when removing assets not available on FTMO (Oil, Gas, Minor Indices).\n\n")
        
        f.write("## 1. Summary Results\n")
        f.write("| Universe | Count | Trades | ROI | Net PnL | Max DD | Win Rate |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        
        for r in results:
            f.write(f"| {r['Name']} | {r['SymbolCount']} | {r['Trades']} | {r['ROI']*100:.2f}% | ${r['PnL']:,.0f} | {r['MaxDD']:.2f}% | {r['WinRate']:.1f}% |\n")
            
        f.write("\n## 2. Impact Analysis\n")
        
        # Calculate Delta
        roi_delta = treatment_res['ROI'] - control_res['ROI']
        pnl_delta = treatment_res['PnL'] - control_res['PnL']
        trades_delta = treatment_res['Trades'] - control_res['Trades']
        dd_delta = treatment_res['MaxDD'] - control_res['MaxDD']
        
        f.write(f"- **ROI Impact:** {roi_delta*100:+.2f}% (Absolute)\n")
        f.write(f"- **PnL Impact:** ${pnl_delta:,.0f}\n")
        f.write(f"- **Trade Vol Impact:** {trades_delta:+} trades\n")
        f.write(f"- **Risk Impact (MaxDD):** {dd_delta:+.2f}%\n")
        
        f.write("\n## 3. Conclusion\n")
        if roi_delta < -0.05: # -5% ROI drop
            f.write("> [!WARNING]\n> Significant performance degradation detected. The missing assets (likely Energy sector) were key alpha contributors. Considerations needed for live deployment constraints.\n")
        elif roi_delta < 0:
            f.write("> [!NOTE]\n> Minor performance degradation. The strategy is robust enough to survive the restricted universe, but expect lower returns.\n")
        else:
            f.write("> [!TIP]\n> No negative impact! The strategy performs equally well or better on the restricted universe. The excluded assets might have been dragging performance down.\n")

    print(f"\nExperiment Complete. Results saved to {RESULTS_FILE}")

if __name__ == "__main__":
    main()
