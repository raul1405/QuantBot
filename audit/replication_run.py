
import sys
import os
import pandas as pd
import numpy as np
sys.path.append(os.path.abspath(".."))
sys.path.append(os.path.abspath("."))
sys.path.append(os.path.abspath("config"))

from quant_backtest import Config, DataLoader, FeatureEngine, RegimeEngine, AlphaEngine, EnsembleSignal, CrisisAlphaEngine, Backtester
from config.universe_definitions import UNIVERSE_MAJORS, UNIVERSE_LIQUID

def run_replication(universe, start_date, end_date, label="Test"):
    print(f"\n[{label}] Running Replication Backtest...")
    print(f"  Range: {start_date} -> {end_date}")
    print(f"  Universe: {len(universe)} pairs")
    
    cfg = Config()
    cfg.symbols = universe
    cfg.timeframe = "1h"
    
    # Force WFO params to be suitable for 1-year run
    # 1 year ~ 250 days ~ 6000 hours.
    # We use 180 day train (4300 bars), 30 day test.
    cfg.wfo_train_bars = 4000
    cfg.wfo_test_bars = 700 
    
    # Date buffer for training (need 180 days BEFORE start_date)
    # yfinance fetch needs to start earlier
    fetch_start = pd.Timestamp(start_date) - pd.Timedelta(days=200)
    fetch_start_str = fetch_start.strftime("%Y-%m-%d")
    end_str = end_date
    
    try:
        loader = DataLoader(cfg)
        data = loader.load_data(fetch_start_str, end_str)
        
        fe = FeatureEngine(cfg)
        re = RegimeEngine(cfg)
        alpha = AlphaEngine(cfg)
        
        data = fe.add_features_all(data)
        data = re.add_regimes_all(data)
        
        # Manually run Backtest Logic constrained to [start_date, end_date]
        # Actually quant_backtest.py does WFO automatically.
        # But we want to capture the specific stats.
        
        # Let's instantiate AlphaEngine and run WFO, but we rely on its internal loop.
        # It loops through the loaded data.
        # We just need to make sure we measure performance mainly on the OOS part.
        
        # Override Alpha's train/predict to return processed data
        processed = alpha.train_predict_walk_forward(data)
        
        # Add Ensemble
        ens = EnsembleSignal(cfg)
        processed = ens.add_ensemble_all(processed)
        
        # Crisis
        crisis = CrisisAlphaEngine(cfg)
        processed = crisis.add_crisis_signals(processed)
        
        # Run Backtest
        bt = Backtester(cfg)
        
        # Filter Data to strict test window?
        # WFO starts training after wfo_train_bars.
        # We want to measure performance from 'start_date' roughly.
        # But WFO handles the 'validity' start.
        # We just run on the resulting processed data.
        
        # Slice processed data to start strictly at start_date for PnL measurement
        test_data = {}
        for sym, df in processed.items():
            test_data[sym] = df.loc[start_date:end_date]
            
        bt.run_backtest(test_data)
        
        final_bal = bt.account.balance
        ret_pct = ((final_bal - cfg.initial_balance) / cfg.initial_balance) * 100
        trades = len(bt.account.trade_history)
        
        # Stats
        max_dd = 0.0
        equity_curve = [cfg.initial_balance]
        peak = cfg.initial_balance
        for t in bt.account.trade_history:
            bal = t['Balance']
            if bal > peak: peak = bal
            dd = (bal - peak) / peak
            if dd < max_dd: max_dd = dd
            
        win_rate = 0.0
        if trades > 0:
            wins = len([t for t in bt.account.trade_history if t['PnL'] > 0])
            win_rate = (wins / trades) * 100
            
        results = {
            "Return": ret_pct,
            "MaxDD": max_dd * 100,
            "Trades": trades,
            "WinRate": win_rate
        }
        print(f"  Result: {results}")
        return results
        
    except Exception as e:
        print(f"  ERROR: {e}")
        return None

def main():
    print("=== REPLICATION AUDIT (2023-2025 Limits) ===")
    
    # yfinance 1h limit is ~730 days from today.
    # If today is Dec 2025, 730 days ago is Dec 2023.
    # WAIT. The user system time is 2025.
    # So 730 days ago is late 2023.
    # 2023-01 is out of range.
    
    # Adjusted Windows:
    # Window A: Dec 2023 -> Nov 2024 (1 Year)
    # Window B: Dec 2024 -> Dec 2025 (YTD/Current)
    
    print("Adjusting windows to fit yfinance 730-day limit (approx late 2023 start)...")
    
    # Window A (History)
    res_A_maj = run_replication(UNIVERSE_MAJORS, "2024-01-01", "2024-12-31", "Window A (2024) Majors")
    res_A_liq = run_replication(UNIVERSE_LIQUID, "2024-01-01", "2024-12-31", "Window A (2024) Liquid")
    
    # Window B (Current Regime)
    # yfinance data might not go to today in this sim, but let's try.
    # User time is Dec 2025. So 2025 is valid.
    res_B_maj = run_replication(UNIVERSE_MAJORS, "2025-01-01", "2025-11-30", "Window B (2025 YTD) Majors")
    res_B_liq = run_replication(UNIVERSE_LIQUID, "2025-01-01", "2025-11-30", "Window B (2025 YTD) Liquid")
    
    print("\n\n=== FINAL REPLICATION REPORT DATA ===")
    print(f"Window A 2024 Majors: {res_A_maj}")
    print(f"Window A 2024 Liquid: {res_A_liq}")
    print(f"Window B 2025 Majors: {res_B_maj}")
    print(f"Window B 2025 Liquid: {res_B_liq}")

if __name__ == "__main__":
    main()
