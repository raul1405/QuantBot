
import sys
import os
import pandas as pd
import numpy as np

# Adjust path to import from current directory (sandbox)
sys.path.append(os.path.abspath("."))
sys.path.append(os.path.abspath("experiments/high_vol_family_c"))

from sandbox_engine import Config, DataLoader, FeatureEngine, RegimeEngine, AlphaEngine, EnsembleSignal, CrisisAlphaEngine, Backtester
from config.high_vol_universes import UNIVERSE_CRYPTO, UNIVERSE_INDICES_METALS

def run_family_c_scan(universe, start_date, end_date, label="Test"):
    print(f"\n[{label}] Running Family C Viability Scan (DAILY)...")
    print(f"  Range: {start_date} -> {end_date}")
    print(f"  Universe: {len(universe)} assets")
    
    cfg = Config()
    cfg.symbols = universe
    cfg.timeframe = "1d"  # DAILY to bypass 730d limit and capture trends
    
    # WFO Pars: 
    # Daily bars. 
    # Train: 4 Years? Or 2 Years?
    # 1 Year ~ 252 bars.
    # Let's train on 500 bars (2 years) and test on 60 bars (1 quarter).
    cfg.wfo_train_bars = 500
    cfg.wfo_test_bars = 60
    
    # Date buffer
    fetch_start = pd.Timestamp(start_date) - pd.Timedelta(days=750) # Need ~2 years buffer
    fetch_start_str = fetch_start.strftime("%Y-%m-%d")
    end_str = end_date
    
    try:
        loader = DataLoader(cfg)
        data = loader.load_data(fetch_start_str, end_str)
        
        if not data:
            print("  ERROR: No data loaded.")
            return None
            
        fe = FeatureEngine(cfg)
        re = RegimeEngine(cfg)
        alpha = AlphaEngine(cfg)
        
        data = fe.add_features_all(data)
        data = re.add_regimes_all(data)
        
        # Train & Predict (WFO)
        processed = alpha.train_predict_walk_forward(data)
        
        # Ensemble
        ens = EnsembleSignal(cfg)
        processed = ens.add_ensemble_all(processed)
        
        # Crisis
        crisis = CrisisAlphaEngine(cfg)
        processed = crisis.add_crisis_signals(processed)
        
        # 1. BASELINE BACKTEST
        bt = Backtester(cfg)
        
        test_data = {}
        for sym, df in processed.items():
            test_data[sym] = df.loc[start_date:end_date]
            
        bt.run_backtest(test_data)
        
        res_baseline = calculate_metrics(bt, cfg.initial_balance)
        print(f"  > Baseline Result: {res_baseline}")
        
        # 2. RANDOM BASELINE (The Control)
        # We re-run backtest but scramble signals
        print("  > Running Random Control...")
        
        # Copy test data to avoid mutation
        random_data = {s: d.copy() for s, d in test_data.items()}
        
        for sym, df in random_data.items():
            s_len = len(df)
            rand_sigs = np.random.choice([-1, 0, 1], size=s_len, p=[0.1, 0.8, 0.1])
            df['S_Alpha'] = rand_sigs
            df['S_Ensemble'] = rand_sigs # Override ensemble too
            
        bt_rand = Backtester(cfg)
        bt_rand.run_backtest(random_data)
        
        res_random = calculate_metrics(bt_rand, cfg.initial_balance)
        print(f"  > Random Result:   {res_random}")
        
        return {
            "Baseline": res_baseline,
            "Random": res_random
        }
        
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None

def calculate_metrics(bt, initial_balance):
    final_bal = bt.account.balance
    ret_pct = ((final_bal - initial_balance) / initial_balance) * 100
    trades = len(bt.account.trade_history)
    
    max_dd = 0.0
    peak = initial_balance
    sqn = 0.0
    
    if trades > 0:
        pnls = [t['PnL'] for t in bt.account.trade_history]
        wins = len([p for p in pnls if p > 0])
        win_rate = (wins / trades) * 100
        mean_pnl = np.mean(pnls)
        std_pnl = np.std(pnls)
        if std_pnl > 0:
            sqn = (mean_pnl / std_pnl) * np.sqrt(trades)
    else:
        win_rate = 0.0
    
    curr = initial_balance
    for t in bt.account.trade_history:
        curr = t['Balance']
        if curr > peak: peak = curr
        dd = (curr - peak) / peak
        if dd < max_dd: max_dd = dd
        
    return {
        "Return": round(ret_pct, 2),
        "MaxDD": round(max_dd * 100, 2),
        "Trades": trades,
        "WinRate": round(win_rate, 2),
        "SQN": round(sqn, 2)
    }

def main():
    print("=== HIGH-VOL FAMILY C: VIABILITY SCAN (DAILY) ===")
    
    # Use a long, safe history window (2 Years)
    start_date = "2023-01-01"
    end_date = "2024-11-30" 
    
    # 1. CRYPTO SCAN
    res_crypto = run_family_c_scan(UNIVERSE_CRYPTO, start_date, end_date, "Crypto Majors")
    
    # 2. INDICES SCAN
    res_indices = run_family_c_scan(UNIVERSE_INDICES_METALS, start_date, end_date, "Indices/Metals")
    
    print("\n\n=== FINAL FAMILY C VIABILITY REPORT DATA ===")
    print(f"Crypto: {res_crypto}")
    print(f"Indices: {res_indices}")

if __name__ == "__main__":
    main()
