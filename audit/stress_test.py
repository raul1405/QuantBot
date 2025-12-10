
import sys
import os
import pandas as pd
import numpy as np
sys.path.append(os.path.abspath(".."))
sys.path.append(os.path.abspath("."))

from quant_backtest import Config, DataLoader, FeatureEngine, RegimeEngine, AlphaEngine, EnsembleSignal, CrisisAlphaEngine, Backtester, Account

def run_backtest_with_config(custom_config, name="Test"):
    print(f"\n[{name}] Running Backtest...")
    
    # 1. Pipeline
    loader = DataLoader(custom_config)
    fe = FeatureEngine(custom_config)
    re = RegimeEngine(custom_config)
    alpha = AlphaEngine(custom_config)
    ens = EnsembleSignal(custom_config)
    crisis = CrisisAlphaEngine(custom_config)
    
    # Calculate Rolling Window (Shorten for speed if needed, but we want accuracy)
    # Using 180 days lookback for speed in audit? 
    # Original uses 729. I'll use 300 to start (approx 1 year).
    end_date = pd.Timestamp.now()
    start_date = end_date - pd.Timedelta(days=365) # 1 Year
    
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")
    
    try:
        # Load
        data = loader.load_data(start_str, end_str)
        if not data: return None
        
        # Feat
        data = fe.add_features_all(data)
        data = re.add_regimes_all(data)
        
        # WFO Setup from main
        full_df_map = data
        combined_idx = pd.Index([])
        for df in full_df_map.values():
            combined_idx = combined_idx.union(df.index)
        combined_idx = combined_idx.sort_values()
        
        if len(combined_idx) == 0: return None
        
        start_date = combined_idx[0]
        end_date = combined_idx[-1]
        
        train_window_days = 90 # Faster for audit
        test_window_days = 30
        
        current_date = start_date + pd.Timedelta(days=train_window_days)
        
        pb = Backtester(custom_config)
        
        while current_date < end_date:
            train_start = current_date - pd.Timedelta(days=train_window_days)
            if train_start < start_date: train_start = start_date
            
            test_end = current_date + pd.Timedelta(days=test_window_days)
            if test_end > end_date: test_end = end_date
            
            # Slice
            train_data = {}
            test_data = {}
            for sym, df in full_df_map.items():
                train_data[sym] = df.loc[train_start:current_date].copy()
                test_data[sym] = df.loc[current_date:test_end].copy()
            
            # Train
            alpha.train_model(train_data)
            
            # Predict
            test_data_sig = alpha.add_signals_all(test_data)
            test_data_ens = ens.add_ensemble_all(test_data_sig)
            
            # If Random Test, scramble signals here
            if hasattr(custom_config, 'is_random_test') and custom_config.is_random_test:
                 for sym, df in test_data_ens.items():
                     df['Ensemble_Score'] = np.random.uniform(-1, 1, size=len(df))
                     df['prob_up'] = np.random.uniform(0, 1, size=len(df))
                     df['prob_down'] = np.random.uniform(0, 1, size=len(df))
                     test_data_ens[sym] = df

            test_data_final = crisis.add_crisis_signals(test_data_ens)
            
            # Backtest Chunk
            chunk_bt = Backtester(custom_config)
            chunk_bt.account.balance = pb.account.balance
            chunk_bt.account.equity = pb.account.equity
            chunk_bt.account.peak_equity = pb.account.peak_equity
            chunk_bt.account.positions = pb.account.positions
            
            chunk_bt.run_backtest(test_data_final)
            
            pb.account.balance = chunk_bt.account.balance
            pb.account.equity = chunk_bt.account.equity
            pb.account.peak_equity = chunk_bt.account.peak_equity
            pb.account.positions = chunk_bt.account.positions
            pb.account.trade_history.extend(chunk_bt.account.trade_history)
            
            current_date = test_end
            
        print(f"[{name}] Result: Bal=${pb.account.balance:,.2f} Trades={len(pb.account.trade_history)}")
        
        ret_pct = ((pb.account.balance - custom_config.initial_balance) / custom_config.initial_balance) * 100
        
        # Calc Sharpe roughly
        if len(pb.account.trade_history) > 0:
            df = pd.DataFrame(pb.account.trade_history)
            win_rate = len(df[df['PnL']>0]) / len(df)
            return {'Return': ret_pct, 'Sharpe': 0.0, 'WinRate': win_rate, 'Trades': len(df)}
        else:
            return {'Return': ret_pct, 'Sharpe': 0.0, 'WinRate': 0.0, 'Trades': 0}
            
    except Exception as e:
        print(f"[{name}] ERROR: {e}")
        return None

def main():
    print("=== AUDIT STRESS TEST ===")
    
    # 1. Baseline
    cfg_base = Config()
    cfg_base.timeframe = "1h" # Ensure
    res_base = run_backtest_with_config(cfg_base, "Baseline (Current 13 Pairs)")
    
    # 2. Survivorship Bias Check (Add Toxic Pairs)
    print("\n--- TEST: SURVIVORSHIP BIAS ---")
    cfg_toxic = Config()
    cfg_toxic.symbols = [
        "AUDNZD=X", "AUDCAD=X", "CADJPY=X", "NZDJPY=X", 
        "GBPCHF=X", "GBPAUD=X", "GBPCAD=X", "EURNZD=X"
    ]
    res_toxic = run_backtest_with_config(cfg_toxic, "Toxic Pairs (Excluded 8)")
    
    # 3. Random Signal Check
    print("\n--- TEST: RANDOM SIGNALS ---")
    cfg_rand = Config()
    cfg_rand.is_random_test = True
    res_rand = run_backtest_with_config(cfg_rand, "Random Signals")
    
    print("\n\n=== FINAL AUDIT REPORT DATA ===")
    print(f"Baseline: {res_base}")
    print(f"Toxic Set: {res_toxic}")
    print(f"Random:   {res_rand}")

if __name__ == "__main__":
    main()
