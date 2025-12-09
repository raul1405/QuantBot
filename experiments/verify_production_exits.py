import sys
import os
import pandas as pd
import numpy as np

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from quant_backtest import Config, DataLoader, FeatureEngine, RegimeEngine, AlphaEngine, EnsembleSignal, CrisisAlphaEngine, Backtester

def verify_exits():
    print("=" * 70)
    print("VERIFYING PRODUCTION SIGNALS-DRIVEN EXITS")
    print("=" * 70)
    
    # 1. Setup Config
    config = Config()
    config.symbols = config.symbols[:5] # Test on subset for speed
    config.mode = "BACKTEST"
    
    # 2. Check Parameters
    print(f"Signal Decay Threshold: {getattr(config, 'signal_decay_threshold', 'MISSING')}")
    print(f"Emergency SL Mult: {getattr(config, 'emergency_sl_mult', 'MISSING')}")
    
    if not hasattr(config, 'signal_decay_threshold'):
        print("FAIL: Config missing signal_decay_threshold")
        return

    # 3. Load Data
    loader = DataLoader(config)
    # Use standard 2024 range
    data = loader.load_data('2024-06-01', '2024-12-01')
    
    # 4. Pipeline
    fe = FeatureEngine(config)
    re = RegimeEngine(config)
    ae = AlphaEngine(config)
    es = EnsembleSignal(config)
    
    data = fe.add_features_all(data)
    data = re.add_regimes_all(data)
    ae.train_model(data)
    data = ae.add_signals_all(data)
    data = es.add_ensemble_all(data)
    
    # 5. Run Backtest
    bt = Backtester(config)
    bt.run_backtest(data)
    
    # 6. Analyze Exits
    trades = pd.DataFrame(bt.account.trade_history)
    
    if trades.empty:
        print("No trades generated.")
        return
        
    print("\nEXIT REASON DISTRIBUTION:")
    exit_counts = trades['Reason'].value_counts()
    print(exit_counts)
    
    total = len(trades)
    sl_count = exit_counts.get('EmergencySL', 0) + exit_counts.get('SL', 0)
    signal_count = exit_counts.get('SignalDecay', 0) + exit_counts.get('SignalFlip', 0)
    
    print(f"\nStats:")
    print(f"Total Trades: {total}")
    print(f"Signal Exits: {signal_count} ({signal_count/total:.1%})")
    print(f"Emergency SL: {sl_count} ({sl_count/total:.1%})")
    print(f"Time Exits: {exit_counts.get('TimeExit', 0)} ({exit_counts.get('TimeExit', 0)/total:.1%})")
    
    if sl_count / total < 0.05:
        print("\n✅ PASS: Emergency SL is rare (<5%)")
    else:
        print("\n⚠️ WARNING: SL rate seems high")

    if signal_count > 0:
         print("✅ PASS: Signal Exits are triggering")
    else:
         print("❌ FAIL: No Signal Exits triggered")

if __name__ == "__main__":
    verify_exits()
