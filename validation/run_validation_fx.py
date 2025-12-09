
"""
RUN VALIDATION LAB (FX STRATEGY)
================================
Executes the FX Strategy and feeds the results into the Institutional Validator.
"""

import sys
import os
import pandas as pd
import numpy as np

# fix path to import quant_backtest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quant_backtest import (
    Config, DataLoader, FeatureEngine, RegimeEngine, 
    AlphaEngine, EnsembleSignal, Backtester
)

from validation.strategy_validator import StrategyValidator

def main():
    print("="*60)
    print("RUNNING INSTITUTIONAL VALIDATION (FX STRATEGY)")
    print("="*60)
    
    config = Config()
    config.risk_per_trade = 0.015
    config.symbols = ["EURUSD=X", "USDJPY=X", "GBPUSD=X", 
               "USDCHF=X", "USDCAD=X", "AUDUSD=X", "NZDUSD=X", 
               "EURGBP=X", "EURJPY=X", "GBPJPY=X", 
               "AUDJPY=X", "EURAUD=X", "EURCHF=X"]
               
    print(f"[SETUP] Testing {len(config.symbols)} Symbols. Risk {config.risk_per_trade*100}%.")

    # 1. Backtest Pipeline
    loader = DataLoader(config)
    try: data = loader.load_data("2024-01-01", "2024-12-01")
    except: return

    fe = FeatureEngine(config)
    re = RegimeEngine(config)
    ae = AlphaEngine(config)
    es = EnsembleSignal(config)
    
    print("[PIPELINE] Processing Features & Signals...")
    data = fe.add_features_all(data)
    data = re.add_regimes_all(data)
    ae.train_model(data)
    data = ae.add_signals_all(data)
    data = es.add_ensemble_all(data)
    
    risks = [0.035, 0.045, 0.050, 0.060]
    
    for r in risks:
        print(f"\n[TEST] Simulating Risk: {r*100}%")
        config.risk_per_trade = r
        
        # Re-run pipeline? No, signals are same. Just Backtest logic changes (Size).
        # We need to re-run Size calc. Which happens inside run_backtest.
        # So we can just call bt.run_backtest(data) again!
        
        # Note: 'data' already has signals.
        bt = Backtester(config) # New Config
        equity_curve = pd.Series(bt.run_backtest(data))
        trades_df = pd.DataFrame(bt.account.trade_history)

        validator = StrategyValidator(trades_df, equity_curve, initial_balance=config.initial_balance)
        mc_res = validator.run_monte_carlo(sims=1000)
        a_metrics = validator.compute_account_metrics()
        
        print(f"  -> Return: {((equity_curve.iloc[-1]/config.initial_balance)-1)*100:.2f}%")
        print(f"  -> Max DD: {a_metrics.get('Max DD', 0)*100:.2f}%")
        print(f"  -> Pass Rate: {mc_res.get('Pass Probability (>10% Profit, No Bust)', 0)*100:.1f}%")
        print(f"  -> Bust Rate: {mc_res.get('Bust Probability (max DD > 10%)', 0)*100:.1f}%")

    # Final Run at 3.5% for Report
    config.risk_per_trade = 0.035

if __name__ == "__main__":
    main()
