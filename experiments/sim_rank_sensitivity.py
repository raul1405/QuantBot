
"""
RANK SENSITIVITY ANALYSIS
=========================
Tests the impact of the "Top N" filter.
Hypothesis:
- Top 1: Highest quality, lowest frequency (Current).
- Top 3: More trades, potential diversification benefit?
- Top 13 (All): Lower quality, higher transaction costs.

"""

import sys
import os
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quant_backtest import (
    Config, DataLoader, FeatureEngine, RegimeEngine, 
    AlphaEngine, EnsembleSignal, Backtester
)

def run_sim(config, data, top_n):
    print(f"\n[SIMULATION] Top {top_n} Assets...")
    
    # Update Config
    config.rank_top_n = top_n
    
    # Re-Run Alpha Engine Ranking Logic?
    # AlphaEngine stores signals in 'S_Alpha'.
    # We must regenerate 'S_Alpha'.
    # But AlphaEngine.add_signals_all calls add_rank_signals.
    # So we just re-run add_rank_signals.
    
    # We need to instantiate AlphaEngine to access methods?
    # Actually add_rank_signals is instance method.
    ae = AlphaEngine(config) 
    
    # Note: 'data' has 'prob_up', 'prob_down' from initial training.
    # We just need to re-apply rank logic.
    processed = ae.add_rank_signals(data.copy())
    
    # Re-run Ensemble (if it depends on S_Alpha)
    es = EnsembleSignal(config)
    processed = es.add_ensemble_all(processed)
    
    # Backtest
    bt = Backtester(config)
    bt.run_backtest(processed)
    
    stats = {
        'Top_N': top_n,
        'Return': (bt.account.balance - config.initial_balance) / config.initial_balance * 100,
        'MaxDD': 0.0, # Todo
        'Trades': len(bt.account.trade_history),
        'Sharpe': 0.0,
        'WinRate': 0.0
    }
    
    # Quick Sharpe/DD calc
    if bt.account.trade_history:
        trades = pd.DataFrame(bt.account.trade_history)
        if not trades.empty:
            stats['WinRate'] = (trades['PnL'] > 0).mean()
            if trades['PnL'].std() > 0:
                stats['Sharpe'] = trades['PnL'].mean() / trades['PnL'].std() * np.sqrt(len(trades)/252 * 252) # Roughly Trade Sharpe * sqrt(Trades)
                # Proper Annualized Sharpe:
                # Daily returns needed. 
                # Let's use Trade Sharpe as proxy for now or recompute equity curve.
                pass
            
    # MaxDD from equity curve
    if isinstance(bt.account.equity_curve, list):
         ec = pd.Series(bt.account.equity_curve)
         dd = (ec - ec.cummax()) / ec.cummax()
         stats['MaxDD'] = dd.min() * 100
        
    print(f"  > Ret: {stats['Return']:.2f}% | DD: {stats['MaxDD']:.2f}% | Trades: {stats['Trades']}")
    return stats

def main():
    print("Running Rank Sensitivity...")
    config = Config()
    # Ensure Rank Logic ON
    config.use_rank_logic = True
    config.risk_per_trade = 0.033 # 3.3% as per Live
    
    # 1. Load & Train ONCE
    loader = DataLoader(config)
    try:
        data = loader.load_data("2024-01-01", "2024-12-01") 
    except:
        return

    fe = FeatureEngine(config)
    re = RegimeEngine(config)
    ae = AlphaEngine(config)
    
    data = fe.add_features_all(data)
    data = re.add_regimes_all(data)
    ae.train_model(data)
    # ae.add_signals_model(data) # REMOVED
    # Wait, add_signals_all calls add_rank_signals.
    # Let's call a method that gets probs but doesn't rank?
    # 'generate_signals_with_probs' is per DF.
    
    # We'll just run add_signals_all (which ranks with default N=1). 
    # Then in loop we overwrite S_Alpha with new rank.
    # But wait, add_signals_all populates 'prob_up'.
    
    print("[TRAINING COMPLETE] Probs Generated.")
    
    # Pre-calculate Probs
    probs_data = {}
    for sym, df in data.items():
         sig, prob_df = ae.generate_signals_with_probs(df)
         # Merge probs into df
         df = df.copy()
         df['prob_up'] = prob_df['prob_up']
         df['prob_down'] = prob_df['prob_down']
         probs_data[sym] = df
    
    results = []
    
    # Test Cases
    for n in [1, 3, 13]:
        res = run_sim(config, probs_data, n)
        results.append(res)
        
    print("\n" + "="*60)
    print("RANK SENSTIVITY RESULTS")
    print("="*60)
    print(f"{'Top N':<6} | {'Return':<8} | {'Max DD':<8} | {'Trades':<6} | {'Win Rate':<8}")
    print("-" * 60)
    for r in results:
        print(f"{r['Top_N']:<6} | {r['Return']:<8.2f}% | {r['MaxDD']:<8.2f}% | {r['Trades']:<6} | {r['WinRate']:<8.1%}")

if __name__ == "__main__":
    main()
