"""
UNIVERSE COST ANALYSIS
======================
Hypothesis: The 'Toxic' pairs (Minor Crosses) failed because their Transaction Costs (Spreads)
exceeded the Strategy's Alpha, not because the Alpha didn't exist.

Test: Run the discarded pairs with:
1. Realistic Costs (Benchmark)
2. ZERO Costs (Theoretical Alpha)

If Zero Cost run is profitable, the filter is justified by Economics (Liquidity).
If Zero Cost run is loss, the Strategy is fundamentally broken on these assets.
"""

import sys
import os
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quant_backtest import (
    Config, DataLoader, FeatureEngine, RegimeEngine, 
    AlphaEngine, EnsembleSignal, CrisisAlphaEngine, Backtester
)

def run_test(name, custom_symbols, use_zero_cost=False):
    print(f"\n[TEST: {name}] Cost={'ZERO' if use_zero_cost else 'REAL'}...")
    
    config = Config()
    config.symbols = custom_symbols
    config.use_signal_decay_exit = True
    config.risk_per_trade = 0.01
    
    # Overrides
    if use_zero_cost:
        config.transaction_cost = 0.0 # Zero Commission
        # We also need to patch Account.close_position to ignore spread?
        # Creating a subclass or monkeypatching is hard.
        # But quant_backtest.py uses `transaction_cost` param inside `close_position` 
        # specifically for the 'fallback' or spread logic?
        # Let's check quant_backtest.py logic.
        pass
        
    loader = DataLoader(config)
    try:
        # Use short range for speed
        data = loader.load_data("2024-01-01", "2024-06-01") 
    except Exception as e:
        print(e)
        return 0, 0
    
    fe = FeatureEngine(config)
    re = RegimeEngine(config)
    ae = AlphaEngine(config)
    
    # Train
    data = fe.add_features_all(data)
    data = re.add_regimes_all(data)
    ae.train_model(data)
    data = ae.add_signals_all(data)
    
    # Ensemble
    es = EnsembleSignal(config)
    data = es.add_ensemble_all(data)
    
    # Backtest
    bt = Backtester(config)
    
    # HACK: If Zero Cost, we must ensure Account doesn't apply spread.
    # The current Account.close_position might hardcode spread logic.
    # Let's see if we can perform a Hot-Fix on the instance method?
    if use_zero_cost:
        # Monkey patch close_position for this instance
        original_close = bt.account.close_position
        
        def mock_close(pos, price, time, reason):
            # No Cost Close
            gross_pnl = (price - pos.entry_price) * pos.size * pos.direction
            bt.account.balance += gross_pnl
            bt.account.equity = bt.account.balance
            bt.account.trade_history.append({
                'Symbol': pos.symbol,
                'PnL': gross_pnl,
                # ...
            })
            if pos in bt.account.positions:
                bt.account.positions.remove(pos)
                
        bt.account.close_position = mock_close
        
    # Also Execution Logic in Backtester loop uses costs? No.
    # But Entry logic might? No.
    
    bt.run_backtest(data)
    
    # Metrics
    final_balance = bt.account.balance
    ret = (final_balance - config.initial_balance) / config.initial_balance * 100
    trades = len(bt.account.trade_history)
    
    print(f"  > Return: {ret:.2f}% | Trades: {trades}")
    return ret, trades

def main():
    # LIST OF REJECTED PAIRS
    toxic_pairs = [
        "AUDNZD=X", "AUDCAD=X", "CADJPY=X", "NZDJPY=X", 
        "GBPCHF=X", "GBPAUD=X", "GBPCAD=X", "EURNZD=X"
    ]
    
    print("="*60)
    print(f"ANALYZING {len(toxic_pairs)} DISCARDED PAIRS")
    print("="*60)
    
    # Run 1: Real Costs
    r1, t1 = run_test(" REAL COSTS", toxic_pairs, use_zero_cost=False)
    
    # Run 2: Zero Costs
    r2, t2 = run_test(" ZERO COSTS", toxic_pairs, use_zero_cost=True)
    
    print("\n" + "="*60)
    print("CONCLUSION")
    print("="*60)
    print(f"Real Cost Return: {r1:.2f}%")
    print(f"Zero Cost Return: {r2:.2f}%")
    
    diff = r2 - r1
    print(f"Cost Drag:        {diff:.2f}%")
    
    if r2 > 0:
        print("\n✅ VERDICT: LIQUIDITY CONSTRAINT.")
        print("The strategy HAS EDGE on these pairs (Zero Cost > 0).")
        print("But Transaction Costs eat the entire profit.")
        print("Filtering them is scientifically valid (Economic Filter).")
    else:
        print("\n❌ VERDICT: OVERFITTING RISK.")
        print("The strategy CLOSES RED even with Zero Costs.")
        print("These pairs simply do not behave like the Majors.")
        print("Filtering them might be cherry-picking.")

if __name__ == "__main__":
    main()
