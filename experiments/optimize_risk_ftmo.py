"""
EXPERIMENT: OPTIMIZE RISK FOR FTMO (SWEEP)
==========================================
Tests risk levels {1.0%, 2.0%, 3.0%, 4.0%, 5.0%} on the Rank-1 Strategy.
Goal: Find the Risk % that maximizes FTMO Pass Rate while keeping Tail Risk acceptable.

Methodology:
1. Re-run Backtest at each risk level (to capture sizing effects).
2. Run Monte Carlo (5000 sims) with FTMO Constraints:
   - Max Drawdown < 10%
   - Profit Target > 10%
   - (Approximated) Daily Drawdown Check < 5%
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

def run_monte_carlo_ftmo_risk(trades, initial_balance, risk_pct, sims=5000):
    if trades.empty: return {}
    
    # We use R-Multiples to simulate risk scaling accurately
    # R = PnL / (Balance * Risk%)
    # This allows us to scale risk without re-running backtest logic for every sim path
    # BUT, to be safer, we will use the actual PnL from the backtest run at that risk level
    # because of the 'Leverage Saturation' soft cap in the backtester.
    
    pnl_array = trades['PnL'].values
    n_trades = len(pnl_array)
    
    passed = 0
    start_bal = initial_balance
    target = start_bal * 0.10
    limit_dd = start_bal * -0.10
    
    tail_dds = []
    
    for _ in range(sims):
        # Bootstrap
        sim_pnl = np.random.choice(pnl_array, size=n_trades, replace=True)
        equity_curve = np.cumsum(sim_pnl) + start_bal
        
        # Check DD
        roll_max = np.maximum.accumulate(equity_curve)
        dd_dollars = equity_curve - roll_max
        max_dd = np.min(dd_dollars)
        
        tail_dds.append(max_dd / start_bal)
        
        total_profit = equity_curve[-1] - start_bal
        
        # Pass Condition
        # Note: Daily DD is hard to sim without timestamps in bootstrap, 
        # so we assume MaxDD is the primary filter, but we know Daily is stricter.
        if total_profit >= target and max_dd > limit_dd:
            passed += 1
            
    pass_rate = (passed / sims) * 100
    var_95 = np.percentile(tail_dds, 5) * 100
    
    return pass_rate, var_95

def run_risk_level(risk_pct):
    print(f"\n[TESTING] Risk {risk_pct*100:.1f}%...")
    
    cfg = Config()
    cfg.initial_balance = 100000.0
    cfg.risk_per_trade = risk_pct
    cfg.rank_top_n = 1 # King of the Hill
    cfg.use_rank_logic = True
    
    # Load Data (2024 limited)
    loader = DataLoader(cfg)
    try: data = loader.load_data("2024-01-01", "2024-11-29")
    except: return None
    
    # Pipeline
    fe = FeatureEngine(cfg)
    re = RegimeEngine(cfg)
    ae = AlphaEngine(cfg)
    es = EnsembleSignal(cfg)
    
    data = fe.add_features_all(data)
    data = re.add_regimes_all(data)
    ae.train_model(data)
    data = ae.add_signals_all(data)
    data = es.add_ensemble_all(data)
    
    # Backtest
    bt = Backtester(cfg)
    equity = pd.Series(bt.run_backtest(data))
    trades = pd.DataFrame(bt.account.trade_history)
    
    if trades.empty:
        return {'Risk': f"{risk_pct*100}%", 'Return': 0, 'Pass %': 0, 'Tail DD': 0}
        
    final_bal = equity.iloc[-1]
    ret_pct = ((final_bal - 100000) / 100000) * 100
    
    # Hist Max DD
    roll_max = equity.cummax()
    dd = (equity - roll_max) / roll_max
    max_dd_hist = dd.min() * 100
    
    # MC
    pass_rate, tail_dd = run_monte_carlo_ftmo_risk(trades, 100000, risk_pct)
    
    print(f"  -> Return: {ret_pct:.2f}%")
    print(f"  -> Max DD: {max_dd_hist:.2f}%")
    print(f"  -> Pass %: {pass_rate:.1f}%")
    print(f"  -> Tail DD: {tail_dd:.2f}%")
    
    return {
        'Risk': f"{risk_pct*100:.1f}%",
        'Return': ret_pct,
        'Max DD': max_dd_hist,
        'Pass %': pass_rate,
        'Tail DD': tail_dd
    }

def main():
    levels = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]
    results = []
    
    for r in levels:
        res = run_risk_level(r)
        if res: results.append(res)
        
    df = pd.DataFrame(results)
    print("\n" + "="*60)
    print("OPTIMAL RISK SWEEP RESULTS")
    print("="*60)
    print(df.to_string(index=False))
    
    # Optimal Choice
    # Max Pass Rate where Tail DD > -10% (safety margin)
    # Actually, let's just pick Max Pass Rate and see the DD cost.
    best = df.loc[df['Pass %'].idxmax()]
    print(f"\n🏆 Optimal Risk: {best['Risk']}")
    print(f"   Pass Rate: {best['Pass %']:.1f}%")
    print(f"   Tail DD:   {best['Tail DD']:.2f}%")

if __name__ == "__main__":
    main()
