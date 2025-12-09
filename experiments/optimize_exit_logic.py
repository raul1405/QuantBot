"""
EXPERIMENT: OPTIMIZE EXIT LOGIC (RIGOROUS TOP-N AUDIT)
======================================================
Tests "Rank-1" vs "Top-N" with strictly normalized risk.
Goal: Determine the mathematically optimal policy for (A) FTMO Passing and (B) Long-term Growth.

Methodology:
1. Total Risk Budget: 2.0% per setup (NORMALIZED).
   - Rank 1: 2.0% on top asset.
   - Top 2: 1.0% on top 2 assets.
   - Top 3: 0.67% on top 3 assets.
2. Period: Full WFO Range (2023-2024 to capture regimes).
3. Metrics: Return, Max DD, Sharpe, SQN, Trade Count, Tail Risk (VaR-95).
4. Validation: Monte Carlo Bootstrap for statistical significance.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import timedelta
import random

# fix path to import quant_backtest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quant_backtest import (
    Config, DataLoader, FeatureEngine, RegimeEngine, 
    AlphaEngine, EnsembleSignal, Backtester
)

def run_monte_carlo_ftmo(trades, initial_balance, sims=5000):
    if trades.empty: 
        return {
            'FTMO Pass %': 0.0,
            'Exp Return': 0.0,
            'Tail DD 5%': 0.0,
            'Tail PnL 5%': 0.0
        }
    
    # R-Multiples bootstrap is better, but PnL bootstrap captures dollar sizing
    pnl_array = trades['PnL'].values
    n_trades = len(pnl_array)
    
    passed_count = 0
    final_pnls = []
    max_drawdowns = []
    
    # FTMO Targets
    TARGET_PROFIT = initial_balance * 0.10
    MAX_DD_LIMIT = initial_balance * -0.10
    
    for _ in range(sims):
        # 1. Bootstrap Trades
        sim_pnl = np.random.choice(pnl_array, size=n_trades, replace=True)
        
        # 2. Build Curve
        cum_pnl = np.cumsum(sim_pnl)
        equity_curve = initial_balance + cum_pnl
        
        # 3. Calc Metrics
        total_pnl = cum_pnl[-1]
        
        # Drawdown
        roll_max = np.maximum.accumulate(equity_curve)
        dd = (equity_curve - roll_max) / roll_max
        max_dd = np.min(dd)
        
        max_drawdowns.append(max_dd)
        final_pnls.append(total_pnl)
        
        # 4. FTMO Check (Simplified: Profit > 10% AND DD > -10%)
        # Ignoring Daily DD in MC for speed, as it requires daily bars
        if total_pnl >= TARGET_PROFIT and max_dd > -0.10:
            passed_count += 1
            
    pass_rate = (passed_count / sims) * 100
    avg_return = np.mean(final_pnls)
    tail_dd_5 = np.percentile(max_drawdowns, 5) * 100 # 5th percentile worst DD
    tail_pnl_5 = np.percentile(final_pnls, 5)         # 5th percentile worst PnL
    
    return {
        'FTMO Pass %': pass_rate,
        'Exp Return': avg_return,
        'Tail DD 5%': tail_dd_5,
        'Tail PnL 5%': tail_pnl_5
    }

def run_rigorous_test(name, top_n, total_risk=0.02):
    print(f"\n[TEST] {name} (Top {top_n}, Risk {total_risk*100}%)")
    
    cfg = Config()
    cfg.initial_balance = 100000.0
    
    # RISK NORMALIZATION
    # We split the total risk budget across N positions.
    cfg.risk_per_trade = total_risk / top_n 
    cfg.max_concurrent_trades = 15 # Allow up to 15 to not block Top-N
    cfg.max_exposure_per_currency = 15 # Relax constraint
    
    # Strategy Config
    cfg.rank_top_n = top_n
    cfg.use_rank_logic = True
    
    # Load Data (Full available range for better sample)
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
        print("  -> No trades generated.")
        return None
        
    # Metrics
    final_balance = equity.iloc[-1]
    ret_pct = ((final_balance - cfg.initial_balance) / cfg.initial_balance) * 100
    
    # Robust DD Calculation
    roll_max = equity.cummax()
    drawdown = (equity - roll_max) / roll_max
    max_dd = drawdown.min() * 100
    
    # Sharpe (Monthly Ann.)
    returns = equity.pct_change().dropna()
    sharpe = (returns.mean() / returns.std()) * np.sqrt(252 * 6) if returns.std() > 0 else 0
    
    # SQN
    r_multiples = trades['R'].fillna(0) if 'R' in trades.columns else (trades['PnL'] / (cfg.initial_balance * cfg.risk_per_trade)).replace([np.inf, -np.inf], 0)
    sqn = (r_multiples.mean() / r_multiples.std()) * np.sqrt(len(r_multiples)) if len(r_multiples) > 0 else 0
    
    # Monte Carlo Stats (FTMO)
    mc_stats = run_monte_carlo_ftmo(trades, cfg.initial_balance)
    
    print(f"  -> Trades: {len(trades)}")
    print(f"  -> Return: {ret_pct:.2f}%")
    print(f"  -> Max DD: {max_dd:.2f}%")
    print(f"  -> Sharpe: {sharpe:.2f}")
    print(f"  -> SQN:    {sqn:.2f}")
    print(f"  -> Pass %: {mc_stats['FTMO Pass %']:.1f}%")
    print(f"  -> Tail DD:{mc_stats['Tail DD 5%']:.2f}%")
    
    return {
        'Top N': top_n,
        'Trades': len(trades),
        'Return': ret_pct,
        'Max DD': max_dd,
        'Sharpe': sharpe,
        'SQN': sqn,
        'Pass %': mc_stats['FTMO Pass %'],
        'Tail DD': mc_stats['Tail DD 5%'],
        'Tail PnL': mc_stats['Tail PnL 5%']
    }

def main():
    print("="*60)
    print("RIGOROUS RANK-1 VS TOP-N AUDIT (FTMO EDITION)")
    print("Risk Budget: 2.0% Total per Setup")
    print("Data: 2024-01-01 -> 2024-11-29 (Max Avail)")
    print("="*60)
    
    results = []
    
    # Comparison Set
    results.append(run_rigorous_test("Rank 1 (Concentrated)", top_n=1))
    results.append(run_rigorous_test("Top 2 (Split Risk)", top_n=2))
    results.append(run_rigorous_test("Top 3 (Diversified)", top_n=3))
    
    print("\n" + "="*60)
    print("FINAL VERDICT")
    print("="*60)
    
    # Simple formatting
    df = pd.DataFrame([r for r in results if r])
    if not df.empty:
        print(df.to_string(index=False))
        
        # Verdict Logic
        best_pass = df.loc[df['Pass %'].idxmax()]
        best_risk = df.loc[df['Tail DD'].idxmax()] # Tail DD is negative, max is closest to 0
        
        print(f"\n🏆 Best FTMO Pass Rate: Top {best_pass['Top N']} ({best_pass['Pass %']:.1f}%)")
        print(f"🛡️ Safest (Tail Risk):  Top {best_risk['Top N']} ({best_risk['Tail DD']:.2f}% Worst Case)")


if __name__ == "__main__":
    main()
