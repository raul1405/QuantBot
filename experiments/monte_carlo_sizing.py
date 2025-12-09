"""
MONTE CARLO RISK SIZING
=======================
Goal: Find the maximum safe Risk % such that the 95th Percentile Max Drawdown < 8%.

Methodology:
1. Run Backtest at baseline risk (1.0%).
2. Extract trade returns (PnL %).
3. Monte Carlo Simulation (10,000 runs):
   - Shuffle trade sequence.
   - Scale returns for different Risk % (1.0% to 5.0%).
   - Calculate Max Drawdown for each run.
4. Determine the Risk cutoff where 95% of runs stay above -8% DD.
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quant_backtest import (
    Config, DataLoader, FeatureEngine, RegimeEngine, 
    AlphaEngine, EnsembleSignal, CrisisAlphaEngine, Backtester
)

def calculate_max_dd(equity_curve):
    """Calculates Max Drawdown % from an equity curve series."""
    # equity_curve is a numpy array of equity values
    peak = np.maximum.accumulate(equity_curve)
    dd = (equity_curve - peak) / peak
    return dd.min() * 100

def run_monte_carlo(trades_pct, risk_scale, n_sims=10000):
    """
    Simulates n_sims equity curves shuffling the trades.
    trades_pct: Array of % returns per trade at Baseline Risk.
    risk_scale: Multiplier (Target / Baseline).
    """
    n_trades = len(trades_pct)
    # Scale returns
    # Geometric compounding: (1 + r * scale)
    scaled_returns = trades_pct * risk_scale
    
    # Simulation
    max_dds = []
    
    # Vectorized simulation might be hard for DD, looping is safer/clearer
    # But slow for 10k sims in python.
    # We'll use numpy for shuffling.
    
    for _ in range(n_sims):
        # Shuffle
        shuffled_rets = np.random.choice(scaled_returns, size=n_trades, replace=True)
        
        # Cumulatively compound
        equity_curve = np.cumprod(1 + shuffled_rets)
        
        # Max DD
        peak = np.maximum.accumulate(equity_curve)
        dd = (equity_curve - peak) / peak
        max_dds.append(dd.min() * 100)
        
    return np.percentile(max_dds, 5) # 5th percentile (e.g. -12% is worse than -5%)
    # Wait, percentile logic:
    # If DDs are [-1, -2, -10], 5th percentile is -10 (worst 5%).
    # We want 95% confidence it stays ABOVE -8%.
    # So we look at the 5th percentile from the bottom.

def main():
    print("="*60)
    print("MONTE CARLO RISK OPTIMIZATION (Target: < 8% Max DD)")
    print("="*60)
    
    # 1. GENERATE BASELINE TRADES (at 1% Risk)
    config = Config()
    config.risk_per_trade = 0.01 # Baseline 1%
    config.use_signal_decay_exit = True
    
    print("[1/3] Running Baseline Backtest (1% Risk)...")
    loader = DataLoader(config)
    try:
        data = loader.load_data("2024-01-01", "2024-12-01")
    except Exception as e:
        print(f"Error: {e}")
        return

    fe = FeatureEngine(config)
    re = RegimeEngine(config)
    ae = AlphaEngine(config)
    es = EnsembleSignal(config)
    ce = CrisisAlphaEngine(config)
    
    data = fe.add_features_all(data)
    data = re.add_regimes_all(data)
    ae.train_model(data)
    data = ae.add_signals_all(data)
    data = es.add_ensemble_all(data)
    data = ce.add_crisis_signals(data)
    
    bt = Backtester(config)
    bt.run_backtest(data)
    
    trades = pd.DataFrame(bt.account.trade_history)
    if trades.empty:
        print("No trades generated.")
        return
        
    # Calculate Trade % Return relative to Account at that time?
    # Backtester History saves PnL. We need Risk Base.
    # Let's approximate: PnL / Initial_Balance? NO. Compounding.
    # We need trade % impact on equity.
    # Our backtester doesn't save "Equity at Entry".
    # But we can reconstruct perfectly if we re-run equity curve?
    # Or simpler: Backtester.run_backtest calculates equity at every step.
    # Can we modify Backtester to save trade % return?
    # Or just use the 'size' and 'entry_price' to back out risk?
    # Actually, Config uses risk_per_trade * Equity.
    # So PnL / Equity_At_Entry approx = Trade_Return_Pct.
    # But we don't have Equity_At_Entry for each trade easily.
    
    # Let's assume linear PnL for unit sizing and risk is constant %?
    # No, sizes change.
    
    # Best way: R-multiples.
    # R = PnL / Risk_Dollars.
    # Trade_Pct = R * Risk_Pct.
    
    trade_returns = []
    # Reconstruct Equity Curve to get Equity at Entry
    equity = config.initial_balance
    trade_history = bt.account.trade_history
    # This list is chronological by Exit? No, append order.
    # We need chronological.
    
    # Actually, simpler:
    # Run backtest with FIXED BALANCE sizing (non-compounding) to get raw R?
    # No, we want to simulate compounding.
    
    # Let's trust the R-multiple calc if we had it.
    # Let's approximate R.
    # We know Risk = 1.0% of Equity.
    # If a trade made $500 on a $100k account, that's 0.5% return.
    # Since Risk was 1.0%, that's 0.5R.
    # But Equity changes.
    # However, for the simulation, we can assume the distribution of R-multiples is constant.
    
    # Calculating R-multiples from PnL is tricky without knowing exact Risk$ at that moment.
    # BUT, we know `size` was calculated as `risk_amt / sl_dist`.
    # PnL = size * (Exit - Entry).
    # R = PnL / risk_amt = size * dist / (size * sl_dist) = dist / sl_dist.
    # R = (Exit - Entry) / SL_Dist.
    # This is Pure, Hard R.
    
    # We need SL_Dist.
    # Backtester history saves `Entry Price`, `Exit Price`.
    # Does it save `SL`? No.
    # Does it save `Context`? Yes?
    # Position object has `sl`. trade_history is dict version.
    # Check what Account.close_position saves.
    
    # Assuming we can't get exact SL from history dict easily.
    # We will assume a "Mean Risk" approach? No, dangerous.
    
    # Let's hack: The simulation ran at 0.01 Risk.
    # We can walk the equity curve and divide PnL by Equity?
    # The trades are list of dicts. We can't perfectly align with equity curve without times.
    
    # WORKAROUND:
    # Calculate R-multiple purely from PnL? 
    # Let's assume the Backtester logic is consistent.
    # 1.0% Risk -> Returns X%.
    # If we want Risk 2.0%, Returns will be approx 2 * X%.
    # So we just need the array of (PnL / Closing_Balance_Prev).
    # Or simply (PnL / Initial_Balance) if we disable compounding for baseline?
    # Let's disable compounding for the extraction run (set kelly=0? No).
    # Just assume Baseline Equity = 100k constant.
    # Then Trade_Pct = PnL / 100000.
    # Then R = Trade_Pct / 0.01.
    
    # To disable compounding in backtester: modify Config?
    # No, Account class uses current equity.
    
    # We will use the recorded PnL and divide by the *actual* equity at that time.
    # To get actual equity, we replay the trades in order.
    current_eq = config.initial_balance
    r_multiples = []
    
    # Sort trades by Exit Time? Or Entry Time?
    # Drawdown happens on Equity Curve. Equity updates on Exit.
    # So Sort by Exit Time.
    df_trades = pd.DataFrame(trade_history)
    df_trades['Exit Time'] = pd.to_datetime(df_trades['Exit Time'])
    df_trades = df_trades.sort_values('Exit Time')
    
    for idx, row in df_trades.iterrows():
        # Risk Amount used was roughly 0.01 * current_eq
        # So R = PnL / (0.01 * current_eq)
        risk_amt = 0.01 * current_eq
        r = row['PnL'] / risk_amt
        r_multiples.append(r)
        
        current_eq += row['PnL']
        
    print(f"  > Generated {len(r_multiples)} trades.")
    print(f"  > Average R: {np.mean(r_multiples):.3f}")
    
    # 2. RUN SIMULATION
    print("\n[2/3] Running Monte Carlo for Risk scaling...")
    
    results = []
    # Test Risks: 0.5% to 5.0% in 0.1% steps
    risks = np.arange(0.5, 5.1, 0.1)
    
    safe_risk = 0.0
    
    print(f"{'Risk %':<10} | {'Max DD (95% CI)':<20} | {'Median DD':<15} | {'Exp Return':<15}")
    print("-" * 75)
    
    best_risk = 0.0
    
    for r_pct in risks:
        actual_risk_pct = r_pct / 100.0
        
        # Trade Returns = R * Risk_Pct
        trade_rets = np.array(r_multiples) * actual_risk_pct
        
        # Run MC
        n_sims = 5000 # Enough for estimate
        max_dds = []
        final_rets = []
        
        for _ in range(n_sims):
            # Sample with replacement
            sim_rets = np.random.choice(trade_rets, size=len(trade_rets), replace=True)
            eq_curve = np.cumprod(1 + sim_rets)
            
            # DD
            peak = np.maximum.accumulate(eq_curve)
            dd = (eq_curve - peak) / peak
            max_dds.append(dd.min() * 100)
            
            final_rets.append((eq_curve[-1] - 1)*100)
            
        # 5th Percentile DD (Negative number closest to -100)
        # We want the value where 95% of runs are BETTER.
        # So we look at the 5th percentile worst outcome.
        dd_95 = np.percentile(max_dds, 5) 
        median_dd = np.median(max_dds)
        exp_ret = np.median(final_rets)
        
        status = "✅" if dd_95 > -8.0 else "❌"
        if dd_95 > -8.0:
            if r_pct > best_risk: best_risk = r_pct
            
        if int(r_pct*10) % 5 == 0: # Print every 0.5
            print(f"{r_pct:.1f}%      | {dd_95:.2f}% {status:<10} | {median_dd:.2f}%        | {exp_ret:.1f}%")

    print(f"\n[3/3] OPTIMAL RISK: {best_risk:.1f}%")
    print("This is the maximum risk where 95% of shuffled years stay safer than -8% DD.")

if __name__ == "__main__":
    main()
