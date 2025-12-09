import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

def run_monte_carlo(csv_path="backtest_results.csv", iterations=2000, initial_equity=100000):
    print(f"\n[MONTE CARLO] Loading trades from {csv_path}...")
    
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    if df.empty:
        print("Trade log is empty.")
        return

    # Extract PnL series
    # Usually 'PnL' column or (Exit_Price - Entry_Price) * Size
    # Check if 'PnL' exists
    if 'Profit' in df.columns:
        pnl = df['Profit'].values
    elif 'PnL' in df.columns:
        pnl = df['PnL'].values
    else:
        print("Could not find 'Profit' or 'PnL' column.")
        print(f"Columns: {df.columns}")
        return

    n_trades = len(pnl)
    print(f"Loaded {n_trades} trades. Running {iterations} iterations...")

    # Storage for terminal equity and max drawdown
    final_equities = []
    max_drawdowns = []
    
    # 1. Trade Shuffling (Resampling without replacement)
    # This destroys serial correlation but preserves the PnL distribution.
    # It answers: "Did the order of trades matter?" (i.e. did we get lucky with streaks?)
    
    for i in range(iterations):
        shuffled_pnl = np.random.permutation(pnl)
        
        # Equity Curve
        curve = np.zeros(n_trades + 1)
        curve[0] = initial_equity
        curve[1:] = initial_equity + np.cumsum(shuffled_pnl)
        
        final_eq = curve[-1]
        final_equities.append(final_eq)
        
        # Max Drawdown
        peak = np.maximum.accumulate(curve)
        drawdown = (curve - peak) / peak
        max_dd = np.min(drawdown)
        max_drawdowns.append(max_dd)

    # 2. Stats
    final_equities = np.array(final_equities)
    max_drawdowns = np.array(max_drawdowns)
    
    # Percentiles
    eq_05 = np.percentile(final_equities, 5)
    eq_50 = np.percentile(final_equities, 50)
    eq_95 = np.percentile(final_equities, 95)
    
    dd_05 = np.percentile(max_drawdowns, 5) # Best case DD (closest to 0)
    dd_50 = np.percentile(max_drawdowns, 50)
    dd_95 = np.percentile(max_drawdowns, 95) # Worst case DD (most negative)

    # Original Metric
    orig_curve = np.zeros(n_trades + 1)
    orig_curve[0] = initial_equity
    orig_curve[1:] = initial_equity + np.cumsum(pnl)
    orig_dd = np.min((orig_curve - np.maximum.accumulate(orig_curve)) / np.maximum.accumulate(orig_curve))
    orig_profit = orig_curve[-1]

    print("\n" + "="*50)
    print("MONTE CARLO RESULTS (Shuffle Test)")
    print("="*50)
    print(f"Start Equity:     ${initial_equity:,.2f}")
    print(f"Original Outcome: ${orig_profit:,.2f} (DD: {orig_dd*100:.2f}%)")
    print("-" * 50)
    print(f"Simulation Mean:  ${np.mean(final_equities):,.2f}")
    print(f"Global Win Probability: {(final_equities > initial_equity).mean()*100:.1f}%")
    print(f"Risk of Ruin (<50% Eq): {(final_equities < initial_equity*0.5).mean()*100:.1f}%")
    print("-" * 50)
    print("TERMINAL EQUITY DISTRIBUTION:")
    print(f"  95% (Best):     ${eq_95:,.2f}")
    print(f"  50% (Median):    ${eq_50:,.2f}")
    print(f"   5% (Worst):     ${eq_05:,.2f}")
    print("-" * 50)
    print("MAX DRAWDOWN DISTRIBUTION:")
    print(f"   5% (Best):      {dd_05*100:.2f}%")
    print(f"  50% (Median):    {dd_50*100:.2f}%")
    print(f"  95% (Worst):     {dd_95*100:.2f}%")
    print("="*50)
    
    # Ruin Check
    if eq_05 < initial_equity:
        print("\n[WARNING] Strategy has >5% chance of losing money in random shuffle.")
        if eq_05 < initial_equity * 0.8:
            print("[CRITICAL] Strategy is fragile. 5th percentile outcome is significant loss.")
    else:
        print("\n[PASS] Strategy is robust across shuffles.")

if __name__ == "__main__":
    csv = sys.argv[1] if len(sys.argv) > 1 else "backtest_results.csv"
    run_monte_carlo(csv)
