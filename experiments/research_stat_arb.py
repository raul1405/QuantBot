"""
Statistical Arbitrage Research (Pairs Trading)
==============================================
Objective:
1. Find Cointegrated Pairs in the FTMO Universe.
2. Backtest Mean-Reversion Strategy on these pairs.
3. Check correlation with existing Momentum Strategy.

Hypothesis: Pairs trading offers "Market Neutral" returns that are uncorrelated 
with our existing Directional/Momentum strategy, improving the Portfolio Sharpe.
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint, adfuller
from datetime import datetime, timedelta

# Add parent directory to path to import quant_backtest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from quant_backtest import Config, DataLoader, Account

# --- CONFIG ---
INITIAL_BALANCE = 100000
Start_Date = "2024-01-01"
End_Date = "2025-01-01" 
MIN_CORRELATION = 0.85
MAX_P_VALUE = 0.05
Z_ENTRY = 2.0
Z_EXIT = 0.0
Z_STOP = 4.0
COST_PER_TRADE_PCT = 0.0002 # 2bps per leg (Spread + Comm) -> 4bps roundtrip per pair

def find_cointegrated_pairs(data_map):
    """
    Scan universe for pairs.
    1. Filter by Correlation > MIN_CORRELATION
    2. Test Cointegration (Engle-Granger)
    """
    print("\n[1] Scanning for Cointegrated Pairs...")
    symbols = list(data_map.keys())
    # Extract Closes
    closes = pd.DataFrame({sym: df['Close'] for sym, df in data_map.items()})
    closes = closes.dropna()
    
    # 1. Correlation Matrix
    corr_matrix = closes.corr()
    
    pairs = []
    
    # Check all combinations
    for s1, s2 in combinations(symbols, 2):
        if s1 == s2: continue
        
        # Pre-filter: Correlation
        corr = corr_matrix.loc[s1, s2]
        if abs(corr) < MIN_CORRELATION:
            continue
            
        # Cointegration Test
        # Null hypothesis: No cointegration. 
        # If p < 0.05, we reject null -> Cointegration exists.
        score, pvalue, _ = coint(closes[s1], closes[s2])
        
        if pvalue < MAX_P_VALUE:
            print(f"  FOUND: {s1} vs {s2} | Corr: {corr:.2f} | p-value: {pvalue:.4f}")
            pairs.append({
                's1': s1,
                's2': s2,
                'corr': corr,
                'pvalue': pvalue,
                'score': score
            })
            
    print(f"  Total Pairs Found: {len(pairs)}")
    return pairs, closes

def calculate_spread_zscore(s1_series, s2_series):
    """
    Calculate spread = Y - beta * X
    Then Z-Score of spread.
    Uses Rolling Linear Regression to adapt beta? 
    For research, start with Static Beta over the period to test pure relationship.
    Actually, Rolling window is more realistic for trading.
    Let's use Rolling OLS (Lookback 500 hours ~ 1 month trading).
    """
    # Rolling OLS
    lookback = 100
    
    # Simple Hedge Ratio: Price Ratio? Or Regression?
    # Regression is better.
    
    # We need rolling beta.
    # To be efficient, we can use rolling_ols from statsmodels or simple covariance math.
    # beta = cov(y, x) / var(x)
    
    roll_cov = s1_series.rolling(lookback).cov(s2_series)
    roll_var = s2_series.rolling(lookback).var()
    beta = roll_cov / roll_var
    
    spread = s1_series - beta * s2_series
    
    # Z-Score
    spread_mean = spread.rolling(lookback).mean()
    spread_std = spread.rolling(lookback).std()
    z_score = (spread - spread_mean) / spread_std
    
    return z_score, spread, beta

def backtest_pair(s1_data, s2_data, z_score, pair_name):
    """
    Simple Vectorized Backtest.
    Long Spread: Buy S1, Sell S2 (when Z < -Entry)
    Short Spread: Sell S1, Buy S2 (when Z > +Entry)
    Exit: Z crosses 0.
    """
    signals = pd.Series(0, index=z_score.index)
    signals[z_score > Z_ENTRY] = -1 # Short Spread
    signals[z_score < -Z_ENTRY] = 1 # Long Spread
    signals[abs(z_score) < 0.1] = 0 # Exit? (Regime based)
    
    # This is "Target Position". We need to hold until exit condition.
    # Loop implementation is safer for state.
    
    position = 0 # 0, 1, -1
    entry_price_spread = 0.0
    
    pnl = []
    equity = [INITIAL_BALANCE]
    
    # Reconstruct spread price relative for PnL is tricky with dynamic beta.
    # PnL = (S1_chg - beta * S2_chg) * Size
    
    # Let's iterate
    s1_close = s1_data
    s2_close = s2_data
    
    curr_eq = INITIAL_BALANCE
    
    trades = []
    
    for i in range(1, len(z_score)):
        ts = z_score.index[i]
        z = z_score.iloc[i]
        
        if pd.isna(z): continue
        
        # Current Returns (approx)
        r1 = (s1_close.iloc[i] - s1_close.iloc[i-1]) / s1_close.iloc[i-1]
        r2 = (s2_close.iloc[i] - s2_close.iloc[i-1]) / s2_close.iloc[i-1]
        
        # PnL logic if in position
        # We assume Equal Dollar allocation? NO, pair trading is beta hedged.
        # Invest $1000 in Leg 1. Short Beta * $1000 in Leg 2?
        # Actually: Dollar Neutral? Or Beta Neutral?
        # Standard: Beta Neutral.
        # Amt1 = Capital/2. Amt2 = Amt1 * Beta?
        # Simplified: Fixed Sizing $10,000 per leg per trade
        
        bet_size = 10000 
        
        # Mark to Market
        if position != 0:
            # Long Spread: Long S1, Short S2
            # Short Spread: Short S1, Long S2
            
            # Note: Beta is from T-1 usually.
            
            leg1_pnl = bet_size * r1 * position
            # leg2_pnl = bet_size * r2 * (-position) # If we did $ neutral
            # But we do beta neutral?
            # Let's stick to simple assumption: Spread PnL ~ Z-Score change?
            # No, explicit calculation.
            
            leg2_pnl = bet_size * r2 * (-position)
            
            # Costs? Calculated at Entry/Exit only.
            
            trade_pnl = leg1_pnl + leg2_pnl
            curr_eq += trade_pnl
            
        equity.append(curr_eq)
        
        # Entry/Exit Logic
        if position == 0:
            if z > Z_ENTRY:
                position = -1 # Short Spread
                # Pay Costs
                curr_eq -= bet_size * COST_PER_TRADE_PCT * 2 
                trades.append({'time': ts, 'type': 'ENTRY_SHORT', 'z': z})
            elif z < -Z_ENTRY:
                position = 1 # Long Spread
                # Pay Costs
                curr_eq -= bet_size * COST_PER_TRADE_PCT * 2
                trades.append({'time': ts, 'type': 'ENTRY_LONG', 'z': z})
                
        elif position == 1: # Long
            if z >= Z_EXIT:
                position = 0 # Exit
                curr_eq -= bet_size * COST_PER_TRADE_PCT * 2
                trades.append({'time': ts, 'type': 'EXIT', 'z': z})
            elif z < -Z_STOP:
                 position = 0 # Stop Loss (Divergence)
                 curr_eq -= bet_size * COST_PER_TRADE_PCT * 2
                 trades.append({'time': ts, 'type': 'STOP', 'z': z})
                 
        elif position == -1: # Short
            if z <= -Z_EXIT: # Wait, Z goes down to 0
                position = 0
                curr_eq -= bet_size * COST_PER_TRADE_PCT * 2
                trades.append({'time': ts, 'type': 'EXIT', 'z': z})
            elif z > Z_STOP:
                 position = 0
                 curr_eq -= bet_size * COST_PER_TRADE_PCT * 2
                 trades.append({'time': ts, 'type': 'STOP', 'z': z})

    return pd.Series(equity, index=z_score.index[len(z_score)-len(equity):]), trades

def load_universe_data():
    config = Config()
    # Define a focused universe for research
    config.symbols = [
        "EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "NZDUSD=X", "USDCAD=X", "USDCHF=X",
        "ES=F", "NQ=F", "YM=F", "GC=F", "SI=F", "CL=F" # Added SI (Silver) if available? 
    ]
    loader = DataLoader(config)
    
    # Load 1 year
    start = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    end = datetime.now().strftime("%Y-%m-%d")
    
    print(f"\n[LOADING DATA] {start} to {end}")
    data_map = loader.load_data(start, end)
    return data_map

def main():
    print("STATISTICAL ARBITRAGE RESEARCH")
    print("==============================")
    
    data_map = load_universe_data()
    
    # 1. Find Pairs
    pairs, closes = find_cointegrated_pairs(data_map)
    
    if not pairs:
        print("No cointegrated pairs found.")
        return

    # 2. Backtest Pairs with Parameter Sweep
    print("\n[2] Backtesting Pairs (Parameter Sweep)...")
    
    z_entries = [2.0, 2.5, 3.0]
    best_results = []
    
    for z_thresh in z_entries:
        print(f"\n--- Testing Z_ENTRY = {z_thresh} ---")
        global Z_ENTRY
        Z_ENTRY = z_thresh # Hacky global set
        
        for p in pairs:
            s1 = p['s1']
            s2 = p['s2']
            
            z, spread, beta = calculate_spread_zscore(closes[s1], closes[s2])
            eq, trades = backtest_pair(closes[s1], closes[s2], z, f"{s1}-{s2}")
            
            total_ret = (eq.iloc[-1] / eq.iloc[0]) - 1 if len(eq) > 0 else 0.0
            print(f"  {s1}-{s2}: Return: {total_ret*100:.2f}% | Trades: {len(trades)}")
            
            if total_ret > 0:
                best_results.append({
                    'pair': f"{s1}-{s2}",
                    'z_entry': z_thresh,
                    'return': total_ret,
                    'trades': len(trades)
                })

    print("\n[3] Profitable Configs Found:")
    if best_results:
        res_df = pd.DataFrame(best_results)
        print(res_df.sort_values('return', ascending=False))
    else:
        print("None.")
    
    # Skip plotting for sweep
    return

if __name__ == "__main__":
    main()
