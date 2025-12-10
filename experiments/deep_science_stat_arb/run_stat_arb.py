
import sys
import os
import pandas as pd
import numpy as np
from itertools import combinations
from statsmodels.tsa.stattools import coint

# Adjust path to import from root and High-Vol config
sys.path.append(os.path.abspath("."))
sys.path.append(os.path.abspath("experiments/high_vol_family_c"))

from sandbox_engine import Config, DataLoader, Backtester # Reuse Basic Infra
# UNIVERSE
from config.high_vol_universes import UNIVERSE_CRYPTO, UNIVERSE_INDICES_METALS

# --- STAT ARB PARAMETERS ---
TRAIN_WINDOW_DAYS = 180     # Lookback for Cointegration
TEST_WINDOW_DAYS = 30       # Forward Trade Window
MIN_CORRELATION = 0.80      # Minimum Correlation to even check Coint
MAX_P_VALUE = 0.05          # Cointegration Threshold
Z_ENTRY = 2.0               # Z-Score to Open
Z_EXIT = 0.0                # Z-Score to Close
Z_STOP = 4.0                # Z-Score Stop Loss (Divergence)
COST_BPS = 0.0006           # 6bps (Spread+Comm) for Crypto Features

class PairsTrader:
    def __init__(self, data_map):
        self.data_map = data_map
        self.symbols = list(data_map.keys())
        self.results = []
        self.trade_log = []
        
    def get_prices(self, symbol, start, end):
        df = self.data_map.get(symbol)
        if df is None: return None
        mask = (df.index >= start) & (df.index < end)
        return df.loc[mask]

    def scan_for_pairs(self, start_date, end_date):
        """
        Find Valid Pairs in the Training Window.
        """
        valid_pairs = []
        
        # Get aligned Closes for Correlation
        frames = {}
        for sym in self.symbols:
            p = self.get_prices(sym, start_date, end_date)
            if p is not None and len(p) > 50:
                frames[sym] = p['Close']
                
        if len(frames) < 2: return []
        
        closes = pd.DataFrame(frames).dropna()
        if closes.empty: return []
        
        corr_matrix = closes.corr()
        
        for s1, s2 in combinations(self.symbols, 2):
            if s1 not in closes.columns or s2 not in closes.columns: continue
            
            # 1. Correlation Filter
            corr = corr_matrix.loc[s1, s2]
            if abs(corr) < MIN_CORRELATION: continue
            
            # 2. Cointegration Test (Engle-Granger)
            try:
                score, pvalue, _ = coint(closes[s1], closes[s2])
                if pvalue < MAX_P_VALUE:
                    # Calculate Beta for Trading
                    # OLS: s1 = beta * s2 + alpha
                    # beta = cov(s1, s2) / var(s2) -- Simple Proxy
                    beta = closes[s1].cov(closes[s2]) / closes[s2].var()
                    
                    valid_pairs.append({
                        's1': s1, 's2': s2, 
                        'pvalue': pvalue, 'corr': corr,
                        'beta': beta
                    })
            except:
                continue
                
        return valid_pairs

    def trade_pair(self, pair, start_date, end_date):
        """
        Execute Mean Reversion on Pair in Test Window.
        """
        s1 = pair['s1']
        s2 = pair['s2']
        beta = pair['beta'] # Use Historical Beta? Or Rolling?
        # Rolling Beta is safer for live trading. Let's stick to Historical for "Forward Test" validity/simplicity first.
        # Actually, standard practice is Rolling Beta/Z-Score.
        # Let's use Rolling Z-Score on Spreads calculated with Rolling Ratio?
        # To keep it consistent with "Regime", let's use the Beta found in Training (Static for the month).
        
        df1 = self.get_prices(s1, start_date, end_date)
        df2 = self.get_prices(s2, start_date, end_date)
        
        if df1 is None or df2 is None or df1.empty or df2.empty: return 0.0
        
        # Align
        common = df1.index.intersection(df2.index)
        if len(common) == 0: return 0.0
        
        p1 = df1.loc[common]['Close']
        p2 = df2.loc[common]['Close']
        
        # Calculate Spread
        spread = p1 - (beta * p2)
        
        # Calculate Z-Score (Rolling 20-period? Or Historical Mean?)
        # Strategy: Use Rolling stats from the Test Data itself to adapt to regime
        roll_mean = spread.rolling(20).mean()
        roll_std = spread.rolling(20).std()
        z_score = (spread - roll_mean) / roll_std
        
        # Vectorized PnL Backtest
        pos = 0
        balance = 10000.0 # Virtual $10k allocation
        start_bal = 10000.0
        
        # We need loop for stateful entry/exit
        entry_price_spread = 0.0
        
        for i in range(1, len(z_score)):
            z = z_score.iloc[i]
            ts = z_score.index[i]
            
            if pd.isna(z): continue
            
            current_spread = spread.iloc[i]
            
            # PnL Calculation (Approximate)
            # PnL = Delta Spread * Size
            # Notional Size = $10k. 
            # If Z goes 2 -> 0, we make money.
            # Delta PnL = Position * (Spread_t - Spread_t-1)
            # Size Multiplier?
            # Roughly: Size = Capital / Price1
            # But Spread is Price difference potentially small or large.
            # Simplified: Use % Returns of the assets.
            
            r1 = (p1.iloc[i] - p1.iloc[i-1])/p1.iloc[i-1]
            r2 = (p2.iloc[i] - p2.iloc[i-1])/p2.iloc[i-1]
            
            if pos != 0:
                # Long Spread = Long S1, Short Beta*S2
                # Short Spread = Short S1, Long Beta*S2
                net_ret = (pos * r1) - (pos * r2) # Assuming 50/50 allocation roughly? 
                # Stat Arb PnL is complex.
                # Let's use Spread Dollar Value change.
                # Trade 1 Unit of Spread. 
                # If Spread = $50. We hold 1 Spread.
                # PnL = Spread_new - Spread_old.
                
                spread_chg = current_spread - spread.iloc[i-1]
                pnl = pos * spread_chg
                
                # Normalize to Capital?
                # Assume we enter with size such that 1 Z-score move ~ 1% impact?
                # Or Just assume fixed notional.
                # Let's do % Return = PnL / AssetPrice1
                
                pct_impact = pnl / p1.iloc[i-1] 
                balance *= (1 + pct_impact)
            
            # Logic
            if pos == 0:
                if z > Z_ENTRY:
                    pos = -1 # Short Spread
                    balance *= (1 - COST_BPS) # Entry Cost
                elif z < -Z_ENTRY:
                    pos = 1 # Long Spread
                    balance *= (1 - COST_BPS)
            elif pos == 1:
                if z > Z_EXIT:
                    pos = 0 # Take Profit
                    balance *= (1 - COST_BPS)
                elif z < -Z_STOP:
                    pos = 0 # Stop Loss
                    balance *= (1 - COST_BPS)
            elif pos == -1:
                if z < -Z_EXIT:
                    pos = 0
                    balance *= (1 - COST_BPS)
                elif z > Z_STOP:
                    pos = 0
                    balance *= (1 - COST_BPS)
                    
        return balance - start_bal

    def run_wfo(self, start_date_str, end_date_str):
        print(f"--- Running WFO Stat Arb: {start_date_str} -> {end_date_str} ---")
        
        current_date = pd.Timestamp(start_date_str)
        end_date = pd.Timestamp(end_date_str)
        
        total_pnl = 0.0
        
        while current_date < end_date:
            train_start = current_date - pd.Timedelta(days=TRAIN_WINDOW_DAYS)
            test_end = current_date + pd.Timedelta(days=TEST_WINDOW_DAYS)
            
            # Ensure we have data for training
            # 1. Scan Pairs (Train)
            pairs = self.scan_for_pairs(train_start, current_date)
            # print(f"  Date: {current_date.date()} | Found Pairs: {len(pairs)}")
            
            month_pnl = 0.0
            
            # 2. Trade Pairs (Test)
            count = 0
            for p in pairs:
                pnl = self.trade_pair(p, current_date, test_end)
                month_pnl += pnl
                if pnl != 0: count += 1
                
            if len(pairs) > 0:
                # Portfolio sizing: Divide capital among pairs?
                # Or assume we trade all signals independently?
                # Let's sum raw dollar PnL from the virtual $10k per pair allocations
                # But normalize to a single $10k account?
                # Avg PnL * Num Pairs?
                
                # Simplified: Total PnL is sum of all pair PnLs
                total_pnl += month_pnl
                print(f"  Window {current_date.date()} -> {test_end.date()}: Pairs={len(pairs)} | PnL=${month_pnl:.2f}")
            
            current_date = test_end
            
        print(f"Total WFO PnL: ${total_pnl:.2f}")
        return total_pnl

def load_data_safe(universe):
    cfg = Config()
    cfg.symbols = universe
    cfg.timeframe = "1d"
    loader = DataLoader(cfg)
    # Load 2 years for training buffers
    start = "2023-01-01" 
    end = "2025-01-01" 
    try:
        data = loader.load_data(start, end)
        return data
    except Exception as e:
        print(f"Data Load Error: {e}")
        return None

def main():
    print("=== PROJECT ALPHA: DEEP SCIENCE (STAT ARB) ===")
    
    # 1. Crypto Experiment
    print("\n[EXPERIMENT 1: CRYPTO MAJORS]")
    data_crypto = load_data_safe(UNIVERSE_CRYPTO)
    if data_crypto:
        trader = PairsTrader(data_crypto)
        # Run WFO on 2024
        trader.run_wfo("2024-01-01", "2024-11-30")
        
    # 2. Indices Experiment
    print("\n[EXPERIMENT 2: INDICES/METALS]")
    data_indices = load_data_safe(UNIVERSE_INDICES_METALS)
    if data_indices:
        trader = PairsTrader(data_indices)
        trader.run_wfo("2024-01-01", "2024-11-30")

if __name__ == "__main__":
    main()
