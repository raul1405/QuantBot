
"""
VERIFY LIVE SIGNALS (YFinance Shadow)
=====================================
Fetches 1.5 years of history from Yahoo Finance for Core 13 pairs.
Trains a fresh Alpha Engine (mimicking LiveTrader).
Prints the LAST BAR probabilities to compare with Live Dashboard.
"""

import sys
import os
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quant_backtest import (
    Config, FeatureEngine, RegimeEngine, 
    AlphaEngine, EnsembleSignal
)

# Yahoo Finance Mapping
SYMBOL_MAP_YF = {
    "EURUSD": "EURUSD=X", "USDJPY": "USDJPY=X", "GBPUSD": "GBPUSD=X",
    "USDCHF": "USDCHF=X", "USDCAD": "USDCAD=X", "AUDUSD": "AUDUSD=X",
    "NZDUSD": "NZDUSD=X", "EURGBP": "EURGBP=X", "EURJPY": "EURJPY=X",
    "GBPJPY": "GBPJPY=X", "AUDJPY": "AUDJPY=X", "EURAUD": "EURAUD=X",
    "EURCHF": "EURCHF=X"
}

def main():
    print("="*60)
    print("SHADOW SIGNAL VERIFICATION (SOURCE: YAHOO FINANCE)")
    print("="*60)
    
    config = Config()
    
    # 1. Fetch Data (Last 1.5 Years ~ 550 days)
    start_date = (datetime.now() - timedelta(days=550)).strftime('%Y-%m-%d')
    end_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
    
    data_map = {}
    print(f"\n[FETCHING] {start_date} -> Present...")
    
    for sym_yf in SYMBOL_MAP_YF.values():
        print(f"  Downloading {sym_yf}...", end="\r")
        try:
            # interval='1h' to match LiveTrader H1
            df = yf.download(sym_yf, start=start_date, end=end_date, interval='1h', progress=False)
            if df.empty:
                print(f"  [WARN] {sym_yf} Empty.")
                continue
                
            # Formatting
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
            
            # YF Volume is sometimes 0 for FX. Fix it?
            # AlphaEngine uses Vol_Intensity which needs Volume.
            # If Volume is 0, Vol_Intensity might break or be 0.
            # We'll leave it as is, LiveTrader MT5 has Tick Volume. YF has weird volume.
            # But let's verify.
            
            data_map[sym_yf] = df
        except Exception as e:
            print(f"  [ERR] {sym_yf}: {e}")

    print(f"\n[OK] Downloaded {len(data_map)} pairs.")
    
    if not data_map: return

    # 2. Pipeline (Same as LiveTrader)
    print("\n[PIPELINE] Feature Eng -> Regime -> Alpha Train -> Signal...")
    
    fe = FeatureEngine(config)
    re = RegimeEngine(config)
    
    # Feature Eng
    data_map = fe.add_features_all(data_map)
    data_map = re.add_regimes_all(data_map)
    
    # Train Alpha (Fresh Model)
    ae = AlphaEngine(config)
    ae.train_model(data_map)
    
    # Generate Probabilities
    data_map = ae.add_signals_all(data_map)
    
    # 3. Print Results (Last Bar)
    print("\nLATEST SIGNAL SNAPSHOT (Verify against Dashboard)")
    print("NOTE: Prices/Times might differ slighty due to Broker vs YF feed.")
    print("-" * 95)
    print(f"{'SYMBOL':<10} | {'TIME':<16} | {'CLOSE':<9} | {'UP':<5} | {'DN':<5} | {'NT':<5} | {'ACT':<4}")
    print("-" * 95)
    
    results = []
    
    for sym_yf, df in data_map.items():
        if df.empty: continue
        last = df.iloc[-1]
        
        # Extract Probs
        p_up = last.get('prob_up', 0.0)
        p_dn = last.get('prob_down', 0.0)
        p_nt = 1.0 - p_up - p_dn
        
        # Signal
        sig = last.get('S_Alpha', 0)
        act = "BUY" if sig == 1 else ("SELL" if sig == -1 else "-")
        
        results.append({
            'sym': sym_yf,
            'time': str(last.name)[5:16], # mm-dd HH:MM
            'close': last['Close'],
            'p_up': p_up,
            'p_dn': p_dn,
            'p_nt': p_nt,
            'act': act
        })
        
    # Sort by UP prob for easier viewing
    results.sort(key=lambda x: x['p_up'], reverse=True)
    
    for r in results:
        print(f"{r['sym']:<10} | {r['time']:<16} | {r['close']:<9.4f} | {r['p_up']:>5.2f} | {r['p_dn']:>5.2f} | {r['p_nt']:>5.2f} | {r['act']:<4}")

    print("-" * 95)

if __name__ == "__main__":
    main()
