
"""
CHECK DATA LATENCY (Indices vs FX)
==================================
Checks if Yahoo Finance data for Indices (NQ=F, ES=F) is Real-Time or Delayed.
Compares 'Last Quote Time' vs 'System Time'.
"""

import yfinance as yf
import datetime
import pytz
import time

def check_symbol(symbol):
    print(f"\nChecking {symbol}...")
    try:
        # Request 1d 1m
        ticker = yf.Ticker(symbol)
        df = ticker.history(period="1d", interval="1m")
        
        if df.empty:
            print("  No data found.")
            return

        last_idx = df.index[-1]
        now = datetime.datetime.now(pytz.utc)
        
        # Convert last_idx to UTC if needed
        if last_idx.tzinfo is None:
            last_idx = last_idx.replace(tzinfo=pytz.utc)
        else:
            last_idx = last_idx.astimezone(pytz.utc)
            
        diff =  now - last_idx
        minutes = diff.total_seconds() / 60
        
        print(f"  Last Bar: {last_idx.strftime('%H:%M:%S')} UTC")
        print(f"  Current:  {now.strftime('%H:%M:%S')} UTC")
        print(f"  Lag:      {minutes:.1f} minutes")
        
        if minutes > 20: 
            print("  ⚠️ COMPLETE DELAY (>20m). Likely 15m Delayed Feed.")
        elif minutes > 5:
             print("  ⚠️ MINOR LAG (>5m).")
        else:
             print("  ✅ REAL-TIME (<5m).")
             
    except Exception as e:
        print(f"  Error: {e}")

def main():
    print("="*60)
    print("DATA LATENCY CHECK")
    print("="*60)
    
    # Check FX (Reference)
    check_symbol("EURUSD=X")
    
    # Check Indices
    check_symbol("NQ=F")  # Nasdaq Futures
    check_symbol("ES=F")  # S&P Futures
    check_symbol("YM=F")  # Dow Futures
    
    # Check Index Tickers directly?
    check_symbol("^IXIC") # Nasdaq Composite
    check_symbol("^GSPC") # S&P 500

if __name__ == "__main__":
    main()
