import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def fetch_pseudo_tick_alpha(symbol, days=5):
    """
    Approximates 'Tick Delta' using 1-minute data from Yahoo Finance.
    Delta ~= Sum of (Close - Open) * Volume for all 1m bars inside the hour.
    """
    print(f"\n[RESEARCH] Fetching {days}d of 1m data for {symbol} (Pseudo-Tick)...")
    
    # 1. Fetch 1m data (Max 7 days allowed by Yahoo)
    df_1m = yf.download(symbol, period=f"{days}d", interval="1m", progress=False)
    
    if df_1m is None or df_1m.empty:
        print("No data found.")
        return None
        
    # Flatten MultiIndex columns if necessary (yfinance v0.2+)
    if isinstance(df_1m.columns, pd.MultiIndex):
        df_1m.columns = df_1m.columns.get_level_values(0)
        
    # 2. Calculate "Micro-Pressure" per minute
    # If Close > Open, assumed Buy Volume. Else Sell Volume.
    # We weight by Volume.
    # Formula: (2*Close - High - Low) / (High - Low) * Volume  (Money Flow Multiplier)
    # Simplified: (Close - Open) * Volume? No, uses range location.
    # Let's use Accumulation/Distribution (A/D) logic per bar.
    
    # Handle NaN in High/Low equality
    hl_range = df_1m['High'] - df_1m['Low']
    hl_range = hl_range.replace(0, 1e-9) 
    
    mf_mult = ((df_1m['Close'] - df_1m['Low']) - (df_1m['High'] - df_1m['Close'])) / hl_range
    df_1m['money_flow_vol'] = mf_mult * df_1m['Volume']
    
    # 3. Resample to 1H to get "Hourly Delta"
    # Yahoo 1m data has datetime index localized?
    if df_1m.index.tz is None:
        df_1m.index = df_1m.index.tz_localize('UTC')
    else:
        df_1m.index = df_1m.index.tz_convert('UTC')
        
    # Aggregate
    df_h1 = df_1m.resample('1h').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum',
        'money_flow_vol': 'sum' # This is our "Pseudo Delta"
    })
    
    df_h1.dropna(inplace=True)
    
    # 4. Target: Next Return
    df_h1['ret_next'] = df_h1['Close'].shift(-1).pct_change()
    
    # 5. Features
    df_h1['pseudo_delta'] = df_h1['money_flow_vol']
    df_h1['delta_ratio'] = df_h1['pseudo_delta'] / (df_h1['Volume'] + 1e-9)
    
    df_h1.dropna(inplace=True)
    return df_h1

def analyze_results(df):
    if df is None or df.empty: return
    
    # Correlation
    print("\n[ANALYSIS] Correlation with Future Return (Next H1):")
    cols = ['pseudo_delta', 'delta_ratio', 'Volume']
    for c in cols:
        if c in df.columns:
            corr = df[c].corr(df['ret_next'])
            print(f"  {c:<15}: {corr:+.4f}")
            
    # Simple Strategy
    df['sig'] = np.sign(df['delta_ratio'])
    df['strat_ret'] = df['sig'] * df['ret_next']
    cum_ret = df['strat_ret'].cumsum()
    print(f"\n[BACKTEST] Cumulative Return (Pseudo-Delta): {cum_ret.iloc[-1]*100:.2f}% (over {len(df)} bars)")

if __name__ == "__main__":
    assets = ["EURUSD=X", "JPY=X", "GBPUSD=X", "SPY", "BTC-USD"]
    
    for a in assets:
        try:
            df = fetch_pseudo_tick_alpha(a, days=5)
            analyze_results(df)
        except Exception as e:
            print(f"Error on {a}: {e}")
