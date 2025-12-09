import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

# Load credentials
load_dotenv()
LOGIN = os.getenv("MT5_LOGIN", "")
PASSWORD = os.getenv("MT5_PASSWORD", "")
SERVER = os.getenv("MT5_SERVER", "")

def init_mt5():
    if not mt5.initialize():
        print("initialize() failed, error code =", mt5.last_error())
        return False
    
    if LOGIN and PASSWORD and SERVER:
        authorized = mt5.login(int(LOGIN), password=PASSWORD, server=SERVER)
        if authorized:
            print(f"Connected to account #{LOGIN}")
        else:
            print("failed to connect at account #{}, error code: {}".format(LOGIN, mt5.last_error()))
            return False
    return True

def fetch_tick_alpha(symbol, hours=24):
    print(f"\n[RESEARCH] Fetching {hours}h of TICK data for {symbol}...")
    
    # 1. Get H1 Candles (Target)
    utc_to = datetime.now()
    candles = mt5.copy_rates_from(symbol, mt5.TIMEFRAME_H1, utc_to, hours)
    if candles is None:
        print("No candles found.")
        return None
        
    df_h1 = pd.DataFrame(candles)
    df_h1['time'] = pd.to_datetime(df_h1['time'], unit='s')
    df_h1.set_index('time', inplace=True)
    df_h1['ret_next'] = df_h1['close'].shift(-1).pct_change() # Target: Next bar return
    
    # 2. Get Ticks (Features)
    # We need to loop because copy_ticks_from range might be limited? 
    # Actually copy_ticks_from takes a count or date. Let's use date range loop for safety or just bulk.
    # 1 hour ~ 100k ticks max? For 24h it's maybe 2.4M ticks. MT5 handles it.
    
    ticks = mt5.copy_ticks_from(symbol, utc_to - timedelta(hours=hours), hours * 100000, mt5.COPY_TICKS_ALL)
    if ticks is None:
        print("No ticks found.")
        return None
        
    df_tick = pd.DataFrame(ticks)
    df_tick['time'] = pd.to_datetime(df_tick['time'], unit='s')
    
    # 3. Aggregate Ticks into H1 Bins
    # Feature 1: Trade Count
    # Feature 2: Delta (Buy Vol - Sell Vol)
    # MT5 Flags: 
    # TICK_FLAG_BUY (32) / TICK_FLAG_SELL (64)
    # Check flags for direction
    
    print(f"  > Processed {len(df_tick)} ticks...")
    
    resampled = []
    
    for t_idx, row in df_h1.iterrows():
        # Define bin range
        t_start = t_idx
        t_end = t_idx + timedelta(hours=1)
        
        # Slice ticks
        mask = (df_tick['time'] >= t_start) & (df_tick['time'] < t_end)
        bin_ticks = df_tick[mask]
        
        if len(bin_ticks) == 0:
            resampled.append({'count': 0, 'delta': 0, 'buy_vol': 0, 'sell_vol': 0})
            continue
            
        # Calc Delta
        # MT5 flags are bitmasks. 
        # BUY = 32 (or 32+others), SELL = 64
        # Note: Logic depends on broker. Often (flags & mt5.TICK_FLAG_BUY) == mt5.TICK_FLAG_BUY
        
        buys = bin_ticks[ (bin_ticks['flags'] & mt5.TICK_FLAG_BUY) == mt5.TICK_FLAG_BUY ]
        sells = bin_ticks[ (bin_ticks['flags'] & mt5.TICK_FLAG_SELL) == mt5.TICK_FLAG_SELL ]
        
        buy_vol = buys['volume'].sum() if not buys.empty else 0.0
        sell_vol = sells['volume'].sum() if not sells.empty else 0.0
        
        # If flags are empty (some brokers dont populate), try Price vs Bid/Ask?
        # Fallback: Tick Direction?
        # For FTMO, flags usually work.
        
        delta = buy_vol - sell_vol
        
        resampled.append({
            'count': len(bin_ticks),
            'delta': delta,
            'buy_vol': buy_vol,
            'sell_vol': sell_vol,
            'tick_vol_ratio': buy_vol / (buy_vol + sell_vol + 1e-9)
        })
        
    df_features = pd.DataFrame(resampled, index=df_h1.index)
    
    # 4. Merge
    df_final = df_h1.join(df_features)
    df_final.dropna(inplace=True)
    return df_final

def analyze_results(df):
    if df is None or df.empty: return
    
    # Correlation
    print("\n[ANALYSIS] Correlation with Future Return (Next H1):")
    cols = ['count', 'delta', 'tick_vol_ratio', 'tick_volume'] # tick_volume is from candles
    for c in cols:
        if c in df.columns:
            corr = df[c].corr(df['ret_next'])
            print(f"  {c:<15}: {corr:+.4f}")
            
    # Simple Strategy Test (Sign of Delta)
    df['sig'] = np.sign(df['delta'])
    df['strat_ret'] = df['sig'] * df['ret_next']
    cum_ret = df['strat_ret'].cumsum()
    print(f"\n[BACKTEST] Cumulative Return (Delta Follower): {cum_ret.iloc[-1]*100:.2f}% (over {len(df)} bars)")

if __name__ == "__main__":
    if init_mt5():
        # Test on volatile assets
        assets = ["XAUUSD", "EURUSD", "US500", "BTCUSD"] 
        # Check available symbols on broker
        all_syms = [s.name for s in mt5.symbols_get()]
        
        for a in assets:
            if a not in all_syms:
                # Try finding mapped symbol (e.g. BTCUSD could be Bitcoin)
                pass 
                
            if a in all_syms:
                df = fetch_tick_alpha(a, hours=48) # Last 48h
                analyze_results(df)
            else:
                print(f"[SKIP] {a} not found.")
                
        mt5.shutdown()
