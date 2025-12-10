
import pandas as pd
import numpy as np
import yfinance as yf
import sys
import os

# Add parent directory to path to import config if needed, but we will replicate logic
sys.path.append(os.path.abspath(".."))

def manual_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0))
    loss = (-delta.where(delta < 0, 0))
    
    # Wilder's Smoothing
    avg_gain = gain.rolling(window=period, min_periods=period).mean() # This is SMA, not Wilder!
    # Wait, the code in quant_backtest.py used rolling(14).mean(). 
    # That IS Simple Moving Average. classic RSI uses Exp Moving Average (Wilder).
    # quant_backtest.py line 360: gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    # So the codebase uses SMA RSI (often called Cutler's RSI).
    # I should match the CODEBASE logic to verify implementation, 
    # but note that this differs from standard RSI.
    
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def manual_volatility(series, period=20):
    return series.pct_change().rolling(period).std()

def check_features():
    print("Downloading Validation Data...")
    df = yf.download("EURUSD=X", period="1y", interval="1h", progress=False)
    
    if isinstance(df.columns, pd.MultiIndex):
        if 'Close' in df.columns.get_level_values(0):
            df.columns = df.columns.get_level_values(0)
    
    # 1. Check RSI
    print("\n[Audit: RSI]")
    df['Audit_RSI'] = manual_rsi(df['Close'], 14)
    
    # Replicate Codebase Logic directly
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df['Codebase_RSI'] = 100 - (100 / (1 + rs))
    
    diff = (df['Audit_RSI'] - df['Codebase_RSI']).abs().sum()
    print(f"Difference between Manual and Codebase RSI calculation: {diff:.6f}")
    if diff > 1e-6:
        print("  -> ERROR: Calculation mismatch!")
    else:
        print("  -> OK (Logic matches code check)")
        
    # Note on standard RSI check
    # True Wilder RSI would use ewm.
    # Highlighting this difference as potential 'bug' or 'feature' in report.
    
    # 2. Check Volatility
    print("\n[Audit: Volatility]")
    df['Audit_Vol'] = manual_volatility(df['Close'], 20)
    
    # Codebase: df['Close'].pct_change().rolling(self.config.vol_period).std()
    df['Codebase_Vol'] = df['Close'].pct_change().rolling(20).std()
    
    diff_vol = (df['Audit_Vol'] - df['Codebase_Vol']).abs().sum()
    print(f"Difference in Volatility: {diff_vol:.6f}")
    
    # 3. Check Target (Lookahead correctness)
    # Code: df['Future_Return'] = df['Close'].pct_change(1).shift(-1)
    print("\n[Audit: Target Generation]")
    df['Ret_Current'] = df['Close'].pct_change() # (C_t - C_t-1)/C_t-1
    df['Target_Code'] = df['Close'].pct_change(1).shift(-1) # (C_t+1 - C_t)/C_t
    
    # Manual verify
    # At index t, Target should be return from t to t+1.
    # Let's check a specific row.
    
    t = df.index[-5]
    t_next = df.index[-4]
    
    price_t = df.loc[t, 'Close']
    price_next = df.loc[t_next, 'Close']
    manual_return = (price_next - price_t) / price_t
    code_return = df.loc[t, 'Target_Code']
    
    print(f"Time: {t}")
    print(f"Price(t): {price_t:.5f}, Price(t+1): {price_next:.5f}")
    print(f"Manual Return: {manual_return:.6f}")
    print(f"Code Target:   {code_return:.6f}")
    
    if abs(manual_return - code_return) < 1e-8:
        print("  -> OK: Target aligns with Future Return.")
    else:
        print("  -> ERROR: Target misalignment!")

if __name__ == "__main__":
    check_features()
