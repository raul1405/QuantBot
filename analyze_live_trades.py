import pandas as pd
import numpy as np
import os

LOG_FILE = "live_logs/FT_001_trades.csv"

def load_data():
    if not os.path.exists(LOG_FILE):
        print(f"Log file {LOG_FILE} not found.")
        return None
    df = pd.read_csv(LOG_FILE)
    df['Entry_Time'] = pd.to_datetime(df['Entry_Time'])
    return df

def analyze_forensics(df):
    if df is None or len(df) == 0: return

    print("\n" + "="*40)
    print("LIVE TRADE FORENSICS (Deep Dive)")
    print("="*40)
    
    # 1. By Trend Regime
    if 'Entry_Trend_Regime' in df.columns:
        print("\n[PERFORMANCE BY TREND REGIME]")
        grp = df.groupby('Entry_Trend_Regime')['R_Multiple'].agg(['count', 'mean', 'sum']).sort_values('mean', ascending=False)
        print(grp)
        
    # 2. By Volatility Bucket (Quantiles)
    if 'Entry_Vol_Intensity' in df.columns:
        print("\n[PERFORMANCE BY VOL INTENSITY]")
        try:
            df['Vol_Bucket'] = pd.qcut(df['Entry_Vol_Intensity'], 3, labels=["Low", "Med", "High"])
            grp = df.groupby('Vol_Bucket')['R_Multiple'].agg(['count', 'mean', 'sum'])
            print(grp)
        except:
            print("Not enough data for Vol Buckets.")
            
    # 3. By Hour of Day
    if 'Entry_Hour' in df.columns:
        print("\n[PERFORMANCE BY HOUR]")
        grp = df.groupby('Entry_Hour')['R_Multiple'].agg(['count', 'mean', 'sum']).sort_index()
        print(grp)
        
    # 4. Long vs Short
    print("\n[PERFORMANCE BY DIRECTION]")
    grp = df.groupby('Direction')['R_Multiple'].agg(['count', 'mean', 'sum'])
    print(grp)

if __name__ == "__main__":
    df = load_data()
    analyze_forensics(df)
