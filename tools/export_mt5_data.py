
import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime
import os
import json

# Credentials from live_config.json
LOGIN = 99832013
PASSWORD = "1xPsWuD@"
SERVER = "MetaQuotes-Demo"

TARGET_NAME = "US500"
TIMEFRAME = mt5.TIMEFRAME_H4
TIMEFRAME_STR = "H4"

def init_mt5():
    if not mt5.initialize():
        print(f"initialize() failed, error code = {mt5.last_error()}")
        return False
    
    print(f"MT5 Initialized. Version: {mt5.version()}")
    
    authorized = mt5.login(LOGIN, password=PASSWORD, server=SERVER)
    if authorized:
        print(f"Connected to account #{LOGIN}")
    else:
        print(f"failed to connect at account #{LOGIN}, error code: {mt5.last_error()}")
        return False
    return True

def find_symbol(target):
    # Try direct select
    if mt5.symbol_select(target, True):
        return target
        
    # Search
    symbols = mt5.symbols_get()
    for s in symbols:
        if target in s.name and "500" in s.name:
            print(f"Found match: {s.name}")
            mt5.symbol_select(s.name, True)
            return s.name
            
    # Fallback search for S&P500 aliases
    aliases = ["SP500", "S&P500", "US500", "US.500", "SPX500"]
    for a in aliases:
        for s in symbols:
            if a in s.name:
                print(f"Found alias match: {s.name}")
                mt5.symbol_select(s.name, True)
                return s.name
    
    return None

def main():
    if not init_mt5():
        return

    symbol = find_symbol(TARGET_NAME)
    if not symbol:
        print(f"Could not find symbol matching {TARGET_NAME}")
        mt5.shutdown()
        return

    print(f"Targeting Symbol: {symbol}")
    
    # Download Max History
    # From 2000 to now
    utc_from = datetime(2010, 1, 1)
    utc_to = datetime.now()
    
    print(f"Downloading {TIMEFRAME_STR} data from {utc_from} to {utc_to}...")
    
    rates = mt5.copy_rates_range(symbol, TIMEFRAME, utc_from, utc_to)
    
    if rates is None or len(rates) == 0:
        print("No data received.")
        mt5.shutdown()
        return
        
    print(f"Received {len(rates)} bars.")
    
    # Create DataFrame
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    
    filename = f"{symbol}_{TIMEFRAME_STR}_{len(df)}_bars.csv"
    df.to_csv(filename, index=False)
    print(f"Saved to {filename}")
    
    mt5.shutdown()

if __name__ == "__main__":
    main()
