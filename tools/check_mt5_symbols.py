import MetaTrader5 as mt5
import pandas as pd
import os
import json

# Load Config for Credentials
try:
    with open("live_config.json", "r") as f:
        config = json.load(f)
        creds = config.get("credentials", {})
        LOGIN = int(creds.get("login", 0))
        PASSWORD = creds.get("password", "")
        SERVER = creds.get("server", "")
except:
    print("Could not load live_config.json")
    LOGIN = 0
    PASSWORD = ""
    SERVER = ""

def main():
    print(f"Connecting to {SERVER} as {LOGIN}...")
    if not mt5.initialize():
        print("initialize() failed, error code =", mt5.last_error())
        return

    if LOGIN and PASSWORD:
        authorized = mt5.login(LOGIN, password=PASSWORD, server=SERVER)
        if not authorized:
            print("failed to connect at account #{}, error code: {}".format(LOGIN, mt5.last_error()))
            return
        print("Connected.")

    # Get all symbols
    symbols = mt5.symbols_get()
    if symbols is None:
        print("No symbols found.")
        return

    print(f"Total Symbols available: {len(symbols)}")
    print("-" * 50)

    # Keywords to search for
    keywords = ["US100", "NAS100", "NASDAQ", "US500", "SPX", "S&P", 
                "US30", "DOW", "DJ30", "USOIL", "WTI", "CRUDE", "BRENT", "UKOIL",
                "NATGAS", "GAS", "NG", 
                "BTC", "BITCOIN", "ETH", "ETHER", "XAU", "GOLD"]

    found_count = 0
    print(f"{'SYMBOL':<20} | {'PATH':<30} | {'DESCRIPTION'}")
    print("-" * 80)
    
    for s in symbols:
        match = False
        # Exact checks or substring
        for k in keywords:
            if k in s.name.upper() or k in s.path.upper():
                match = True
                break
        
        if match:
            print(f"{s.name:<20} | {s.path:<30} | {s.description}")
            found_count += 1
            
    print("-" * 80)
    print(f"Found {found_count} potential matches.")
    
    mt5.shutdown()

if __name__ == "__main__":
    main()
