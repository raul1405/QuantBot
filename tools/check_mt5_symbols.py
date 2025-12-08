import MetaTrader5 as mt5
import pandas as pd
import os
import json

    # HARDCODED CREDENTIALS (USER PROVIDED)
    LOGIN = 99832013
    PASSWORD = "1xPsWuD@"
    SERVER = "MetaQuotes-Demo"
    
    print(f"Initializing MT5 with Account {LOGIN}...")
    # Try initializing WITH credentials to force correct account load
    if not mt5.initialize(login=LOGIN, password=PASSWORD, server=SERVER):
        print("initialize() failed, error code =", mt5.last_error())
        print("Retrying without credentials (using existing terminal state)...")
        if not mt5.initialize():
             print("initialize() fallback failed too.")
             return

    print("MT5 Initialized. Verifying connection...")
    
    # Force Login
    authorized = mt5.login(LOGIN, password=PASSWORD, server=SERVER)
    if authorized:
        print(f"Connected to {LOGIN} on {SERVER}")
    else:
        print(f"Login failed: {mt5.last_error()}")
        # Continue anyway, listing symbols might still work if partial auth exists


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
