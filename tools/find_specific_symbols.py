import MetaTrader5 as mt5
import json

def main():
    # Credentials (fallback)
    LOGIN = 99832013
    PASSWORD = "1xPsWuD@"
    SERVER = "MetaQuotes-Demo"

    print("Initializing...")
    if not mt5.initialize():
        print(f"Init failed: {mt5.last_error()}, retrying with creds...")
        if not mt5.initialize(login=LOGIN, password=PASSWORD, server=SERVER):
            print(f"Retry failed: {mt5.last_error()}")
            return
            
    # Targets to find
    queries = {
        "Nasdaq": ["US100", "NAS100", "NASDAQ", "NQ", "USTECH", "US 100"],
        "Oil": ["USOIL", "WTI", "CRUDE", "OIL", "UKOIL", "BRENT"],
        "Gas": ["NATGAS", "GAS", "NG", "HENRY"],
        "Crypto": ["BTC", "ETH", "BITCOIN", "ETHER", "BTCUSD", "ETHUSD"]
    }

    all_symbols = mt5.symbols_get()
    if not all_symbols:
        print("No symbols found.")
        return

    print(f"Scanned {len(all_symbols)} symbols.")
    
    for category, keywords in queries.items():
        print(f"\n--- Searching for {category} ---")
        found = []
        for s in all_symbols:
            for k in keywords:
                if k in s.name.upper() or k in s.path.upper() or k in s.description.upper():
                    # Filter out options/ETFs if possible (usually have weird paths or names)
                    if "ETF" not in s.path.upper() and "OPTION" not in s.path.upper():
                         found.append(f"{s.name} ({s.path}) - {s.description}")
                    break
        
        # Limit output
        for f in found[:20]:
            print(f)
        if not found:
            print("No matches found.")

    mt5.shutdown()

if __name__ == "__main__":
    main()
