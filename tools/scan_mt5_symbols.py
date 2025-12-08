import MetaTrader5 as mt5
import pandas as pd

def scan_symbols():
    if not mt5.initialize():
        print("initialize() failed")
        mt5.shutdown()
        return

    print(f"Connected to: {mt5.terminal_info().name}")
    print(f"Account: {mt5.account_info().login} exclude_hidden=False")
    
    # Get all symbols
    symbols = mt5.symbols_get()
    print(f"Total Symbols Found: {len(symbols)}")
    
    # Filter for interesting ones or just dump all
    data = []
    for s in symbols:
        data.append({
            'Symbol': s.name,
            'Path': s.path,
            'Description': s.description,
            'Digits': s.digits
        })
        
    df = pd.DataFrame(data)
    
    # Save to CSV for analysis
    df.to_csv("mt5_symbols_dump.csv", index=False)
    print("Saved symbol list to 'mt5_symbols_dump.csv'")
    
    # Print partial matches for our universe
    search_terms = ['USD', 'EUR', 'GBP', 'JPY', 'BTC', 'ETH', 'US500', 'ES', 'GOLD', 'XAU']
    print("\n--- Potential Matches ---")
    for term in search_terms:
        matches = df[df['Symbol'].str.contains(term, case=False)]
        if not matches.empty:
            print(f"\nMatches for '{term}':")
            print(matches['Symbol'].tolist()[:10]) # Show top 10

    mt5.shutdown()

if __name__ == "__main__":
    scan_symbols()
