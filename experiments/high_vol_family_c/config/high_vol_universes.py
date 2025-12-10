
# ==============================================================================
# FAMILY C: HIGH-VOLATILITY UNIVERSES
# ==============================================================================
# "Ex-Ante" selection based on liquidity and volatility profile.
# No performance-based filtering.

# 1. CRYPTO MAJORS (24/7, High Volatility, Inefficient)
# Selected Top 5 by Market Cap (excluding Stablecoins) available on Yahoo Finance.
UNIVERSE_CRYPTO = [
    "BTC-USD",  # Bitcoin
    "ETH-USD",  # Ethereum
    "SOL-USD",  # Solana
    "BNB-USD",  # Binance Coin
    "XRP-USD"   # Ripple
]

# 2. INDICES & METALS (Futures)
# High structural volatility and mean-reversion tendencies.
UNIVERSE_INDICES_METALS = [
    "GC=F",     # Gold Futures
    "NQ=F",     # Nasdaq 100 Futures
    "ES=F",     # S&P 500 Futures
    "YM=F"      # Dow Jones Futures
]
