import pandas as pd
from datetime import datetime, timezone
# Mockup of the issue
# We need to load REAL data to reproduce.
# But loading all data is slow.
# We will load ONE symbol.

from quant_backtest import Config, DataLoader, Backtester

config = Config()
loader = DataLoader(config)
df = loader.load_data("2025-01-01", "2025-01-10") # Short range
sym = list(df.keys())[0]
print(f"Data Index Sample: {df[sym].index[:5]}")
print(f"Data Timezone: {df[sym].index.dtype}")

# Create dummy trade with timezone
trade_time = df[sym].index[0].replace(hour=14) # modify slightly
print(f"Trade Time: {trade_time}")

# Test matching
idx = df[sym].index.get_indexer([trade_time], method='nearest')
print(f"Match Index: {idx}")
