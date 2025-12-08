"""
Skeleton for Exp_008 – DO NOT EXECUTE YET.
RESEARCH ONLY.

Idea:
  - Import quant_backtest.Config
  - Override universe_name = "ftmo_generic_universe" (once implemented)
  - Run walk-forward with v2.1 settings
  - Save results to exp_008_ftmo_universe.md

# ==============================================================================
# class Exp008_FTMO_Universe(Experiment):
#     def run(self):
#         print("[EXP 008] Starting Universe Restriction Test...")
#         
#         # 1. Define Restricted Universe (FTMO Only)
#         ftmo_subset = [
#             'EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'USDCHF=X', 'USDCAD=X',
#             'AUDUSD=X', 'NZDUSD=X', 'EURJPY=X', 'GBPJPY=X',
#             'GC=F', 'ES=F', 'NQ=F', 'YM=F', 'BTC-USD', 'ETH-USD'
#         ]
#         
#         # 2. Configure Engine
#         config = Config()
#         config.universe = ftmo_subset
#         # ... override other params if needed ...
#         
#         # 3.Run Backtest
#         # engine = BacktestEngine(config)
#         # results = engine.run_full_backtest()
#         
#         # 4. Compare with Baseline (Full Universe)
#         # ... plot comparison ...
#         
#         print("[EXP 008] Complete. Results saved.")
# ==============================================================================
"""
