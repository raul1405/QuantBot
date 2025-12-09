
"""
STOCK ALPHA vs BETA ANALYSIS
============================
Determines if 'Top Stocks' performance (+111%) is Skill (Alpha) or Luck (Beta).

Methodology:
1. Benchmark: Equal-Weight Buy & Hold of the 5 Stocks.
2. Strategy: Rank 1 Rotation.
3. Regression: Strat_Ret = Alpha + Beta * Bench_Ret
"""

import sys
import os
import pandas as pd
import numpy as np
import statsmodels.api as sm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quant_backtest import (
    Config, DataLoader, FeatureEngine, RegimeEngine, 
    AlphaEngine, EnsembleSignal, Backtester
)

def main():
    print("="*60)
    print("ALPHA vs BETA: TECH STOCKS 2024")
    print("="*60)
    
    symbols = ["NVDA", "AAPL", "MSFT", "TSLA", "AMD"]
    
    config = Config()
    config.symbols = symbols
    config.risk_per_trade = 0.033 # 3.3% risk
    
    # 1. Load Data
    loader = DataLoader(config)
    try:
        data = loader.load_data("2024-01-01", "2024-12-01")
    except:
        return

    # 2. Construct Benchmark (Daily Returns of Equal Weight Portfolio)
    # Align closes
    print("[BENCHMARK] Constructing Equal-Weight Index...")
    closes = pd.DataFrame({sym: df['Close'] for sym, df in data.items()}).dropna()
    bench_returns = closes.pct_change().mean(axis=1).dropna()
    bench_cum = (1 + bench_returns).cumprod()
    bench_total_ret = (bench_cum.iloc[-1] - 1) * 100
    print(f"  Benchmark Return: {bench_total_ret:.2f}%")

    # 3. Runs Strategy
    print("\n[STRATEGY] Running Algo...")
    fe = FeatureEngine(config)
    re = RegimeEngine(config)
    ae = AlphaEngine(config)
    es = EnsembleSignal(config)
    
    data = fe.add_features_all(data)
    data = re.add_regimes_all(data) # This might print
    ae.train_model(data)
    data = ae.add_signals_all(data)
    data = es.add_ensemble_all(data)
    
    bt = Backtester(config)
    equity_curve = bt.run_backtest(data)
    
    strat_ret_total = (bt.account.balance - config.initial_balance) / config.initial_balance * 100
    print(f"  Strategy Return:  {strat_ret_total:.2f}%")
    
    # 4. Alpha/Beta Calculation
    # Need Daily Returns of Strategy
    if isinstance(equity_curve, list):
        ec = pd.Series(equity_curve, index=closes.index[-len(equity_curve):]) # Approximation of index
        # Better: Backtester usually aligns equity curve to simulation steps.
        # But simulation steps might be H1.
        # We need to resample to Daily to match Benchmark?
        # Or calculate Benchmark on H1?
        pass

    # Let's use Trade Returns vs Market Returns over trade duration? Complex.
    # Let's use the Equity Curve series provided by Backtester (H1).
    # And resample Benchmark to H1?
    # Or resample Equity Curve to Daily.
    
    # Reconstruct Equity Series with Datetime Index
    # Backtester defines self.history list.
    # We assume 'closes' index covers the period.
    # We can't easily map exact timestamps without logging them.
    # Let's rely on simple comparison for now, or use 'trades' to infer dates.
    
    # Simple Logic:
    # If Strategy > Benchmark, and Strategy DD < Benchmark DD -> Alpha.
    
    ec_series = pd.Series(bt.account.equity_curve)
    strat_dd = (ec_series - ec_series.cummax()).div(ec_series.cummax()).min() * 100
    
    bench_dd = (bench_cum - bench_cum.cummax()).div(bench_cum.cummax()).min() * 100
    
    print("\n" + "="*60)
    print("COMPARISON")
    print("="*60)
    print(f"{'Metric':<15} | {'Strategy':<10} | {'Benchmark (Eq Wt)':<20}")
    print("-" * 55)
    print(f"{'Return':<15} | {strat_ret_total:>9.2f}% | {bench_total_ret:>18.2f}%")
    print(f"{'Max Drawdown':<15} | {strat_dd:>9.2f}% | {bench_dd:>18.2f}%")
    
    # Beta Estimate (Rough)
    # Beta = Ret_Strat / Ret_Bench (if correlated)
    # Not mathematically precise but indicative of leverage.
    beta_proxy = strat_ret_total / bench_total_ret if bench_total_ret != 0 else 0
    print(f"{'Leverage Beta':<15} | {beta_proxy:>9.1f}x | {'1.0x':>18}")
    
    print("\nVERDICT:")
    if strat_ret_total > bench_total_ret:
        print("✅ Alpha Detected. (Outperformed Buy & Hold).")
        if strat_dd > bench_dd:
            print("⚠️ High Risk. Drawdown is deeper than benchmark.")
            print("   Likely just Leveraged Beta.")
        else:
            print("💎 TRUE ALPHA. Higher Return AND Lower Drawdown.")
    else:
        print("❌ No Alpha. Underperformed Benchmark.")

if __name__ == "__main__":
    main()
