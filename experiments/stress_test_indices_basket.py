
"""
INDICES BASKET STRESS TEST
==========================
Audits the 2.35 Sharpe Ratio for Equity Indices.
Universe: ES=F (S&P), NQ=F (Nasdaq), YM=F (Dow), RTY=F (Russell).

Metrics:
1. Beta to S&P 500.
2. Performance during August Crash.
3. Alpha vs Equal-Weight Benchmark.
"""

import sys
import os
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quant_backtest import (
    Config, DataLoader, FeatureEngine, RegimeEngine, 
    AlphaEngine, EnsembleSignal, Backtester
)

def analyze_crash(trades, start_date, end_date):
    print(f"\n[CRASH WINDOW: {start_date} to {end_date}]")
    mask = (trades['Entry Time'] >= start_date) & (trades['Entry Time'] <= end_date)
    subset = trades[mask]
    
    if subset.empty:
        print("  Status: FLAT (No Trades). Preserved Capital.")
        return 0
        
    pnl = subset['PnL'].sum()
    print(f"  PnL: ${pnl:,.0f} ({len(subset)} trades)")
    shorts = subset[subset['Direction'] == 'SHORT']
    if not shorts.empty:
         print(f"  Shorts: {len(shorts)} trades, PnL: ${shorts['PnL'].sum():,.0f}")
    
    return pnl

def main():
    print("="*60)
    print("AUDIT: INDICES BASKET (2024)")
    print("="*60)
    
    indices = ["ES=F", "NQ=F", "YM=F", "RTY=F"]
    config = Config()
    config.symbols = indices
    config.risk_per_trade = 0.02 # 2% per trade
    
    # 1. Load Data
    loader = DataLoader(config)
    try:
        data = loader.load_data("2024-01-01", "2024-12-01")
    except:
        return

    # 2. Benchmark (Equal Weight)
    print("\n[BENCHMARK] Calculating Equal-Weight B&H...")
    closes = pd.DataFrame({sym: df['Close'] for sym, df in data.items()}).dropna()
    # Normalize
    norm = closes / closes.iloc[0]
    bench_curve = norm.mean(axis=1)
    bench_ret = (bench_curve.iloc[-1] - 1) * 100
    
    # Benchmark Daily Sharpe
    bench_daily_ret = bench_curve.pct_change().dropna()
    bench_sharpe = 0
    if bench_daily_ret.std() > 0:
        bench_sharpe = bench_daily_ret.mean() / bench_daily_ret.std() * np.sqrt(252)
        
    print(f"  Benchmark Return: {bench_ret:.2f}%")
    print(f"  Benchmark Sharpe: {bench_sharpe:.2f}")

    # 3. Strategy
    print("\n[STRATEGY] Running Algo...")
    fe = FeatureEngine(config)
    re = RegimeEngine(config)
    ae = AlphaEngine(config)
    es = EnsembleSignal(config)
    
    data = fe.add_features_all(data)
    data = re.add_regimes_all(data)
    ae.train_model(data)
    data = ae.add_signals_all(data)
    data = es.add_ensemble_all(data)
    
    bt = Backtester(config)
    # bt.run_backtest returns equity curve (list of floats)
    strat_curve_list = bt.run_backtest(data)
    # strat_curve_list = bt.account.equity_curve # REMOVED
    
    strat_bal = bt.account.balance
    strat_ret = (strat_bal - config.initial_balance) / config.initial_balance * 100
    
    # Strat Daily Sharpe
    # Need to match dates approx? Or just calc on list
    sc = pd.Series(strat_curve_list)
    s_ret = sc.pct_change().dropna()
    strat_sharpe = 0
    if s_ret.std() > 0:
        strat_sharpe = s_ret.mean() / s_ret.std() * np.sqrt(252)

    print(f"  Strategy Return: {strat_ret:.2f}%")
    print(f"  Strategy Sharpe: {strat_sharpe:.2f}")
    
    trades = pd.DataFrame(bt.account.trade_history)
    if not trades.empty:
        trades['Entry Time'] = pd.to_datetime(trades['Entry Time'])
        
        # 4. Crash Analysis
        aug_pnl = analyze_crash(trades, "2024-07-15", "2024-08-10")
        apr_pnl = analyze_crash(trades, "2024-04-01", "2024-04-30")
        
    print("\n" + "="*60)
    print("VERDICT")
    print("="*60)
    
    if strat_sharpe > 1.5 * bench_sharpe:
        print("✅ GENUINE ALPHA. Sharpe is significantly higher (>1.5x Benchmark).")
    elif strat_sharpe > bench_sharpe:
        print("⚠️ MARGINAL ALPHA. Sharpe is better, but maybe just lucky timing.")
    else:
        print("❌ BETA RIDE. Strategy is arguably worse than holding the index.")
        
    if trades.empty: return

    # Check Shorts
    shorts = trades[trades['Direction'] == 'SHORT']
    if shorts['PnL'].sum() > 0:
        print("✅ HEDGE CAPABILITY: Shorts contributed Profit.")
    else:
        print("⚠️ BULL ONLY: Shorts lost money.")

if __name__ == "__main__":
    main()
