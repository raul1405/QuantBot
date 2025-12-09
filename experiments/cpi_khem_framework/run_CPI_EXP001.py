"""
EXP001: Pure CPI Overlay Backtest
=================================
Objective: Verify the mechanics of the CPI strategy isolation.
Output: Equity Curve, Trades List.
"""

import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
from cpi_engine import CPIEngine, CPIConfig

def main():
    print("Running EXP001: Pure CPI Backtest")
    print("=================================")
    
    # 1. Setup
    cfg = CPIConfig(
        symbol_equity="ES=F",
        symbol_inflation="GC=F",
        horizon_days=5,
        risk_cap_per_trade=5000
    )
    
    engine = CPIEngine(cfg)
    engine.load_data()
    
    # 2. Run
    trades_df = engine.run_backtest()
    
    if trades_df.empty:
        print("No Valid Trades generated.")
        return

    # 3. Analysis
    # Construct Equity Curve
    # Starts at Initial Capital.
    # PnL is realized at 'Date'. (Simplification: Realized on Entry? No, usually exit. 
    # But for "Overlay" visualization, let's step it.)
    
    # trades_df has 'Date' (Entry) and 'Payout', 'Premium'.
    # Net PnL is realized at... Exit?
    # Let's say realized at Exit for curve.
    
    trades_df['Exit_Date'] = pd.to_datetime(trades_df['Date']) + pd.Timedelta(days=cfg.horizon_days)
    trades_df = trades_df.sort_values('Exit_Date')
    
    equity = [cfg.initial_capital]
    dates = [trades_df['Date'].min() - pd.Timedelta(days=1)]
    
    cum_pnl = 0
    for _, t in trades_df.iterrows():
        payload = t['Net_PnL']
        cum_pnl += payload
        equity.append(cfg.initial_capital + cum_pnl)
        dates.append(t['Exit_Date'])
        
    # Plot
    plt.figure(figsize=(10, 6))
    plt.step(dates, equity, where='post')
    plt.title("EXP001: CPI Overlay Equity Curve (Gold)")
    plt.xlabel("Date")
    plt.ylabel("Equity ($)")
    plt.grid(True)
    plt.savefig("exp_CPI_001_chart.png")
    print("Chart saved to exp_CPI_001_chart.png")

    # 4. Report
    report = f"""# EXP001: CPI Overlay Results (Gold)

## Parameters
- Asset: {cfg.symbol_inflation}
- Horizon: {cfg.horizon_days} Days
- Trigger: 2/3 Acceleration
- Risk: ${cfg.risk_cap_per_trade}

## Performance
- Total Trades: {len(trades_df)}
- Total PnL: ${trades_df['Net_PnL'].sum():.2f}
- Win Rate: {len(trades_df[trades_df['Net_PnL']>0]) / len(trades_df):.1%}
- Hit Dates: {trades_df['Date'].dt.date.tolist()}

## Interpretation
This strategy trades INFREQUENTLY ({len(trades_df)} times in ~1 year).
It is a "Sniper" overlay, not a continuous strategy.
"""
    with open("exp_CPI_001_results.md", "w") as f:
        f.write(report)
    
    print("Report saved to exp_CPI_001_results.md")

if __name__ == "__main__":
    main()
