
"""
INDICES TRADE DISTRIBUTION ANALYSIS
===================================
Objective: Evaluate the mathematical edge (Expectancy) of the Indices Strategy.
User Argument: "I don't care about monthly stats, look at the trades."
Metrics:
1. Win Rate.
2. Avg Win / Avg Loss (Risk:Reward).
3. Expectancy (Edge per Trade).
4. SQN (System Quality Number).
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

def main():
    print("="*60)
    print("INDICES: TRADE EXPECTANCY ANALYSIS")
    print("="*60)
    
    config = Config()
    config.symbols = ["ES=F", "NQ=F", "YM=F", "RTY=F"]
    config.risk_per_trade = 0.015
    
    loader = DataLoader(config)
    try: data = loader.load_data("2024-01-01", "2024-12-01")
    except: return

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
    bt.run_backtest(data)
    trades = pd.DataFrame(bt.account.trade_history)
    
    if trades.empty:
        print("No trades check.")
        return
        
    # Analyze Trade Distribution
    pnl = trades['PnL']
    wins = pnl[pnl > 0]
    losses = pnl[pnl <= 0]
    
    n_trades = len(trades)
    win_rate = len(wins) / n_trades * 100
    avg_win = wins.mean() if not wins.empty else 0
    avg_loss = losses.mean() if not losses.empty else 0
    
    # Expectancy = (Win% * AvgWin) - (Loss% * AvgLoss) -> NO abs(loss) usually
    if avg_loss == 0: avg_loss = -1.0 # Avoid div zero
    
    payoff_ratio = abs(avg_win / avg_loss)
    expectancy = (len(wins)/n_trades * avg_win) + (len(losses)/n_trades * avg_loss)
    
    print(f"\nTotal Trades: {n_trades}")
    print(f"Win Rate:     {win_rate:.1f}%")
    print(f"Avg Win:      ${avg_win:,.2f}")
    print(f"Avg Loss:     ${avg_loss:,.2f}")
    print(f"Payoff Ratio: {payoff_ratio:.2f} (Reward:Risk)")
    print("-" * 40)
    print(f"EXPECTANCY:   ${expectancy:,.2f} per trade")
    
    # SQN = sqrt(N) * Mean / Std
    sqn = np.sqrt(n_trades) * pnl.mean() / pnl.std()
    print(f"SQN Score:    {sqn:.2f}")
    
    print("\nVERDICT:")
    if expectancy > 0 and sqn > 1.0:
        print("✅ POSITIVE EDGE. The math works long term.")
        if sqn < 2.0:
            print("⚠️ VOLATILE. It is profitable but rocky (SQN < 2).")
    else:
        print("❌ NEGATIVE EDGE. You lose money on average.")

if __name__ == "__main__":
    main()
