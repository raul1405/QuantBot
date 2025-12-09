
"""
FTMO PASS PROBABILITY (Monte Carlo)
===================================
Calculates the likelihood of passing the FTMO Swing Challenge (-10% Max DD, +10% Profit).

Methodology:
1. Run Strategy Comparison backtest to get 'Real Trade Distribution'.
2. Bootstrap (Resample) this distribution 10,000 times.
3. Simulate Equity Curves.
4. Check Constraints:
   - Fail: Equity < -10% (Max DD)
   - Fail: Daily Loss > 5% (Approximate by checking trade clustering)
   - Pass: Equity > +10%
"""

import sys
import os
import pandas as pd
import numpy as np
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quant_backtest import (
    Config, DataLoader, FeatureEngine, RegimeEngine, 
    AlphaEngine, EnsembleSignal, Backtester
)

def run_simulation(trades_pnl_pct):
    """
    Simulates a single account journey.
    Returns: 'PASS', 'FAIL_MAX_DD', 'FAIL_DAILY', 'STUCK'
    """
    equity = 1.0
    peak = 1.0
    daily_start_equity = 1.0
    
    # Approx 2 trades per day max? Or cluster trades?
    # We'll just iterate trades. To check Daily DD, we need dates or assume density.
    # Assumption: 1 Trade per "Step". Max 3 bad trades in a day?
    # Conservative: Sample random "Daily Volume" of trades (e.g. 0 to 3).
    
    # We loop "Days"
    limit_days = 252 * 2 # 2 years limit (effectively unlimited)
    
    bag = list(trades_pnl_pct)
    
    for day in range(limit_days):
        # Determine trades today (Poisson distribution avg 0.5 trades/day?)
        # Let's say Prob of trade = 40%.
        # If trade, sample from bag.
        
        n_trades = np.random.poisson(0.5) # Avg 0.5 trades per day
        
        daily_pnl = 0
        
        for _ in range(n_trades):
            trade_ret = random.choice(bag)
            daily_pnl += trade_ret
            
            equity *= (1 + trade_ret)
            
            # Check Max DD
            if equity > peak: peak = equity
            dd = (equity - peak) / peak
            if dd < -0.10: return 'FAIL_MAX_DD'
            
            # Check Profit Target
            if equity >= 1.10: return 'PASS'
            
        # Check Daily DD
        # Daily DD is calculated from Equity at Start of Day
        day_drawdown = (equity - daily_start_equity) / daily_start_equity
        if day_drawdown < -0.05: return 'FAIL_DAILY'
        
        daily_start_equity = equity
        
    return 'STUCK'

def main():
    print("="*60)
    print("FTMO PASS PROBABILITY SIMULATION")
    print("="*60)
    
    config = Config()
    config.risk_per_trade = 0.015 # 1.5% Conservative Risk for consistency
    # (Live is 3.3%, but allow user to see probabilities at 1.5% too?)
    # Let's use 2.0% as 'Standard'.
    config.risk_per_trade = 0.02
    
    print(f"Risk Per Trade: {config.risk_per_trade*100:.1f}%")
    
    # 1. Get Real Trade Data (Backtest)
    print("\n[BACKTEST] Generating Trade Distribution...")
    # Using Core 13 FX
    config.symbols = ["EURUSD=X", "USDJPY=X", "GBPUSD=X", "AUDUSD=X", "NZDUSD=X", 
                      "USDCAD=X", "USDCHF=X", "EURGBP=X", "EURJPY=X", "GBPJPY=X", 
                      "AUDJPY=X", "EURAUD=X", "EURCHF=X"]
                      
    loader = DataLoader(config)
    try:
        data = loader.load_data("2024-01-01", "2024-12-01")
    except Exception as e:
        print(e)
        return

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
        print("No trades generated.")
        return
        
    # Extract Trade PnL % (Account impact)
    # trade['PnL'] is checks out. We need % of Balance at that time?
    # Approximation: PnL / 100,000 (Initial). 
    # Or Risk % * R-Multiple.
    # Let's use raw PnL / Initial Balance for distribution shape.
    trade_returns = trades['PnL'] / config.initial_balance
    
    print(f"  Trades: {len(trade_returns)}")
    print(f"  Avg Win: {trade_returns[trade_returns>0].mean()*100:.2f}%")
    print(f"  Avg Loss: {trade_returns[trade_returns<0].mean()*100:.2f}%")
    print(f"  Win Rate: {(trade_returns>0).mean()*100:.1f}%")
    
    # 2. Monte Carlo
    print("\n[MONTE CARLO] Simulating 10,000 Accounts...")
    results = {'PASS': 0, 'FAIL_MAX_DD': 0, 'FAIL_DAILY': 0, 'STUCK': 0}
    
    N_SIMS = 10000
    for i in range(N_SIMS):
        res = run_simulation(trade_returns)
        results[res] += 1
        
    pass_rate = results['PASS'] / N_SIMS * 100
    fail_dd = results['FAIL_MAX_DD'] / N_SIMS * 100
    fail_daily = results['FAIL_DAILY'] / N_SIMS * 100
    stuck = results['STUCK'] / N_SIMS * 100
    
    print("\n" + "="*60)
    print(f"RESULTS (Risk {config.risk_per_trade*100:.1f}%)")
    print("="*60)
    print(f"✅ PASS RATE:       {pass_rate:.1f}%")
    print(f"❌ FAIL (Max DD):   {fail_dd:.1f}%")
    print(f"❌ FAIL (Daily DD): {fail_daily:.1f}%")
    print(f"⏳ STUCK (>2 Years): {stuck:.1f}%")
    
    prob_fail = fail_dd + fail_daily
    print("-" * 60)
    print(f"Certainty: {pass_rate / (pass_rate + prob_fail + 0.001) * 100:.1f}% of finished games are Wins.")

if __name__ == "__main__":
    main()
