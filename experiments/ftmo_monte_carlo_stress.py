
"""
FTMO MONTE CARLO (STRESSED)
===========================
Calculates passing probability under adverse conditions (Edge Decay).
Scenario A: 2024 Performance (Optimistic)
Scenario B: Win Rate -10%, Avg Win -10% (Pessimistic)
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
    equity = 1.0
    peak = 1.0
    daily_start_equity = 1.0
    
    # Limit 500 trading days (2 years)
    limit_days = 500
    
    bag = list(trades_pnl_pct)
    
    for day in range(limit_days):
        # Trade Frequency: Poisson(0.5)
        n_trades = np.random.poisson(0.5)
        
        for _ in range(n_trades):
            if not bag: break # Safety
            trade_ret = random.choice(bag)
            equity *= (1 + trade_ret)
            
            if equity > peak: peak = equity
            dd = (equity - peak) / peak
            
            # FTMO Constraints
            if dd < -0.10: return 'FAIL_MAX_DD'
            if equity >= 1.10: return 'PASS'
            
        daily_dd = (equity - daily_start_equity) / daily_start_equity
        if daily_dd < -0.05: return 'FAIL_DAILY'
        
        daily_start_equity = equity
        
    return 'STUCK'

def degrade_distribution(returns):
    """
    Simulate "Edge Decay":
    1. Reduce Avg Win magnitude by 10%.
    2. Turn 10% of Winners into Losers (Win Rate drop).
    """
    new_rets = []
    
    # 1. Reduce magnitude of winners
    returns = [r * 0.9 if r > 0 else r for r in returns]
    
    # 2. Rate Decay
    for r in returns:
        if r > 0:
            # 10% chance to flip to loss (slippage/whipsaw)
            if random.random() < 0.10:
                new_rets.append(-1 * abs(r)) # Flip sign
            else:
                new_rets.append(r)
        else:
            new_rets.append(r)
            
    return pd.Series(new_rets)

def main():
    print("="*60)
    print("FTMO STRESS TEST (Monte Carlo)")
    print("="*60)
    
    config = Config()
    config.risk_per_trade = 0.02 # 2%
    
    # Load Real Data (Core 13)
    # Exclude Toxic? No, use full Core 13 to be realistic about current state.
    config.symbols = ["EURUSD=X", "USDJPY=X", "GBPUSD=X", "AUDUSD=X", "NZDUSD=X", 
                      "USDCAD=X", "USDCHF=X", "EURGBP=X", "EURJPY=X", "GBPJPY=X", 
                      "AUDJPY=X", "EURAUD=X", "EURCHF=X"]
                      
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
    
    real_returns = trades['PnL'] / config.initial_balance
    stressed_returns = degrade_distribution(real_returns)
    
    print(f"Stats (Real)    | Count: {len(real_returns)} | Win Rate: {(real_returns>0).mean()*100:.1f}%")
    print(f"Stats (Stressed)| Count: {len(stressed_returns)} | Win Rate: {(stressed_returns>0).mean()*100:.1f}% (-Edge)")
    
    # Run
    N = 5000
    
    print("\n[SCENARIO A: OPTIMISTIC (2024 Conditions)]")
    res_a = {'PASS':0, 'FAIL':0, 'STUCK':0}
    for _ in range(N):
        r = run_simulation(real_returns)
        if 'FAIL' in r: res_a['FAIL'] += 1
        elif r == 'PASS': res_a['PASS'] += 1
        else: res_a['STUCK'] += 1
        
    print(f"  PASS: {res_a['PASS']/N*100:.1f}%")
    print(f"  FAIL: {res_a['FAIL']/N*100:.1f}%")
    print(f"  STUCK: {res_a['STUCK']/N*100:.1f}%")
    
    print("\n[SCENARIO B: PESSIMISTIC (Edge Decay)]")
    res_b = {'PASS':0, 'FAIL':0, 'STUCK':0}
    for _ in range(N):
        r = run_simulation(stressed_returns)
        if 'FAIL' in r: res_b['FAIL'] += 1
        elif r == 'PASS': res_b['PASS'] += 1
        else: res_b['STUCK'] += 1
        
    print(f"  PASS: {res_b['PASS']/N*100:.1f}%")
    print(f"  FAIL: {res_b['FAIL']/N*100:.1f}%")
    print(f"  STUCK: {res_b['STUCK']/N*100:.1f}%")
    
    print("\nVERDICT:")
    if res_b['PASS']/N > 0.60:
        print("✅ ROBUST. Even in bad conditions, you likely Pass.")
    else:
        print("⚠️ FRAGILE. If edge decays, you fail/stuck.")

if __name__ == "__main__":
    main()
