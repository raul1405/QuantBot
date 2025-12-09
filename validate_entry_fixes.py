"""Validate entry quality fixes."""
import sys
sys.path.insert(0, '.')
from quant_backtest import Config, DataLoader, FeatureEngine, RegimeEngine, AlphaEngine, EnsembleSignal, CrisisAlphaEngine, Backtester
import pandas as pd
import numpy as np

print('=' * 60)
print('ENTRY QUALITY FIX VALIDATION')
print('Changes: 1) Block HIGH vol entries, 2) SL widened to 2.0x ATR')
print('=' * 60)

config = Config()
config.symbols = config.symbols[:10]

loader = DataLoader(config)
data = loader.load_data('2024-01-01', '2024-12-01')

fe = FeatureEngine(config)
re = RegimeEngine(config)
ae = AlphaEngine(config)
es = EnsembleSignal(config)
ce = CrisisAlphaEngine(config)

data = fe.add_features_all(data)
data = re.add_regimes_all(data)
ae.train_model(data)
data = ae.add_signals_all(data)
data = es.add_ensemble_all(data)
data = ce.add_crisis_signals(data)

bt = Backtester(config)
bt.run_backtest(data)

trades = pd.DataFrame(bt.account.trade_history)

if not trades.empty:
    print()
    print('=' * 60)
    print('RESULTS COMPARISON')
    print('=' * 60)
    
    print()
    print('BY EXIT REASON:')
    print('-' * 40)
    for reason in trades['Reason'].unique():
        subset = trades[trades['Reason'] == reason]
        wr = (subset['PnL'] > 0).mean() * 100
        mean_r = subset['R_Multiple'].mean()
        pct = len(subset) / len(trades) * 100
        icon = 'OK' if mean_r > 0 else 'BAD'
        print(f'  {reason:<12}: n={len(subset):>3} ({pct:>4.0f}%), WR={wr:>5.1f}%, Mean R={mean_r:>+.3f} {icon}')
    
    print('-' * 40)
    total_wr = (trades['PnL'] > 0).mean() * 100
    total_r = trades['R_Multiple'].mean()
    print(f'  TOTAL        : n={len(trades):>3}, WR={total_wr:>5.1f}%, Mean R={total_r:>+.3f}')
    
    sharpe = trades['PnL'].mean() / trades['PnL'].std() * np.sqrt(252) if trades['PnL'].std() > 0 else 0
    
    # SL rate calculation
    sl_rate = len(trades[trades['Reason']=='SL']) / len(trades) * 100
    chandelier_rate = len(trades[trades['Reason']=='Chandelier']) / len(trades) * 100 if 'Chandelier' in trades['Reason'].values else 0
    
    print()
    print('=' * 60)
    print('BEFORE vs AFTER FIXES')
    print('=' * 60)
    print()
    print(f'  Metric          |  BEFORE  |  AFTER   ')
    print(f'  ----------------|----------|----------')
    print(f'  SL Rate         |  60%     |  {sl_rate:.0f}%     ')
    print(f'  Chandelier Rate |  N/A     |  {chandelier_rate:.0f}%     ')
    print(f'  Win Rate        |  36.7%   |  {total_wr:.1f}%   ')
    print(f'  Mean R          |  0.012   |  {total_r:.3f}  ')
    print(f'  Trade Sharpe    |  ~0.0    |  {sharpe:.2f}   ')
    print(f'  Total Trades    |  813     |  {len(trades)}     ')
    print()
    
    # Verdict
    if total_r > 0.05 and sl_rate < 55:
        print('VERDICT: IMPROVEMENT - Continue with these settings')
    elif total_r > 0.01:
        print('VERDICT: MARGINAL - May need more tuning')
    else:
        print('VERDICT: NO IMPROVEMENT - Consider reverting')
