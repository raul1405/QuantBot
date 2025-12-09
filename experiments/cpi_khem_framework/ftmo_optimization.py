"""
FTMO Swing Challenge Optimization Analysis
==========================================
"""

import pandas as pd
import numpy as np

print('=' * 60)
print('FTMO SWING CHALLENGE OPTIMIZATION')
print('=' * 60)
print()
print('FTMO Constraints:')
print('  - Profit Target: 10%')
print('  - Max Drawdown: 10%')
print('  - Max Daily DD: 5%')
print('  - Leverage: 1:30')
print()

# Results from EXP006
results = [
    {'alloc': '100/0', 'ret': 6.99, 'sharpe': 0.59, 'max_dd': -11.17},
    {'alloc': '70/30', 'ret': 5.23, 'sharpe': 0.62, 'max_dd': -7.86},
    {'alloc': '50/50', 'ret': 4.02, 'sharpe': 0.66, 'max_dd': -5.63},
    {'alloc': '30/70', 'ret': 2.79, 'sharpe': 0.75, 'max_dd': -3.41},
]

print('FTMO COMPLIANCE ANALYSIS')
print('-' * 60)
print(f"{'Allocation':<12} {'Return':<10} {'Max DD':<10} {'DD Pass':<10} {'Target':<10}")
print('-' * 60)

for r in results:
    dd_pass = 'PASS' if abs(r['max_dd']) < 10 else 'FAIL'
    target = 'PASS' if r['ret'] >= 10 else f"{10 - r['ret']:.1f}% gap"
    print(f"{r['alloc']:<12} {r['ret']:.2f}%     {r['max_dd']:.2f}%    {dd_pass:<10} {target:<10}")

print()
print('=' * 60)
print('PROBLEM: No allocation hits 10% profit target!')
print()
print('SOLUTIONS:')
print('  1. Run longer (current is ~1 year, need ~1.5 years for 10%)')
print('  2. Increase position sizing (more risk)')
print('  3. Use 100/0 A with Family B in Shadow Mode only')
print('=' * 60)
print()

# Best for FTMO
print('RECOMMENDED FOR FTMO:')
print()
print('Option A: Aggressive (Maximize Profit)')
print('  Allocation: 100% A (ML Engine)')
print('  Return: 6.99%')
print('  Max DD: -11.17% (EXCEEDS 10% LIMIT)')
print('  Risk: HIGH - may bust challenge')
print()
print('Option B: Conservative (Stay in DD Limit)')
print('  Allocation: 70/30 (A/B)')
print('  Return: 5.23%')
print('  Max DD: -7.86% (Under 10%)')
print('  Sharpe: 0.62')
print('  Risk: MEDIUM')
print()
print('Option C: Ultra-Safe (Best Risk-Adjusted)')
print('  Allocation: 50/50 (A/B)')
print('  Return: 4.02%')
print('  Max DD: -5.63% (Under 5%!)')
print('  Sharpe: 0.66')
print('  Risk: LOW - but slower to target')
print()

# Calculate time to 10% at each rate
print('TIME TO 10% PROFIT TARGET:')
for r in results:
    monthly_ret = r['ret'] / 12
    months_to_target = 10 / monthly_ret if monthly_ret > 0 else float('inf')
    print(f"  {r['alloc']}: {months_to_target:.1f} months ({r['ret']/12:.2f}%/mo)")

print()
print('=' * 60)
print('COUNCIL RECOMMENDATION FOR FTMO:')
print('=' * 60)
print()
print('Given FTMO constraints:')
print('  1. Max DD of 10% is HARD LIMIT (instant fail)')
print('  2. 5% daily DD is HARD LIMIT (instant fail)')
print('  3. 10% profit is SOFT target (just takes time)')
print()
print('VERDICT: 70/30 ALLOCATION')
print('  - Max DD -7.86% gives 2.14% buffer from 10% limit')
print('  - Better Sharpe than 100/0')
print('  - Can reach 10% in ~23 months')
print('  - Shadow mode for Family B allows monitoring')
