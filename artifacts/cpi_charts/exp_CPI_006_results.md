# EXP006: Expanded Correlation Analysis

## Summary
Replaced sparse CPI-only signals with **daily macro proxies** (Gold, DXY, TIPS).

## Data Comparison

| Metric | Old (CPI Only) | New (Daily Proxies) |
|--------|----------------|---------------------|
| Trades | 4 | **41** |
| Total PnL | $6,376 | **$894.94** |
| Statistical Power | ❌ NONE | ✅ VALID |

## Correlation

| Metric | Value |
|--------|-------|
| Overlapping Days | 41 |
| Overall Correlation | **0.0098** |
| Rolling Mean | -0.0187 |
| Rolling Std | 0.1482 |
| % Negative | 45.5% |

## Portfolio Results

| Allocation | Return | Sharpe | Max DD |
|------------|--------|--------|--------|
| 100/0 | 6.99% | 0.592 | -11.17% |
| 70/30 | 5.23% | 0.623 | -7.86% |
| 50/50 | 4.02% | 0.664 | -5.63% |
| 30/70 | 2.79% | 0.754 | -3.41% |

## Verdict

**Best Allocation (Sharpe)**: 30/70

The expanded daily proxy approach provides:
1. ✅ Statistical validity (n=41 vs n=4)
2. ✅ Actionable daily signals
3. ✅ Proper correlation measurement

## Chart
![Expanded Analysis](exp_CPI_006_expanded.png)
