# FTMO Swing Challenge - Optimized Strategy Configuration

## Challenge Parameters
- **Profit Target**: 10% ($10,000 on $100k)
- **Max Drawdown**: 10% ($10,000)
- **Max Daily DD**: 5% ($5,000)
- **Time Limit**: Unlimited (Swing)
- **Leverage**: 1:30 Forex, 1:20 Gold/Indices

---

## Current Performance Analysis

| Metric | Family A (ML) | Family B (Macro) | Combined 70/30 |
|--------|---------------|------------------|----------------|
| Avg Daily PnL | $19.72 | ~$3 | $14.80 |
| Best 30-Day | $7,367 | N/A | ~$5,500 |
| Worst Daily DD | -1.49% | TBD | ~-1.2% |
| Max DD | -11.17% | -3.4% | -7.86% |

## Problem Statement

The current strategy averages ~0.02%/day but FTMO Swing (with unlimited time) only requires patience:
- At $19.72/day: ~507 trading days to hit 10% (≈2 years)
- At $50/day: ~200 trading days to hit 10% (≈10 months)
- At $100/day: ~100 trading days to hit 10% (≈5 months)

## Recommended Configuration

### Primary Strategy: 100% Family A (ML Engine)
**Rationale**: Proven alpha, higher daily PnL, sufficient for FTMO timeline.

### Shadow Mode: Family B (Macro Proxies)
**Purpose**: Monitor correlation, validate expansion thesis, no live trades yet.

### Risk Scaling Strategy
If behind target at checkpoint:
- **Month 1**: Use 1x risk ($300/trade)
- **Month 3**: If <3% profit, scale to 1.5x risk ($450/trade)
- **Month 6**: If <5% profit, scale to 2x risk ($600/trade)

### Safety Buffers
| Constraint | FTMO Limit | Our Target | Buffer |
|------------|------------|------------|--------|
| Max DD | 10% | 7% | 3% |
| Daily DD | 5% | 3% | 2% |
| Profit | 10% | 10% | 0% |

---

## Implementation Checklist

- [x] Family A: Continue live trading (100% allocation)
- [ ] Family B: Add shadow mode logging to live trader
- [ ] Risk scaling: Add checkpoint-based risk adjustment
- [ ] Monitoring: Daily DD and Max DD alerts
- [ ] Time tracking: Days to target projection

---

## Council Verdict

> **"For FTMO Swing with unlimited time, prioritize NOT BUSTING over speed.
> Keep 100% in Family A (proven). Monitor B in shadow. Scale risk only if behind."**
