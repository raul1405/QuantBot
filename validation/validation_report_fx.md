# 🏛️ INSTITUTIONAL VALIDATION REPORT
**Strategy Analysis** | Init Balance: $100,000 | Model: FX Alpha V5

> **Note on Stats**: All metrics are estimates based on 2024 data (1h). Future performance may vary significantly by regime.

## 1. Trade Edge (Micro)
- **SQN Score**: `1.83` (Good, but not Holy Grail)
- **Expectancy**: `0.09 R` per trade
- **Win Rate**: `~67%`
- **Payoff Ratio**: `~0.8` (Scalper-like profile)

## 2. Account Health (Macro) (Rank 1 / 2% Total Risk)
- **Sharpe Ratio**: `0.98`
- **Max Drawdown**: `-1.98%` (Historical)
- **Total Return**: `+7.25%` (below FTMO target, but viable with scaling)

## 3. Statistical Proof (Null Hypothesis)
- **P-Value (Mean > 0)**: `0.012` (Significant)
- **Edge Presence**: Confirmed (Positive Drift)

## 4. Monte Carlo (FTMO Simulation)
- **Pass Rate (>10%)**: `23.5%` (Probabilistic shot, not a guarantee)
- **Tail DD (95% VaR)**: `-3.78%` (Safe buffer vs -10% limit)
- **Recommendation**: Acceptable odds for a high-variance challenge.

## 5. Exit Logic Audit (Rank 1 vs Diversification)
We tested "King of the Hill" (Rank 1) vs "Diversified" (Top 3) with normalized risk (2% total per setup).

| Metric | Rank 1 (Current) | Top 3 (Diversified) | Verdict |
| :--- | :--- | :--- | :--- |
| **Pass %** | **23.5%** | 0.2% | **Rank 1** dominates. |
| **Exp Return** | **+7.3%** | +1.1% | Alpha is scarce and concentrated in Rank 1. |
| **Tail DD (95%)** | **-3.78%** | -4.47% | Diversification added noise/risk here. |

**Conclusion**: Stick to **Rank 1** (Concentrated). This alpha engine is a "Sniper". Diversifying into Rank 2/3 dilutes edge and increases drawdown.

## 6. Council Guidelines
> Interpret all statistics as fragile estimates. 23.5% pass rate means ~1 in 4 attempts succeed, implying multiple months/retries may be needed. The strategy "improves odds", it does not "print money".