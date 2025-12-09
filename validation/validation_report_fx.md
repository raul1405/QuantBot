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
## 7. Optimal Risk Audit (Exp_RISK)
We swept risk from 1% to 6% to find the FTMO sweet spot.

| Risk | Return | Pass % | Tail DD (95%) |
| :--- | :--- | :--- | :--- |
| 1.0% | +2.7% | 0.0% | -2.76% |
| 2.0% | +7.2% | 23.4% | -3.88% |
| 3.0% | +12.1% | 64.7% | -4.63% |
| **5.0%** | **+17.4%** | **83.1%** | **-6.29%** |
| 6.0% | +20.7% | 90.8% | -6.17% |

**Key Insight**: **Leverage Saturation**.
At 5-6% risk, the "Soft Cap" (`Total Notional < 6x Equity`) acts as a dynamic circuit breaker. It prevents Drawdown from scaling linearly with Risk.
- *Naive Projection*: 2% -> 5% implies DD -3.9% -> -9.7% (Danger Zone).
- *Actual Reality*: 5% Risk had Tail DD of only -6.3% because the cap blocked marginal trades during high exposure.

**Decision**: Run at **5.0% Risk** for Live Trading.