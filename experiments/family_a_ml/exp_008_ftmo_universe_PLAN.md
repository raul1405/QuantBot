# 🧪 Experiment 008: FTMO Universe Impact

**Status:** Plan / Research
**Objective:** Quantify the "loss of edge" when moving from our ideal 30-asset Global Universe (Yahoo) to the restricted FTMO-only Universe.

## 1. Hypothesis
Restricting the universe will **lower Sharpe Ratio** and **Frequency** because we lose high-alpha assets (e.g. Natural Gas, certain crosses) or diversified uncorrelated assets.
*   **Null Hypothesis:** Perf Change < -10%.
*   **Alternative Hypothesis:** Perf Change >= -10% (Robust edge).

## 2. Configuration (v2.1 Baseline)
*   **Horizon:** 5 Hours
*   **Threshold:** 0.001 (0.1%)
*   **Sizing:** Volatility-Adjusted Risk (0.30% Base)
*   **Features:** Lean Set (v2.1)

## 3. The Universes
| Population | Description | Count |
| :--- | :--- | :--- |
| **A (Control)** | Full Yahoo Universe (Inc. NATGAS, obscure pairs). | 30 |
| **B (Treatment)** | FTMO-Generic Universe (Majors + Gold + Indices + BTC). | ~18 |

## 4. Metrics & comparison
| Metric | Full Universe (Baseline) | FTMO Universe (Exp) | Delta (%) |
| :--- | :--- | :--- | :--- |
| **Total Return** | | | |
| **Sharpe Ratio** | | | |
| **Max Drawdown** | | | |
| **Trade Count** | | | |
| **Win Rate** | | | |
| **Avg Trade R** | | | |

## 5. Execution Plan
1.  Create `run_EXP008.py` (inheriting from `quant_backtest`).
2.  Define `FTMO_UNIVERSE = ['EURUSD=X', 'GBPUSD=X', 'ES=F', ...]` manually in the script.
3.  Run Backtest B (Treatment).
4.  Compare against cached Baseline A.
5.  If performance drops > 20%, investigate: **Are we missing a specific asset class (e.g. Energy)?**

## 6. Constraints
DO NOT execute until Approved. This is a paper plan.
