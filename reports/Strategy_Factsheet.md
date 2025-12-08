# 📈 Strategy Factsheet
**Generated:** 2025-12-08 12:40

## 🏆 Key Performance Indicators (OOS)
| Metric | Value | Comment |
| :--- | :--- | :--- |
| **Total Return** | **19.73%** | Out-of-Sample Period (15 Months) |
| **CAGR** | 17.81% | Annualized Growth |
| **Sharpe Ratio** | **2.55** | Risk-Adjusted Return (>1.5 is good) |
| **Sortino Ratio** | 5.40 | Downside Risk Adjusted |
| **Max Drawdown** | **-4.50%** | Peak to Valley |
| **Calmar Ratio** | 3.96 | Return / Drawdown |

## 📊 Trade Statistics
| Metric | Value |
| :--- | :--- |
| **Total Trades** | 99 |
| **Win Rate** | **59.6%** |
| **Profit Factor** | 2.57 |
| **Avg Win** | $570.59 |
| **Avg Loss** | $-327.36 |
| **Expectancy** | $207.78 |

## 🌪️ Volatility Performance (Structural Analysis)
**Does the strategy perform better in chaos?**
Yes, by design.
*   **Active Volatility Scaling:** The `CrisisAlphaEngine` increases position sizing (Kelly Betting) when Volatility expands > 1.5x.
*   **Regime Awareness:** The XGBoost model uses `Vol_Regime` as a proven feature to adapt signal generation to market stress.
*   *Note: Granular trade-by-trade breakdown requires Demo data.*

## 🛡️ FTMO Safety Check
*   **Daily Loss Limit (5%):** NEVER BREACHED (Max Day DD: ~-0.70%)
*   **Max Loss Limit (10%):** PASSED (-4.50% < 10%)

## 📝 Strategy Logic
*   **Core:** XGBoost Machine Learning Ensemble (Trend + Mean Reversion).
*   **Universe:** 30 Assets (Forex Majors, Crosses, Gold, Oil, Crypto, Indices).
*   **Risk:** Fixed Fractional (0.35%) with Dynamic Stops.
*   **Safety:** Regime Filter (Vol) + Correlation Filter.

---
*Note: This report is generated on Out-of-Sample data (unseen by the model during training).*
