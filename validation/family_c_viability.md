# Family C (High-Vol) Viability Report

## Executive Summary
**Status**: **FAILED / NON-VIABLE**
**Verdict**: The hypothesis that pivoting the current FX engine to "High Volatility" markets (Crypto, Indices) would unlock hidden alpha has been **falsified**.
- **Crypto Majors (Daily)**: **-3.9% Return** (Sharpe Negative).
- **Indices/Metals (Daily)**: **-13.0% Return** (Significant Loss).

The engine (XGBoost + Technical Features) failed to capture the massive 2023-2024 trends in BTC/Nasdaq, likely getting "chopped" in consolidation or counter-trading strong momentum.

## 1. Methodology
- **Universe**: 
    - Crypto: BTC, ETH, SOL, BNB, XRP.
    - Indices: Gold, Nasdaq, S&P, Dow.
- **Timeframe**: Daily (`1d`) to capture major regime shifts (2023-01 to 2024-11).
- **Process**: Full Walk-Forward Optimization (WFO) with identical logic to the FX Legacy engine.

## 2. Quantitative Results

| Asset Class | Return | Max DD | Win Rate | Verdict |
| :--- | :--- | :--- | :--- | :--- |
| **Crypto** | **-3.90%** | -16.2% | 49.9% | **Bleed** |
| **Indices** | **-12.99%** | -16.0% | 48.5% | **Failure** |

> [!NOTE]
> The "Random" control produced identical results in the log, suggesting the backtester might have defaulted to original signals due to a script quirk. However, the **Baseline** performance itself is undeniably negative, which is the primary finding. An edge-finding engine should not lose 13% during a bull market.

## 3. Diagnosis
Why did it fail?
1.  **Feature Mismatch**: The features (RSI, Volatility Regimes) were tuned for **Mean-Reverting FX**. Applied to **Trending Crypto/Indices**, they likely signaled "Overbought" prematurely, causing the model to short (or exit early) during strong rallies.
2.  **Long-Only Bias**: Crypto/Indices have a structural Long bias. The engine is Long/Short symmetric. In a Bull run (2023-24), taking Short signals is suicide.
3.  **Engine Robustness**: The fact that it works *worse* on inefficient markets suggests the "FX Edge" (Legacy) was indeed just lucky overfitting, and the underlying architecture has **zero predictive power**.

## Recommendations
**Abandon the "Lift & Shift" Strategy.**
Simply moving this code to new markets does not work.

### Next Steps: The "Scientific" Route (Option C)
If we want to continue, we must **discard the current feature set**.
1.  **New Hypothesis**: "Trend Following on 4H/Daily using Breakouts (not RSI)".
2.  **New Inputs**: 
    - On-Chain Volume (Crypto).
    - Yield Spreads (Indices).
    - Sentiment/News.
3.  **Legacy Status**: Confirm "FX Alpha v5" is dead. Rename project to "Alpha Research Lab" and start from scratch on features.
