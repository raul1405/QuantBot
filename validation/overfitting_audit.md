# Audit Report: FX Alpha Framework Assessment

## Executive Summary
**Status**: **SUSPECT (High Overfitting Risk)**
**Verdict**: The strategy possesses a *genuine but modest* edge on major pairs (+3.6% baseline), but the claim of "20%+ Returns" is heavily driven by **Survivorship Bias** (excluding losing pairs) and likely **Regime Specificity**. The "Holy Grail" metrics are not robust.

## 1. Data & Feature Integrity ("Garbage In" Check)
- **Data Loading**: Passed. Properly aligned timestamps. No future lookahead in `load_data`.
- **Feature Calculation**:
    - **Methodology**: Passed. Features use strictly rolling/lagged windows.
    - **Anomaly**: `RSI` implementation is "Naive" (Simple Moving Average) rather than standard Wilder's (EMA). This is not a bug but a quirk.
    - **Target Generation**: Passed. `Future_Return` is correctly shifted and NOT leaked into input features `X`.

## 2. Walk-Forward Logic ("Crystal Ball" Check)
- **Training/Testing Separation**: Passed. strict separation between Train (180 days) and Test (30 days).
- **Leakage**: **No Logic Leakage Found**. The backtester executes at `Open(t+1)` based on Signal at `Close(t)`. This is conservative and correct.

## 3. Reliability & Robustness ("Lucky Monkey" Check)
### Experiment 1: Survivorship Bias (The "Toxic Pairs" Test)
> [!WARNING]
> This is the primary failure point.

When the 8 excluded "Toxic" pairs (e.g., AUDNZD, GBPCHF) are re-added to the portfolio:
- **Baseline (13 Selected Pairs)**: **+3.6% Return**, 55% Win Rate.
- **Full Universe (Selected + Toxic)**: **-4.6% Return**, 51% Win Rate.

**Implication**: The strategy does not generalize to the asset class (FX). It works *only* on the specific subset of pairs chosen *after* seeing the results. This is classic "Selection Bias".

### Experiment 2: Random Signals
- **Random Entry**: **-1.8% Return**.
- **Implication**: The risk management and cost model correctly penalize random trading (via spread/commission). The Baseline's positive performance *is* statistically distinguishable from noise, confirming a non-zero alpha on Majors.

## 4. Execution Consistency ("Production Gap" Check)
### Discrepancy Found: Repainting Risk
- **Backtest**: Executes at `Open[t+1]` using features from `Close[t]`. (Safe/Lagged)
- **Live Trader**: Executes on `Current Bar` (Index -1).
    - If `Index -1` refers to the *forming* bar, features like `Close` (current price) and `S_Alpha` will fluctuate.
    - **Risk**: The live bot trades "mid-candle". If the signal disappears by close, the backtest would NOT have taken the trade, but the Live Bot DID.
    - This breaks the "Identical Logic" assumption and makes live trading *riskier* than the backtest.

## 5. Cost Model
- **Findings**: The backtest assumes a spread of ~1.0 pip ($10/lot) for majors.
- **Reality**: Institutional spreads are lower (0-0.5 pips), but commissions add up. The 1.0 pip assumption is a reasonable/conservative proxy. The positive baseline result is robust to costs.

## Recommendations
1.  **Quantify the Edge Correctly**: Stop claiming "20% Return". The realistic edge is ~5-10% annualized unleveraged (or ~30% with 1:30 leverage *if* drawdown limits aren't hit).
2.  **Fix Live Execution**: Change `live_trader_mt5.py` to execute only on **Bar Close** (or check `time` to ensure new bar start) to match Backtest.
3.  **Generalization**: The model fails on choppy crosses (AUDNZD). Do not deploy on them. Acknowledge this limitation in the Investment Policy.
4.  **Standardize Indicators**: Switch to Wilder's RSI if standard behavior is expected.

## Conclusion
The **FX Alpha v5** is **NOT** a scam, but it is **Over-Optimized**. The "Toxic Pair Removal" is the smoking gun for curve-fitting. The strategy is tradeable on Majors, but expect significantly higher variance and lower returns than the "Marketing" backtest suggests.
