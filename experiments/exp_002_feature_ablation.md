# EXP_002 – Feature Ablation (The "Noise" Hunt)

## 1. Hypothesis
Our feature vector has grown large (>30 features).
Hypothesis: Many features (especially Basic and Cross-Sectional) might be adding noise, causing the XGBoost model to overfit. Removing "dead weight" should improve OOS performance.

## 2. Change Implemented
- **Method**: Group-wise Ablation (Remove one group at a time).
- **Groups**:
  - `Basic`: Z_Score, ATR, Momentum, Volatility
  - `Advanced`: Vol_Ratio, Trend_Dist, RangeNorm, Asset_DD_200, Regimes
  - `Lags`: Ret_Lag1, Ret_Lag2, Ret_Lag3
  - `CrossSectional`: Rank features (Mom, Vol)
  - `TimeMeta`: Hour, Session
  - `ContinuousRegime`: Vol_Intensity, Vol_Pct
- **Base Config**: v1.0 Baseline (H=3, T=0.001).

## 3. Test Battery Results

| Removed Group | Mean R | Trades | Profit | Verdict |
|:---|---:|---:|---:|:---|
| **(Baseline / None)** | 0.1098 | 418 | +$15,213 | Reference. |
| **Remove Advanced** | **0.0761** | 387 | +$9,744 | **CRITICAL LOSS.** This group contains the Alpha. |
| **Remove TimeMeta** | 0.0877 | 406 | +$11,796 | Loss. Time features are valuable. |
| Remove Lags | 0.1306 | 383 | +$12,953 | Slight R improvement, Profit drop. Neural. |
| **Remove Basic** | **0.1509** | 340 | +$11,650 | **Quality Win.** 'Basic' features dilute signal. |
| **Remove CrossSec** | **0.1801** | 380 | +$15,491 | **Strong Win.** R jumps from 0.11 -> 0.18. |
| **Remove ContRegime** | **0.2128** | 403 | **+$24,932** | **🚀 MONSTER WIN.** Alpha Logic is confused by Vol features. |

### 3.2 Key Findings
1.  **"Less is More"**: The best features are the **Advanced** ones (`Vol_Ratio`, `Trend_Dist`).
2.  **Toxic Features**:
    *   **Continuous Regime Features** (`Vol_Intensity`, `Vol_Pct`) are excellent for *Risk Management* (Sizing), but terrible for *Alpha Prediction*. Removing them from the model doubled the R-multiple.
    *   **Cross-Sectional Features** appear to be pure noise in this setup.
    *   **Basic Features** (raw RSI/ATR) also degrade performance.

## 4. Comparison vs Baseline (Frozen v1)
*   **Baseline**: R 0.11, Profit $15k.
*   **Result (Remove Regimes)**: R 0.21, Profit $25k. **Delta: +65% Profit, +100% Quality.**
*   **Conclusion**: v2.0 should essentially strip out Basic, CrossSectional, and Regime features from the *Model*, keeping Regimes only for the *Risk Engine*.

## 5. Decision
- [ ] Reject change
- [ ] Needs more testing
- [x] **Candidate for next frozen spec (v2.0)**:
    *   **Alpha Model**: Use ONLY `Advanced` + `TimeMeta` + `Lags`.
    *   **Risk Engine**: Continue using `ContinuousRegime` for sizing.
