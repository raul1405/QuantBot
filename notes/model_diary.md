# Model Diary

## 2025-12-08
*   **Frozen Spec v1.0**: Baseline established.
    *   **Core**: XGBoost Alpha, Universal Model.
    *   **Risk**: 0.30% per trade, 1.5 ATR SL, 2.0 ATR TP.
    *   **Filters**: Anti-Bull Trend, Low-Confidence Block (<0.10).
    *   **Regime**: Continuous Volatility Sizing (`1 / (1 + z^2)`).
    *   **Performance**: 96.7% Pass Rate, -6.5% Max DD in diagnostics.
