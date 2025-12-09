# CPI-Only Inflation Product Framework (Family B)

> [!IMPORTANT]
> **Sandbox Only**: This framework lives in `experiments/cpi_khem_framework/`.
> It MUST NOT interact with or modify the existing ML Alpha Engine (Family A).

## 1. Core Logic & Notation

### Triggers & Timeline
-   **Event ($t_m$)**: The CPI Release Date/Time.
-   **Horizon ($h$)**: Holding period in days (e.g., 1, 3, 5, 10).
-   **Acceleration Trigger ($A_m$)**:
    -   We track last 3 CPI prints.
    -   If 2 out of 3 show "Acceleration" (Actual > Forecast OR Actual > Previous), $A_m \ge 2$.
    -   **Action**: If $A_m \ge 2$, we enter the trade.

### Assets
-   **Equity Basket ($E_t$)**: Proxy via `US500` (SPX) or `US30`.
-   **Inflation Factor ($F_t$)**: Proxy via `Gold` (XAUUSD) or Energy (XLE).
    -   *Simplification*: We will use **Gold** as our Inflation Proxy for FTMO.

### Payoff Structure (Convexity)
We aim to replicate a **Call Option** on Inflation Volatility or Direction.
For a "CPI Overlay", we want to be Long Volatility or Long Trend if Inflation accelerates.

**Client Payoff ($\pi_m$)**:
$$ \pi_m = \max(0, R_{t_m \to t_m+h} - K) $$
Where $K$ is a volatility-adapted strike:
$$ K = \sigma_{realized} \times \alpha $$

**Global Cap**:
PnL is capped at $C_{glob}$ to prevent "Black Swan" blowups against the Issuer (us).

## 2. Implementation Architecture

### `cpi_engine.py`
-   **Class `CPIWindow`**:
    -   State machine: `WAITING` -> `ACTIVE` -> `EXPIRED`.
    -   Manages triggers and PnL accrual.
-   **Class `CPIConfig`**:
    -   Separate from `quant_backtest.Config`.
    -   Stores `horizon`, `trigger_threshold`, `risk_cap`.

### Data Requirements
-   `cpi_events.csv`: Date, Forecast, Previous, Actual.
-   Price Data: `US500`, `XAUUSD` loaded via `DataLoader` (Shared utility, but Read-Only).

## 3. Risk Management (Issuer Side)
-   **Inventory Limit**: Max N open CPI windows.
-   **VaR Limit**: If VaR(95) > Limit, reject new windows.
-   **Sizing**: Fixed Dollar Amount (e.g., \$5,000 per event) or Kelly logic (separate from Family A).

## 4. Success Metrics
-   **Hit Rate**: % of Triggers that yield positive return.
-   **Convexity**: Does PnL scale non-linearly with Volatility?
-   **Correlation**: Is it uncorrelated with Family A (Momentum)?
