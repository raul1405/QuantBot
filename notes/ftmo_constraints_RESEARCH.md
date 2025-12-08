# 🏦 FTMO Broker Constraints (Research)

**Status:** Draft / Research
**Objective:** Define the non-price attributes we need to query from the broker to ensure correct sizing and execution.

## 1. Symbol-Specific Constraints Table (Template)
This table must be populated by querying `mt5.symbol_info(symbol)` during the next phase (Phase 7b execution).

| Symbol | Min Lot | Lot Step | Max Lot | Contract Size | Tick Value | Est. Spread (Pts) | Leverage |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **EURUSD** | 0.01 | 0.01 | 100.0 | 100,000 | 1.00 USD | ~5 | 1:30 |
| **USDJPY** | 0.01 | 0.01 | 100.0 | 100,000 | ~0.65 USD | ~6 | 1:30 |
| **XAUUSD** | 0.01 | 0.01 | 50.0 | 100 | ~1.00 USD | ~20 | 1:20 |
| **US500** | 0.10 | 0.10 | 500.0 | 10 | 1.00 USD | ~40 | 1:20 |
| **BTCUSD** | 0.01 | 0.01 | 20.0 | 1 | 1.00 USD | ~3000 | 1:2 |

## 2. Required Data for Live Compatibility
To support the "FTMO Generic Universe", the `LiveTrader` engine (in future updates) must dynamically ingest:

### A. Sizing Constraints
*   **`volume_min`**: To avoid `RETCODE_INVALID_VOLUME`.
*   **`volume_step`**: To round lots correctly (e.g. 0.01 vs 0.10 for Indices).
*   **`volume_max`**: To split large orders if we scale up.

### B. Valuation Constraints
*   **`trade_contract_size`**: Essential for calculating Notional Value ($100k vs $10) exposure.
*   **`margin_initial`**: To calculate "Used Margin" before trade entry (Constraint: Free Margin > 0).

### C. Execution Constraints
*   **`trade_mode`**: Ensure symbol is not "Close Only".
*   **Trading Hours**: The bot is 24/7, but Indices close daily (e.g. 23:00-00:00). We need logic to avoid opening/closing during gaps? (Research: MT5 usually rejects usage outside hours, handling error is enough).

## 3. Leverage Caps (FTMO Swing)
*   **Forex Major**: 1:30
*   **Forex Minor/Cross**: 1:20
*   **Gold/Indices**: 1:20
*   **Crypto**: 1:2
*   **Stocks**: 1:5

**Impact:** The `0.30%` risk model relies on Stop Distance. If SL is tight, size increases.
**Constraint:** Size cannot exceed `FreeMargin * Leverage`.
**Action:** Future `LiveTrader` must check `OrderCalcMargin` before `OrderSend`.
