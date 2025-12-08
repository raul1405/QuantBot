# Deployment Manifest: Forward Test v2.1

**System Version**: v2.1 Frozen
**Date**: 2025-12-08
**Status**: READY FOR FORWARD TEST

---

## 1. Critical Artifacts
The following files define your trading operation. Do not edit them manually.

*   **[Frozen Strategy Spec v2.1](frozen_specs/frozen_v2_spec.md)**
    *   The "Bible". H=5, T=0.001, Lean Features, Vol Sizing.
*   **[Forward Testing Protocol](../forward_testing_protocol.md)**
    *   The "Rules of Engagement". Trading hours, Kill Switches, Pass/Fail criteria.
*   **[Deployment Monitor](monitor_forward_test.py)**
    *   Run this daily to track your Equity Curve vs Kill Switches.

## 2. Updated Codebase
*   **`quant_backtest.py`**:
    *   **Status**: FROZEN.
    *   **Config**: Synced to v2.1.
    *   **Logic**: Vol Sizing Enabled. Transaction Cost added (1bp).
*   **`live_trader_mt5.py`**:
    *   **Status**: ACTIVE.
    *   **Updates**: 
        *   Added `TradeLogger` (logs to `live_logs/FT_001_trades.csv`).
        *   Added State Persistence (tracks risk context across restarts).
        *   Added FTMO Daily Loss Kill-Switch (-4.5%).

## 3. Stress Test Results (Exp 006)
We stress-tested the system with **Double Transaction Costs (2bp)**.
*   **Baseline (1bp)**: $12,607 Net Profit.
*   **Stress (2bp)**: $12,170 Net Profit (-3.5%).
*   **Conclusion**: **PASSED**. The strategy edge is robust to slippage and high friction.

## 3b. Margin & Leverage (Exp 008)
We stress-tested v2.1 on **$10k, $25k, $100k** accounts at **1:30 Leverage**.
*   **Trades Blocked**: 0.
*   **Safety logic**:
    *   **Max Used Margin**: 30% of Equity.
    *   **Max Notional (USD)**: 400% of Equity.
*   **Conclusion**: **PASSED**. Safe for small accounts.

## 4. Instructions

### Phase A: Shadow Mode (1 Week)
**Goal**: Verify connection & logs (No real orders).
1.  **Check Config**: `live_config.json` should have `"mode": "shadow"`.
2.  **Run**: `python live_trader_mt5.py`.
3.  **Verify**: Log file `live_logs/FT_001_shadow.csv` should grow. Console should say `[SHADOW] Would Open...`.
4.  **Monitor**: `python monitor_forward_test.py`.

### Phase B: Paper Campaign (FT_001)
**Goal**: 90 Days / 300 Trades.
1.  **Switch Mode**: Edit `live_config.json` -> `"mode": "paper"`.
2.  **Daily Run**: 
    *   Start `live_trader_mt5.py` (London Open).
    *   Stop `live_trader_mt5.py` (US Close).
    *   Run `python monitor_forward_test.py`.
3.  **Governance**:
    *   **DO NOT TOUCH** `quant_backtest.py`.
    *   **DO NOT TOUCH** `live_trader_mt5.py`.
    *   New ideas -> `experiments/v3_ideas/`.

**Good Luck.**
