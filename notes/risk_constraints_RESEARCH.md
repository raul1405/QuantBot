# ⚖️ Risk & Exposure Constraints (Research)

**Status:** Draft / Research
**Context:** FTMO "Swing" Account Limits.

## 1. Global Licensing Limits (FTMO Rules)
*   **Max Daily Loss:** 5% (Soft Breach -> Hard Breach)
    *   *Agent Rule:* Hard Stop at **4.5%**.
*   **Max Total Loss:** 10%
    *   *Agent Rule:* Hard Stop at **9.0%**.
*   **News Trading:** Allowed (Swing Account).
*   **Holding Over Weekend:** Allowed (Swing Account).

## 2. Portfolio Exposure Limits (Design)
To prevent "Cluster Blowups" (e.g. Long EURUSD + Long EURJPY + Long EURNZD = 3x Short Euro Risk).

### A. Nominal Exposure Cap
*   **Current Logic:** `max_net_lots_usd`.
*   **Proposed Logic (Research):** Cap Total Notional Value as % of Equity.
    *   *Formula:* `Total_Notional_Value / Equity <= Max_Leverage_Used`
    *   *Target:* Max effective leverage of **1:10** (even if broker allows 1:30).
    *   *Why?* To survive a 10% Flash Crash without hitting the 10% Drawdown limit.

### B. Single Asset Cap
*   Max Risk per Trade: **0.30%** (Fixed in v2.1).
*   Max Open Positions per Symbol: **1** (No stacking/grid).

### C. "Worst Case" Margin Calculation
Before opening a trade, compute:
1.  Current Used Margin.
2.  New Trade Requirement (`Lots * ContractSize / Leverage`).
3.  **Constraint:** `Used_Margin + New_Trade_Margin < 0.80 * Equity`.
    *   (Leave 20% buffer for floating PnL swings to avoid Margin Call).

## 3. Derived Formulas

### Max Position Size (Risk Based)
$$ Lots_{Risk} = \frac{Equity \times 0.003}{SlDist \times TickValue} $$

### Max Position Size (Margin Based)
$$ Lots_{Margin} = \frac{FreeMargin \times 0.90 \times Leverage}{ContractSize} $$

### Final Size
$$ Lots_{Final} = \min(Lots_{Risk}, Lots_{Margin}, VolumeMax) $$
