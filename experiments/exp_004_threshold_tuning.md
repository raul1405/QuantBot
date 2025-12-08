# EXP_004: Threshold Tuning (v2 spec)

| Threshold | Mean R | Trades | Profit | Comment |
|---|---|---|---|---|
| 0.0005 | 0.0653 | 956 | $17,717 | High Volume, Low R (0.06). "Churn & burn". |
| 0.00075 | 0.1006 | 735 | $19,199 | **Highest Profit.** Good balance. |
| **0.0010** | **0.1204** | **577** | **$18,127** | **Best Quality (R > 0.12). Safer.** |

## 2. Findings
1.  **Quality vs Quantity**: Raising the threshold from 0.0005 to 0.001 reduced trade volume by ~40% (956 -> 577), but **Doubled** the trade quality (Mean R 0.06 -> 0.12).
2.  **Profit Paradox**: Usually, higher volume leads to higher profit via compounding. Here, T=0.001 actually made *more money* ($18.1k) than T=0.0005 ($17.7k) because the trades were so much better.
3.  **T=0.00075**: Technically the highest profit ($19.2k), but R=0.10 is lower than T=0.001.

## 3. Decision
- [ ] T=0.0005 (Growth)
- [ ] T=0.00075 (Max Profit)
- [x] **T=0.0010 (Quality Champion)**
    *   **Rationale**: In prop trading (FTMO), **Drawdown Protection** is king. A system with Mean R=0.12 is far less likely to hit a drawdown limit than one with R=0.06, even if they make similar money.
    *   **Verdict**: v2.1 Configuration = **H=5 / T=0.001 / Lean Features**.

