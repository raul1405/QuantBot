# EXP_005: Alpha vs Risk Separation (v2.1)

## 1. Setup
*   **Config A (Proposed v2.1)**: Lean Alpha Features + Continuous Vol Sizing in Risk Engine.
*   **Config B (Flat Risk)**: Lean Alpha Features + Fixed Risk (No Vol Sizing).
*   **Goal**: Prove that separating Alpha (Signal) from Risk (Sizing) is superior.

## 2. Results (2-Year Walk-Forward)

| Metric | A: v2.1 (Vol Sizing) | B: v2.1 (Flat Risk) | Delta |
|:---|:---:|:---:|:---:|
| **Profit** | **$17,046** | $12,607 | **+$4,439 (+35%)** |
| **Mean R** | 0.076 | 0.078 | Neutral |
| **Max DD** | 9.5% | 9.8% | **Slightly Safer (A)** |
| **Trades** | 560 | 543 | Similar |

## 3. Verdict
*   **Vol Sizing Works**: By sizing UP in low-vol regimes and DOWN in high-vol regimes, we squeezed out **35% more profit** while actually *reducing* the max drawdown slightly.
*   **Separation Validated**: We can successfully strip Vol features from the Alpha model (to prevent overfitting) while still using them for Sizing.

## 4. Decision
- [x] **Adop Config A (Vol Sizing)** as the Frozen v2.1 Spec.
- [ ] Revert to Flat Risk.
