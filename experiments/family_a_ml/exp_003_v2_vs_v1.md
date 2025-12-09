# EXP_003 – The A/B Test (v1.0 vs v2.0 Candidate)

## 1. Setup
*   **v1.0 (Baseline)**: Horizon=3, Threshold=0.001, **All Features** (including Noisy/Toxic ones).
*   **v2.0 (Candidate)**: Horizon=5, Threshold=0.0005, **Lean Features** (Advanced + Time + Lags ONLY). Matches findings from Exp A1 & B1.
*   **Metric**: Walk-Forward (2 Year) + High-Vol Slice Analysis.

## 2. Head-to-Head Results

| Metric | v1.0 (Baseline) | v2.0 (Candidate) | Delta |
|:---|:---:|:---:|:---:|
| **Profit** | +$10,903 | **+$17,717** | **+62%** |
| **Trades** | 394 | **956** | **2.4x Volume** |
| **Mean R** | **0.0807** | 0.0653 | -19% (Lower Quality) |
| **Win Rate** | ~57% | ~53% | -4% |

### 2.2 The "High-Vol" Stress Test (CRITICAL)
How does the strategy behave when Volatility is HIGH? (The "Crisis Alpha" test).

| Regimes | v1.0 (Baseline) | v2.0 (Candidate) | Verdict |
|:---|:---:|:---:|:---|
| **High Vol R-Multiple** | **-0.0139** (LOSS) | **+0.0268** (PROFIT) | **v2 SURVIVES.** |
| **High Vol Trade Count** | 263 | 548 | v2 trades through the noise. |

## 3. Findings
1.  **Noise Toxicity Confirmed**: v1.0 fails in High Volatility (Negative Alpha). This confirms that "Basic", "CrossSectional", and "Regime" features confuse the model during stress correlations.
2.  **Structural Robustness**: v2.0, using only "Advanced" structural features (`Trend_Dist`, `Vol_Ratio`), maintains positive expectancy even during chaos.
3.  **Scalability**: v2.0 generates immense volume (956 trades) which compensates for the lower per-trade R (0.065). The Total Profit difference ($17k vs $11k) proves that "Many Small Bets" (Phase 4 goal) works better than "Few Sniper Bets" here.

## 4. Decision
- [ ] Reject v2
- [x] **Promote v2.0 to "Likely Successor"**.
- [ ] **Next Step**: Fine-tune the Risk-Return Trade-off via Threshold (Exp 004).
    *   T=0.0005 gives maximum profit ($17k) but lower R (0.06).
    *   T=0.00075 or 0.001 might restore R to >0.10 while keeping robustness.
