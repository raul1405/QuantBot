# Research Reset: The "Null Hypothesis" Pivot

## 1. Post-Mortem: Why Families A & C Failed

Our previous alpha engines (**Family A: FX Intra-day** and **Family C: High-Vol**) have been scientifically falsified.

### The Failure Mode
The architecture relied on:
1.  **Technical Features**: RSI, ATR, Moving Averages, Volatility Regimes.
2.  **Model**: XGBoost (Tree-based Gradient Boosting).
3.  **Target**: Intra-day directional prediction (1h/Daily).

**The Reality Check**:
-   **FX (Family A)**: Initial success (+20%) was proven to be **100% Selection Bias** (picking winners ex-post) and **Execution Artifacts** (repainting). On a fixed universe with safe execution, the edge collapsed to noise (+0.7%).
-   **High-Vol (Family C)**: Lifting the same logic to Crypto/Indices failed even harder (-4% to -13%). The features tuned for mean-reverting FX did not generalize to trending/momentum assets.

**Conclusion**: *Standard technical analysis features, fed into standard ML models, have zero predictive edge in efficient markets (FX) or semi-efficient markets (Crypto Majors) in 2024.*

---

## 2. The Reset: "Null Hypothesis" Protocol

From this point forward, we operate under the **Null Hypothesis**:
> *"There is no edge. The market is efficient. I will lose spread+commission on every trade unless I can prove otherwise."*

We will no longer "sweep" for signals. We will test specific **Hypotheses**.

---

## 3. Proposed Research Vectors

We have two distinct paths for the next phase.

### Path A: Deep Markets (Hard Mode)
*Stay in Financial Markets, but dig deeper for Alpha.*

*   **Hypothesis**: Price history alone (OHLC) is mined out. Edge requires non-price information.
*   **New Directions**:
    *   **Microstructure/Order Flow**: Using tick data to predict imbalances (requires hft-like features).
    *   **Alternative Data**: Sentiment (News/Twitter), On-Chain (Crypto), Macro (Yields/Calendar).
    *   **Structural Arb**: Funding Rate arbitrage, Basis trading, Liquidity Provision (Market Neutral).
*   **Pros**: Infinite scale, professional skillset.
*   **Cons**: Extremely competitive, expensive data, high barrier to entry.

### Path B: Inefficient Domains (Soft Mode)
*Pivot the ML Infra to "Softer" Markets.*

*   **Hypothesis**: Financial markets are shark tanks. Sports betting or Poker markets are populated by amateurs and fans. The same ML rigor (WFO, Kelly Sizing, Risk Mgmt) applied there yields higher ROI.
*   **New Directions**:
    *   **Sports Betting (EPL/NBA)**: Predicting match outcomes/totals using team stats + betting liquidity.
    *   **Poker**: Solving decision trees or exploiting population tendencies.
*   **Pros**: Higher potential edge (ROI > 5%), less institutional competition.
*   **Cons**: Liquidity limits (can't scale to $100M), operational friction (betting accounts), moral/regulatory constraints.

---

## 4. Next Steps

**Decision Required**:
Do we commit to **Path A (Deep Science)** or **Path B (Domain Pivot)**?

*   If **Path A**: We scrap the old features and start a "Data Acquisition" sprint.
*   If **Path B**: We configure the engine to ingest Sports/Game data and run a viability scan on a new domain.

*Legacy Code (FX/Crypto) is archived in `experiments/` and will not be deployed.*
