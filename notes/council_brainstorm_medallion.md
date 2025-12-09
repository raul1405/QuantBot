# 🧠 Council Session: "The Medallion Roadmap"

**Topic:** Improving Edge & Statistical Robustness (Alpha Hunt v3)
**Goal:** Identifying the path from "Prop Firm Bot" to "Institutional Alpha".

---

## 🗣️ The Debate (Minutes)

### 1. 👤 Quant Researcher (The Math)
> "You are mimicking the *form* of Medallion (ML, ranking) but missing the *substance*. RenTec doesn't just rank price returns. They predict the *residual*."

*   **Critique:** Our features (`Z_Score`, `RSI`, `Slope`) are **linear** and **commoditized**. Everyone has them.
*   **The Medallion Way:**
    *   **Signal Orthogonality:** Don't just predict "Up/Down". Predict "Tech vs Energy" (Sector Neutrality) or "Move vs SPY" (Beta Neutrality).
    *   **Target transformation:** Predict the *Innovation* (the noise), not the Trend.
*   **Proposal:**
    *   **Residual Modeling:** Train target `(Return - Beta * Market_Return)`. Isolate the idiosyncratic alpha.
    *   **Frequency:** Medallion trades noise. We trade hourly bars. We are too slow for "Arbitrage" but too fast for "Macro". We are in the "Zone of Pain" (High noise, high cost).

### 2. 👤 Overfitting Guard (The Skeptic)
> "If you torture the data long enough, it will confess. v3 'works' because we got lucky with the Rank-1 parameter."

*   **Critique:** We found v3 by trial and error (v1->v2->v3). This is **p-hacking**.
*   **The Medallion Way:**
    *   **Single Model:** RenTec famously uses ONE giant model for everything. We have one model per asset (via `add_signals_all` iterating). Wait, we do train a single global model? *Check code.* Yes, `train_model` trains on the concatenated dataset. That is actually good (Medallion-style).
*   **Proposal:**
    *   **Deflated Sharpe Test:** Adjust our Sharpe (0.12) for the number of trials we ran (4). Real Sharpe is probably ~0.05.
    *   **Feature Purge:** Delete half the features. If `Vol_Intensity` doesn't work on Gold, it shouldn't be in the model for EURUSD.

### 3. 👤 Realism Checker (The Floor Trader)
> "Jim Simons didn't pay spread. You do."

*   **Critique:** We pay 5bps (spread + comms). Medallion has negative costs (rebates).
*   **The Medallion Way:**
    *   **Execution Alpha:** They don't take market orders. They *make* markets. We are taking liquidity.
    *   **Capacity:** Our edge is thin. If we scale to $10M, we move the market.
*   **Proposal:**
    *   **Limit Orders:** Stop using Market/IOC. Place Limit orders inside the spread? (Hard with Python/MT5 latency).
    *   **Cost Filter:** Only trade if `Expected_Return > 2.5 * Cost`. Currently we trade if `Prob > 0.5`.

### 4. 👤 Hacker (The Engineer)
> "Python is slow. MT5 is an abstraction. We are trading through a straw."

*   **Critique:** `live_trader.py` sleeps for 15s. The market moves in milliseconds.
*   **The Medallion Way:**
    *   **Colocation:** Their servers are next to the exchange. Ours are on... a Windows VPS?
    *   **Data:** They use Order Book (L3) data. We use Candles (L1).
*   **Proposal:**
    *   We can't be HFT. We must be **MFT (Medium Frequency)**.
    *   **Alternative Data:** We need inputs that *others don't have*. (e.g., scraping FTMO sentiment? News feed sentiment?). Using only Price/Vol is a crowded trade.

---

## 🚀 The Consensus Roadmap

The Council agrees: **Alpha Hunt v3 is "Good Enough" for a retail Prop Challenge, but "Toy Grade" compared to Medallion.**

### Immediate Action Plan (The "Poor Man's Medallion")
1.  **Feature Orthogonality:**
    *   Modify `FeatureEngine`: Add **Sector/Market Relative** features.
    *   *Idea:* `Dist_from_SPY_Correlation`. Trade when asset *de-couples* from the market.
2.  **Exectution Optimization:**
    *   Modify `LiveTrader`: Implement **Passive Entry**. Place limit orders 1 pip better than BBO? (Risk: Missed fills).
3.  **Data Expansion (The Hardest Part):**
    *   We are blind to the Order Book. We cannot see "Pressure".
    *   *Action:* Can we get **Tick Data**? (MT5 has `copy_ticks`). We could aggregate Tick Flow (Buy Vol vs Sell Vol) into 1H bars ourselves to get a "Delta" signal.

### The Vote
*   **Quant:** APPROVE (Feature work).
*   **Hacker:** APPROVE (Tick data aggregation).
*   **Risk:** CAUTION (Don't break the current profitable loop).

**Recommendation:** Let v3 run (Baseline). Fork to `research_v5_medallion.py` to test **Tick-Flow Features**.
