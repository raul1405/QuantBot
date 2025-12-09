# Council Chokepoints (Triggers)

This document defines the specific events in the development lifecycle that **MUST** trigger a Council Review.

## 1. Spec Freeze (`FREEZE_SPEC`)

**Trigger:** Creating or Modifying a file in `frozen_specs/` (e.g., `v3_strategy.md`).
**Goal:** Ensure we don't commit to a strategy design that is fundamentally flawed or unimplementable.

**Input Requirements:**
*   **The Spec File:** The full markdown content of the new spec.
*   **Proof of Concept Data:** Backtest results justifying the spec (if available).
*   **Code Diff:** (Optional) If prototype code exists.

**Key Question:** "Is this strategy specification theoretically sound, compliant with FTMO rules, and implementable given our data constraints?"

## 2. Config Change (`UPDATE_CONFIG`)

**Trigger:** Modifying default values in `Config` class in `quant_backtest.py` or `live_config.json`.
**Goal:** Prevent silent risk drift (e.g., accidentally increasing leverage, removing stop-losses, changing symbols).

**Input Requirements:**
*   **Diff:** Shows exactly what parameter changed (e.g., `risk_per_trade: 0.003 -> 0.005`).
*   **Impact Analysis:** A text summary of why this change is needed.

**Key Question:** "Does this configuration change increase risk of ruin, violate correlation limits, or introduce untested parameters?"

## 3. Deployment Manifest Update (`UPDATE_MANIFEST`)

**Trigger:** Modifying `deployment_manifest.md`.
**Goal:** The manifest controls what is actually running in production. This is the highest-stakes file.

**Input Requirements:**
*   **The Manifest:** Full content.
*   **Reference Artifacts:** Links to the "Passed" backtests/specs referenced in the manifest.

**Key Question:** "Is the version of code/strategy being promoted to Live actually the one that passed validation? Are all checksums/hashes correct?"

## 4. Declaration of Victory (`DECLARE_PASS`)

**Trigger:** Use of the keyword "FTMO PASS" or "VALIDATED" in a research report or `scoreboard.md`.
**Goal:** Prevent premature celebration based on limited or biased data.

**Input Requirements:**
*   **Comprehensive Validation Report:** (e.g., `notes/strategy_evaluation_checklist.md`).
*   **OOS Verification:** Proof of strict Train/Test separation.
*   **Monte Carlo Results.**

**Key Question:** "Is this result statistically significant, strictly Out-of-Sample, and robust to stress testing? Or is it a fluke?"

## 5. Major Refactor (`CODE_REFACTOR`)

**Trigger:** Significant changes (>50 lines) to core engines (`quant_backtest.py`, `live_trader_mt5.py`).
**Goal:** Ensure refactoring doesn't introduce regression bugs or change logic inadvertantly.

**Input Requirements:**
*   **Full `git diff`.**
*   **Test Cases:** Output of unit tests or regression backtests.

**Key Question:** "Does the new code implement the EXACT same logic as the old code (if pure refactor), or if logic changed, is it intentional and documented?"
