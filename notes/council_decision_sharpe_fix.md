# Council Deliberation: Sharpe Error Fixes

## 📋 Request Summary
**Request ID:** `fix_sharpe_2024_001`  
**Change Type:** `CODE_REFACTOR`  
**Proposal:** Fix transaction costs and Sharpe annualization in backtest engine

---

## Evidence Presented

### Diagnostic Results (2025-12-09)
| Issue | Current Value | Correct Value | Impact |
|-------|---------------|---------------|--------|
| Transaction Cost | $0.0005/trade | $20/trade | 40,000x underestimation |
| Sharpe K-factor | 1764 | 2699 | 1.24x inflation |
| True Sharpe (trade-based) | N/A | 0.48 | Reality check |

### Proposed Fixes
1. **Fix A:** Transaction costs = `notional × spread × 2` (round-trip)
2. **Fix B:** Calculate K dynamically from actual bars/day
3. **Fix C:** Separate Sharpe by asset class (FX vs Equity)

---

## 🧑‍🔬 ROLE 1: Quant Researcher (The Skeptic)

**Verdict:** ✅ **APPROVE**  
**Confidence:** 0.95

### Checklist Results
| Question | Answer | Pass |
|----------|--------|------|
| Sample size sufficient? | 813 trades | ✅ Yes |
| Train/Test separation? | WFO implemented | ✅ Yes |
| Is this alpha or beta? | Need to verify post-fix | ⚠️ TBD |
| P-hacking risk? | Fix doesn't tune params | ✅ Yes |

### Concerns
- After fixing costs, strategy may show **negative** expected value
- FX component already shows -$539 loss on only 6 trades

### Required Actions
1. **Fix transaction costs FIRST** - this is the ROOT CAUSE
2. Re-run backtest to get true performance before any other changes
3. If strategy is unprofitable, abandon - don't try to "fix" it

---

## 🛡️ ROLE 2: Risk Officer (The Constable)

**Verdict:** ✅ **APPROVE**  
**Confidence:** 0.90

### Checklist Results
| Question | Answer | Pass |
|----------|--------|------|
| Weekend holding? | Strategy allows it (Swing) | ✅ Yes |
| Max DD within limit? | TBD after fix | ⚠️ TBD |
| Leverage < 30? | Config shows 30x | ✅ Yes |
| News event handling? | Not addressed here | ⚠️ N/A |

### Concerns
- With 40,000x cost increase, drawdowns will be MUCH worse
- Current $7,238 profit becomes potentially **negative** after $8,000 costs

### Required Actions
1. Fix costs first, then re-evaluate if strategy survives
2. Do NOT deploy to live until realistic costs are validated

---

## 💻 ROLE 3: Code Auditor (The Compiler)

**Verdict:** ✅ **APPROVE**  
**Confidence:** 0.85

### Checklist Results
| Question | Answer | Pass |
|----------|--------|------|
| Code matches spec? | Cost formula is WRONG | ❌ No |
| Variables unused? | N/A | ✅ N/A |
| Look-ahead bias? | Not in cost calc | ✅ Yes |
| Index alignment? | Verified in diagnostic | ✅ Yes |

### Concerns
- Current: `cost = size * 0.0005` - **units are nonsensical**
- Size is in lots (0.01-1.0), cost should be in dollars proportional to notional

### Required Actions
```python
# FIX - Replace line quant_backtest.py:1323
# OLD: cost = pos.size * self.config.transaction_cost
# NEW: 
pip_value = 10 if '=X' in pos.symbol else 1  # FX vs other
spread_pips = 1.0  # Conservative 1 pip spread
cost = pos.size * pip_value * spread_pips * 2  # Entry + Exit
```

---

## 🎯 ROLE 4: Realism Checker (The Trader)

**Verdict:** ✅ **APPROVE** (with strong endorsement)  
**Confidence:** 0.98

### Checklist Results
| Question | Answer | Pass |
|----------|--------|------|
| Costs included? | **NO - This is the bug** | ❌ No |
| Exact high/low fills? | Uses Close price | ✅ Yes |
| Overlapping sessions? | Hour-based features present | ✅ Yes |
| Liquid assets? | FX majors/crosses | ✅ Yes |

### Concerns
- This is EXACTLY my domain. The cost model is broken.
- $0.0005 per trade is **10,000x less** than a single pip
- Real FTMO accounts pay ~0.5-2 pips spread!

### Required Actions
1. **URGENT:** Fix costs immediately - all other analysis is meaningless without this
2. For FTMO realism, use:
   - FX Majors: 0.5-1.0 pip spread
   - FX Crosses: 1.0-2.0 pip spread
   - Indices: 0.5-1.0 point spread

---

## 🔧 ROLE 5: Integration Engineer (The Architect)

**Verdict:** ✅ **APPROVE**  
**Confidence:** 0.92

### Checklist Results
| Question | Answer | Pass |
|----------|--------|------|
| New pip install? | No | ✅ Yes |
| Breaks live_state.json? | No | ✅ Yes |
| Logging sufficient? | Costs should be logged | ⚠️ Add |
| Conflicts with frozen spec? | Yes - spec should be unfrozen | ⚠️ Risk |

### Concerns
- `quant_backtest.py` is marked "FROZEN FOR FT_001"
- But a fundamental cost bug MUST be fixed

### Required Actions
1. Log cost calculation for debugging
2. Update frozen spec comment to acknowledge bug fix
3. Increment version to v2.2

---

## 📊 COUNCIL DECISION

| Role | Verdict | Confidence |
|------|---------|------------|
| Quant Researcher | APPROVE | 0.95 |
| Risk Officer | APPROVE | 0.90 |
| Code Auditor | APPROVE | 0.85 |
| Realism Checker | APPROVE | 0.98 |
| Integration Engineer | APPROVE | 0.92 |

### **FINAL OUTCOME: ✅ APPROVE**

### Axis Scores
| Axis | Score | Notes |
|------|-------|-------|
| Assumptions | 9/10 | Clear evidence of bug |
| Applicability | 10/10 | Directly fixes core issue |
| Code Correctness | 9/10 | Simple fix, low risk |
| Realistic/Newness | 10/10 | Makes backtest realistic |
| Overfitting | 8/10 | Doesn't change strategy logic |
| Implications | 7/10 | May reveal unprofitable strategy |

### Implementation Order (Council Consensus)
1. **FIRST:** Fix transaction costs in `quant_backtest.py`
2. **SECOND:** Run diagnostic to verify fix applied
3. **THIRD:** Fix Sharpe K-factor calculation (separate by asset class)
4. **FOURTH:** Re-run full backtest with realistic params
5. **FIFTH:** Evaluate if strategy is still viable

### Blocking Issues
- None - all roles approve

### Risk Warning
> ⚠️ After fixing costs, the strategy may show **negative expected value**. Be prepared for this outcome. If unprofitable, do NOT try to "tune" parameters to make it profitable - that's overfitting.
