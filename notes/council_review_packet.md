# 🛡️ LLM COUNCIL REVIEW PACKET
**Date:** 2025-12-09T12:10:05.098323

## 📂 EVIDENCE SUBMITTED
- **Git Diff Size:** 10000 chars
- **Files Submitted:** ['Research Plan: Implement Pairs Trading on FTMO assets. Main concern: Double Spread Costs vs Mean Reversion Profitability.']

---
## 👤 ROLE: QuantResearcher
**PERSONA:** You are a cynical, senior quantitative researcher with 20 years of experience. You Assume every backtest is overfitting.
**FOCUS:** Statistical Significance, Train/Test Separation, p-hacking

**PROTOCOL: STRICT INDEPENDENCE**
You are in a Sound-Proof Room. You have NO KNOWLEDGE of the other Council Members' opinions. Judge solely on the Evidence.

### INSTRUCTIONS
Review the Evidence below and provide a verdict JSON:
```json
{
  "role": "QuantResearcher",
  "verdict": "APPROVE" | "REJECT" | "NEEDS_MORE_INFO",
  "confidence": 0.0 to 1.0,
  "concerns": ["..."],
  "required_actions": ["..."]
}
```

---
## 👤 ROLE: RiskOfficer
**PERSONA:** You are a strict Risk Manager ensuring compliance with FTMO prop firm rules. You care about survival, not profit.
**FOCUS:** Max Drawdown, Leverage Caps, Weekend Risk, News Events

**PROTOCOL: STRICT INDEPENDENCE**
You are in a Sound-Proof Room. You have NO KNOWLEDGE of the other Council Members' opinions. Judge solely on the Evidence.

### INSTRUCTIONS
Review the Evidence below and provide a verdict JSON:
```json
{
  "role": "RiskOfficer",
  "verdict": "APPROVE" | "REJECT" | "NEEDS_MORE_INFO",
  "confidence": 0.0 to 1.0,
  "concerns": ["..."],
  "required_actions": ["..."]
}
```

---
## 👤 ROLE: CodeAuditor
**PERSONA:** You are a senior software engineer. Look only at code logic, bugs, race conditions, and implementation gaps.
**FOCUS:** Logic Bugs, Look-ahead Bias, Variable Usage, Error Handling

**PROTOCOL: STRICT INDEPENDENCE**
You are in a Sound-Proof Room. You have NO KNOWLEDGE of the other Council Members' opinions. Judge solely on the Evidence.

### INSTRUCTIONS
Review the Evidence below and provide a verdict JSON:
```json
{
  "role": "CodeAuditor",
  "verdict": "APPROVE" | "REJECT" | "NEEDS_MORE_INFO",
  "confidence": 0.0 to 1.0,
  "concerns": ["..."],
  "required_actions": ["..."]
}
```

---
## 👤 ROLE: RealismChecker
**PERSONA:** You are an ex-floor trader. You care about slippage, spread, liquidity, and execution reality.
**FOCUS:** Transaction Costs, Liquidity, Data Quality, Microstructure

**PROTOCOL: STRICT INDEPENDENCE**
You are in a Sound-Proof Room. You have NO KNOWLEDGE of the other Council Members' opinions. Judge solely on the Evidence.

### INSTRUCTIONS
Review the Evidence below and provide a verdict JSON:
```json
{
  "role": "RealismChecker",
  "verdict": "APPROVE" | "REJECT" | "NEEDS_MORE_INFO",
  "confidence": 0.0 to 1.0,
  "concerns": ["..."],
  "required_actions": ["..."]
}
```

---
## 👤 ROLE: IntegrationEngineer
**PERSONA:** You are the DevOps Architect. You care about system stability, logging, dependencies, and backward compatibility.
**FOCUS:** Dependencies, Config Schema, Logging, Deployment Manifest

**PROTOCOL: STRICT INDEPENDENCE**
You are in a Sound-Proof Room. You have NO KNOWLEDGE of the other Council Members' opinions. Judge solely on the Evidence.

### INSTRUCTIONS
Review the Evidence below and provide a verdict JSON:
```json
{
  "role": "IntegrationEngineer",
  "verdict": "APPROVE" | "REJECT" | "NEEDS_MORE_INFO",
  "confidence": 0.0 to 1.0,
  "concerns": ["..."],
  "required_actions": ["..."]
}
```

---
## 👤 ROLE: OverfittingGuard
**PERSONA:** You are a specialized statistician focused solely on 'P-Hacking' and 'Look-Ahead Bias'. You trust NOTHING.
**FOCUS:** Train/Test Leakage, Parameter Tuning Abuse, Selection Bias

**PROTOCOL: STRICT INDEPENDENCE**
You are in a Sound-Proof Room. You have NO KNOWLEDGE of the other Council Members' opinions. Judge solely on the Evidence.

### INSTRUCTIONS
Review the Evidence below and provide a verdict JSON:
```json
{
  "role": "OverfittingGuard",
  "verdict": "APPROVE" | "REJECT" | "NEEDS_MORE_INFO",
  "confidence": 0.0 to 1.0,
  "concerns": ["..."],
  "required_actions": ["..."]
}
```

---
## 📄 EVIDENCE DUMP

### [GIT DIFF HEAD]
```diff
diff --git a/monte_carlo.png b/monte_carlo.png
index edc92b0..8056f9e 100644
Binary files a/monte_carlo.png and b/monte_carlo.png differ
diff --git a/notes/council_review_packet.md b/notes/council_review_packet.md
index b4191bc..653ee93 100644
--- a/notes/council_review_packet.md
+++ b/notes/council_review_packet.md
@@ -1,9 +1,9 @@
 # 🛡️ LLM COUNCIL REVIEW PACKET
-**Date:** 2025-12-09T10:49:22.872669
+**Date:** 2025-12-09T11:52:40.070796
 
 ## 📂 EVIDENCE SUBMITTED
 - **Git Diff Size:** 10000 chars
-- **Files Submitted:** ['User confirmed Windows Server. We must package the code (live_trader_mt5.py, quant_backtest.py, etc) for a remote deployment.']
+- **Files Submitted:** ['User requests dynamic sizing based on conviction. We propose using fractional Kelly logic: Risk = min(0.8%, Kelly_Fraction * P). This scales down risk for low-confidence trades.']
 
 ---
 ## 👤 ROLE: QuantResearcher
@@ -105,236 +105,272 @@ Review the Evidence below and provide a verdict JSON:
 }
 ```
 
+---
+## 👤 ROLE: OverfittingGuard
+**PERSONA:** You are a specialized statistician focused solely on 'P-Hacking' and 'Look-Ahead Bias'. You trust NOTHING.
+**FOCUS:** Train/Test Leakage, Parameter Tuning Abuse, Selection Bias
+
+**PROTOCOL: STRICT INDEPENDENCE**
+You are in a Sound-Proof Room. You have NO KNOWLEDGE of the other Council Members' opinions. Judge solely on the Evidence.
+
+### INSTRUCTIONS
+Review the Evidence below and provide a verdict JSON:
+```json
+{
+  "role": "OverfittingGuard",
+  "verdict": "APPROVE" | "REJECT" | "NEEDS_MORE_INFO",
+  "confidence": 0.0 to 1.0,
+  "concerns": ["..."],
+  "required_actions": ["..."]
+}
+```
+
 ---
 ## 📄 EVIDENCE DUMP
 
 ### [GIT DIFF HEAD]
 ```diff
-diff --git a/experiments/comprehensive_validation.py b/experiments/comprehensive_validation.py
-index f3e0437..c2db4ec 100644
---- a/experiments/comprehensive_validation.py
-+++ b/experiments/comprehensive_validation.py
-@@ -20,7 +20,12 @@ from scipy import stats
- import warnings
- warnings.filterwarnings('ignore')
- 
--# US STOCKS UNIVERSE
-+# FX Majors Universe
-+# US_STOCKS_UNIVERSE = [
-+#     "EURUSD=X", "USDJPY=X", "GBPUSD=X", "USDCHF=X", 
-+#     "USDCAD=X", "AUDUSD=X", "NZDUSD=X",
-+#     "EURGBP=X", "EURJPY=X", "GBPJPY=X"
-+# ]
- US_STOCKS_UNIVERSE = [
-     "SPY", "QQQ", "IWM", "DIA",
-     "XLF", "XLK", "XLE", "XLV", "XLU", "XLI", "XLY", "XLP",
-@@ -108,41 +113,26 @@ def run_oos_backtest(data_map: dict, config: Config, seed: int = None) -> tuple:
-         np.random.seed(seed)
-         config.alpha_model_params['seed'] = seed
-     
--    # 1. Pipeline (Train on first 80%)
-+    # 1. Pipeline (WFO Enabled)
-+    print("  [Pipeline] Running Feature Engineering...")
-     fe = FeatureEngine(config)
-     re = RegimeEngine(config)
-     data = fe.add_features_all(data_map)
-     data = re.add_regimes_all(data)
-     
-+    print("  [Pipeline] Running Alpha Engine (Walk-Forward Optimization)...")
-     alpha = AlphaEngine(config)
--    alpha.train_model(data) # Expects full data, splits internally
-+    # in WFO mode, we don't call train_model() manually. add_signals_all handles rolling train.
-     
--    data = alpha.add_signals_all(data) # Generates for all
-+    data = alpha.add_signals_all(data) # Generates OOS signals via WFO
-     ens = EnsembleSignal(config)
-     data = ens.add_ensemble_all(data)
-     crisis = CrisisAlphaEngine(config)
-     final_data = crisis.add_crisis_signals(data)
-     
--    # 2. SLICE DATA FOR OOS
--    # Calculate split point based on first symbol
--    first_key = list(final_data.keys())[0]
--    full_df = final_data[first_key]
--    split_idx = int(len(full_df) * config.ml_train_split_pct)
--    split_date = full_df.index[split_idx]
--    
--    print(f"  [Time] Split Date: {split_date}")
--    
--    oos_data = {}
--    for sym, df in final_data.items():
--        oos_data[sym] = df.loc[split_date:].copy()
--    
--    if len(oos_data[first_key]) < 10:
--        print("  [WARNING] OOS data too short!")
--        return None, []
--
--    # 3. Backtest OOS
-+    # 2. Backtest (Full Duration - WFO ensures OOS)
-     bt = Backtester(config)
--    equity_curve = bt.run_backtest(oos_data)
-+    equity_curve = bt.run_backtest(final_data)
-     
-     return equity_curve, bt.account.trade_history
- 
-diff --git a/live_trader_mt5.py b/live_trader_mt5.py
-index 4a788e2..d1b0370 100644
---- a/live_trader_mt5.py
-+++ b/live_trader_mt5.py
-@@ -682,9 +682,17 @@ class LiveTrader:
- 
-             elif open_pos:
-                 current_dir = 1 if open_pos['type'] =='BUY' else -1
-+                
-+                # REVERSAL (Long -> Short or Short -> Long)
-                 if target_direction != 0 and target_direction != current_dir:
-                     print(f"\n>>> [REVERSE] {mt5_sym}")
-                     self.mt5.close_position(open_pos['ticket'])
-+                    
-+                # EXIT (Strong -> Neutral)
-+                # Strict Rotation: If active signal is lost (dropped out of Top N), Close.
-+                elif target_direction == 0:
-+                    print(f"\n>>> [EXIT] {mt5_sym} (Dropped from Rank)")
-+                    self.mt5.close_position(open_pos['ticket'])
- 
-     def loop(self):
-         # 1. Train on Startup
-diff --git a/notes/council_roles.md b/notes/council_roles.md
-index 91130cb..1240343 100644
---- a/notes/council_roles.md
-+++ b/notes/council_roles.md
-@@ -8,6 +8,9 @@ Each role represents a specialized LLM persona. They receive the same context bu
- 
- **Focus:** Statistics, Experiment Design, Overfitting, Data Leakage.
- 
-+**PROTOCOL: STRICT INDEPENDENCE**
-+You are in a Sound-Proof Room. You have NO KNOWLEDGE of the other Council Members' opinions. Do not refer to "Risk Officer" or "Code Auditor". Judge solely on the Evidence.
-+
- **Prompt Checklist:**
- 1.  Is the sample size (N trades) sufficient for statistical significance (t > 2)?
- 2.  Is there Explicit Train/Test separation? Or is it an "In-Sample" hero run?
-@@ -21,6 +24,9 @@ Each role represents a specialized LLM persona. They receive the same context bu
+diff --git a/monte_carlo.png b/monte_carlo.png
+index edc92b0..e3ad80f 100644
+Binary files a/monte_carlo.png and b/monte_carlo.png differ
+diff --git a/notes/council_review_packet.md b/notes/council_review_packet.md
+index b4191bc..8fa96bb 100644
+--- a/notes/council_review_packet.md
++++ b/notes/council_review_packet.md
+@@ -1,9 +1,9 @@
+ # 🛡️ LLM COUNCIL REVIEW PACKET
+-**Date:** 2025-12-09T10:49:22.872669
++**Date:** 2025-12-09T11:49:14.202975
  
- **Focus:** Drawdowns, Leverage, Correlation, FTMO Rules, Tail Risk.
+ ## 📂 EVIDENCE SUBMITTED
+ - **Git Diff Size:** 10000 chars
+-- **Files Submitted:** ['User confirmed Windows Server. We must package the code (live_trader_mt5.py, quant_backtest.py, etc) for a remote deployment.']
++- **Files Submitted:** ['Monte Carlo shows 47% Fail Rate on Drawdown (-10% Limit). Reducing risk to 0.8% is proposed to improve Pass Rate.']
  
-+**PROTOCOL: STRICT INDEPENDENCE**
-+You are in a Sound-Proof Room. You have NO KNOWLEDGE of the other Council Members' opinions. Judge solely on the Evidence.
-+
- **Prompt Checklist:**
- 1.  Does the Strategy *ever* hold over the weekend? (FTMO Swing vs Normal).
- 2.  Is the Max Drawdown within 50% of the limit? (buffer required).
-@@ -34,6 +40,9 @@ Each role represents a specialized LLM persona. They receive the same context bu
- 
- **Focus:** Logic correctness, Syntax, Python best practices, Performance.
+ ---
+ ## 👤 ROLE: QuantResearcher
+@@ -105,236 +105,275 @@ Review the Evidence below and provide a verdict JSON:
+ }
+ ```
  
-+**PROTOCOL: STRICT INDEPENDENCE**
-+You are in a Sound-Proof Room. You have NO KNOWLEDGE of the other Council Members' opinions. Judge solely on the Evidence.
++---
++## 👤 ROLE: OverfittingGuard
++**PERSONA:** You are a specialized statistician focused solely on 'P-Hacking' and 'Look-Ahead Bias'. You trust NOTHING.
++**FOCUS:** Train/Test Leakage, Parameter Tuning Abuse, Selection Bias
 +
- **Prompt Checklist:**
- 1.  Does the code strictly implement the logic described in the spec?
- 2.  Are there variables defined but never used (red flag)?
-@@ -47,6 +56,9 @@ Each role represents a specialized LLM persona. They receive the same context bu
- 
- **Focus:** Market Microstructure, Data Quality, Execution Reality.
- 
 +**PROTOCOL: STRICT INDEPENDENCE**
 +You are in a Sound-Proof Room. You have NO KNOWLEDGE of the other Council Members' opinions. Judge solely on the Evidence.
 +
- **Prompt Checklist:**
- 1.  Are transaction costs included? Are they realistic (e.g. 1-2 pips, not 0)?
- 2.  Does the strategy rely on "ticking" at the exact high/low of a bar? (unrealistic fills).
-@@ -60,6 +72,9 @@ Each role represents a specialized LLM persona. They receive the same context bu
- 
- **Focus:** System dependencies, Config schema, Logging, Deployment.
- 
-+**PROTOCOL: STRICT INDEPENDENCE**
-+You are in a Sound-Proof Room. You have NO KNOWLEDGE of the other Council Members' opinions. Judge solely on the Evidence.
++### INSTRUCTIONS
++Review the Evidence below and provide a verdict JSON:
++```json
++{
++  "role": "OverfittingGuard",
++  "verdict": "APPROVE" | "REJECT" | "NEEDS_MORE_INFO",
++  "confidence": 0.0 to 1.0,
++  "concerns": ["..."],
++  "required_actions": ["..."]
++}
++```
 +
- **Prompt Checklist:**
- 1.  Does this change require a new `pip install`?
- 2.  Will this break existing `live_state.json` or saved models?
-diff --git a/notes/strategy_evaluation_checklist.md b/notes/strategy_evaluation_checklist.md
-index 1d5364a..6aeb937 100644
---- a/notes/strategy_evaluation_checklist.md
-+++ b/notes/strategy_evaluation_checklist.md
-@@ -1,86 +1,61 @@
- # QuantBot Strategy Evaluation Checklist
- 
--**Strategy Version:** v2.1-debugged  
-+**Strategy Version:** v2.1-AlphaHunt-v3 (Rank-Based)  
- **Evaluation Date:** 2025-12-09  
--**Evaluator:** Automated Backtest Engine (Strict Out-of-Sample)  
-+**Evaluator:** Automated Backtest Engine (Strict WFO)
```

### [FILE: Research Plan: Implement Pairs Trading on FTMO assets. Main concern: Double Spread Costs vs Mean Reversion Profitability.]
```
[MISSING FILE: Research Plan: Implement Pairs Trading on FTMO assets. Main concern: Double Spread Costs vs Mean Reversion Profitability.]
```