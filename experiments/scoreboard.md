# Experiment Scoreboard

| ID | Description | Mean R | Trades | Max DD (5% Tail) | Pass Rate (MC2) | Decision/Comment |
|:---|:------------|:-------|:-------|:-----------------|:----------------|:-----------------|
| **V1.0** | **Baseline Frozen Spec** | **0.0807** | **394** | **-6.50%** | **96.7%** | **FROZEN (Benchmark).** |
| A1-A | Target H=5, Thr=0.001 | 0.1461 | 684 | TBD | TBD | **Strong Quality Upgrade. R almost doubles.** |
| 004-C | v2 Tuning (H=5, T=0.0010) | 0.1204 | 577 | TBD | TBD | **🏆 Quality Winner. Best R, Safe.** |
| 005-B | v2.1 Flat Risk (No Vol Sizing) | 0.0785 | 543 | TBD | TBD | **Baseline (Flat Risk).** |
| 005-A | **v2.1 Full (Vol Sizing)** | **0.0761** | **560** | TBD | TBD | **🏆 Winner. Vol Sizing adds +35% Profit.** |
| **006-A** | **v2.1 (1bp Cost)** | **0.0785** | **543** | **$12,607** | **?** | **Validated Net Profit. Consistent.** |
| **006-B** | **v2.1 (2bp Cost)** | **0.0754** | **543** | **$12,170** | **?** | **PASSED. High Friction (-3.5% Profit).** |
| **007-B** | **v2.1 (Flash Crash)** | **TBD** | **232** | **$63,154** | **TBD** | **PASSED. Survived Black Swan with huge profit.** |
| **008-A** | **v2.1 (10k Margin Test)** | **--** | **124** | **$2,560** | **--** | **PASSED. No trades blocked at 1:30.** |
| **FT_001** | **Live/Paper Fwd Test (v2.1)** | **TBD** | **0** | **TBD** | **--** | **STARTED 2025-12-08. TARGET: 10% DD Limit.** |3x Trades.** |
| B1-A | Remove CrossSectional | 0.1801 | 380 | TBD | TBD | **Strong Quality Upgrade.** |
| B1-B | Remove ContinuousRegime | 0.2128 | 403 | TBD | TBD | **🚀 ALPHA UNLOCKED. Vol features confused the model.** |
| 003-v1 | Baseline (H=3/T=0.001/All) | 0.0807 | 394 | TBD | TBD | **Baseline Re-Run. Fails High-Vol Test (-0.01 R).** |
| 003-v2 | **Candidate v2.0 (H=5/T=0.0005/Lean)** | **0.0653** | **956** | TBD | TBD | **🏆 Winner. +60% Profit, Survives High Vol (+0.02 R).** |
