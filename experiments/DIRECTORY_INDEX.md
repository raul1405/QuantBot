# Experiments Directory Structure

## Current Organization

```
experiments/
├── family_a_ml/         # ← ML Engine experiments (EXP001-008, A1, B1)
│   ├── run_*.py         # Experiment runners
│   └── exp_*.md         # Results & reports
│
├── cpi_khem_framework/  # ← Family B (CPI Macro Strategy)
│   ├── cpi_engine.py    # Core logic
│   ├── correlation_research.py  # Research module
│   ├── run_CPI_EXP*.py  # CPI experiments
│   └── *.csv            # CPI event data
│
├── archive_legacy/      # ← Old/deprecated scripts
│   ├── debug_split_test.py
│   ├── run_EXP008_SKETCH.py
│   └── research_v5_*.py
│
├── v3_ideas/            # ← Future strategy ideas
│
├── scoreboard.md        # ← Master experiment tracker
├── research_sandbox.py  # ← Main research engine
├── research_validation.py
└── comprehensive_validation.py
```

```
artifacts/
└── cpi_charts/          # ← All CPI experiment outputs
    ├── exp_CPI_001_*.png/md
    ├── exp_CPI_002_*.png/md
    ├── exp_CPI_003_*.png/md
    ├── exp_CPI_004_*.png/md
    └── exp_CPI_005_*.png/md
```

## Quick Reference

| Folder | Purpose | Status |
|--------|---------|--------|
| `family_a_ml/` | ML alpha experiments | ACTIVE |
| `cpi_khem_framework/` | CPI macro strategy | ACTIVE (Shadow Mode) |
| `archive_legacy/` | Old/unused scripts | ARCHIVED |
| `artifacts/cpi_charts/` | CPI research outputs | REFERENCE |
