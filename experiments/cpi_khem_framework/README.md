# Family B: CPI Inflation Framework

This directory contains the **Event-Driven CPI Convexity Overlay**.
It is a "Sandbox" for researching inflation-linked strategies.

## ⚠️ Guardrails
1.  This code is **isolated** from the main ML Alpha Engine (Family A).
2.  It does NOT affect `quant_backtest.py` or `live_trader_mt5.py`.
3.  Backtests are run via `run_CPI_EXP001.py`.

## Directory Structure
-   `cpi_spec.md`: Mathematical & Logical Specification.
-   `cpi_engine.py`: The logic Core (CPIWindow, Trigger, Payoff).
-   `cpi_events.csv`: Historical CPI Data (Mock or Real).
-   `run_CPI_EXP001.py`: First Experiment Driver.

## Goal
To determine if a "Macro Overlay" focusing *only* on CPI events provides **uncorrelated alpha** to our primary Momentum/ML strategies.
