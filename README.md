# QuantBot v2.1 (FTMO Challenge)

## Overview
This is a professional-grade quantitative trading system designed for the FTMO Swing Challenge.
It features:
- **Architecture**: Market-Neutral Statistical Arbitrage.
- **Risk Engine**: Continuous Volatility Sizing (Kelly Criterion).
- **Governance**: Automated Kill-Switches for Daily DD (-4.5%) and Max DD (-9.5%).

## Deployment (Windows VPS)
See `notes/vps_checklist.md` for step-by-step deployment instructions.

## Quick Start (Shadow Mode)
1.  Install dependencies: `pip install -r requirements.txt`
2.  Configure secrets in `.env`.
3.  Run: `python live_trader_mt5.py`.

## Status
- Strategy: **Frozen v2.1**
- Deployment: **Ready**
