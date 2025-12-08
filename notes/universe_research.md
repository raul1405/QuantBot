# 🌌 Universe & Broker Research

**Status:** Draft / Research
**Context:** Preparing to streamline the 30-asset global universe into an FTMO-compatible subset for execution.

## 1. Current Global Universe (30 Assets)
This is the full universe currently used in `quant_backtest.py` and `live_trader_mt5.py` (via Yahoo Finance tickers).

| Internal / Yahoo Ticker | Instrument | Sector | Notes |
| :--- | :--- | :--- | :--- |
| **EURUSD=X** | EUR/USD | Forex | Major |
| **USDJPY=X** | USD/JPY | Forex | Major |
| **GBPUSD=X** | GBP/USD | Forex | Major |
| **USDCHF=X** | USD/CHF | Forex | Major |
| **USDCAD=X** | USD/CAD | Forex | Major |
| **AUDUSD=X** | AUD/USD | Forex | Major |
| **NZDUSD=X** | NZD/USD | Forex | Major |
| **EURGBP=X** | EUR/GBP | Forex | Cross |
| **EURJPY=X** | EUR/JPY | Forex | Cross |
| **GBPJPY=X** | GBP/JPY | Forex | Cross |
| **AUDJPY=X** | AUD/JPY | Forex | Cross |
| **EURAUD=X** | EUR/AUD | Forex | Cross |
| **EURCHF=X** | EUR/CHF | Forex | Cross |
| **AUDNZD=X** | AUD/NZD | Forex | Cross |
| **AUDCAD=X** | AUD/CAD | Forex | Cross |
| **CADJPY=X** | CAD/JPY | Forex | Cross |
| **NZDJPY=X** | NZD/JPY | Forex | Cross |
| **GBPCHF=X** | GBP/CHF | Forex | Cross |
| **GBPAUD=X** | GBP/AUD | Forex | Cross |
| **GBPCAD=X** | GBP/CAD | Forex | Cross |
| **EURNZD=X** | EUR/NZD | Forex | Cross |
| **ES=F** | S&P 500 Futures | Index | Highly Liquid |
| **YM=F** | Dow Jones Futures | Index | Highly Liquid |
| **NQ=F** | Nasdaq 100 Futures | Index | Highly Liquid |
| **RTY=F** | Russell 2000 Futures | Index | Small Cap |
| **GC=F** | Gold Futures | Metal | Safe Haven |
| **CL=F** | Crude Oil (WTI) | Energy | Volatile |
| **NG=F** | Natural Gas | Energy | Extremely Volatile |
| **BTC-USD** | Bitcoin | Crypto | 24/7 |
| **ETH-USD** | Ethereum | Crypto | 24/7 |

## 2. FTMO_Generic_Universe_Draft (Target Subset)
A proposed lean universe optimized for FTMO Swing constraints (low fees, high liquidity, standard spreads).

| Underlying | Draft MT5/CFD Symbol | Category | Priority (1-3) | Rationale |
| :--- | :--- | :--- | :--- | :--- |
| **EUR/USD** | `EURUSD` | FX | 1 | Lowest spread, high liquidity. |
| **GBP/USD** | `GBPUSD` | FX | 1 | Good volatility/spread ratio. |
| **USD/JPY** | `USDJPY` | FX | 1 | Critical Asian session pair. |
| **USD/CHF** | `USDCHF` | FX | 2 | Hedge pair. |
| **USD/CAD** | `USDCAD` | FX | 2 | Commodity correlation (Oil). |
| **AUD/USD** | `AUDUSD` | FX | 2 | Proxy for China/Risk. |
| **NZD/USD** | `NZDUSD` | FX | 2 | Kiwi correlation. |
| **EUR/JPY** | `EURJPY` | FX | 2 | High volatility cross. |
| **GBP/JPY** | `GBPJPY` | FX | 2 | "The Beast" - High Alpha potential. |
| **Gold** | `XAUUSD` | Metal | 1 | Essential specific-risk diversifier. |
| **Silver** | `XAGUSD` | Metal | 3 | Correlated to Gold, optional. |
| **S&P 500** | `US500` / `US500.cash` | Index | 1 | Global equity benchmark. |
| **Nasdaq** | `US100` / `US100.cash` | Index | 1 | Tech/Growth factor. |
| **Dow Jones** | `US30` / `US30.cash` | Index | 2 | Old economy factor. |
| **DAX** | `GER40` / `DE40` | Index | 2 | European session beta. |
| **FTSE 100** | `UK100` | Index | 3 | UK exposure, optional. |
| **Bitcoin** | `BTCUSD` | Crypto | 1 | Uncorrelated asset class. |
| **Ethereum** | `ETHUSD` | Crypto | 2 | Alt-beta. |

**Selection Criteria:**
1.  **Spread:** Must be tight (< 2 pips avg for FX, < 1.0 index points).
2.  **Liquidity:** Must accept reasonably large lot sizes without massive slippage.
3.  **Correlation:** Reduced redundant crosses (e.g. `EURAUD`, `GBPCAD`) to simplify risk model.
