# Chandelier Exit - Technical Explanation

## What Is It?

The **Chandelier Exit** is a volatility-based trailing stop developed by Charles Le Beau. It "hangs" from the highest high (for longs) or lowest low (for shorts) like a chandelier from a ceiling.

## The Formula

### For LONG positions:
```
Chandelier_Stop = Highest_High(N) - (ATR × Multiplier)
```

### For SHORT positions:
```
Chandelier_Stop = Lowest_Low(N) + (ATR × Multiplier)
```

Where:
- **N** = Lookback period (typically 22 bars = 1 trading month)
- **ATR** = Average True Range (volatility measure)
- **Multiplier** = Typically 3.0 (3× ATR from extreme)

## Visual Example (Long Trade)

```
Price
 ▲
 │     *** Highest High (ceiling)
 │    *   *
 │   *     *
 │  *       *  ← Price action
 │ *         *
 │*           *
 │─────────────────── Chandelier Stop (ceiling - 3×ATR)
 │
 │  Entry ●
 └────────────────────────────────▶ Time
```

## Why It Works

1. **Captures Trend Extremes**: Trails from the highest point, not entry
2. **Volatility Adaptive**: Uses ATR, so wider stops in volatile markets
3. **Never Tightens Prematurely**: Stop only moves in profit direction
4. **Lets Winners Run**: 3× ATR gives enough room for normal fluctuations

## Comparison to Other Exits

| Exit Type | How It Trails | Problem |
|-----------|---------------|---------|
| Fixed TP | Doesn't trail | Misses big moves |
| ATR Trail | From current price | Too tight, stopped by noise |
| Chandelier | From highest high | Captures full swing |

## Parameters We're Using

| Parameter | Value | Reason |
|-----------|-------|--------|
| Lookback (N) | 22 bars | ~1 day in H1 timeframe |
| Multiplier | 3.0 | Standard, proven robust |
| Max Hold | 30 bars | Safety limit |

## Exit Logic Flow

```
1. Enter LONG at $100
2. Price rises to $110 (new highest high)
3. ATR = $2
4. Chandelier Stop = $110 - (3 × $2) = $104
5. Price drops to $107 → HOLD (above $104)
6. Price rises to $115 (new highest high)
7. Chandelier Stop = $115 - (3 × $2) = $109 ← Stop moved UP
8. Price drops to $108 → EXIT at $108 (hit stop)
9. Captured: $108 - $100 = $8 profit (instead of $7 if exited earlier)
```

## Key Insight

The Chandelier trails from the **ceiling** (highest point), not the floor (entry).
This means you lock in profits from the peak, not from where you started.
