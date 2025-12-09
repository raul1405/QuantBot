# Council Guidelines: Statistical Interpretation

> "Interpret all Monte Carlo / FTMO statistics as probabilistic and fragile estimates based on limited, non-stationary history; never claim that a configuration 'guarantees' or 'comfortably' achieves a challenge, only that it improves the odds relative to alternatives, and always accompany such claims with explicit caveats about sample size, regime coverage, and risk scaling."

## Key Principles
1. **No Magic**: Edges are thin, noisy, and regime-dependent.
2. **Path Dependence**: Averages hide the risk of ruin. Path-dependent rules (like FTMO daily limits) are harder to pass than MC averages suggest.
3. **Alpha Scarcity**: Diversification only works if you have multiple positive-alpha assets. If you only have one good idea, diversification is "deworsification".
4. **Sample Limits**: 80 trades in 1 year is a "hint", not a proof.

## Sanity Check Questions
- Did we normalize risk?
- Is the sample size statistically significant (N > 30 is bare minimum, N > 100 preferred)?
- Does the "improvement" survive transaction costs?
- Are we overfitting to a specific regime (e.g., 2024 low vol)?
