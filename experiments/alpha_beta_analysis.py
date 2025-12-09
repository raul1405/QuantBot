"""
Alpha-Beta Analysis: Separating Skill from Market Exposure
============================================================
This script calculates TRUE ALPHA by adjusting for market beta.

Alpha = Strategy_Return - (Beta × Market_Return)

For FX strategies, we use a neutral market assumption (beta ≈ 0)
since FX is a zero-sum market without directional drift.

Author: Antigravity
"""

import sys
import os
import numpy as np
import pandas as pd
from scipy import stats
import yfinance as yf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quant_backtest import (
    Config, DataLoader, FeatureEngine, RegimeEngine, 
    AlphaEngine, EnsembleSignal, CrisisAlphaEngine, Backtester
)


def calculate_beta(strategy_returns: pd.Series, market_returns: pd.Series) -> tuple:
    """
    Calculate beta coefficient via OLS regression.
    Returns: (beta, alpha, r_squared, p_value)
    """
    # Normalize timezones - convert both to tz-naive for alignment
    strat_returns = strategy_returns.copy()
    mkt_returns = market_returns.copy()
    
    if strat_returns.index.tz is not None:
        strat_returns.index = strat_returns.index.tz_localize(None)
    if mkt_returns.index.tz is not None:
        mkt_returns.index = mkt_returns.index.tz_localize(None)
    
    # Align the series
    aligned = pd.concat([strat_returns, mkt_returns], axis=1).dropna()
    if len(aligned) < 10:
        return 0.0, 0.0, 0.0, 1.0
    
    strat = aligned.iloc[:, 0].values
    mkt = aligned.iloc[:, 1].values
    
    # Linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(mkt, strat)
    
    return slope, intercept, r_value**2, p_value


def get_fx_dollar_index():
    """
    Get USD Index (DXY) as market factor for FX.
    If unavailable, return None (assume zero beta for FX).
    """
    try:
        dxy = yf.download("DX-Y.NYB", period="2y", interval="1d", progress=False)
        if dxy.empty:
            return None
        if isinstance(dxy.columns, pd.MultiIndex):
            dxy.columns = dxy.columns.get_level_values(0)
        returns = dxy['Close'].pct_change().dropna()
        return returns
    except:
        return None


def run_strategy_and_get_returns(symbols, start_date, end_date):
    """Run backtest and return daily strategy returns."""
    config = Config()
    config.symbols = symbols
    
    loader = DataLoader(config)
    try:
        data = loader.load_data(start_date, end_date)
    except Exception as e:
        print(f"  Error loading data: {e}")
        return None, None
    
    if not data:
        return None, None
    
    fe = FeatureEngine(config)
    re = RegimeEngine(config)
    ae = AlphaEngine(config)
    es = EnsembleSignal(config)
    ce = CrisisAlphaEngine(config)
    
    data = fe.add_features_all(data)
    data = re.add_regimes_all(data)
    ae.train_model(data)
    data = ae.add_signals_all(data)
    data = es.add_ensemble_all(data)
    data = ce.add_crisis_signals(data)
    
    bt = Backtester(config)
    equity_curve = bt.run_backtest(data)
    trades = pd.DataFrame(bt.account.trade_history)
    
    # Convert equity curve to daily returns
    if hasattr(equity_curve, 'index'):
        eq_df = pd.DataFrame({'equity': equity_curve})
    else:
        eq_df = pd.DataFrame({'equity': equity_curve})
    
    # Resample to daily
    eq_df.index = pd.to_datetime(eq_df.index)
    daily_equity = eq_df['equity'].resample('D').last().dropna()
    daily_returns = daily_equity.pct_change().dropna()
    
    return daily_returns, trades


def main():
    print("=" * 70)
    print("ALPHA-BETA ANALYSIS: SEPARATING SKILL FROM MARKET EXPOSURE")
    print("=" * 70)
    
    # FX Pairs
    fx_all = [
        "EURUSD=X", "USDJPY=X", "GBPUSD=X", "USDCHF=X", 
        "USDCAD=X", "AUDUSD=X", "NZDUSD=X",
        "EURGBP=X", "EURJPY=X", "GBPJPY=X", "AUDJPY=X",
        "EURAUD=X", "EURCHF=X"
    ]
    
    start_date = "2024-01-01"
    end_date = "2024-12-01"
    
    print("\n[1] RUNNING FX STRATEGY BACKTEST...")
    strategy_returns, trades = run_strategy_and_get_returns(fx_all, start_date, end_date)
    
    if strategy_returns is None or len(strategy_returns) < 10:
        print("  ERROR: Could not get strategy returns.")
        return
    
    print(f"  Got {len(strategy_returns)} daily return observations")
    
    # [2] Get Market Factor (DXY)
    print("\n[2] FETCHING USD INDEX (DXY) AS MARKET FACTOR...")
    dxy_returns = get_fx_dollar_index()
    
    if dxy_returns is not None:
        print(f"  Got {len(dxy_returns)} DXY return observations")
    else:
        print("  DXY unavailable - will assume beta = 0 for FX")
    
    # [3] Calculate Beta
    print("\n[3] CALCULATING BETA...")
    
    if dxy_returns is not None:
        beta, reg_alpha, r_sq, p_val = calculate_beta(strategy_returns, dxy_returns)
        print(f"  Beta (vs DXY): {beta:.4f}")
        print(f"  Regression Alpha: {reg_alpha:.6f} (daily)")
        print(f"  R-squared: {r_sq:.4f}")
        print(f"  P-value: {p_val:.4f}")
    else:
        beta = 0.0
        print(f"  Assuming Beta = 0 (FX is theoretically beta-neutral)")
    
    # [4] Calculate Alpha
    print("\n[4] CALCULATING TRUE ALPHA...")
    
    # Strategy stats
    ann_factor = np.sqrt(252)
    strat_mean = strategy_returns.mean() * 252  # Annualized
    strat_std = strategy_returns.std() * ann_factor
    strat_sharpe = strat_mean / strat_std if strat_std > 0 else 0
    
    print(f"\n  --- RAW STRATEGY METRICS ---")
    print(f"  Total Return: {(strategy_returns + 1).prod() - 1:.2%}")
    print(f"  Annualized Return: {strat_mean:.2%}")
    print(f"  Annualized Vol: {strat_std:.2%}")
    print(f"  Sharpe Ratio: {strat_sharpe:.2f}")
    
    # Market stats (if available)
    if dxy_returns is not None:
        # Normalize timezones for alignment
        strat_tz_naive = strategy_returns.copy()
        dxy_tz_naive = dxy_returns.copy()
        if strat_tz_naive.index.tz is not None:
            strat_tz_naive.index = strat_tz_naive.index.tz_localize(None)
        if dxy_tz_naive.index.tz is not None:
            dxy_tz_naive.index = dxy_tz_naive.index.tz_localize(None)
        
        # Align dates
        aligned = pd.concat([strat_tz_naive, dxy_tz_naive], axis=1, join='inner')
        aligned.columns = ['strategy', 'market']
        
        mkt_return = aligned['market'].sum()  # Total period return
        beta_contribution = beta * mkt_return
        
        # Alpha = strategy return - beta * market return
        total_strat_return = aligned['strategy'].sum()
        alpha_return = total_strat_return - beta_contribution
        
        print(f"\n  --- BETA-ADJUSTED (TRUE ALPHA) ---")
        print(f"  Market (DXY) Return: {mkt_return:.2%}")
        print(f"  Beta Contribution: {beta_contribution:.2%}")
        print(f"  TRUE ALPHA: {alpha_return:.2%}")
        
        if abs(beta) < 0.1:
            print(f"\n  ✅ LOW BETA ({beta:.3f}) - Strategy is market-neutral")
            print(f"     Returns are mostly alpha, not market exposure")
        elif beta > 0.3:
            print(f"\n  ⚠️ POSITIVE BETA ({beta:.3f}) - Some market exposure")
            print(f"     Part of returns may be from market direction")
        elif beta < -0.3:
            print(f"\n  ⚠️ NEGATIVE BETA ({beta:.3f}) - Counter-trend exposure")
    else:
        print(f"\n  --- ALPHA ASSESSMENT (NO MARKET FACTOR) ---")
        print(f"  For FX, we assume beta ≈ 0 (zero-sum market)")
        print(f"  TRUE ALPHA ≈ TOTAL RETURN = {(strategy_returns + 1).prod() - 1:.2%}")
    
    # [5] Trade-Level Analysis
    print("\n[5] TRADE-LEVEL ALPHA ANALYSIS...")
    
    if trades is not None and not trades.empty:
        total_pnl = trades['PnL'].sum()
        avg_pnl = trades['PnL'].mean()
        win_rate = (trades['PnL'] > 0).mean()
        
        # Direction breakdown
        if 'Direction' in trades.columns:
            long_trades = trades[trades['Direction'] == 'LONG']
            short_trades = trades[trades['Direction'] == 'SHORT']
            
            print(f"\n  LONG Trades: {len(long_trades)}, PnL: ${long_trades['PnL'].sum():,.0f}")
            print(f"  SHORT Trades: {len(short_trades)}, PnL: ${short_trades['PnL'].sum():,.0f}")
            
            # For FX pairs trading, balanced long/short is good (true alpha)
            long_pct = len(long_trades) / len(trades) * 100
            short_pct = len(short_trades) / len(trades) * 100
            
            print(f"\n  Direction Balance: {long_pct:.1f}% Long / {short_pct:.1f}% Short")
            
            if 40 <= long_pct <= 60:
                print(f"  ✅ BALANCED - Strategy is not directionally biased")
            elif long_pct > 70:
                print(f"  ⚠️ LONG BIASED - May have upward beta hidden")
            elif long_pct < 30:
                print(f"  ⚠️ SHORT BIASED - May have downward beta hidden")
    
    # [6] Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    if strat_sharpe > 0.5:
        alpha_verdict = "POSITIVE ALPHA DETECTED"
        emoji = "✅"
    elif strat_sharpe > 0:
        alpha_verdict = "MARGINAL ALPHA (Statistically Weak)"
        emoji = "⚠️"
    else:
        alpha_verdict = "NO ALPHA (Negative Performance)"
        emoji = "❌"
    
    print(f"\n  {emoji} VERDICT: {alpha_verdict}")
    print(f"\n  Sharpe: {strat_sharpe:.2f}")
    print(f"  Total Return: {(strategy_returns + 1).prod() - 1:.2%}")
    
    if dxy_returns is not None and abs(beta) > 0.1:
        print(f"  Beta-Adjusted Alpha: {alpha_return:.2%}")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
