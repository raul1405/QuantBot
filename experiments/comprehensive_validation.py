"""
COMPREHENSIVE STRATEGY VALIDATION
==================================
Full validation suite including:
1. All performance metrics (Sharpe, Sortino, Calmar, VaR, CVaR, etc.)
2. Monte Carlo simulations (vary random seeds)
3. Random Walk comparison (baseline)
4. Synthetic Bear Market stress tests
5. Data integrity checks
"""
import sys
sys.path.insert(0, '/Users/raulschalkhammer/Desktop/Costum Portfolio Backtest/FTMO Challenge')

from quant_backtest import (
    Config, DataLoader, FeatureEngine, AlphaEngine, 
    RegimeEngine, EnsembleSignal, CrisisAlphaEngine, Backtester
)
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# US STOCKS & ETFs UNIVERSE
# ============================================================================
US_STOCKS_UNIVERSE = [
    "SPY", "QQQ", "IWM", "DIA",
    "XLF", "XLK", "XLE", "XLV", "XLU", "XLI", "XLY", "XLP",
    "EEM", "EFA", "TLT",
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "NFLX",
    "JPM", "BAC", "V", "MA", "JNJ", "XOM", "WMT",
]


def compute_all_metrics(equity_curve: pd.Series, trades: list, rf_rate: float = 0.05) -> dict:
    """Compute comprehensive performance metrics."""
    
    # Basic returns
    returns = equity_curve.pct_change().dropna()
    N = len(returns)
    K = 252 * 7  # Hourly bars per year (approx)
    
    # 1. CAGR
    total_days = (equity_curve.index[-1] - equity_curve.index[0]).days
    years = total_days / 365.25
    total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
    cagr = (1 + total_return) ** (1/years) - 1 if years > 0 else 0
    
    # 2. Volatility
    vol_period = returns.std()
    vol_ann = vol_period * np.sqrt(K)
    
    # 3. Sharpe Ratio
    excess_return = returns.mean() * K - rf_rate
    sharpe = excess_return / vol_ann if vol_ann > 0 else 0
    
    # 4. Sortino Ratio (downside deviation)
    downside_returns = returns[returns < 0]
    downside_std = np.sqrt((downside_returns ** 2).mean()) * np.sqrt(K)
    sortino = excess_return / downside_std if downside_std > 0 else 0
    
    # 5. Max Drawdown
    rolling_max = equity_curve.cummax()
    drawdowns = (equity_curve - rolling_max) / rolling_max
    max_dd = drawdowns.min()
    
    # 6. Calmar Ratio
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0
    
    # 7. Skewness
    skewness = stats.skew(returns)
    
    # 8. Kurtosis (excess)
    kurtosis = stats.kurtosis(returns)
    
    # 9. VaR (95% and 99%)
    var_95 = np.percentile(returns, 5)
    var_99 = np.percentile(returns, 1)
    
    # 10. CVaR/Expected Shortfall
    cvar_95 = returns[returns <= var_95].mean()
    cvar_99 = returns[returns <= var_99].mean()
    
    # 11. Tail Ratio
    upper_tail = np.percentile(returns, 95)
    lower_tail = np.percentile(returns, 5)
    tail_ratio = abs(upper_tail / lower_tail) if lower_tail != 0 else 0
    
    # 12. Win Rate & Profit Factor
    if trades:
        wins = [t for t in trades if t.get('PnL', 0) > 0]
        losses = [t for t in trades if t.get('PnL', 0) < 0]
        win_rate = len(wins) / len(trades) if trades else 0
        
        total_wins = sum(t.get('PnL', 0) for t in wins)
        total_losses = abs(sum(t.get('PnL', 0) for t in losses))
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        
        avg_win = np.mean([t.get('PnL', 0) for t in wins]) if wins else 0
        avg_loss = np.mean([t.get('PnL', 0) for t in losses]) if losses else 0
        payoff_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
    else:
        win_rate = 0
        profit_factor = 0
        payoff_ratio = 0
    
    # 13. t-statistic of mean return
    t_stat = (returns.mean() / (returns.std() / np.sqrt(N))) if N > 1 else 0
    
    # 14. Ulcer Index
    dd_squared = drawdowns ** 2
    ulcer_index = np.sqrt(dd_squared.mean())
    
    # 15. Average Drawdown Duration
    in_drawdown = drawdowns < 0
    dd_changes = in_drawdown.astype(int).diff().fillna(0)
    dd_starts = dd_changes == 1
    dd_ends = dd_changes == -1
    # Approximate by counting consecutive DD bars
    dd_lengths = []
    current_dd_len = 0
    for is_dd in in_drawdown:
        if is_dd:
            current_dd_len += 1
        elif current_dd_len > 0:
            dd_lengths.append(current_dd_len)
            current_dd_len = 0
    if current_dd_len > 0:
        dd_lengths.append(current_dd_len)
    avg_dd_duration = np.mean(dd_lengths) if dd_lengths else 0
    
    return {
        'total_return_pct': total_return * 100,
        'cagr': cagr * 100,
        'volatility_ann': vol_ann * 100,
        'sharpe': sharpe,
        'sortino': sortino,
        'calmar': calmar,
        'max_dd_pct': max_dd * 100,
        'skewness': skewness,
        'kurtosis': kurtosis,
        'var_95': var_95 * 100,
        'var_99': var_99 * 100,
        'cvar_95': cvar_95 * 100,
        'cvar_99': cvar_99 * 100,
        'tail_ratio': tail_ratio,
        'win_rate': win_rate * 100,
        'profit_factor': profit_factor,
        'payoff_ratio': payoff_ratio,
        't_stat': t_stat,
        'ulcer_index': ulcer_index * 100,
        'avg_dd_duration_bars': avg_dd_duration,
        'n_trades': len(trades),
        'n_bars': N,
    }


def run_single_backtest(data_map: dict, config: Config, seed: int = None) -> tuple:
    """Run a single backtest with optional seed variation."""
    
    if seed is not None:
        np.random.seed(seed)
        config.alpha_model_params = {
            'objective': 'multi:softprob',
            'num_class': 3,
            'max_depth': 6,
            'eta': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'seed': seed
        }
    
    # Make copies to avoid mutation
    data = {k: v.copy() for k, v in data_map.items()}
    
    fe = FeatureEngine(config)
    re = RegimeEngine(config)
    
    data = fe.add_features_all(data)
    data = re.add_regimes_all(data)
    
    alpha = AlphaEngine(config)
    alpha.train_model(data)
    data = alpha.add_signals_all(data)
    
    ens = EnsembleSignal(config)
    data = ens.add_ensemble_all(data)
    
    crisis = CrisisAlphaEngine(config)
    final_data = crisis.add_crisis_signals(data)
    
    bt = Backtester(config)
    equity_curve = bt.run_backtest(final_data)
    
    return equity_curve, bt.account.trade_history


def run_random_walk_baseline(equity_curve: pd.Series, n_sims: int = 1000) -> dict:
    """Compare against random walk (no-skill) baseline."""
    
    returns = equity_curve.pct_change().dropna()
    actual_total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
    
    # Generate random walks with same vol
    vol = returns.std()
    n_bars = len(returns)
    
    random_returns_list = []
    for _ in range(n_sims):
        random_returns = np.random.normal(0, vol, n_bars)
        random_total = (1 + random_returns).prod() - 1
        random_returns_list.append(random_total)
    
    random_returns_array = np.array(random_returns_list)
    
    # Percentile rank of actual return vs random
    percentile = (random_returns_array < actual_total_return).mean() * 100
    
    return {
        'actual_return': actual_total_return * 100,
        'random_mean': random_returns_array.mean() * 100,
        'random_std': random_returns_array.std() * 100,
        'random_5th': np.percentile(random_returns_array, 5) * 100,
        'random_95th': np.percentile(random_returns_array, 95) * 100,
        'percentile_rank': percentile,
        'beats_random_pct': (random_returns_array < actual_total_return).mean() * 100,
    }


def generate_synthetic_bear_market(data_map: dict, crash_pct: float = -0.30, 
                                   crash_duration_days: int = 30) -> dict:
    """Apply synthetic bear market shock to price data."""
    
    shocked_data = {}
    
    for sym, df in data_map.items():
        df_copy = df.copy()
        n_bars = len(df_copy)
        
        # Apply crash in the middle of the data
        crash_start = n_bars // 3
        crash_bars = crash_duration_days * 7  # Approx hourly bars per day
        crash_end = min(crash_start + crash_bars, n_bars)
        
        # Linear decline
        decline_factor = np.linspace(1.0, 1 + crash_pct, crash_end - crash_start)
        
        # Apply to OHLC
        for col in ['Open', 'High', 'Low', 'Close']:
            if col in df_copy.columns:
                original = df_copy[col].values.copy()
                for i, factor in enumerate(decline_factor):
                    original[crash_start + i] = original[crash_start + i] * factor
                # Recovery phase (slow)
                if crash_end < n_bars:
                    recovery_bars = n_bars - crash_end
                    recovery_factor = np.linspace(1 + crash_pct, 0.95, recovery_bars)
                    for i, factor in enumerate(recovery_factor):
                        original[crash_end + i] = original[crash_start] * factor
                df_copy[col] = original
        
        shocked_data[sym] = df_copy
    
    return shocked_data


def run_monte_carlo(data_map: dict, config: Config, n_runs: int = 20) -> dict:
    """Run Monte Carlo simulations with varied random seeds."""
    
    print(f"\n[MONTE CARLO] Running {n_runs} simulations...")
    
    results = []
    for i in range(n_runs):
        try:
            eq, trades = run_single_backtest(data_map.copy(), config, seed=i*42)
            metrics = compute_all_metrics(eq, trades)
            results.append(metrics)
            if (i + 1) % 5 == 0:
                print(f"  Completed {i+1}/{n_runs}...")
        except Exception as e:
            print(f"  Run {i} failed: {e}")
    
    if not results:
        return None
    
    # Aggregate
    summary = {}
    for key in results[0].keys():
        values = [r[key] for r in results if r[key] is not None and not np.isinf(r[key])]
        if values:
            summary[f'{key}_mean'] = np.mean(values)
            summary[f'{key}_std'] = np.std(values)
            summary[f'{key}_5pct'] = np.percentile(values, 5)
            summary[f'{key}_95pct'] = np.percentile(values, 95)
    
    return summary


def verify_data_integrity(data_map: dict) -> dict:
    """Check data for common issues."""
    
    issues = []
    stats = {}
    
    for sym, df in data_map.items():
        # Check for NaN
        nan_count = df.isna().sum().sum()
        if nan_count > 0:
            issues.append(f"{sym}: {nan_count} NaN values")
        
        # Check for zero/negative prices
        if 'Close' in df.columns:
            zero_prices = (df['Close'] <= 0).sum()
            if zero_prices > 0:
                issues.append(f"{sym}: {zero_prices} zero/negative prices")
        
        # Check for gaps
        if len(df) > 1:
            time_diffs = df.index.to_series().diff().dropna()
            max_gap = time_diffs.max()
            if max_gap > pd.Timedelta(hours=100):  # More than 4 days gap
                issues.append(f"{sym}: Large gap of {max_gap}")
        
        # Check for duplicate timestamps
        dup_count = df.index.duplicated().sum()
        if dup_count > 0:
            issues.append(f"{sym}: {dup_count} duplicate timestamps")
    
    # Overall stats
    all_lens = [len(df) for df in data_map.values()]
    stats['n_symbols'] = len(data_map)
    stats['bars_per_symbol'] = all_lens[0] if all_lens else 0
    stats['all_aligned'] = len(set(all_lens)) == 1
    stats['issues'] = issues
    stats['is_clean'] = len(issues) == 0
    
    return stats


def print_metrics_table(metrics: dict, title: str = "METRICS"):
    """Pretty print metrics table."""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")
    
    # Group by category
    categories = {
        'Returns': ['total_return_pct', 'cagr', 'volatility_ann'],
        'Risk-Adjusted': ['sharpe', 'sortino', 'calmar'],
        'Drawdown': ['max_dd_pct', 'ulcer_index', 'avg_dd_duration_bars'],
        'Tail Risk': ['skewness', 'kurtosis', 'var_95', 'var_99', 'cvar_95', 'cvar_99', 'tail_ratio'],
        'Trading': ['n_trades', 'win_rate', 'profit_factor', 'payoff_ratio'],
        'Significance': ['t_stat', 'n_bars'],
    }
    
    for cat_name, keys in categories.items():
        print(f"\n{cat_name}:")
        print("-" * 40)
        for key in keys:
            if key in metrics:
                val = metrics[key]
                if isinstance(val, float):
                    if abs(val) > 100:
                        print(f"  {key:<25} {val:>12,.2f}")
                    else:
                        print(f"  {key:<25} {val:>12.4f}")
                else:
                    print(f"  {key:<25} {val:>12}")


if __name__ == "__main__":
    print("\n" + "="*70)
    print(" COMPREHENSIVE STRATEGY VALIDATION")
    print(" US Stocks & ETFs Universe (30 assets)")
    print("="*70)
    
    # ===== 1. LOAD DATA =====
    print("\n[1/6] LOADING DATA...")
    config = Config()
    config.symbols = US_STOCKS_UNIVERSE
    config.transaction_cost = 0.0001
    config.initial_balance = 100000.0
    
    loader = DataLoader(config)
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=500)).strftime('%Y-%m-%d')
    print(f"Date range: {start_date} to {end_date}")
    
    data_map = loader.load_data(start_date, end_date)
    print(f"Loaded {len(data_map)} symbols")
    
    # ===== 2. DATA INTEGRITY CHECK =====
    print("\n[2/6] VERIFYING DATA INTEGRITY...")
    integrity = verify_data_integrity(data_map)
    print(f"  Symbols: {integrity['n_symbols']}")
    print(f"  Bars per symbol: {integrity['bars_per_symbol']}")
    print(f"  All aligned: {integrity['all_aligned']}")
    print(f"  Data clean: {integrity['is_clean']}")
    if integrity['issues']:
        print("  Issues found:")
        for issue in integrity['issues'][:5]:
            print(f"    - {issue}")
    
    # ===== 3. BASE BACKTEST =====
    print("\n[3/6] RUNNING BASE BACKTEST...")
    equity_curve, trades = run_single_backtest(data_map.copy(), config, seed=42)
    base_metrics = compute_all_metrics(equity_curve, trades)
    print_metrics_table(base_metrics, "BASE BACKTEST METRICS")
    
    # ===== 4. RANDOM WALK COMPARISON =====
    print("\n[4/6] RANDOM WALK COMPARISON (1000 simulations)...")
    rw = run_random_walk_baseline(equity_curve, n_sims=1000)
    print(f"\n  Actual Return:     {rw['actual_return']:+.2f}%")
    print(f"  Random Mean:       {rw['random_mean']:+.2f}%")
    print(f"  Random Std:        {rw['random_std']:.2f}%")
    print(f"  Random 5th-95th:   [{rw['random_5th']:+.2f}%, {rw['random_95th']:+.2f}%]")
    print(f"  Percentile Rank:   {rw['percentile_rank']:.1f}%")
    print(f"  Beats Random:      {rw['beats_random_pct']:.1f}% of random walks")
    
    if rw['percentile_rank'] > 95:
        print("  ✅ PASS: Strategy significantly outperforms random (>95th percentile)")
    elif rw['percentile_rank'] > 75:
        print("  ⚠️ MARGINAL: Strategy performs better than average random")
    else:
        print("  ❌ FAIL: Strategy does not clearly outperform random walk")
    
    # ===== 5. MONTE CARLO =====
    print("\n[5/6] MONTE CARLO SIMULATIONS...")
    mc_results = run_monte_carlo(data_map.copy(), config, n_runs=15)
    
    if mc_results:
        print("\n" + "="*60)
        print(" MONTE CARLO RESULTS (15 runs)")
        print("="*60)
        key_metrics = ['total_return_pct', 'sharpe', 'max_dd_pct', 'win_rate', 'n_trades']
        print(f"\n{'Metric':<20} {'Mean':>12} {'Std':>12} {'5th%':>12} {'95th%':>12}")
        print("-"*70)
        for m in key_metrics:
            mean_key = f'{m}_mean'
            std_key = f'{m}_std'
            p5_key = f'{m}_5pct'
            p95_key = f'{m}_95pct'
            if mean_key in mc_results:
                print(f"{m:<20} {mc_results[mean_key]:>12.2f} {mc_results[std_key]:>12.2f} "
                      f"{mc_results[p5_key]:>12.2f} {mc_results[p95_key]:>12.2f}")
    
    # ===== 6. SYNTHETIC BEAR MARKET STRESS TEST =====
    print("\n[6/6] SYNTHETIC BEAR MARKET STRESS TESTS...")
    
    stress_scenarios = [
        ("Moderate Crash", -0.15, 20),
        ("Severe Crash", -0.30, 30),
        ("Extreme Crash (2008-style)", -0.50, 60),
    ]
    
    print(f"\n{'Scenario':<30} {'Return':>12} {'Max DD':>12} {'Trades':>10} {'Win Rate':>10}")
    print("-"*75)
    
    for name, crash_pct, crash_days in stress_scenarios:
        try:
            shocked_data = generate_synthetic_bear_market(data_map.copy(), crash_pct, crash_days)
            eq_stress, trades_stress = run_single_backtest(shocked_data, config, seed=42)
            stress_metrics = compute_all_metrics(eq_stress, trades_stress)
            print(f"{name:<30} {stress_metrics['total_return_pct']:>+11.2f}% "
                  f"{stress_metrics['max_dd_pct']:>11.2f}% "
                  f"{stress_metrics['n_trades']:>10} "
                  f"{stress_metrics['win_rate']:>9.1f}%")
        except Exception as e:
            print(f"{name:<30} ERROR: {e}")
    
    # ===== FINAL SUMMARY =====
    print("\n" + "="*70)
    print(" VALIDATION SUMMARY")
    print("="*70)
    
    print(f"""
┌─────────────────────────────────────────────────────────────────────┐
│ DATA INTEGRITY                                                       │
├─────────────────────────────────────────────────────────────────────┤
│ Symbols Loaded:        {integrity['n_symbols']:>5}                                          │
│ Bars per Symbol:       {integrity['bars_per_symbol']:>5}                                          │
│ Data Clean:            {'✅ YES' if integrity['is_clean'] else '❌ NO ':<6}                                        │
├─────────────────────────────────────────────────────────────────────┤
│ BASE PERFORMANCE                                                     │
├─────────────────────────────────────────────────────────────────────┤
│ Total Return:          {base_metrics['total_return_pct']:>+7.2f}%                                      │
│ CAGR:                  {base_metrics['cagr']:>+7.2f}%                                      │
│ Sharpe Ratio:          {base_metrics['sharpe']:>7.2f}                                       │
│ Sortino Ratio:         {base_metrics['sortino']:>7.2f}                                       │
│ Calmar Ratio:          {base_metrics['calmar']:>7.2f}                                       │
│ Max Drawdown:          {base_metrics['max_dd_pct']:>7.2f}%                                      │
│ Win Rate:              {base_metrics['win_rate']:>7.2f}%                                      │
│ Total Trades:          {base_metrics['n_trades']:>7}                                       │
├─────────────────────────────────────────────────────────────────────┤
│ STATISTICAL SIGNIFICANCE                                             │
├─────────────────────────────────────────────────────────────────────┤
│ t-statistic:           {base_metrics['t_stat']:>7.2f}       {'✅ >2' if base_metrics['t_stat'] > 2 else '⚠️ <2':<20}         │
│ vs Random Walk:        {rw['percentile_rank']:>6.1f}%       {'✅ >95%' if rw['percentile_rank'] > 95 else '⚠️ <95%':<20}        │
├─────────────────────────────────────────────────────────────────────┤
│ TAIL RISK                                                            │
├─────────────────────────────────────────────────────────────────────┤
│ VaR (95%):             {base_metrics['var_95']:>7.4f}%                                      │
│ CVaR (95%):            {base_metrics['cvar_95']:>7.4f}%                                      │
│ Skewness:              {base_metrics['skewness']:>7.2f}        {'✅ >0' if base_metrics['skewness'] > 0 else '⚠️ <0 (left tail)':<20}       │
│ Kurtosis:              {base_metrics['kurtosis']:>7.2f}        {'⚠️ Fat tails' if base_metrics['kurtosis'] > 3 else '✅ Normal':<20}       │
└─────────────────────────────────────────────────────────────────────┘
""")
    
    # Final verdict
    score = 0
    checks = [
        base_metrics['total_return_pct'] > 0,
        base_metrics['sharpe'] > 1.0,
        base_metrics['max_dd_pct'] > -15,
        base_metrics['win_rate'] > 50,
        base_metrics['t_stat'] > 2,
        rw['percentile_rank'] > 90,
        integrity['is_clean'],
    ]
    score = sum(checks)
    
    print(f"\nOVERALL SCORE: {score}/7 checks passed")
    if score >= 6:
        print("✅ STRATEGY VALIDATED - Ready for deployment")
    elif score >= 4:
        print("⚠️ STRATEGY MARGINAL - Review concerns before deployment")
    else:
        print("❌ STRATEGY FAILED - Significant issues detected")
    
    print("\n" + "="*70)
    print(" VALIDATION COMPLETE")
    print("="*70)
