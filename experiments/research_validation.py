"""
COMPREHENSIVE STRATEGY VALIDATION (STRICT OUT-OF-SAMPLE)
========================================================
Updated to strictly separate Train (80%) and Test (20%) data to avoid
look-ahead bias/overfitting.

Metrics are reported ONLY on the Out-of-Sample (Test) period.
"""
import sys
sys.path.insert(0, '/Users/raulschalkhammer/Desktop/Costum Portfolio Backtest/FTMO Challenge')

from experiments.research_sandbox import (
    Config, DataLoader, FeatureEngine, AlphaEngine, 
    RegimeEngine, EnsembleSignal, CrisisAlphaEngine, Backtester
)
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# FX Majors Universe
# US_STOCKS_UNIVERSE = [
#     "EURUSD=X", "USDJPY=X", "GBPUSD=X", "USDCHF=X", 
#     "USDCAD=X", "AUDUSD=X", "NZDUSD=X",
#     "EURGBP=X", "EURJPY=X", "GBPJPY=X"
# ]
US_STOCKS_UNIVERSE = [
    "SPY", "QQQ", "IWM", "DIA",
    "XLF", "XLK", "XLE", "XLV", "XLU", "XLI", "XLY", "XLP",
    "EEM", "EFA", "TLT",
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "NFLX",
    "JPM", "BAC", "V", "MA", "JNJ", "XOM", "WMT",
]

def compute_all_metrics(equity_curve: pd.Series, trades: list, rf_rate: float = 0.05) -> dict:
    """Compute comprehensive performance metrics."""
    if len(equity_curve) < 2:
        return {}
        
    # Basic returns
    returns = equity_curve.pct_change().dropna()
    N = len(returns)
    K = 252 * 7  # Hourly bars per year (approx)
    
    # 1. CAGR
    total_days = (equity_curve.index[-1] - equity_curve.index[0]).days
    if total_days < 1: total_days = 1
    years = total_days / 365.25
    total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
    cagr = (1 + total_return) ** (1/years) - 1 if years > 0 else 0
    
    # 2. Volatility
    vol_period = returns.std()
    vol_ann = vol_period * np.sqrt(K)
    
    # 3. Sharpe Ratio
    excess_return = returns.mean() * K - rf_rate
    sharpe = excess_return / vol_ann if vol_ann > 0 else 0
    
    # 4. Sortino Ratio
    downside_returns = returns[returns < 0]
    downside_std = np.sqrt((downside_returns ** 2).mean()) * np.sqrt(K)
    sortino = excess_return / downside_std if downside_std > 0 else 0
    
    # 5. Max Drawdown
    rolling_max = equity_curve.cummax()
    drawdowns = (equity_curve - rolling_max) / rolling_max
    max_dd = drawdowns.min()
    
    # 6. Calmar Ratio
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0
    
    # 7. Skewness/Kurtosis
    skewness = stats.skew(returns)
    kurtosis = stats.kurtosis(returns)
    
    # 8. VaR (95%)
    var_95 = np.percentile(returns, 5)
    cvar_95 = returns[returns <= var_95].mean()
    
    # 9. Win Rate
    if trades:
        wins = [t for t in trades if t.get('PnL', 0) > 0]
        losses = [t for t in trades if t.get('PnL', 0) < 0]
        win_rate = len(wins) / len(trades) if trades else 0
    else:
        win_rate = 0

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
        'cvar_95': cvar_95 * 100,
        'win_rate': win_rate * 100,
        'n_trades': len(trades),
        'days': total_days
    }


def run_oos_backtest(data_map: dict, config: Config, seed: int = None) -> tuple:
    """Run backtest ONLY on Out-of-Sample data."""
    
    if seed is not None:
        np.random.seed(seed)
        config.alpha_model_params['seed'] = seed
    
    # 1. Pipeline (WFO Enabled)
    print("  [Pipeline] Running Feature Engineering...")
    fe = FeatureEngine(config)
    re = RegimeEngine(config)
    data = fe.add_features_all(data_map)
    data = re.add_regimes_all(data)
    
    print("  [Pipeline] Running Alpha Engine (Walk-Forward Optimization)...")
    alpha = AlphaEngine(config)
    # in WFO mode, we don't call train_model() manually. add_signals_all handles rolling train.
    
    data = alpha.add_signals_all(data) # Generates OOS signals via WFO
    ens = EnsembleSignal(config)
    data = ens.add_ensemble_all(data)
    crisis = CrisisAlphaEngine(config)
    final_data = crisis.add_crisis_signals(data)
    
    # 2. Backtest (Full Duration - WFO ensures OOS)
    bt = Backtester(config)
    equity_curve = bt.run_backtest(final_data)
    
    return equity_curve, bt.account.trade_history


if __name__ == "__main__":
    print("\n" + "="*70)
    print(" COMPREHENSIVE STRATEGY VALIDATION (STRICT OOS)")
    print(" US Stocks & ETFs Universe (30 assets)")
    print("="*70)
    
    # Config
    config = Config()
    config.symbols = US_STOCKS_UNIVERSE
    config.transaction_cost = 0.0001
    config.initial_balance = 100000.0
    config.ml_train_split_pct = 0.80 # 80% Train, 20% Test
    
    # Load
    loader = DataLoader(config)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=500)
    data_map = loader.load_data(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
    
    # Run OOS Backtest
    print("\n[RUNNING STRICT OOS BACKTEST]...")
    eq, trades = run_oos_backtest(data_map, config, seed=42)
    
    if eq is not None:
        metrics = compute_all_metrics(eq, trades)
        
        print("\n" + "="*60)
        print(" FINAL OOS RESULTS (Unseen Data)")
        print("="*60)
        print(f"{'Metric':<25} {'Value':>20}")
        print("-"*45)
        print(f"{'Test Duration (Days)':<25} {metrics['days']:>20}")
        print(f"{'Total Return %':<25} {metrics['total_return_pct']:>+19.2f}%")
        print(f"{'Max Drawdown %':<25} {metrics['max_dd_pct']:>19.2f}%")
        print(f"{'Sharpe Ratio':<25} {metrics['sharpe']:>20.2f}")
        print(f"{'Sortino Ratio':<25} {metrics['sortino']:>20.2f}")
        print(f"{'Win Rate %':<25} {metrics['win_rate']:>19.1f}%")
        print(f"{'Total Trades':<25} {metrics['n_trades']:>20}")
        print(f"{'Skewness':<25} {metrics['skewness']:>20.2f}")
        
        print("\n" + "="*60)
        
        if metrics['sharpe'] > 5.0:
             print("❌ Result is still suspiciously high. Model might be overfitting even on OOS (time leakage?).")
        elif metrics['sharpe'] > 1.0:
             print("✅ Result looks realistic and good.")
        else:
             print("⚠️ Performance dropped significantly in OOS (Expected).")

    print("\n" + "="*70)
    print(" VALIDATION COMPLETE")
    print("="*70)
