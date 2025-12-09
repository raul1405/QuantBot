"""
Experiment: Expanded Universe Test (FX/Futures Only - 24h assets)
Goal: Test if backtest results stay comparable when scaling from ~30 to 60 assets
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

# ============================================================================
# EXPANDED UNIVERSE: ~60 Assets (24h tradeable only)
# ============================================================================

EXPANDED_UNIVERSE = [
    # --- FOREX MAJORS (7) ---
    "EURUSD=X", "USDJPY=X", "GBPUSD=X", "USDCHF=X", 
    "USDCAD=X", "AUDUSD=X", "NZDUSD=X",
    
    # --- FOREX CROSSES (21) ---
    "EURGBP=X", "EURJPY=X", "GBPJPY=X", "AUDJPY=X",
    "EURAUD=X", "EURCHF=X", "AUDNZD=X", "AUDCAD=X",
    "CADJPY=X", "NZDJPY=X", "GBPCHF=X", "GBPAUD=X",
    "GBPCAD=X", "EURNZD=X", "CHFJPY=X", "NZDCAD=X",
    "GBPNZD=X", "EURCAD=X", "AUDCHF=X", "NZDCHF=X",
    "CADCHF=X",
    
    # --- EXOTIC FOREX (12) ---
    "USDZAR=X", "USDMXN=X", "USDTRY=X", "USDHKD=X", 
    "USDSGD=X", "USDSEK=X", "USDNOK=X", "USDDKK=X",
    "EURPLN=X", "EURHUF=X", "EURCZK=X", "EURSEK=X",
    
    # --- INDICES FUTURES (4) ---
    "ES=F", "NQ=F", "YM=F", "RTY=F",
    
    # --- COMMODITIES (10) ---
    "GC=F", "SI=F", "CL=F", "NG=F", "BZ=F",
    "HG=F", "PL=F", "PA=F", "ZC=F", "ZW=F",
    
    # --- CRYPTO (6) ---
    "BTC-USD", "ETH-USD", "XRP-USD", "SOL-USD", "BNB-USD", "ADA-USD",
]

# Current baseline universe (from quant_backtest.py)
BASELINE_UNIVERSE = [
    "EURUSD=X", "USDJPY=X", "GBPUSD=X", "USDCHF=X", 
    "USDCAD=X", "AUDUSD=X", "NZDUSD=X",
    "EURGBP=X", "EURJPY=X", "GBPJPY=X", "AUDJPY=X",
    "EURAUD=X", "EURCHF=X", "AUDNZD=X", "AUDCAD=X",
    "CADJPY=X", "NZDJPY=X", "GBPCHF=X", "GBPAUD=X",
    "GBPCAD=X", "EURNZD=X",
    "ES=F", "NQ=F", "YM=F", "RTY=F",
    "GC=F", "CL=F", "NG=F",
    "BTC-USD", "ETH-USD",
]


def run_backtest_scenario(universe: list, name: str) -> dict:
    """Run backtest on a given universe and return results."""
    print(f"\n{'='*60}")
    print(f"RUNNING: {name}")
    print(f"Assets requested: {len(universe)}")
    print(f"{'='*60}\n")
    
    config = Config()
    config.symbols = universe
    config.transaction_cost = 0.0001
    config.initial_balance = 100000.0
    
    try:
        # Load Data
        loader = DataLoader(config)
        data_map = loader.load_data("2024-01-01", "2025-12-01")
        print(f"Loaded {len(data_map)} symbols")
        
        if len(data_map) == 0:
            print("ERROR: No data loaded!")
            return None
        
        # Pipeline (matching EXP008 flow)
        fe = FeatureEngine(config)
        re = RegimeEngine(config)
        
        data_map = fe.add_features_all(data_map)
        data_map = re.add_regimes_all(data_map)
        
        alpha = AlphaEngine(config)
        alpha.train_model(data_map)
        
        data_map = alpha.add_signals_all(data_map)
        
        ens = EnsembleSignal(config)
        data_map = ens.add_ensemble_all(data_map)
        
        crisis = CrisisAlphaEngine(config)
        final_data = crisis.add_crisis_signals(data_map)
        
        # Backtest
        bt = Backtester(config)
        equity_curve = bt.run_backtest(final_data)
        
        # Stats
        trades = len(bt.account.trade_history)
        balance = bt.account.balance
        pnl = balance - config.initial_balance
        roi = (pnl / config.initial_balance) * 100
        
        # Calculate Max DD from Equity Curve
        peak = equity_curve.cummax()
        drawdown = (equity_curve - peak) / peak
        max_dd = drawdown.min() * 100
        
        wins = [t for t in bt.account.trade_history if t.get('PnL', 0) > 0]
        win_rate = (len(wins) / trades * 100) if trades > 0 else 0.0
        
        results = {
            'name': name,
            'n_assets': len(universe),
            'n_assets_loaded': len(data_map),
            'total_return_pct': roi,
            'max_drawdown_pct': max_dd,
            'n_trades': trades,
            'win_rate_pct': win_rate,
            'final_equity': balance,
        }
        
        print(f"\nRESULTS: {name}")
        print(f"  Assets Loaded: {len(data_map)}/{len(universe)}")
        print(f"  Total Return: {roi:+.2f}%")
        print(f"  Max Drawdown: {max_dd:.2f}%")
        print(f"  Total Trades: {trades}")
        print(f"  Win Rate: {win_rate:.1f}%")
        
        return results
        
    except Exception as e:
        print(f"ERROR in {name}: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    print("\n" + "="*70)
    print("EXPERIMENT: Expanded Universe Test (FX + Futures + Crypto Only)")
    print("="*70)
    print(f"\nBaseline Universe: {len(BASELINE_UNIVERSE)} assets")
    print(f"Expanded Universe: {len(EXPANDED_UNIVERSE)} assets")
    
    # Run baseline backtest
    baseline_results = run_backtest_scenario(BASELINE_UNIVERSE, "Baseline (~30 assets)")
    
    # Run expanded backtest
    expanded_results = run_backtest_scenario(EXPANDED_UNIVERSE, "Expanded (~60 assets)")
    
    # Comparison
    if baseline_results and expanded_results:
        print("\n" + "="*70)
        print("COMPARISON: Baseline vs Expanded Universe")
        print("="*70)
        print(f"\n{'Metric':<25} {'Baseline':>15} {'Expanded':>15}")
        print("-"*55)
        print(f"{'Assets Loaded':<25} {baseline_results['n_assets_loaded']:>15} {expanded_results['n_assets_loaded']:>15}")
        print(f"{'Total Return %':<25} {baseline_results['total_return_pct']:>+14.2f}% {expanded_results['total_return_pct']:>+14.2f}%")
        print(f"{'Max Drawdown %':<25} {baseline_results['max_drawdown_pct']:>14.2f}% {expanded_results['max_drawdown_pct']:>14.2f}%")
        print(f"{'Total Trades':<25} {baseline_results['n_trades']:>15} {expanded_results['n_trades']:>15}")
        print(f"{'Win Rate %':<25} {baseline_results['win_rate_pct']:>14.1f}% {expanded_results['win_rate_pct']:>14.1f}%")
        
        # Compute deltas
        print(f"\n{'CHANGE ANALYSIS':>25}")
        print("-"*55)
        ret_delta = expanded_results['total_return_pct'] - baseline_results['total_return_pct']
        dd_delta = expanded_results['max_drawdown_pct'] - baseline_results['max_drawdown_pct']
        trade_delta = expanded_results['n_trades'] - baseline_results['n_trades']
        wr_delta = expanded_results['win_rate_pct'] - baseline_results['win_rate_pct']
        print(f"{'Return Delta':<25} {ret_delta:>+14.2f}%")
        print(f"{'Max DD Delta':<25} {dd_delta:>+14.2f}%")
        print(f"{'Trade Count Delta':<25} {trade_delta:>+15}")
        print(f"{'Win Rate Delta':<25} {wr_delta:>+14.1f}%")
        
        # Trades per asset
        trades_per_asset_base = baseline_results['n_trades'] / baseline_results['n_assets_loaded']
        trades_per_asset_exp = expanded_results['n_trades'] / expanded_results['n_assets_loaded']
        print(f"\n{'Trades per Asset':<25} {trades_per_asset_base:>14.1f} {trades_per_asset_exp:>14.1f}")
    
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE")
    print("="*70)
