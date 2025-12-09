"""
SHARPE RATIO ERROR DIAGNOSTIC
=============================
Identifies calculation errors and data issues in the current backtest.

Checks:
1. Sharpe annualization factor (K) for different asset classes
2. Transaction cost calculation realism
3. Cross-sectional feature look-ahead bias
4. Data alignment issues

Run: python experiments/diagnose_sharpe_errors.py
"""
import sys
sys.path.insert(0, '/Users/raulschalkhammer/Desktop/Costum Portfolio Backtest/FTMO Challenge')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from quant_backtest import Config, DataLoader, FeatureEngine, RegimeEngine, AlphaEngine

def diagnose_annualization_factor():
    """Check what K should be for FX vs Equities."""
    print("\n" + "="*70)
    print(" DIAGNOSTIC 1: SHARPE ANNUALIZATION FACTOR")
    print("="*70)
    
    config = Config()
    loader = DataLoader(config)
    
    # Load sample data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=60)
    
    try:
        data = loader.load_data(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
    except Exception as e:
        print(f"  [ERROR] Failed to load data: {e}")
        return
    
    fx_symbols = [s for s in data.keys() if '=X' in s]
    equity_symbols = [s for s in data.keys() if '=X' not in s and '-USD' not in s]
    
    print(f"\n  FX Symbols: {len(fx_symbols)}")
    print(f"  Other Symbols: {len(equity_symbols)}")
    
    # Analyze bar counts
    fx_bars_per_day = []
    equity_bars_per_day = []
    
    for sym in fx_symbols[:3]:  # Sample 3
        df = data[sym]
        days = (df.index[-1] - df.index[0]).days
        if days > 0:
            bars_per_day = len(df) / days
            fx_bars_per_day.append(bars_per_day)
            print(f"    {sym}: {len(df)} bars over {days} days = {bars_per_day:.1f} bars/day")
    
    for sym in equity_symbols[:3]:  # Sample 3
        if sym in data:
            df = data[sym]
            days = (df.index[-1] - df.index[0]).days
            if days > 0:
                bars_per_day = len(df) / days
                equity_bars_per_day.append(bars_per_day)
                print(f"    {sym}: {len(df)} bars over {days} days = {bars_per_day:.1f} bars/day")
    
    # Calculate correct K
    avg_fx_bars = np.mean(fx_bars_per_day) if fx_bars_per_day else 24
    avg_equity_bars = np.mean(equity_bars_per_day) if equity_bars_per_day else 7
    
    # Trading days per year
    fx_trading_days = 252  # FX trades ~252 days but 24h
    equity_trading_days = 252
    
    K_fx = avg_fx_bars * fx_trading_days
    K_equity = avg_equity_bars * equity_trading_days
    K_current = 252 * 7  # What code uses
    
    print(f"\n  📊 ANNUALIZATION FACTORS:")
    print(f"  -" * 35)
    print(f"  CURRENT (in code):  K = {K_current} (assuming 7 bars/day)")
    print(f"  CORRECT for FX:     K = {K_fx:.0f} ({avg_fx_bars:.1f} bars/day * 252)")
    print(f"  CORRECT for Equity: K = {K_equity:.0f} ({avg_equity_bars:.1f} bars/day * 252)")
    
    # Sharpe inflation factor
    if K_fx > 0:
        inflation_factor_fx = np.sqrt(K_fx / K_current)
        print(f"\n  ⚠️ FX Sharpe DEFLATION needed: divide by {inflation_factor_fx:.2f}")
        print(f"     (If reported Sharpe = 2.18, true FX Sharpe ≈ {2.18 / inflation_factor_fx:.2f})")
    
    return {'K_current': K_current, 'K_fx': K_fx, 'K_equity': K_equity}


def diagnose_transaction_costs():
    """Check transaction cost calculation."""
    print("\n" + "="*70)
    print(" DIAGNOSTIC 2: TRANSACTION COST MODEL")
    print("="*70)
    
    # Current model from code
    config = Config()
    transaction_cost = config.transaction_cost  # 0.0005
    
    # Simulate a typical trade
    test_size = 1.0  # 1 standard lot
    test_price = 1.10  # EURUSD price
    
    # Current calculation
    current_cost = test_size * transaction_cost
    
    # Realistic FX costs
    # 1 standard lot = 100,000 units
    # 1 pip for EUR/USD = $10
    # Typical spread = 0.5-1.5 pips
    spread_pips = 1.0  # Conservative estimate
    realistic_cost_per_side = spread_pips * 10  # $10 per side
    realistic_round_trip = realistic_cost_per_side * 2  # $20 round-trip
    
    # Alternative: notional-based
    notional = test_size * test_price * 100000
    notional_cost = notional * transaction_cost * 2  # Entry + exit
    
    print(f"\n  📊 COST COMPARISON (1 lot EURUSD at 1.10):")
    print(f"  -" * 35)
    print(f"  CURRENT FORMULA: size * cost = {test_size} * {transaction_cost} = ${current_cost:.4f}")
    print(f"  ❌ This is {current_cost:.4f} dollars per trade!")
    print(f"")
    print(f"  REALISTIC FX COST (1 pip spread): ${realistic_round_trip:.2f} round-trip")
    print(f"  NOTIONAL-BASED: ${notional_cost:.2f} (notional * {transaction_cost} * 2)")
    print(f"")
    
    error_magnitude = realistic_round_trip / current_cost if current_cost > 0 else float('inf')
    print(f"  ⚠️ COST UNDERESTIMATION: {error_magnitude:,.0f}x too low!")
    print(f"     This MASSIVELY inflates apparent profits.")
    
    # Estimate impact on a typical backtest
    assumed_trades = 400
    assumed_avg_pnl = 50  # $50 per trade
    
    current_total_cost = assumed_trades * current_cost
    realistic_total_cost = assumed_trades * realistic_round_trip
    
    gross_profit = assumed_trades * assumed_avg_pnl
    net_with_current = gross_profit - current_total_cost
    net_with_realistic = gross_profit - realistic_total_cost
    
    print(f"\n  📊 IMPACT ESTIMATE ({assumed_trades} trades, ${assumed_avg_pnl}/trade avg):")
    print(f"  -" * 35)
    print(f"  Gross Profit:         ${gross_profit:,.0f}")
    print(f"  Costs (current):      ${current_total_cost:,.2f}")
    print(f"  Costs (realistic):    ${realistic_total_cost:,.0f}")
    print(f"  Net (current):        ${net_with_current:,.0f}")
    print(f"  Net (realistic):      ${net_with_realistic:,.0f}")
    
    if net_with_realistic < 0:
        print(f"\n  ❌ STRATEGY WOULD BE UNPROFITABLE WITH REALISTIC COSTS!")
    
    return {
        'current_cost': current_cost,
        'realistic_cost': realistic_round_trip,
        'underestimation_factor': error_magnitude
    }


def diagnose_feature_leakage():
    """Check for potential look-ahead bias in features."""
    print("\n" + "="*70)
    print(" DIAGNOSTIC 3: FEATURE LOOK-AHEAD BIAS")
    print("="*70)
    
    config = Config()
    loader = DataLoader(config)
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=100)
    
    try:
        data = loader.load_data(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
    except Exception as e:
        print(f"  [ERROR] Failed to load data: {e}")
        return
    
    # Apply features
    fe = FeatureEngine(config)
    data_feat = fe.add_features_all(data)
    
    print("\n  📊 FEATURE ANALYSIS:")
    print(f"  -" * 35)
    
    issues_found = []
    
    # Check 1: Cross-sectional features (ranks)
    print("\n  [Cross-Sectional Features]")
    sample_sym = list(data_feat.keys())[0]
    df = data_feat[sample_sym]
    
    if 'Mom_24h_rank' in df.columns:
        # Check if rank uses future data
        # The rank is computed across symbols at each timestamp
        # This is safe IF all symbols have the same timestamps
        
        # Check alignment
        all_lengths = [len(data_feat[s]) for s in data_feat.keys()]
        if len(set(all_lengths)) > 1:
            print(f"    ⚠️ POTENTIAL ISSUE: Symbols have different lengths")
            print(f"       Lengths: {min(all_lengths)} to {max(all_lengths)}")
            issues_found.append("Cross-sectional alignment")
        else:
            print(f"    ✅ Symbols aligned ({all_lengths[0]} bars each)")
    
    # Check 2: Rolling calculations with potential edge issues
    print("\n  [Rolling Window Features]")
    
    rolling_features = [
        ('Z_Score', 50),  # Uses MA_50
        ('Volatility', 20),
        ('RSI', 14),
    ]
    
    for feat, window in rolling_features:
        if feat in df.columns:
            # Check for NaN handling
            nan_count = df[feat].isna().sum()
            first_valid_idx = df[feat].first_valid_index()
            
            if first_valid_idx is not None:
                first_valid_pos = df.index.get_loc(first_valid_idx)
                if first_valid_pos < window:
                    print(f"    ⚠️ {feat}: First valid at index {first_valid_pos} (expected {window})")
                else:
                    print(f"    ✅ {feat}: Proper warmup ({first_valid_pos} NaN rows)")
    
    # Check 3: Target creation
    print("\n  [Target Variable]")
    alpha = AlphaEngine(config)
    lookahead = config.alpha_target_lookahead
    print(f"    Target lookahead: {lookahead} bar(s)")
    print(f"    ✅ This is correct IF WFO properly excludes target-overlapping rows at fold boundaries")
    
    # Check 4: WFO Parameters
    print("\n  [Walk-Forward Optimization]")
    print(f"    Train window: {config.wfo_train_bars} bars")
    print(f"    Test window: {config.wfo_test_bars} bars")
    print(f"    Target lookahead: {config.alpha_target_lookahead} bars")
    
    if config.wfo_train_bars < 200:
        print(f"    ⚠️ Train window seems small for robust ML training")
    
    return issues_found


def diagnose_recent_trades():
    """Load recent backtest results and analyze."""
    print("\n" + "="*70)
    print(" DIAGNOSTIC 4: RECENT BACKTEST ANALYSIS")
    print("="*70)
    
    try:
        trades_file = '/Users/raulschalkhammer/Desktop/Costum Portfolio Backtest/FTMO Challenge/backtest_results.csv'
        trades_df = pd.read_csv(trades_file)
        
        print(f"\n  📊 LOADED {len(trades_df)} TRADES")
        print(f"  -" * 35)
        
        if 'PnL' in trades_df.columns:
            total_pnl = trades_df['PnL'].sum()
            avg_pnl = trades_df['PnL'].mean()
            std_pnl = trades_df['PnL'].std()
            
            print(f"  Total PnL:    ${total_pnl:,.2f}")
            print(f"  Avg PnL:      ${avg_pnl:,.2f}")
            print(f"  Std PnL:      ${std_pnl:,.2f}")
            
            # Trade-based Sharpe (not annualized)
            raw_sharpe = avg_pnl / std_pnl if std_pnl > 0 else 0
            print(f"  Raw Sharpe (per-trade): {raw_sharpe:.4f}")
            
            # For proper annualization, we need trade frequency
            if 'Entry Time' in trades_df.columns and 'Exit Time' in trades_df.columns:
                try:
                    trades_df['Entry Time'] = pd.to_datetime(trades_df['Entry Time'])
                    trades_df['Exit Time'] = pd.to_datetime(trades_df['Exit Time'])
                    
                    first_trade = trades_df['Entry Time'].min()
                    last_trade = trades_df['Exit Time'].max()
                    trading_days = (last_trade - first_trade).days
                    
                    trades_per_day = len(trades_df) / trading_days if trading_days > 0 else 0
                    trades_per_year = trades_per_day * 252
                    
                    # Approximate equity-curve Sharpe
                    approx_annual_sharpe = raw_sharpe * np.sqrt(trades_per_year)
                    
                    print(f"\n  Trade Frequency: {trades_per_day:.1f}/day ({trades_per_year:.0f}/year)")
                    print(f"  Annualized Sharpe (approx): {approx_annual_sharpe:.2f}")
                    
                except Exception as e:
                    print(f"  [WARN] Date parsing issue: {e}")
        
        # Analyze by symbol type
        if 'Symbol' in trades_df.columns and 'PnL' in trades_df.columns:
            print(f"\n  📊 BY ASSET CLASS:")
            print(f"  -" * 35)
            
            trades_df['Asset_Class'] = trades_df['Symbol'].apply(
                lambda x: 'FX' if '=X' in str(x) else ('Crypto' if '-USD' in str(x) else 'Commodity/Index')
            )
            
            for asset_class, group in trades_df.groupby('Asset_Class'):
                n = len(group)
                total = group['PnL'].sum()
                avg = group['PnL'].mean()
                std = group['PnL'].std()
                sharpe = avg / std if std > 0 else 0
                
                print(f"  {asset_class}:")
                print(f"    Trades: {n}, Total PnL: ${total:,.0f}, Avg: ${avg:.1f}, Sharpe (raw): {sharpe:.3f}")
        
        return trades_df
        
    except FileNotFoundError:
        print(f"  [WARN] No backtest_results.csv found. Run a backtest first.")
        return None
    except Exception as e:
        print(f"  [ERROR] {e}")
        return None


def main():
    print("\n" + "="*70)
    print(" 🔬 SHARPE RATIO ERROR DIAGNOSTIC SUITE")
    print(" Identifying calculation errors in 2.18 Sharpe result")
    print("="*70)
    
    results = {}
    
    # Run diagnostics
    results['annualization'] = diagnose_annualization_factor()
    results['transaction_costs'] = diagnose_transaction_costs()
    results['feature_leakage'] = diagnose_feature_leakage()
    results['trades'] = diagnose_recent_trades()
    
    # Summary
    print("\n" + "="*70)
    print(" 📋 DIAGNOSTIC SUMMARY")
    print("="*70)
    
    print("\n  🚨 CRITICAL ISSUES FOUND:")
    
    if results.get('annualization'):
        K_current = results['annualization']['K_current']
        K_fx = results['annualization']['K_fx']
        if K_fx > K_current * 1.5:
            print(f"  1. [SHARPE K-FACTOR] Using K={K_current}, should be K={K_fx:.0f} for FX")
            print(f"     → Sharpe is INFLATED by ~{np.sqrt(K_fx/K_current):.1f}x")
    
    if results.get('transaction_costs'):
        factor = results['transaction_costs']['underestimation_factor']
        if factor > 100:
            print(f"  2. [TRANSACTION COSTS] Underestimated by ~{factor:,.0f}x")
            print(f"     → Profits are MASSIVELY overstated")
    
    print("\n  💡 RECOMMENDED FIXES:")
    print("  1. Calculate Sharpe separately for FX (K≈6000) vs Equities (K≈1700)")
    print("  2. Fix transaction cost: cost = notional * spread * 2 (round-trip)")
    print("  3. Re-run backtest with realistic parameters")
    
    print("\n" + "="*70)
    print(" DIAGNOSTIC COMPLETE")
    print("="*70)
    
    return results


if __name__ == "__main__":
    main()
