"""
EXP007: Dynamic Exit Framework Research
========================================
Compare different exit strategies on historical trades.

Frameworks:
1. Baseline (Current): Exit on signal reversal or time
2. ATR Trailing Stop (2x, 3x)
3. Chandelier Exit
4. Regime-Adaptive Trail
5. 1R Profit Lock
6. Hybrid (Regime + Profit Lock)
"""

import sys
import os

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from quant_backtest import (
    Config, DataLoader, FeatureEngine, RegimeEngine, 
    AlphaEngine, EnsembleSignal
)
from dataclasses import dataclass
from typing import Dict, List, Tuple

@dataclass
class ExitConfig:
    """Configuration for exit strategy testing."""
    name: str
    atr_trail_mult: float = 2.0       # For ATR trailing
    chandelier_lookback: int = 22      # For chandelier
    profit_lock_r: float = 1.0         # Move to breakeven after 1R
    regime_trail_low: float = 3.0      # Wider trail in low vol
    regime_trail_normal: float = 2.0   # Normal trail
    regime_trail_high: float = 1.5     # Tighter trail in high vol
    max_bars_hold: int = 30            # Max hold time
    use_profit_lock: bool = False
    use_regime_adaptive: bool = False


class DynamicExitSimulator:
    """Simulate different exit strategies on historical data."""
    
    def __init__(self, data: Dict[str, pd.DataFrame], config: Config):
        self.data = data
        self.config = config
        
    def compute_atr_trail_stop(self, df: pd.DataFrame, entry_price: float, 
                                direction: int, entry_idx: int, 
                                atr_mult: float) -> Tuple[float, int, str]:
        """
        ATR Trailing Stop Exit.
        Returns: (exit_price, exit_idx, reason)
        """
        trail_stop = 0.0
        
        for i in range(entry_idx + 1, min(entry_idx + 30, len(df))):
            row = df.iloc[i]
            price = row['Close']
            atr = row['ATR']
            
            # Update trailing stop
            if direction == 1:  # Long
                new_trail = price - atr * atr_mult
                trail_stop = max(trail_stop, new_trail) if trail_stop > 0 else new_trail
                
                # Check stop hit
                if row['Low'] <= trail_stop:
                    return (trail_stop, i, 'ATR_TRAIL')
                    
            else:  # Short
                new_trail = price + atr * atr_mult
                trail_stop = min(trail_stop, new_trail) if trail_stop > 0 else new_trail
                
                if row['High'] >= trail_stop:
                    return (trail_stop, i, 'ATR_TRAIL')
        
        # Max bars exit
        exit_idx = min(entry_idx + 29, len(df) - 1)
        return (df.iloc[exit_idx]['Close'], exit_idx, 'TIME')
    
    def compute_chandelier_exit(self, df: pd.DataFrame, entry_price: float,
                                  direction: int, entry_idx: int,
                                  lookback: int = 22, atr_mult: float = 3.0) -> Tuple[float, int, str]:
        """
        Chandelier Exit: Trail from highest high (long) or lowest low (short).
        """
        for i in range(entry_idx + 1, min(entry_idx + 30, len(df))):
            row = df.iloc[i]
            atr = row['ATR']
            
            # Get rolling high/low
            start_idx = max(0, i - lookback)
            window = df.iloc[start_idx:i+1]
            
            if direction == 1:  # Long
                highest_high = window['High'].max()
                chandelier_stop = highest_high - atr * atr_mult
                
                if row['Low'] <= chandelier_stop:
                    return (chandelier_stop, i, 'CHANDELIER')
            else:  # Short
                lowest_low = window['Low'].min()
                chandelier_stop = lowest_low + atr * atr_mult
                
                if row['High'] >= chandelier_stop:
                    return (chandelier_stop, i, 'CHANDELIER')
        
        exit_idx = min(entry_idx + 29, len(df) - 1)
        return (df.iloc[exit_idx]['Close'], exit_idx, 'TIME')
    
    def compute_regime_adaptive_exit(self, df: pd.DataFrame, entry_price: float,
                                       direction: int, entry_idx: int,
                                       exit_config: ExitConfig) -> Tuple[float, int, str]:
        """
        Regime-Adaptive Exit: Adjust trail based on volatility regime.
        - LOW vol: Wide trail (3x ATR) - let winners run
        - NORMAL: Standard trail (2x ATR)
        - HIGH vol: Tight trail (1.5x ATR) - take profits fast
        """
        trail_stop = 0.0
        sl_distance = df.iloc[entry_idx]['ATR'] * 2.0  # Initial SL for 1R calc
        
        for i in range(entry_idx + 1, min(entry_idx + 30, len(df))):
            row = df.iloc[i]
            price = row['Close']
            atr = row['ATR']
            vol_regime = row.get('Vol_Regime', 'NORMAL')
            
            # Select trail multiplier based on regime
            if vol_regime == 'LOW':
                mult = exit_config.regime_trail_low
            elif vol_regime == 'HIGH':
                mult = exit_config.regime_trail_high
            else:
                mult = exit_config.regime_trail_normal
            
            # Profit Lock: After 1R, move to breakeven
            current_pnl = (price - entry_price) * direction
            if exit_config.use_profit_lock and current_pnl >= sl_distance * exit_config.profit_lock_r:
                # Lock in profit - trail can't go below entry
                if direction == 1:
                    trail_stop = max(trail_stop, entry_price)
                else:
                    trail_stop = min(trail_stop, entry_price) if trail_stop > 0 else entry_price
            
            # Update trailing stop
            if direction == 1:
                new_trail = price - atr * mult
                trail_stop = max(trail_stop, new_trail) if trail_stop > 0 else new_trail
                
                if row['Low'] <= trail_stop:
                    return (trail_stop, i, 'REGIME_TRAIL')
            else:
                new_trail = price + atr * mult
                # For short, we want higher stop (closer to current price)
                if trail_stop == 0:
                    trail_stop = new_trail
                else:
                    trail_stop = min(trail_stop, new_trail)
                
                if row['High'] >= trail_stop:
                    return (trail_stop, i, 'REGIME_TRAIL')
        
        exit_idx = min(entry_idx + 29, len(df) - 1)
        return (df.iloc[exit_idx]['Close'], exit_idx, 'TIME')
    
    def simulate_exits(self, exit_config: ExitConfig) -> pd.DataFrame:
        """
        Simulate trades using specified exit strategy.
        """
        results = []
        
        for sym, df in self.data.items():
            if 'S_Alpha' not in df.columns and 'Final_Signal' not in df.columns:
                continue
            
            signal_col = 'S_Alpha' if 'S_Alpha' in df.columns else 'Final_Signal'
            
            in_trade = False
            entry_price = 0
            entry_idx = 0
            direction = 0
            entry_atr = 0
            
            for i in range(len(df)):
                row = df.iloc[i]
                signal = row.get(signal_col, 0)
                
                if not in_trade and signal != 0:
                    # Enter trade
                    in_trade = True
                    direction = int(signal)
                    entry_price = row['Close']
                    entry_idx = i
                    entry_atr = row['ATR']
                    sl_distance = entry_atr * 2.0  # Standard 2x ATR SL
                    
                elif in_trade:
                    # Check exit based on strategy
                    if exit_config.name == 'Baseline':
                        # Exit on signal change or max bars
                        if signal != direction or (i - entry_idx) >= exit_config.max_bars_hold:
                            exit_price = row['Close']
                            pnl = (exit_price - entry_price) * direction
                            r_mult = pnl / sl_distance if sl_distance > 0 else 0
                            
                            results.append({
                                'Symbol': sym,
                                'Direction': direction,
                                'Entry': entry_price,
                                'Exit': exit_price,
                                'PnL': pnl,
                                'R': r_mult,
                                'Bars': i - entry_idx,
                                'Reason': 'SIGNAL' if signal != direction else 'TIME'
                            })
                            in_trade = False
                            
                    elif exit_config.name == 'ATR_Trail_2x':
                        exit_price, exit_idx, reason = self.compute_atr_trail_stop(
                            df, entry_price, direction, entry_idx, 2.0)
                        if exit_idx > entry_idx:
                            pnl = (exit_price - entry_price) * direction
                            r_mult = pnl / sl_distance if sl_distance > 0 else 0
                            results.append({
                                'Symbol': sym, 'Direction': direction,
                                'Entry': entry_price, 'Exit': exit_price,
                                'PnL': pnl, 'R': r_mult,
                                'Bars': exit_idx - entry_idx, 'Reason': reason
                            })
                            in_trade = False
                            # Skip to exit_idx
                            continue
                            
                    elif exit_config.name == 'ATR_Trail_3x':
                        exit_price, exit_idx, reason = self.compute_atr_trail_stop(
                            df, entry_price, direction, entry_idx, 3.0)
                        if exit_idx > entry_idx:
                            pnl = (exit_price - entry_price) * direction
                            r_mult = pnl / sl_distance if sl_distance > 0 else 0
                            results.append({
                                'Symbol': sym, 'Direction': direction,
                                'Entry': entry_price, 'Exit': exit_price,
                                'PnL': pnl, 'R': r_mult,
                                'Bars': exit_idx - entry_idx, 'Reason': reason
                            })
                            in_trade = False
                            
                    elif exit_config.name == 'Chandelier':
                        exit_price, exit_idx, reason = self.compute_chandelier_exit(
                            df, entry_price, direction, entry_idx)
                        if exit_idx > entry_idx:
                            pnl = (exit_price - entry_price) * direction
                            r_mult = pnl / sl_distance if sl_distance > 0 else 0
                            results.append({
                                'Symbol': sym, 'Direction': direction,
                                'Entry': entry_price, 'Exit': exit_price,
                                'PnL': pnl, 'R': r_mult,
                                'Bars': exit_idx - entry_idx, 'Reason': reason
                            })
                            in_trade = False
                            
                    elif exit_config.name in ['Regime_Adaptive', 'Hybrid']:
                        exit_price, exit_idx, reason = self.compute_regime_adaptive_exit(
                            df, entry_price, direction, entry_idx, exit_config)
                        if exit_idx > entry_idx:
                            pnl = (exit_price - entry_price) * direction
                            r_mult = pnl / sl_distance if sl_distance > 0 else 0
                            results.append({
                                'Symbol': sym, 'Direction': direction,
                                'Entry': entry_price, 'Exit': exit_price,
                                'PnL': pnl, 'R': r_mult,
                                'Bars': exit_idx - entry_idx, 'Reason': reason
                            })
                            in_trade = False
        
        return pd.DataFrame(results)


def run_exit_comparison():
    print("=" * 60)
    print("EXP007: Dynamic Exit Framework Research")
    print("=" * 60)
    
    # Load data
    config = Config()
    config.symbols = config.symbols[:10]  # Limit for speed
    
    loader = DataLoader(config)
    data = loader.load_data("2024-01-01", "2024-12-01")
    
    # Add features and signals
    fe = FeatureEngine(config)
    re = RegimeEngine(config)
    ae = AlphaEngine(config)
    es = EnsembleSignal(config)
    
    data = fe.add_features_all(data)
    data = re.add_regimes_all(data)
    ae.train_model(data)
    data = ae.add_signals_all(data)
    data = es.add_ensemble_all(data)
    
    # Define exit strategies to test
    strategies = [
        ExitConfig(name='Baseline'),
        ExitConfig(name='ATR_Trail_2x', atr_trail_mult=2.0),
        ExitConfig(name='ATR_Trail_3x', atr_trail_mult=3.0),
        ExitConfig(name='Chandelier', chandelier_lookback=22),
        ExitConfig(name='Regime_Adaptive', use_regime_adaptive=True),
        ExitConfig(name='Hybrid', use_regime_adaptive=True, use_profit_lock=True, profit_lock_r=1.0),
    ]
    
    simulator = DynamicExitSimulator(data, config)
    
    # Run simulations
    results = []
    print("\n[Running Exit Strategy Simulations]")
    print("-" * 60)
    
    for strategy in strategies:
        trades = simulator.simulate_exits(strategy)
        
        if trades.empty:
            print(f"  {strategy.name}: No trades")
            continue
        
        # Calculate metrics
        total_pnl = trades['PnL'].sum()
        mean_r = trades['R'].mean()
        win_rate = len(trades[trades['PnL'] > 0]) / len(trades) * 100
        avg_bars = trades['Bars'].mean()
        
        # Sharpe approximation (assuming daily)
        daily_pnl = trades['PnL']
        sharpe = (daily_pnl.mean() / daily_pnl.std()) * np.sqrt(252) if daily_pnl.std() > 0 else 0
        
        # Profit Factor
        wins = trades[trades['PnL'] > 0]['PnL'].sum()
        losses = abs(trades[trades['PnL'] < 0]['PnL'].sum())
        profit_factor = wins / losses if losses > 0 else 0
        
        results.append({
            'Strategy': strategy.name,
            'Trades': len(trades),
            'Total_PnL': total_pnl,
            'Mean_R': mean_r,
            'Win_Rate': win_rate,
            'Avg_Bars': avg_bars,
            'Sharpe': sharpe,
            'Profit_Factor': profit_factor
        })
        
        print(f"  {strategy.name}: {len(trades)} trades, "
              f"R={mean_r:.3f}, WR={win_rate:.1f}%, PF={profit_factor:.2f}")
    
    # Results table
    print("\n" + "=" * 60)
    print("RESULTS COMPARISON")
    print("=" * 60)
    
    df_results = pd.DataFrame(results)
    print(df_results.to_string(index=False))
    
    # Find best
    if not df_results.empty:
        best_sharpe = df_results.loc[df_results['Sharpe'].idxmax()]
        best_r = df_results.loc[df_results['Mean_R'].idxmax()]
        best_pf = df_results.loc[df_results['Profit_Factor'].idxmax()]
        
        print("\n" + "=" * 60)
        print("BEST PERFORMERS")
        print("=" * 60)
        print(f"Best Sharpe: {best_sharpe['Strategy']} ({best_sharpe['Sharpe']:.3f})")
        print(f"Best Mean R: {best_r['Strategy']} ({best_r['Mean_R']:.3f})")
        print(f"Best Profit Factor: {best_pf['Strategy']} ({best_pf['Profit_Factor']:.2f})")
    
    # Save report
    report_path = os.path.join(os.path.dirname(__file__), "../../artifacts/exp_dynamic_exits_results.md")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    report = f"""# EXP007: Dynamic Exit Framework Research

## Objective
Compare different exit strategies to find optimal profit capture.

## Strategies Tested

| Strategy | Description |
|----------|-------------|
| Baseline | Exit on signal reversal or max time |
| ATR_Trail_2x | Trailing stop at 2x ATR |
| ATR_Trail_3x | Trailing stop at 3x ATR |
| Chandelier | Trail from highest high - 3x ATR |
| Regime_Adaptive | Adjust trail by volatility regime |
| Hybrid | Regime + 1R Profit Lock |

## Results

{df_results.to_markdown(index=False) if not df_results.empty else "No results"}

## Best Performers

- **Best Sharpe**: {best_sharpe['Strategy'] if not df_results.empty else 'N/A'} 
- **Best Mean R**: {best_r['Strategy'] if not df_results.empty else 'N/A'}
- **Best Profit Factor**: {best_pf['Strategy'] if not df_results.empty else 'N/A'}

## Recommendation

Based on results, implement the winner in `quant_backtest.py`.
"""
    
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    run_exit_comparison()
