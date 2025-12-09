"""
Institutional Exit Framework Comparison
=======================================
Test 4 variants:
A) Baseline: Current 2× ATR SL + Chandelier
B) Wide SL: 5× ATR emergency SL + Chandelier  
C) Signal-Driven: 5× ATR + Signal decay exit
D) Pure Time: No SL, only exit at max hold time
"""

import sys
import os

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List
from quant_backtest import (
    Config, DataLoader, FeatureEngine, RegimeEngine, 
    AlphaEngine, EnsembleSignal, CrisisAlphaEngine
)

@dataclass
class ExitVariant:
    name: str
    emergency_sl_mult: float = 5.0
    use_signal_decay: bool = False
    signal_decay_threshold: float = 0.45
    use_chandelier: bool = True
    chandelier_mult: float = 3.0
    max_bars: int = 30


class InstitutionalBacktester:
    """Backtester with configurable institutional-style exits."""
    
    def __init__(self, config: Config, variant: ExitVariant):
        self.config = config
        self.variant = variant
        self.balance = config.initial_balance
        self.positions = []
        self.trade_history = []
        
    def run_backtest(self, data: Dict[str, pd.DataFrame]):
        # Get combined timeline
        combined_index = pd.Index([])
        for df in data.values():
            combined_index = combined_index.union(df.index)
        combined_index = combined_index.sort_values()
        
        for current_time in combined_index:
            # === EXITS ===
            for pos in list(self.positions):
                if pos['symbol'] not in data:
                    continue
                df = data[pos['symbol']]
                if current_time not in df.index:
                    continue
                    
                row = df.loc[current_time]
                current_price = row['Close']
                atr = row.get('ATR', 0.001)
                
                # Calculate P/L
                pnl = (current_price - pos['entry_price']) * pos['direction']
                pnl_atr = pnl / atr if atr > 0 else 0
                bars_held = (current_time - pos['entry_time']).total_seconds() / 3600
                
                exit_reason = None
                exit_price = current_price
                
                # 1. Emergency SL (Catastrophe Protection)
                if pnl_atr < -self.variant.emergency_sl_mult:
                    exit_reason = 'EmergencySL'
                    exit_price = pos['entry_price'] - (atr * self.variant.emergency_sl_mult * pos['direction'])
                
                # 2. Signal Decay Exit (Medallion-style)
                elif self.variant.use_signal_decay:
                    current_signal = row.get('Final_Signal', row.get('Ensemble_Score', 0))
                    
                    # Signal flipped
                    if current_signal * pos['direction'] < -0.1:
                        exit_reason = 'SignalFlip'
                    # Signal decayed below threshold
                    elif abs(current_signal) < self.variant.signal_decay_threshold:
                        exit_reason = 'SignalDecay'
                
                # 3. Chandelier Exit
                elif self.variant.use_chandelier:
                    if pos['direction'] == 1:
                        pos['trailing_high'] = max(pos['trailing_high'], row['High'])
                        chandelier_stop = pos['trailing_high'] - (atr * self.variant.chandelier_mult)
                        if row['Low'] <= chandelier_stop:
                            exit_reason = 'Chandelier'
                            exit_price = chandelier_stop
                    else:
                        pos['trailing_low'] = min(pos['trailing_low'], row['Low'])
                        chandelier_stop = pos['trailing_low'] + (atr * self.variant.chandelier_mult)
                        if row['High'] >= chandelier_stop:
                            exit_reason = 'Chandelier'
                            exit_price = chandelier_stop
                
                # 4. Time Exit
                if exit_reason is None and bars_held >= self.variant.max_bars:
                    exit_reason = 'TimeExit'
                
                # Execute exit
                if exit_reason:
                    trade_pnl = (exit_price - pos['entry_price']) * pos['direction'] * pos['size']
                    r_mult = (exit_price - pos['entry_price']) * pos['direction'] / pos['sl_dist'] if pos['sl_dist'] > 0 else 0
                    
                    self.balance += trade_pnl
                    self.trade_history.append({
                        'Symbol': pos['symbol'],
                        'Direction': pos['direction'],
                        'Entry': pos['entry_price'],
                        'Exit': exit_price,
                        'PnL': trade_pnl,
                        'R_Multiple': r_mult,
                        'Bars': bars_held,
                        'Reason': exit_reason
                    })
                    self.positions.remove(pos)
            
            # === ENTRIES ===
            for sym, df in data.items():
                if current_time not in df.index:
                    continue
                if len([p for p in self.positions if p['symbol'] == sym]) > 0:
                    continue
                if len(self.positions) >= self.config.max_concurrent_trades:
                    break
                    
                row = df.loc[current_time]
                signal = row.get('Final_Signal', row.get('Ensemble_Score', 0))
                
                if abs(signal) < self.config.ensemble_threshold:
                    continue
                    
                direction = 1 if signal > 0 else -1
                entry_price = row['Close']
                atr = row.get('ATR', 0.001)
                
                # Skip HIGH vol (kept from council recommendation)
                vol_regime = row.get('Vol_Regime', 'NORMAL')
                if vol_regime == 'HIGH':
                    continue
                    
                # Skip BULL trend (existing filter)
                trend_regime = row.get('Trend_Regime', 'RANGE')
                if trend_regime == 'BULL':
                    continue
                
                sl_dist = atr * 2.0  # For R calculation
                size = (self.balance * self.config.risk_per_trade) / sl_dist
                
                self.positions.append({
                    'symbol': sym,
                    'direction': direction,
                    'entry_price': entry_price,
                    'entry_time': current_time,
                    'size': size,
                    'sl_dist': sl_dist,
                    'trailing_high': row['High'],
                    'trailing_low': row['Low']
                })
        
        return pd.DataFrame(self.trade_history)


def run_comparison():
    print("=" * 70)
    print("INSTITUTIONAL EXIT FRAMEWORK COMPARISON")
    print("=" * 70)
    
    # Define variants
    variants = [
        ExitVariant(name="A_Baseline", emergency_sl_mult=2.0, use_chandelier=True),
        ExitVariant(name="B_WideSL", emergency_sl_mult=5.0, use_chandelier=True),
        ExitVariant(name="C_SignalDriven", emergency_sl_mult=5.0, use_signal_decay=True, use_chandelier=False),
        ExitVariant(name="D_PureTime", emergency_sl_mult=999.0, use_chandelier=False),
    ]
    
    # Load data
    config = Config()
    config.symbols = config.symbols[:10]
    
    loader = DataLoader(config)
    data = loader.load_data('2024-01-01', '2024-12-01')
    
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
    
    # Run each variant
    results = []
    
    for variant in variants:
        print(f"\n[Testing: {variant.name}]")
        
        bt = InstitutionalBacktester(config, variant)
        trades = bt.run_backtest(data)
        
        if trades.empty:
            print("  No trades")
            continue
        
        # Calculate metrics
        total_pnl = trades['PnL'].sum()
        mean_r = trades['R_Multiple'].mean()
        win_rate = (trades['PnL'] > 0).mean() * 100
        sharpe = trades['PnL'].mean() / trades['PnL'].std() * np.sqrt(252) if trades['PnL'].std() > 0 else 0
        
        # Exit breakdown
        exit_counts = trades['Reason'].value_counts()
        
        results.append({
            'Variant': variant.name,
            'Trades': len(trades),
            'Total_PnL': total_pnl,
            'Mean_R': mean_r,
            'Win_Rate': win_rate,
            'Sharpe': sharpe,
            'Final_Balance': bt.balance
        })
        
        print(f"  Trades: {len(trades)}, Mean R: {mean_r:.3f}, WR: {win_rate:.1f}%, Sharpe: {sharpe:.2f}")
        print(f"  Exits: {dict(exit_counts)}")
    
    # Summary table
    print("\n" + "=" * 70)
    print("RESULTS COMPARISON")
    print("=" * 70)
    
    df_results = pd.DataFrame(results)
    print(df_results.to_string(index=False))
    
    # Find best
    if not df_results.empty:
        best_sharpe = df_results.loc[df_results['Sharpe'].idxmax()]
        best_r = df_results.loc[df_results['Mean_R'].idxmax()]
        
        print("\n" + "=" * 70)
        print("BEST PERFORMERS")
        print("=" * 70)
        print(f"Best Sharpe: {best_sharpe['Variant']} ({best_sharpe['Sharpe']:.2f})")
        print(f"Best Mean R: {best_r['Variant']} ({best_r['Mean_R']:.3f})")
        print()
        
        # Council recommendation
        print("COUNCIL RECOMMENDATION:")
        if best_sharpe['Variant'] == 'D_PureTime':
            print("  Pure TimeExit wins! This confirms: SL is hurting us.")
            print("  ACTION: Remove SL entirely, exit only on time/signal.")
        elif best_sharpe['Variant'] == 'C_SignalDriven':
            print("  Signal-Driven wins! Medallion approach is correct.")
            print("  ACTION: Implement signal decay exits in production.")
        else:
            print(f"  {best_sharpe['Variant']} wins, but may need more testing.")


if __name__ == "__main__":
    run_comparison()
