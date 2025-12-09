"""
Daily Macro Proxy Engine (Family B v2)
======================================
Expands CPI-based signals to daily frequency using macro proxies.

Proxies:
1. TIPS Breakeven (5Y) - Inflation expectations
2. DXY (Dollar Index) - Currency strength (inverse inflation)  
3. Gold Momentum - Real-time inflation indicator

Signal Logic:
- Score = weighted combination of proxy signals
- LONG when inflation rising, SHORT when falling
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from datetime import datetime, timedelta
import os
import sys
import yfinance as yf

# Reuse data infrastructure
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from quant_backtest import DataLoader, Config


@dataclass
class MacroProxyConfig:
    """Configuration for daily macro proxy signals."""
    # Proxies
    symbol_gold: str = "GC=F"       # Gold futures
    symbol_dollar: str = "DX-Y.NYB" # Dollar index
    
    # Signal thresholds
    inflation_long_threshold: float = 0.6   # Score > 0.6 = LONG
    inflation_short_threshold: float = 0.3  # Score < 0.3 = SHORT
    
    # Momentum lookback
    momentum_window: int = 5  # 5-day ROC
    
    # Position sizing
    risk_per_trade: float = 5000.0
    holding_days: int = 5
    
    # Capital
    initial_capital: float = 100000.0


class MacroProxyEngine:
    """Generate daily inflation signals from macro proxies."""
    
    def __init__(self, config: MacroProxyConfig):
        self.config = config
        self.data = {}
        self.signals = None
        self.trades = []
        
    def load_data(self, start_date: str = "2022-01-01", end_date: str = "2024-12-31"):
        """Load price data for all proxies."""
        print(f"[MacroProxy] Loading data from {start_date} to {end_date}...")
        
        # Load Gold
        print(f"  Loading {self.config.symbol_gold}...")
        gold = yf.download(self.config.symbol_gold, start=start_date, end=end_date, progress=False)
        if not gold.empty:
            # Squeeze to Series if needed (yfinance returns DataFrame)
            close_col = gold['Close']
            self.data['gold'] = close_col.squeeze() if hasattr(close_col, 'squeeze') else close_col
            self.data['gold'] = self.data['gold'].dropna()
            print(f"    OK ({len(self.data['gold'])} bars)")
        else:
            print(f"    FAILED - using synthetic")
            dates = pd.date_range(start_date, end_date, freq='B')
            self.data['gold'] = pd.Series(np.cumsum(np.random.randn(len(dates))) + 2000, index=dates)
        
        # Load Dollar Index
        print(f"  Loading {self.config.symbol_dollar}...")
        dxy = yf.download(self.config.symbol_dollar, start=start_date, end=end_date, progress=False)
        if not dxy.empty:
            close_col = dxy['Close']
            self.data['dxy'] = close_col.squeeze() if hasattr(close_col, 'squeeze') else close_col
            self.data['dxy'] = self.data['dxy'].dropna()
            print(f"    OK ({len(self.data['dxy'])} bars)")
        else:
            print(f"    FAILED - using synthetic")
            dates = pd.date_range(start_date, end_date, freq='B')
            self.data['dxy'] = pd.Series(np.cumsum(np.random.randn(len(dates)) * 0.5) + 100, index=dates)
        
        # TIPS Breakeven - use synthetic proxy (Gold-DXY spread as approximation)
        # Real TIPS data requires FRED API for T5YIE series
        print("  Computing TIPS proxy (Gold/DXY spread)...")
        common_idx = self.data['gold'].index.intersection(self.data['dxy'].index)
        gold_norm = self.data['gold'].loc[common_idx] / self.data['gold'].loc[common_idx].iloc[0]
        dxy_norm = self.data['dxy'].loc[common_idx] / self.data['dxy'].loc[common_idx].iloc[0]
        self.data['tips_proxy'] = (gold_norm / dxy_norm) * 2.5  # Scaled to ~breakeven %
        print(f"    OK ({len(self.data['tips_proxy'])} bars)")
        
        return True
    
    def compute_signals(self) -> pd.DataFrame:
        """Compute daily inflation score and signals."""
        print("[MacroProxy] Computing signals...")
        
        window = self.config.momentum_window
        
        # Align all data
        common_idx = self.data['gold'].index.intersection(
            self.data['dxy'].index
        ).intersection(
            self.data['tips_proxy'].index
        )
        
        df = pd.DataFrame(index=common_idx)
        df['gold'] = self.data['gold'].loc[common_idx]
        df['dxy'] = self.data['dxy'].loc[common_idx]
        df['tips'] = self.data['tips_proxy'].loc[common_idx]
        
        # Compute momentum signals
        # Gold: Rising = inflationary
        df['gold_roc'] = df['gold'].pct_change(window)
        df['gold_signal'] = (df['gold_roc'] > 0).astype(float)
        
        # DXY: Falling = inflationary (weak dollar)
        df['dxy_roc'] = df['dxy'].pct_change(window)
        df['dxy_signal'] = (df['dxy_roc'] < 0).astype(float)
        
        # TIPS: High = inflationary expectations
        df['tips_signal'] = (df['tips'] > 2.5).astype(float)
        
        # Combined score (weighted)
        df['inflation_score'] = (
            0.4 * df['tips_signal'] +
            0.3 * df['dxy_signal'] +
            0.3 * df['gold_signal']
        )
        
        # Trading signal
        df['signal'] = 0  # Neutral
        df.loc[df['inflation_score'] > self.config.inflation_long_threshold, 'signal'] = 1   # LONG
        df.loc[df['inflation_score'] < self.config.inflation_short_threshold, 'signal'] = -1  # SHORT
        
        self.signals = df.dropna()
        
        # Stats
        long_days = (df['signal'] == 1).sum()
        short_days = (df['signal'] == -1).sum()
        neutral_days = (df['signal'] == 0).sum()
        
        print(f"  Total Days: {len(df)}")
        print(f"  LONG signals: {long_days} ({long_days/len(df)*100:.1f}%)")
        print(f"  SHORT signals: {short_days} ({short_days/len(df)*100:.1f}%)")
        print(f"  NEUTRAL: {neutral_days} ({neutral_days/len(df)*100:.1f}%)")
        
        return self.signals
    
    def run_backtest(self) -> pd.DataFrame:
        """Backtest the daily macro proxy signals."""
        if self.signals is None:
            self.compute_signals()
        
        print("\n[MacroProxy] Running Backtest...")
        
        capital = self.config.initial_capital
        equity_curve = [capital]
        trades = []
        
        position = 0
        entry_price = 0
        entry_date = None
        hold_count = 0
        
        gold_prices = self.data['gold']
        
        for date, row in self.signals.iterrows():
            current_price = gold_prices.loc[date] if date in gold_prices.index else None
            if current_price is None:
                equity_curve.append(equity_curve[-1])
                continue
            
            # Check if in position
            if position != 0:
                hold_count += 1
                
                # Exit after holding period
                if hold_count >= self.config.holding_days:
                    # Calculate PnL
                    if position == 1:  # Long
                        pnl = (current_price - entry_price) / entry_price * self.config.risk_per_trade
                    else:  # Short
                        pnl = (entry_price - current_price) / entry_price * self.config.risk_per_trade
                    
                    capital += pnl
                    trades.append({
                        'Entry_Date': entry_date,
                        'Exit_Date': date,
                        'Direction': 'LONG' if position == 1 else 'SHORT',
                        'Entry_Price': entry_price,
                        'Exit_Price': current_price,
                        'PnL': pnl,
                        'Score': row['inflation_score']
                    })
                    
                    position = 0
                    hold_count = 0
            
            # Check for new entry (only if flat)
            if position == 0 and row['signal'] != 0:
                position = row['signal']
                entry_price = current_price
                entry_date = date
                hold_count = 0
            
            equity_curve.append(capital)
        
        # Results
        self.trades = pd.DataFrame(trades)
        
        if not self.trades.empty:
            total_pnl = self.trades['PnL'].sum()
            win_rate = len(self.trades[self.trades['PnL'] > 0]) / len(self.trades)
            avg_win = self.trades[self.trades['PnL'] > 0]['PnL'].mean() if len(self.trades[self.trades['PnL'] > 0]) > 0 else 0
            avg_loss = self.trades[self.trades['PnL'] < 0]['PnL'].mean() if len(self.trades[self.trades['PnL'] < 0]) > 0 else 0
            
            print(f"\n[Results]")
            print(f"  Trades: {len(self.trades)}")
            print(f"  Total PnL: ${total_pnl:,.2f}")
            print(f"  Win Rate: {win_rate:.1%}")
            print(f"  Avg Win: ${avg_win:,.2f}")
            print(f"  Avg Loss: ${avg_loss:,.2f}")
            
            # Compute daily returns for correlation analysis
            self.daily_returns = self.trades.set_index('Exit_Date')['PnL'] / self.config.initial_capital
        else:
            print("  No trades generated.")
            self.daily_returns = pd.Series(dtype=float)
        
        return self.trades
    
    def get_daily_returns(self) -> pd.Series:
        """Return daily PnL as returns for correlation analysis."""
        if not hasattr(self, 'daily_returns'):
            self.run_backtest()
        return self.daily_returns


if __name__ == "__main__":
    print("=" * 60)
    print("Daily Macro Proxy Engine - Test Run")
    print("=" * 60)
    
    config = MacroProxyConfig()
    engine = MacroProxyEngine(config)
    
    engine.load_data("2022-01-01", "2024-12-31")
    engine.compute_signals()
    trades = engine.run_backtest()
    
    if not trades.empty:
        print("\n[Sample Trades]")
        print(trades.head(10).to_string())
