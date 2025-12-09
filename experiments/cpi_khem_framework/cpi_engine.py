"""
CPI Framework Engine (Family B)
===============================
This module implements the "Event-Driven CPI Convexity Overlay".
It is ISOLATED from the main ML Alpha Engine.

Core Components:
1. CPIConfig: Parameters for the strategy.
2. CPIWindow: Represents a single active trade window [t_m, t_m + h].
3. CPIEngine: Orchestrates data loading, trigger detection, and PnL simulation.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import os
import sys

# Hack to import DataLoader from parent
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from quant_backtest import DataLoader, Config # Reuse data loader only

@dataclass
class CPIConfig:
    horizon_days: int = 5          # Holding period (h)
    risk_cap_per_trade: float = 5000.0 # $ Risk per event
    symbol_equity: str = "ES=F" # Equity Proxy (ES=F or US500)
    symbol_inflation: str = "GC=F" # Inflation Proxy (Gold)
    trigger_lookback: int = 3      # Look at last 3 prints
    trigger_threshold: int = 2     # Need 2/3 Accelerations
    initial_capital: float = 100000.0

@dataclass
class CPIWindow:
    event_date: pd.Timestamp
    entry_price: float
    strike_price: float
    direction: int # 1 = Call (Long Vol/Inf), -1 = Put
    size: float
    status: str = "ACTIVE" # ACTIVE, EXPIRED
    pnl: float = 0.0
    exit_date: pd.Timestamp = None
    exit_price: float = None

class CPIEngine:
    def __init__(self, config: CPIConfig):
        self.config = config
        self.windows = [] # History of windows
        self.active_windows = []
        self.capital = config.initial_capital
        self.equity_curve = []
        
        # Data
        self.cpi_data = None
        self.price_data = {}

    def load_data(self):
        """Load CPI Events and Price Data (Read-Only)"""
        # 1. Load CPI Events
        csv_path = os.path.join(os.path.dirname(__file__), "cpi_events.csv")
        self.cpi_data = pd.read_csv(csv_path)
        self.cpi_data['Date'] = pd.to_datetime(self.cpi_data['Date']).dt.tz_localize(None) # Align TZ
        
        # 2. Load Price Data via Shared Loader
        print("[CPI Engine] Loading Price Data...")
        loader_config = Config() # Generic config
        loader_config.symbols = [self.config.symbol_equity, self.config.symbol_inflation]
        loader = DataLoader(loader_config)
        
        # Load enough history
        start_date = (self.cpi_data['Date'].min() - timedelta(days=30)).strftime("%Y-%m-%d")
        end_date = (self.cpi_data['Date'].max() + timedelta(days=30)).strftime("%Y-%m-%d")
        
        self.price_data = loader.load_data(start_date, end_date)
        
        # Clean TZ
        for s in self.price_data:
            self.price_data[s].index = self.price_data[s].index.tz_localize(None)

    def check_trigger(self, current_date, row_idx):
        """
        Check triggers based on Khem Kapital logic (Acceleration).
        A_m >= 2 means 2 out of last 3 prints were 'Acceleration'.
        Acceleration: Actual > Forecast OR Actual > Previous.
        """
        if row_idx < self.config.trigger_lookback:
            return False
            
        acceleration_count = 0
        
        # Look at last N events (inclusive of current?)
        # Logic: We trade ON the release day if the trend IS accelerating?
        # Or we trade based on PAST 3 prints?
        # Specification says "We track last 3 prints. If 2/3 show Acceleration... Action: Enter".
        # Assuming we know the CURRENT print (Instant Reaction) or Pre-Position?
        # "Event-Driven" usually implies Pre-Position or Instant.
        # Let's assume we use Previous 3 *before* today to predict today? 
        # Or include today? 
        # "A_m (two-out-of-three acceleration rule)" usually implies the *environment* is inflationary.
        # Let's use [current, prev, prev-1].
        
        subset = self.cpi_data.iloc[row_idx-self.config.trigger_lookback+1 : row_idx+1]
        
        for _, event in subset.iterrows():
            is_accel = (event['Actual'] > event['Forecast']) or (event['Actual'] > event['Previous'])
            if is_accel:
                acceleration_count += 1
                
        return acceleration_count >= self.config.trigger_threshold

    def run_backtest(self):
        print(f"[CPI Engine] Running Backtest on {len(self.cpi_data)} Events...")
        
        # We simulate day-by-day or event-by-event?
        # Event-by-event is faster for "Sandbox".
        
        inflation_symbol = self.config.symbol_inflation
        if inflation_symbol not in self.price_data:
            print(f"Error: {inflation_symbol} not found in price data.")
            return

        prices = self.price_data[inflation_symbol]
        
        daily_equity = pd.Series(index=prices.index, data=self.config.initial_capital)
        current_eq = self.config.initial_capital
        
        # Event Loop
        for i, event in self.cpi_data.iterrows():
            event_date = event['Date']
            
            # Check Trigger
            if self.check_trigger(event_date, i):
                print(f"  [TRIGGER] {event_date.date()} | Accel Mode | Buying {inflation_symbol}")
                
                # Market Data at Event
                try:
                    # Find closest close price
                    idx = prices.index.get_indexer([event_date], method='nearest')[0]
                    entry_price = prices.iloc[idx]['Close']
                    entry_date = prices.index[idx]
                    
                    # Calculate Volatility for Strike (Logic: K = F * e^(alpha * vol))
                    # For simple overlay, let's just use Delta 1 (Linear) for now, capped.
                    # Or replicate Call Option?
                    # Payoff = max(0, S_end - K).
                    # Strike K = Entry Price (ATM) or 1 Sigma OTM?
                    # Spec: "Volatility-adapted strikes". Let's use ATM for simplicity in V1 (Pure Direction).
                    
                    strike = entry_price # ATM Call
                    
                    # Size
                    # Fixed Risk Amount. How many contracts?
                    # If we buy Options, Risk = Premium Paid.
                    # We are trading Futures/CFDs.
                    # So we need a Stop Loss to define Risk.
                    # Stop Loss = 2 * Daily Vol?
                    
                    daily_vol = prices.iloc[idx-20:idx]['Close'].pct_change().std()
                    stop_dist = entry_price * daily_vol * 2.0
                    risk_per_unit = stop_dist
                    units = self.config.risk_cap_per_trade / risk_per_unit if risk_per_unit > 0 else 0
                    
                    # Valid Window?
                    end_idx = min(idx + self.config.horizon_days, len(prices)-1)
                    exit_price = prices.iloc[end_idx]['Close']
                    exit_date = prices.index[end_idx]
                    
                    # PnL Calculation (Linear for now, representing Delta-Hedged Option or just Directional Trade)
                    # "Client Payoff = max(0, Return - Strike)"
                    # This implies we BUY options.
                    # Cost = Premium. (approx 40% of risk cap?)
                    # Let's calculate theoretical Option Payoff.
                    
                    # Intrinsic Value at Expiry
                    intrinsic = max(0, exit_price - strike)
                    
                    # PnL = (Intrinsic - Cost).
                    # Cost of ATM Call ~ 0.8 * Vol * Sqrt(T).
                    # Approximation: Cost = 0.4 * daily_vol * sqrt(h) * S * Units?
                    # Let's simplify: We go LONG Futures.
                    # PnL = (Exit - Entry) * Units.
                    # But "Convexity" means we act like an Option?
                    # Okay, let's simulate the Payout only.
                    # Payout = Intrinsic * Units.
                    # Cost = Premium paid (sunk cost from capital).
                    
                    premium = 0.05 * self.config.risk_cap_per_trade # Mock Premium
                    current_eq -= premium # Pay premium
                    
                    payout = intrinsic * units
                    current_eq += payout
                    
                    self.windows.append({
                        'Date': event_date,
                        'Entry': entry_price,
                        'Exit': exit_price,
                        'Strike': strike,
                        'Units': units,
                        'Premium': premium,
                        'Payout': payout,
                        'Net_PnL': payout - premium
                    })
                    
                except Exception as e:
                    print(f"    Error processing event: {e}")
            else:
                # No Trigger
                pass

        # Summary
        print("\n[CPI Backtest Complete]")
        df_res = pd.DataFrame(self.windows)
        if not df_res.empty:
            print(f"  Trades: {len(df_res)}")
            print(f"  Total PnL: ${df_res['Net_PnL'].sum():.2f}")
            print(f"  Win Rate: {len(df_res[df_res['Net_PnL']>0]) / len(df_res):.2%}")
        else:
            print("  No trades triggered.")
            
        return df_res

if __name__ == "__main__":
    # Test Run
    cfg = CPIConfig()
    engine = CPIEngine(cfg)
    engine.load_data()
    engine.run_backtest()
