
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime, time

# --- Phoenix Engine: FTMO Swing Challenge ---
# A robust, event-based backtester designed for 1:30 leverage and strict risk limits.

@dataclass
class Bar:
    time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0

@dataclass
class Trade:
    id: str
    symbol: str
    direction: str # 'BUY' or 'SELL'
    entry_price: float
    sl: float
    tp: float
    size: float
    entry_time: datetime
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    pnl: float = 0.0
    status: str = 'OPEN' # OPEN, CLOSED
    exit_reason: str = ''

class RiskGuardian:
    """
    Enforces FTMO Constraints:
    1. Max Daily Loss (5% of Initial Balance).
    2. Max Total Drawdown (10% of Initial Balance).
    """
    def __init__(self, initial_balance: float):
        self.initial_balance = initial_balance
        self.max_daily_loss_pct = 0.05
        self.max_total_dd_pct = 0.10
        self.current_daily_loss = 0.0
        self.start_of_day_equity = initial_balance
        self.current_day = None
        self.trading_halted = False

    def on_new_day(self, current_date, current_equity):
        """Reset daily stats."""
        # Check previous day's result? No, just reset baseline.
        self.start_of_day_equity = current_equity
        self.current_daily_loss = 0.0
        self.current_day = current_date
        self.trading_halted = False

    def check_risk(self, current_equity: float, open_pnl: float) -> bool:
        """
        Returns False if risk limits are breached.
        """
        # 1. Total Drawdown
        dd = (self.initial_balance - current_equity) / self.initial_balance
        if dd >= self.max_total_dd_pct:
            return False # Account Blown
            
        # 2. Daily Loss (Equity based)
        # Daily Loss = StartOfDayEquity - CurrentEquity
        daily_loss = self.start_of_day_equity - current_equity
        daily_loss_pct = daily_loss / self.start_of_day_equity
        
        if daily_loss_pct >= self.max_daily_loss_pct:
            self.trading_halted = True
            return False
            
        return True

class PhoenixBacktester:
    def __init__(self, initial_balance=100000, leverage=30, symbol="XAUUSD"):
        self.balance = initial_balance
        self.equity = initial_balance
        self.leverage = leverage
        self.symbol = symbol
        self.guardian = RiskGuardian(initial_balance)
        self.trades: List[Trade] = []
        self.active_trades: List[Trade] = []
        self.equity_curve = []
        
        # Costs (Approx for Gold/Indices)
        # Commission: $7 per lot round trip? Or Spread included?
        # Let's assume standard Spread + $0 Comm for Indices/CFDs usually, or raw.
        # We'll use a transaction cost model.
        self.spread_pips = 2.0 # Standard Gold Spread
        
    def run(self, df: pd.DataFrame, strategy):
        """
        Event-based loop.
        df must have standard OHLC columns and be sorted by Time.
        """
        print(f"Starting Phoenix Backtest on {self.symbol}...")
        
        # Ensure Datetime
        if 'Time' in df.columns:
            df['Time'] = pd.to_datetime(df['Time'])
            
        # Strategy Setup
        strategy.setup(df) 
        
        # Iterate
        for i in range(len(df)):
            row = df.iloc[i]
            current_time = row['Time']
            
            # 1. Update Guardian (New Day Check)
            if self.guardian.current_day != current_time.date():
                self.guardian.on_new_day(current_time.date(), self.equity)
                
            # 2. Update Open Positions (Mark to Market)
            self._update_equity(row['Close'])
            
            # 3. Check Risk Limits
            if not self.guardian.check_risk(self.equity, 0): # 0 because equity already updated
                # Close all trades if daily limit hit? 
                # Ideally yes.
                if self.active_trades:
                    self._close_all(current_time, row['Close'], "RISK_HALT")
                continue # Skip processing
            
            # 4. Process Exits (SL/TP/Trailing)
            self._process_exits(row)
            
            # 5. Get Signal
            if not self.guardian.trading_halted:
                signal = strategy.get_signal(i)
                if signal:
                    self._execute_trade(signal, row)
            
            # Log Equity
            self.equity_curve.append({'Time': current_time, 'Equity': self.equity})
            
        return self._generate_report()

    def _update_equity(self, current_price):
        floating_pnl = 0.0
        for t in self.active_trades:
            if t.direction == 'BUY':
                diff = current_price - t.entry_price
            else:
                diff = t.entry_price - current_price
            
            # Value = Size * Diff
            # For indices/gold, Size usually = Lots. 
            # 1 Lot Gold = 100oz. 1 pip (0.01) = $1.
            # Delta 1.0 = $100 per lot.
            
            # Let's normalize logic:
            # PnL = (Diff / TickSize) * TickValue * Size
            # Simplified: PnL = Diff * Size * Multiplier
            # Gold Multiplier = 100? No, let's assume 'Size' is nominal units (USD).
            # No, MT5 uses Lots. Let's use Units.
            # 1 Unit XAUUSD = Price.
            
            pnl = diff * t.size 
            floating_pnl += pnl
            
        self.equity = self.balance + floating_pnl

    def _process_exits(self, row):
        # Check SL/TP (Intra-bar approximation could use Low/High)
        # Conservative: Check Low for Buy SL, High for Sell SL.
        
        for t in list(self.active_trades):
            closed = False
            exit_px = 0.0
            reason = ""
            
            if t.direction == 'BUY':
                if row['Low'] <= t.sl:
                    closed = True
                    exit_px = t.sl # Slippage ignored for now
                    reason = "SL"
                elif row['High'] >= t.tp and t.tp > 0:
                    closed = True
                    exit_px = t.tp
                    reason = "TP"
            else:
                if row['High'] >= t.sl:
                    closed = True
                    exit_px = t.sl
                    reason = "SL"
                elif row['Low'] <= t.tp and t.tp > 0:
                    closed = True
                    exit_px = t.tp
                    reason = "TP"
                    
            if closed:
                self._close_trade(t, row['Time'], exit_px, reason)

    def _execute_trade(self, signal, row):
        # signal: {'direction': 'BUY', 'sl': 100, 'tp': 200, 'risk_pct': 0.01}
        
        # Calculate Size
        # Risk Amount = Equity * risk_pct
        # Risk Per Unit = |Entry - SL|
        # Units = Risk Amount / Risk Per Unit
        
        risk_amt = self.equity * signal['risk_pct']
        dist = abs(row['Close'] - signal['sl'])
        
        if dist == 0: return
        
        size = risk_amt / dist
        
        # Cap leverage
        max_size = (self.equity * self.leverage) / row['Close']
        size = min(size, max_size)
        
        t = Trade(
            id=f"{len(self.trades)}",
            symbol=self.symbol,
            direction=signal['direction'],
            entry_price=row['Close'],
            sl=signal['sl'],
            tp=signal['tp'],
            size=size,
            entry_time=row['Time']
        )
        self.active_trades.append(t)
        self.trades.append(t)

    def _close_trade(self, trade, time, price, reason):
        trade.exit_time = time
        trade.exit_price = price
        trade.status = 'CLOSED'
        trade.exit_reason = reason
        
        # Calc PnL
        if trade.direction == 'BUY':
            pnl = (price - trade.entry_price) * trade.size
        else:
            pnl = (trade.entry_price - price) * trade.size
            
        trade.pnl = pnl
        self.balance += pnl
        self.active_trades.remove(trade)
    
    def _close_all(self, time, price, reason):
        for t in list(self.active_trades):
            self._close_trade(t, time, price, reason)

    def _generate_report(self):
        df_res = pd.DataFrame([t.__dict__ for t in self.trades if t.status=='CLOSED'])
        if df_res.empty:
            return {'Return': 0.0, 'Trades': 0}
            
        total_pnl = df_res['pnl'].sum()
        ret_pct = (total_pnl / self.guardian.initial_balance) * 100
        
        # Max DD
        eq_curve = pd.DataFrame(self.equity_curve)
        eq_curve['HWM'] = eq_curve['Equity'].cummax()
        eq_curve['DD'] = (eq_curve['HWM'] - eq_curve['Equity']) / eq_curve['HWM']
        max_dd = eq_curve['DD'].max() * 100
        
        return {
            'Return': round(ret_pct, 2),
            'MaxDD': round(max_dd, 2),
            'Trades': len(df_res),
            'WinRate': round(len(df_res[df_res['pnl']>0])/len(df_res)*100, 1),
            'FinalBalance': round(self.balance, 2)
        }
