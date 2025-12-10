
import pandas as pd
import numpy as np
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Optional, Dict

@dataclass
class Event:
    """
    Represents a single sporting event (e.g., Match).
    """
    id: str
    date: datetime
    team_home: str
    team_away: str
    
    # Odds (Decimal)
    odds_home: float
    odds_draw: float
    odds_away: float
    
    # Model Probabilities (Calculated by Strategy)
    prob_home: float = 0.0
    prob_draw: float = 0.0
    prob_away: float = 0.0
    
    # Result (0=Draw, 1=Home, 2=Away) - Standard 1X2
    # Or just store string result
    result: str = None 
    
    def implied_prob_home(self):
        return 1.0 / self.odds_home if self.odds_home > 0 else 0.0

@dataclass
class Bet:
    """
    Represents a placed bet.
    """
    event_id: str
    selection: str # 'HOME', 'DRAW', 'AWAY'
    odds: float
    stake: float
    model_prob: float
    edge: float
    
    result: str = None # 'WIN', 'LOSS', 'VOID'
    pnl: float = 0.0
    
    def resolve(self, actual_result: str):
        """
        Resolve bet PnL.
        Actual result: 'HOME', 'DRAW', 'AWAY'
        """
        if actual_result == self.selection:
            self.result = 'WIN'
            self.pnl = self.stake * (self.odds - 1.0) # Net Profit
        else:
            self.result = 'LOSS'
            self.pnl = -self.stake

class KellySizer:
    """
    Calculates Optimal Stake using Fractional Kelly Criterion.
    """
    def __init__(self, fraction=0.1, max_stake_pct=0.05):
        self.fraction = fraction
        self.max_stake_pct = max_stake_pct # Cap at 5% of bankroll for safety
        
    def calculate_stake(self, bankroll: float, odds: float, prob: float) -> float:
        """
        Kelly f* = (bp - q) / b
        b = decimal_odds - 1 (Net fractional odds)
        p = probability of win
        q = 1 - p
        """
        if odds <= 1.0: return 0.0
        
        b = odds - 1.0
        p = prob
        q = 1.0 - p
        
        f_star = (b * p - q) / b
        
        if f_star <= 0:
            return 0.0
            
        # Apply Fraction (e.g., Half Kelly)
        f_adj = f_star * self.fraction
        
        # Apply Safety Cap
        if f_adj > self.max_stake_pct:
            f_adj = self.max_stake_pct
            
        return bankroll * f_adj

class SportsBacktester:
    """
    Event-Driven Backtester for Sports.
    """
    def __init__(self, initial_balance=10000.0, sizer: KellySizer = None):
        self.balance = initial_balance
        self.initial_balance = initial_balance
        self.sizer = sizer if sizer else KellySizer()
        
        self.bets: List[Bet] = []
        self.equity_curve = [initial_balance]
        self.trade_log = []
        
    def process_event(self, event: Event):
        """
        Evaluate Event for Value and Place Bet if Edge exists.
        Strategy: Bet on HOME if Edge > threshold.
        (Simplified for now - can expand to 3-way).
        """
        # 1. Simple Value Check for HOME
        implied = event.implied_prob_home()
        model = event.prob_home
        
        edge = model - implied
        
        # Value Threshold (e.g. 2% edge)
        if edge > 0.02:
            # Calculate Stake
            stake = self.sizer.calculate_stake(self.balance, event.odds_home, model)
            
            if stake > 0:
                # Place Bet
                bet = Bet(
                    event_id=event.id,
                    selection='HOME',
                    odds=event.odds_home,
                    stake=stake,
                    model_prob=model,
                    edge=edge
                )
                
                # Resolve immediately (since we iterate history)
                # In real sim, we'd queue bets and resolve later.
                bet.resolve(event.result)
                
                # Update Bankroll
                self.balance += bet.pnl
                self.bets.append(bet)
                
                self.trade_log.append({
                    'Date': event.date,
                    'Match': f"{event.team_home} vs {event.team_away}",
                    'Selection': bet.selection,
                    'Odds': bet.odds,
                    'ModelProb': round(model, 3),
                    'ImpliedProb': round(implied, 3),
                    'Edge': round(edge, 3),
                    'Stake': round(stake, 2),
                    'Result': bet.result,
                    'PnL': round(bet.pnl, 2),
                    'Balance': round(self.balance, 2)
                })
        
        self.equity_curve.append(self.balance)

    def get_metrics(self):
        df_log = pd.DataFrame(self.trade_log)
        if df_log.empty:
            return {"Return": 0.0, "Trades": 0, "ROI": 0.0, "WinRate": 0.0}
            
        total_pnl = self.balance - self.initial_balance
        ret_pct = (total_pnl / self.initial_balance) * 100
        turnover = df_log['Stake'].sum()
        roi = (total_pnl / turnover) * 100 if turnover > 0 else 0.0
        win_rate = (df_log['Result'] == 'WIN').mean() * 100
        
        # Max DD
        equity = np.array(self.equity_curve)
        peak = np.maximum.accumulate(equity)
        dd = (equity - peak) / peak
        max_dd = dd.min() * 100
        
        return {
            "Return": round(ret_pct, 2),
            "MaxDD": round(max_dd, 2),
            "Trades": len(df_log),
            "ROI": round(roi, 2),
            "WinRate": round(win_rate, 2),
            "FinalBalance": round(self.balance, 2)
        }
