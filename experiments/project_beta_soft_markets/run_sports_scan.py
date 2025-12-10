
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

# Adjust path 
sys.path.append(os.path.abspath("."))
sys.path.append(os.path.abspath("experiments/project_beta_soft_markets"))

from sports_engine import Event, SportsBacktester, KellySizer

# --- CONFIG ---
DATA_FILE = "experiments/project_beta_soft_markets/data/EPL_Combined_2021_2024.csv"
TRAIN_WINDOW = 50 # matches per team approx? No, simple rolling window of matches.
KELLEY_FRACTION = 0.05 # Very conservative real world
VALUE_THRESHOLD = 0.02 # 2% Edge

class EloEngine:
    """
    Maintains Elo ratings for teams.
    """
    def __init__(self, k_factor=20, initial_rating=1500):
        self.k_factor = k_factor
        self.ratings = {} # Team -> Rating
        self.initial_rating = initial_rating
        
    def get_rating(self, team):
        return self.ratings.get(team, self.initial_rating)
        
    def update(self, home_team, away_team, result_str):
        # Result: H, D, A
        r_home = self.get_rating(home_team)
        r_away = self.get_rating(away_team)
        
        # Expected Score (Logistic Curve)
        # qa = 10^(Ra/400), qb = 10^(Rb/400)
        # Ea = qa / (qa + qb)
        
        qa = 10**(r_home/400)
        qb = 10**(r_away/400)
        
        e_home = qa / (qa + qb)
        e_away = qb / (qa + qb)
        
        # Actual Score (1=Win, 0.5=Draw, 0=Loss)
        if result_str == 'H':
            s_home = 1.0
            s_away = 0.0
        elif result_str == 'D':
            s_home = 0.5
            s_away = 0.5
        else: # 'A'
            s_home = 0.0
            s_away = 1.0
            
        # Update
        new_r_home = r_home + self.k_factor * (s_home - e_home)
        new_r_away = r_away + self.k_factor * (s_away - e_away)
        
        self.ratings[home_team] = new_r_home
        self.ratings[away_team] = new_r_away
        
    def predict_home_prob(self, home_team, away_team):
        # Return expected home win prob?
        # Elo gives Win Expectancy (Win + 0.5 Draw).
        # We need Prob(Win). 
        # In football, Draw is huge. 
        # Simple proxy: P(Home) approx E(Home) - P(Drawbias).
        # Better: Use Poisson/Rating difference to map to 1X2.
        # Shortcuts for "Viability Scan":
        # Just assume P(HomeWin) = EloExpectancy * 0.7 (Discount for Draw)
        # This is a heuristic.
        
        r_home = self.get_rating(home_team) + 100 # Home Advantage
        r_away = self.get_rating(away_team)
        
        qa = 10**(r_home/400)
        qb = 10**(r_away/400)
        
        e_home = qa / (qa + qb)
        
        # Simple heuristic mapping for 1X2
        # P(Home) ~ E_home * 0.8?
        # Actually, let's just bet if Elo says Home is STRONG favorite and Odds are good.
        return e_home

def run_epl_scan():
    print("--- Running EPL Viability Scan (2021-2024) ---")
    
    # 1. Load Data
    if not os.path.exists(DATA_FILE):
        print(f"Data file not found: {DATA_FILE}")
        return
        
    df = pd.read_csv(DATA_FILE)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    
    print(f"Loaded {len(df)} matches.")
    
    # 2. Setup Engines
    elo = EloEngine(k_factor=20)
    sizer = KellySizer(fraction=KELLEY_FRACTION)
    tester = SportsBacktester(initial_balance=10000, sizer=sizer)
    
    # 3. Iterate (Walk Forward)
    # We update Elo after every match.
    # We bet BEFORE updating.
    
    ignored_bets = 0
    placed_bets = 0
    
    for idx, row in df.iterrows():
        home = row['HomeTeam']
        away = row['AwayTeam']
        result = row['FTR'] # H, D, A
        
        odds_h = row.get('B365H', 0.0)
        odds_d = row.get('B365D', 0.0)
        odds_a = row.get('B365A', 0.0)
        
        if odds_h == 0: continue
        
        # 3a. Predict (Before Match)
        # Get Elo-based "Expectancy" (Win+Draw/2)
        # We need P(Win).
        # Let's simple model: P(Win) = 1 / (1 + 10^((Ra-Rh)/400))
        # But this doesn't account for Draw.
        # Fix: We treat football as 3 outcomes.
        # Let's just use Elo Delta as a feature proxy.
        # If HomeAdvantage_Elo > X, and Odds > Y...
        
        # PROPER APPROACH: Model Probability
        # Let's try a naive mapping:
        # P(Home) = 1 / (1 + 10^(-(RatingDiff + 100)/400))
        # But we must normalize for Draw (~25%).
        # So Normalized P(Home) = Raw_Elo_Prob * (1 - P_Draw).
        # Assume P_Draw = 0.25 constant.
        
        r_h = elo.get_rating(home) + 80 # Home Field Advantage (approx 80 points)
        r_a = elo.get_rating(away)
        
        dr = r_h - r_a
        win_expectancy = 1 / (1 + 10**(-dr/400))
        
        # Adjust for Draw
        # Very rough heuristic
        p_home_model = win_expectancy * 0.75 
        
        # Create Event
        # Convert Result FTR to 'HOME', 'AWAY', 'DRAW' for engine
        res_map = {'H': 'HOME', 'A': 'AWAY', 'D': 'DRAW'}
        
        event = Event(
            id=f"{home}-{away}-{idx}",
            date=row['Date'],
            team_home=home,
            team_away=away,
            odds_home=odds_h,
            odds_draw=odds_d,
            odds_away=odds_a,
            prob_home=p_home_model,
            result=res_map.get(result, 'DRAW')
        )
        
        # 3b. Bet Processing (Engine decides value)
        # Engine bets if p_home_model > 1/odds + margin
        tester.process_event(event)
        
        # 3c. Update Knowledge (After Match)
        elo.update(home, away, result)
        
    # 4. Results
    metrics = tester.get_metrics()
    print(f"\nFinal Metrics: {metrics}")
    
    return metrics

if __name__ == "__main__":
    run_epl_scan()
