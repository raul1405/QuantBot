
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
DATA_FILE = "experiments/project_beta_soft_markets/data/NBA_Combined_2022_2024.csv"
KELLEY_FRACTION = 0.05
VALUE_THRESHOLD = 0.02

class EloEngineNBA:
    """
    Elo for 2-Way Market.
    """
    def __init__(self, k_factor=20, initial_rating=1500):
        self.k_factor = k_factor
        self.ratings = {} 
        self.initial_rating = initial_rating
        
    def get_rating(self, team):
        return self.ratings.get(team, self.initial_rating)
        
    def update(self, home, away, result_str):
        # Result: H or A (No Draw)
        r_home = self.get_rating(home)
        r_away = self.get_rating(away)
        
        # Home Adv
        r_home_adj = r_home + 100 
        
        qa = 10**(r_home_adj/400)
        qb = 10**(r_away/400)
        
        e_home = qa / (qa + qb)
        
        s_home = 1.0 if result_str == 'HOME' else 0.0
        
        # Update
        new_r_home = r_home + self.k_factor * (s_home - e_home)
        new_r_away = r_away + self.k_factor * ((1-s_home) - (1-e_home))
        
        self.ratings[home] = new_r_home
        self.ratings[away] = new_r_away
        
    def predict_home_prob(self, home, away):
        r_h = self.get_rating(home) + 100 
        r_a = self.get_rating(away)
        dr = r_h - r_a
        return 1 / (1 + 10**(-dr/400))

def run_nba_scan():
    print("--- Running NBA Viability Scan (Synthetic/Real) ---")
    
    if not os.path.exists(DATA_FILE):
        print(f"Data file not found: {DATA_FILE}")
        return
        
    df = pd.read_csv(DATA_FILE)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    
    print(f"Loaded {len(df)} matches.")
    
    elo = EloEngineNBA(k_factor=20)
    sizer = KellySizer(fraction=KELLEY_FRACTION)
    tester = SportsBacktester(initial_balance=10000, sizer=sizer)
    
    for idx, row in df.iterrows():
        home = row['Home']
        away = row['Away']
        score_h = row['ScoreH']
        score_a = row['ScoreA']
        
        res = 'HOME' if score_h > score_a else 'AWAY'
        
        odds_h = row.get('ML_Home', 0.0)
        odds_a = row.get('ML_Away', 0.0)
        
        if odds_h <= 1.0 or odds_a <= 1.0: continue
        
        # Predict
        p_home_model = elo.predict_home_prob(home, away)
        
        event = Event(
            id=f"{home}-{away}-{idx}",
            date=row['Date'],
            team_home=home,
            team_away=away,
            odds_home=odds_h,
            odds_draw=0.0,
            odds_away=odds_a,
            prob_home=p_home_model,
            result=res
        )
        
        # Bet
        tester.process_event(event)
        
        # Update
        elo.update(home, away, res)
        
    metrics = tester.get_metrics()
    print(f"\nFinal NBA Metrics: {metrics}")
    return metrics

if __name__ == "__main__":
    run_nba_scan()
