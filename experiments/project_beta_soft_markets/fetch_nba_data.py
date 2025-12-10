
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Create Synthetic NBA Data for logic validation
# NBA Characteristics:
# Home Win Rate: ~58-60%
# Scoring: 110-115 points average
# Odds: Efficient (Vig ~4-5%)

def generate_nba_data():
    teams = [f"Team_{i}" for i in range(30)] # 30 NBA Teams
    # Assign true strength to teams
    # Team 0 is best (0.80 win rate vs Team 29)
    # Strength = Elo
    true_ratings = {t: 1500 + (15 - i)*20 for i, t in enumerate(teams)}
    # Team_0: 1500 + 300 = 1800
    # Team_29: 1500 - 280 = 1220
    
    matches = []
    start_date = datetime(2023, 10, 24)
    
    # Simulate Season (82 games * 15 games/day approx)
    # Total ~1230 games
    
    n_games = 1200
    
    print(f"Generating {n_games} Synthetic NBA Games...")
    
    for i in range(n_games):
        date = start_date + timedelta(days=i//10)
        
        # Pick 2 teams randomly
        home, away = random.sample(teams, 2)
        
        r_h = true_ratings[home] + 100 # Home Court
        r_a = true_ratings[away]
        
        dr = r_h - r_a
        win_prob = 1 / (1 + 10**(-dr/400))
        
        # Determine Result
        if random.random() < win_prob:
            score_h = 115
            score_a = 105
            ml_home = -150 # Placeholder odds Favor Home
            # Calculate Efficient Odds
            fair_odds_h = 1/win_prob
            fair_odds_a = 1/(1-win_prob)
        else:
            score_h = 105
            score_a = 115
            fair_odds_h = 1/win_prob
            fair_odds_a = 1/(1-win_prob)
            
        # Add Vig
        vig = 0.05
        # Implied = Fair_Prob + Vig/2
        prob_h_vig = win_prob + 0.025
        prob_a_vig = (1-win_prob) + 0.025
        
        odds_h = 1 / prob_h_vig
        odds_a = 1 / prob_a_vig
        
        # Convert to US Odds? No, use Decimal. SBR uses US usually but Engine uses Decimal.
        # Let's save Decimal for simplicity.
        
        matches.append({
            'Date': date,
            'Home': home,
            'Away': away,
            'ScoreH': score_h,
            'ScoreA': score_a,
            'ML_Home': round(odds_h, 2), # Decimal
            'ML_Away': round(odds_a, 2), # Decimal
            'Season': '2023-2024'
        })
        
    df = pd.DataFrame(matches)
    # Add Random "Inefficiencies" for testing
    # e.g. Randomly make Underdog Odds way too high in 5% of games
    # calculated later? No, let's keep it efficient to see if simple Elo fails (as expected).
    
    outfile = "experiments/project_beta_soft_markets/data/NBA_Combined_2022_2024.csv"
    df.to_csv(outfile, index=False)
    print(f"Saved {len(df)} matches to {outfile}")

if __name__ == "__main__":
    generate_nba_data()
