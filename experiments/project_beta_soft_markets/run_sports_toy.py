
import sys
import os
import random
import pandas as pd
from datetime import datetime, timedelta

# Adjust path to import engine
sys.path.append(os.path.abspath("."))
sys.path.append(os.path.abspath("experiments/project_beta_soft_markets"))

from sports_engine import Event, SportsBacktester, KellySizer

def run_simulation(n_matches=1000, true_prob=0.55, bias=-0.05, vig=0.05):
    """
    Run a Toy Simulation.
    
    Parameters:
    - n_matches: Number of games.
    - true_prob: Real probability of Home Win.
    - bias: How much the Public underestimates the favorite (e.g. -0.05 means Public thinks 0.50).
    - vig: Bookmaker overround (margin).
    """
    print(f"\n--- Running Toy Simulation ---")
    print(f"Matches: {n_matches} | True Prob: {true_prob} | Bias: {bias} | Vig: {vig}")
    
    # 1. Setup Engine
    # Kelly Fraction 0.1 (Conservative)
    sizer = KellySizer(fraction=0.1, max_stake_pct=0.05)
    tester = SportsBacktester(initial_balance=10000, sizer=sizer)
    
    # 2. Generate Events
    events = []
    start_date = datetime(2024, 1, 1)
    
    wins = 0
    losses = 0
    
    for i in range(n_matches):
        date = start_date + timedelta(days=i//10) # 10 games per day
        
        # Determine Outcome based on True Probability
        if random.random() < true_prob:
            result = 'HOME'
            wins += 1
        else:
            result = 'AWAY' # Simplified (Draw ignored for toy)
            losses += 1
            
        # Determine Odds
        # Public Perception
        public_prob = true_prob + bias # e.g. 0.55 + (-0.05) = 0.50
        
        # Bookmaker Odds (Public Prob + Vig)
        # Implied Prob = Public Prob / (1 - Vig)? No, Implied = Public * (1+Vig) usually.
        # Let's say Bookie prices it at Public Prob + Vig.
        bookie_implied_home = public_prob + (vig/2)
        bookie_implied_away = (1 - public_prob) + (vig/2)
        
        # Odds
        odds_home = 1.0 / bookie_implied_home
        odds_away = 1.0 / bookie_implied_away
        
        # Model Estimation (Our Edge)
        # Assume our model is perfect (True Prob)
        model_prob_home = true_prob
        
        event = Event(
            id=f"M{i}",
            date=date,
            team_home=f"Home_{i}",
            team_away=f"Away_{i}",
            odds_home=odds_home,
            odds_draw=0.0,
            odds_away=odds_away,
            prob_home=model_prob_home,
            result=result
        )
        tester.process_event(event)

    # 3. Results
    metrics = tester.get_metrics()
    print(f"Simulation Actual Win Rate: {(wins/n_matches)*100:.2f}% (Exp: {true_prob*100}%)")
    print(f"Strategy Results: {metrics}")
    
    return metrics, tester

def main():
    print("=== PROJECT BETA: TOY SIMULATION ===")
    
    # Scenario 1: The "Value" Bettor
    # We possess Edge. True Prob 55%. Bookie Odds based on 50%.
    # Odds ~ 1.90. (Implied 52.5%).
    # Edge = 55% - 52.5% = 2.5%.
    
    run_simulation(n_matches=2000, true_prob=0.55, bias=-0.05, vig=0.05)
    
    # Scenario 2: The "Gambler" (Negative EV)
    # We have NO Edge. Model = Random noise around 50%.
    # True Prob 50%. Odds based on 50% + Vig.
    # Odds ~ 1.90.
    # We bet anyway.
    
    print("\n[Control Group: Negative EV Gambler]")
    # Bias = 0 (Public is right), but Vig makes it -EV.
    run_simulation(n_matches=2000, true_prob=0.50, bias=0.0, vig=0.05)

if __name__ == "__main__":
    main()
