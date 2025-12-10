
import os
import requests
import pandas as pd

# URLs for English Premier League (EPL) Data
# Source: football-data.co.uk
URLS = {
    "2023-2024": "https://www.football-data.co.uk/mmz4281/2324/E0.csv",
    "2022-2023": "https://www.football-data.co.uk/mmz4281/2223/E0.csv",
    "2021-2022": "https://www.football-data.co.uk/mmz4281/2122/E0.csv"
}

OUTPUT_DIR = "experiments/project_beta_soft_markets/data"

def fetch_data():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    all_data = []
    
    for season, url in URLS.items():
        print(f"Fetching {season} from {url}...")
        try:
            # mimic browser header
            headers = {'User-Agent': 'Mozilla/5.0'}
            r = requests.get(url, headers=headers)
            r.raise_for_status()
            
            # Save raw
            filename = f"{OUTPUT_DIR}/EPL_{season}.csv"
            with open(filename, 'wb') as f:
                f.write(r.content)
                
            # Read into dataframe
            df = pd.read_csv(filename)
            # Add Season column
            df['Season'] = season
            all_data.append(df)
            print(f"  Saved {len(df)} matches.")
            
        except Exception as e:
            print(f"  Error fetching {season}: {e}")
            
    if all_data:
        # Combine
        combined_df = pd.concat(all_data, ignore_index=True)
        # Select important columns
        # Date, HomeTeam, AwayTeam, FTR (Full Time Result), B365H, B365D, B365A (Bet365 Odds)
        cols = ['Date', 'HomeTeam', 'AwayTeam', 'FTR', 'B365H', 'B365D', 'B365A', 'Season']
        # Handle cases where columns might differ slightly or missing
        # Standard football-data.co.uk cols are usually consistent
        
        # Ensure Date parsing
        # Usually dd/mm/yyyy or dd/mm/yy
        combined_df['Date'] = pd.to_datetime(combined_df['Date'], dayfirst=True, errors='coerce')
        
        final_file = f"{OUTPUT_DIR}/EPL_Combined_2021_2024.csv"
        combined_df.to_csv(final_file, index=False)
        print(f"\nCreated Combined Dataset: {final_file} ({len(combined_df)} matches)")
        
if __name__ == "__main__":
    fetch_data()
