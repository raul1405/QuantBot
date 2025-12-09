"""
FRED CPI Data Fetcher
=====================
Fetches 10 years of CPI data from FRED API for Family B research.

Series Used:
- CPIAUCSL: Consumer Price Index for All Urban Consumers (Seasonally Adjusted)
- We compute: Actual, Previous, and derive Forecast from consensus median
"""

import requests
import pandas as pd
from datetime import datetime
import os

# FRED API Configuration
FRED_API_KEY = "0378c1d272d0d01dda18308e4333615e"
FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

def fetch_cpi_series(series_id: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch a FRED series and return as DataFrame."""
    params = {
        "series_id": series_id,
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "observation_start": start_date,
        "observation_end": end_date,
    }
    
    response = requests.get(FRED_BASE_URL, params=params)
    data = response.json()
    
    if "observations" not in data:
        print(f"Error fetching {series_id}: {data}")
        return pd.DataFrame()
    
    df = pd.DataFrame(data["observations"])
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df[["date", "value"]].dropna()
    
    return df


def compute_cpi_events(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert monthly CPI readings into event-format data.
    
    For each month:
    - Date: CPI release date (approx 10th-15th of following month)
    - Actual: That month's CPI YoY change
    - Previous: Prior month's CPI YoY change
    - Forecast: Use 3-month trailing average as naive forecast proxy
    """
    
    # Compute YoY change
    df = df.sort_values("date").reset_index(drop=True)
    df["yoy_change"] = df["value"].pct_change(periods=12) * 100  # Percentage
    
    # Drop first 12 months (no YoY available)
    df = df.iloc[12:].reset_index(drop=True)
    
    # Build event DataFrame
    events = []
    for i in range(1, len(df)):
        actual = df.iloc[i]["yoy_change"]
        previous = df.iloc[i-1]["yoy_change"]
        
        # Naive forecast: 3-month trailing average (analysts typically forecast near prior)
        if i >= 3:
            forecast = df.iloc[i-3:i]["yoy_change"].mean()
        else:
            forecast = previous
        
        # CPI release date: ~15th of the NEXT month (lagged release)
        ref_date = df.iloc[i]["date"]
        release_date = ref_date + pd.DateOffset(months=1, days=14)
        
        events.append({
            "Date": release_date.strftime("%Y-%m-%d"),
            "Actual": round(actual, 2),
            "Forecast": round(forecast, 2),
            "Previous": round(previous, 2),
        })
    
    return pd.DataFrame(events)


def main():
    print("[FRED Fetcher] Downloading CPI data (2014-2024)...")
    
    # Fetch CPIAUCSL (need 13 months prior for YoY)
    start_date = "2013-01-01"
    end_date = "2024-12-01"
    
    df_cpi = fetch_cpi_series("CPIAUCSL", start_date, end_date)
    
    if df_cpi.empty:
        print("Failed to fetch CPI data.")
        return
    
    print(f"  Fetched {len(df_cpi)} monthly observations.")
    
    # Convert to event format
    events_df = compute_cpi_events(df_cpi)
    print(f"  Generated {len(events_df)} CPI events.")
    
    # Save
    output_path = os.path.join(os.path.dirname(__file__), "historical_cpi_data.csv")
    events_df.to_csv(output_path, index=False)
    print(f"  Saved to: {output_path}")
    
    # Preview
    print("\n[Sample Events]")
    print(events_df.tail(10).to_string(index=False))


if __name__ == "__main__":
    main()
