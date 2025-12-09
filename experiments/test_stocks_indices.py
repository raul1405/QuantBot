"""
Experiment: US Stocks & ETFs Universe Test
Goal: Test strategy performance on 30 US stocks/ETFs (same trading hours)
Note: Excludes international indices due to different trading hours.
"""
import sys
sys.path.insert(0, '/Users/raulschalkhammer/Desktop/Costum Portfolio Backtest/FTMO Challenge')

from quant_backtest import (
    Config, DataLoader, FeatureEngine, AlphaEngine, 
    RegimeEngine, EnsembleSignal, CrisisAlphaEngine, Backtester
)
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ============================================================================
# US STOCKS & ETFs UNIVERSE: 30 Assets (same trading hours)
# ============================================================================

US_STOCKS_UNIVERSE = [
    # --- US INDEX ETFs (4) ---
    "SPY",      # S&P 500 ETF
    "QQQ",      # Nasdaq 100 ETF
    "IWM",      # Russell 2000 ETF
    "DIA",      # Dow Jones ETF
    
    # --- SECTOR ETFs (8) ---
    "XLF",      # Financial Sector
    "XLK",      # Technology Sector
    "XLE",      # Energy Sector
    "XLV",      # Healthcare Sector
    "XLU",      # Utilities Sector
    "XLI",      # Industrial Sector
    "XLY",      # Consumer Discretionary
    "XLP",      # Consumer Staples
    
    # --- OTHER ETFs (3) ---
    "EEM",      # Emerging Markets
    "EFA",      # EAFE (Developed Intl)
    "TLT",      # 20+ Year Treasury
    
    # --- MEGA-CAP TECH (8) ---
    "AAPL",     # Apple
    "MSFT",     # Microsoft
    "GOOGL",    # Alphabet
    "AMZN",     # Amazon
    "NVDA",     # Nvidia
    "META",     # Meta
    "TSLA",     # Tesla
    "NFLX",     # Netflix
    
    # --- FINANCIALS (4) ---
    "JPM",      # JPMorgan
    "BAC",      # Bank of America
    "V",        # Visa
    "MA",       # Mastercard
    
    # --- OTHER SECTORS (3) ---
    "JNJ",      # Johnson & Johnson (Healthcare)
    "XOM",      # Exxon Mobil (Energy)
    "WMT",      # Walmart (Retail)
]


def run_backtest_scenario(universe: list, name: str) -> dict:
    """Run backtest on a given universe and return results."""
    print(f"\n{'='*60}")
    print(f"RUNNING: {name}")
    print(f"Assets requested: {len(universe)}")
    print(f"{'='*60}\n")
    
    config = Config()
    config.symbols = universe
    config.transaction_cost = 0.0001
    config.initial_balance = 100000.0
    
    try:
        # Load Data - Use 500-day range (Yahoo limits hourly to 730 days)
        loader = DataLoader(config)
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=500)).strftime('%Y-%m-%d')
        print(f"Date range: {start_date} to {end_date}")
        data_map = loader.load_data(start_date, end_date)
        print(f"Loaded {len(data_map)} symbols")
        
        if len(data_map) == 0:
            print("ERROR: No data loaded!")
            return None
        
        # Pipeline
        fe = FeatureEngine(config)
        re = RegimeEngine(config)
        
        data_map = fe.add_features_all(data_map)
        data_map = re.add_regimes_all(data_map)
        
        alpha = AlphaEngine(config)
        alpha.train_model(data_map)
        
        data_map = alpha.add_signals_all(data_map)
        
        ens = EnsembleSignal(config)
        data_map = ens.add_ensemble_all(data_map)
        
        crisis = CrisisAlphaEngine(config)
        final_data = crisis.add_crisis_signals(data_map)
        
        # Backtest
        bt = Backtester(config)
        equity_curve = bt.run_backtest(final_data)
        
        # Stats
        trades = len(bt.account.trade_history)
        balance = bt.account.balance
        pnl = balance - config.initial_balance
        roi = (pnl / config.initial_balance) * 100
        
        # Calculate Max DD from Equity Curve
        peak = equity_curve.cummax()
        drawdown = (equity_curve - peak) / peak
        max_dd = drawdown.min() * 100
        
        wins = [t for t in bt.account.trade_history if t.get('PnL', 0) > 0]
        win_rate = (len(wins) / trades * 100) if trades > 0 else 0.0
        
        results = {
            'name': name,
            'n_assets': len(universe),
            'n_assets_loaded': len(data_map),
            'total_return_pct': roi,
            'max_drawdown_pct': max_dd,
            'n_trades': trades,
            'win_rate_pct': win_rate,
            'final_equity': balance,
        }
        
        print(f"\nRESULTS: {name}")
        print(f"  Assets Loaded: {len(data_map)}/{len(universe)}")
        print(f"  Total Return: {roi:+.2f}%")
        print(f"  Max Drawdown: {max_dd:.2f}%")
        print(f"  Total Trades: {trades}")
        print(f"  Win Rate: {win_rate:.1f}%")
        
        return results
        
    except Exception as e:
        print(f"ERROR in {name}: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    print("\n" + "="*70)
    print("EXPERIMENT: US Stocks & ETFs Universe Test")
    print("="*70)
    print(f"\nUS Stocks/ETFs Universe: {len(US_STOCKS_UNIVERSE)} assets")
    print("\nAssets:")
    for i, sym in enumerate(US_STOCKS_UNIVERSE):
        print(f"  {i+1:2}. {sym}")
    
    # Run US stocks/ETFs backtest
    results = run_backtest_scenario(US_STOCKS_UNIVERSE, "US Stocks & ETFs (30)")
    
    if results:
        print("\n" + "="*70)
        print("FINAL RESULTS: US Stocks & ETFs Universe")
        print("="*70)
        print(f"\n{'Metric':<25} {'Value':>20}")
        print("-"*45)
        print(f"{'Assets Loaded':<25} {results['n_assets_loaded']:>20}")
        print(f"{'Total Return %':<25} {results['total_return_pct']:>+19.2f}%")
        print(f"{'Max Drawdown %':<25} {results['max_drawdown_pct']:>19.2f}%")
        print(f"{'Total Trades':<25} {results['n_trades']:>20}")
        print(f"{'Win Rate %':<25} {results['win_rate_pct']:>19.1f}%")
        print(f"{'Final Equity':<25} ${results['final_equity']:>18,.2f}")
        
        # Compare with baseline
        print("\n" + "="*70)
        print("COMPARISON WITH FX BASELINE")
        print("="*70)
        print("""
Reference (from previous tests):
┌─────────────────────────┬──────────┬─────────┬────────┬─────────┐
│ Universe                │ Return   │ Max DD  │ Trades │ Win Rate│
├─────────────────────────┼──────────┼─────────┼────────┼─────────┤
│ FX Baseline (30 assets) │ +22.30%  │ -1.11%  │  126   │  66.7%  │
│ FX Expanded (60 assets) │ +21.31%  │ -2.78%  │  158   │  64.6%  │
└─────────────────────────┴──────────┴─────────┴────────┴─────────┘

US Stocks/ETFs performance:""")
        print(f"│ US Stocks/ETFs (30)     │ {results['total_return_pct']:>+7.2f}% │ {results['max_drawdown_pct']:>6.2f}% │ {results['n_trades']:>6} │ {results['win_rate_pct']:>6.1f}% │")
    
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE")
    print("="*70)
