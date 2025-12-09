"""
Correlation Research Module
===========================
Academic-style research tools for Family A/B correlation profitability.

Components:
1. Rolling Correlation Analyzer
2. Combined Portfolio Simulator
3. Monte Carlo Engine with correlation uncertainty
4. Regime Detector
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import os
import sys

# Import CPI Engine
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cpi_engine import CPIEngine, CPIConfig


@dataclass
class ResearchConfig:
    """Configuration for correlation research."""
    # Data paths
    family_a_csv: str = "../../backtest_results.csv"
    
    # Rolling correlation windows
    corr_windows: List[int] = None  # Days: [60, 120, 252]
    
    # Monte Carlo settings
    mc_simulations: int = 10000
    corr_uncertainty: float = 0.3  # +/- noise band
    
    # Portfolio allocation scenarios
    allocation_scenarios: List[Tuple[float, float]] = None  # [(A%, B%)]
    
    # Capital
    initial_capital: float = 100000.0
    
    def __post_init__(self):
        if self.corr_windows is None:
            self.corr_windows = [60, 120, 252]
        if self.allocation_scenarios is None:
            self.allocation_scenarios = [
                (1.0, 0.0),   # 100% A
                (0.9, 0.1),   # 90/10
                (0.8, 0.2),   # 80/20
                (0.7, 0.3),   # 70/30
                (0.5, 0.5),   # 50/50
            ]


class RollingCorrelationAnalyzer:
    """Analyze correlation stability over different time windows."""
    
    def __init__(self, config: ResearchConfig):
        self.config = config
        self.family_a_returns = None
        self.family_b_returns = None
        
    def load_family_a(self) -> pd.Series:
        """Load Family A (ML) daily returns."""
        csv_path = os.path.join(os.path.dirname(__file__), self.config.family_a_csv)
        
        if not os.path.exists(csv_path):
            print(f"Warning: {csv_path} not found. Using mock data.")
            dates = pd.date_range("2014-01-01", "2024-12-01", freq="B")
            returns = np.random.normal(0.0005, 0.01, size=len(dates))
            return pd.Series(index=dates, data=returns, name="Family_A")
        
        df = pd.read_csv(csv_path)
        
        # Parse exit time
        if 'Exit Time' in df.columns:
            df['Exit_Time'] = pd.to_datetime(df['Exit Time'])
        elif 'Exit_Time' in df.columns:
            df['Exit_Time'] = pd.to_datetime(df['Exit_Time'])
        else:
            print(f"Columns: {df.columns}")
            return None
        
        df['Date'] = df['Exit_Time'].dt.date
        
        # Get PnL column
        if 'PnL' in df.columns:
            col = 'PnL'
        elif 'Profit' in df.columns:
            col = 'Profit'
        else:
            return None
        
        daily_pnl = df.groupby('Date')[col].sum()
        daily_pnl.index = pd.to_datetime(daily_pnl.index)
        
        # Convert to returns
        daily_ret = daily_pnl / self.config.initial_capital
        self.family_a_returns = daily_ret
        return daily_ret
    
    def load_family_b(self, use_historical: bool = True) -> pd.Series:
        """Load Family B (CPI) daily returns using expanded data."""
        cfg = CPIConfig(
            symbol_equity="ES=F",
            symbol_inflation="GC=F",
            horizon_days=5
        )
        engine = CPIEngine(cfg)
        
        # Override with historical data if available
        if use_historical:
            hist_path = os.path.join(os.path.dirname(__file__), "historical_cpi_data.csv")
            if os.path.exists(hist_path):
                engine.cpi_data = pd.read_csv(hist_path)
                engine.cpi_data['Date'] = pd.to_datetime(engine.cpi_data['Date']).dt.tz_localize(None)
                print(f"[Research] Loaded {len(engine.cpi_data)} historical CPI events.")
        
        engine.load_data()
        trades_df = engine.run_backtest()
        
        if trades_df.empty:
            return pd.Series(dtype=float, name="Family_B")
        
        # Compute daily returns from trade PnL
        trades_df['Exit_Date'] = pd.to_datetime(trades_df['Date']) + pd.Timedelta(days=cfg.horizon_days)
        daily_pnl = trades_df.groupby(trades_df['Exit_Date'].dt.date)['Net_PnL'].sum()
        daily_pnl.index = pd.to_datetime(daily_pnl.index)
        
        daily_ret = daily_pnl / cfg.initial_capital
        self.family_b_returns = daily_ret
        return daily_ret
    
    def compute_rolling_correlation(self) -> Dict[int, pd.Series]:
        """Compute rolling correlation for each window size."""
        if self.family_a_returns is None or self.family_b_returns is None:
            print("Error: Load both families first.")
            return {}
        
        # Align data
        combined = pd.DataFrame({
            'A': self.family_a_returns,
            'B': self.family_b_returns
        }).fillna(0.0)
        
        results = {}
        for window in self.config.corr_windows:
            rolling_corr = combined['A'].rolling(window=window).corr(combined['B'])
            results[window] = rolling_corr.dropna()
            
            # Summary stats
            mean_corr = rolling_corr.mean()
            std_corr = rolling_corr.std()
            min_corr = rolling_corr.min()
            max_corr = rolling_corr.max()
            
            print(f"[Window {window}d] Mean: {mean_corr:.3f}, Std: {std_corr:.3f}, "
                  f"Range: [{min_corr:.3f}, {max_corr:.3f}]")
        
        return results


class PortfolioSimulator:
    """Simulate combined portfolios with different allocations."""
    
    def __init__(self, family_a_returns: pd.Series, family_b_returns: pd.Series, config: ResearchConfig):
        self.family_a = family_a_returns
        self.family_b = family_b_returns
        self.config = config
        
        # Align
        self.combined = pd.DataFrame({
            'A': self.family_a,
            'B': self.family_b
        }).fillna(0.0)
    
    def simulate_allocation(self, weight_a: float, weight_b: float) -> Dict:
        """Simulate portfolio with given weights."""
        portfolio_returns = weight_a * self.combined['A'] + weight_b * self.combined['B']
        
        # Compute metrics
        equity_curve = (1 + portfolio_returns).cumprod() * self.config.initial_capital
        
        total_return = (equity_curve.iloc[-1] / self.config.initial_capital) - 1
        
        # Sharpe (annualized)
        mean_ret = portfolio_returns.mean()
        std_ret = portfolio_returns.std()
        sharpe = (mean_ret / std_ret) * np.sqrt(252) if std_ret > 0 else 0
        
        # Max Drawdown
        rolling_max = equity_curve.cummax()
        drawdown = (equity_curve - rolling_max) / rolling_max
        max_dd = drawdown.min()
        
        # Calmar Ratio
        calmar = total_return / abs(max_dd) if max_dd != 0 else 0
        
        return {
            'weights': (weight_a, weight_b),
            'total_return': total_return,
            'sharpe': sharpe,
            'max_drawdown': max_dd,
            'calmar': calmar,
            'equity_curve': equity_curve,
            'returns': portfolio_returns
        }
    
    def run_all_scenarios(self) -> List[Dict]:
        """Run all allocation scenarios."""
        results = []
        for weight_a, weight_b in self.config.allocation_scenarios:
            result = self.simulate_allocation(weight_a, weight_b)
            results.append(result)
            
            print(f"[{weight_a*100:.0f}/{weight_b*100:.0f}] "
                  f"Return: {result['total_return']*100:.2f}%, "
                  f"Sharpe: {result['sharpe']:.2f}, "
                  f"MaxDD: {result['max_drawdown']*100:.2f}%")
        
        return results


class MonteCarloEngine:
    """Monte Carlo simulation with correlation uncertainty."""
    
    def __init__(self, family_a_returns: pd.Series, family_b_returns: pd.Series, config: ResearchConfig):
        self.family_a = family_a_returns
        self.family_b = family_b_returns
        self.config = config
        
        # Base correlation
        combined = pd.DataFrame({'A': family_a_returns, 'B': family_b_returns}).fillna(0.0)
        active_days = combined[(combined['A'] != 0) | (combined['B'] != 0)]
        self.base_correlation = active_days['A'].corr(active_days['B'])
    
    def run_simulation(self, weight_a: float = 0.7, weight_b: float = 0.3) -> Dict:
        """Run Monte Carlo with correlation noise injection."""
        
        results = []
        
        for i in range(self.config.mc_simulations):
            # Inject correlation noise
            noise = np.random.uniform(-self.config.corr_uncertainty, self.config.corr_uncertainty)
            simulated_corr = np.clip(self.base_correlation + noise, -1, 1)
            
            # Adjust B returns based on correlation shift
            # Simple approach: scale B's variance contribution
            adjusted_b = self.family_b * (1 + noise * 0.5)
            
            # Compute portfolio
            combined = pd.DataFrame({'A': self.family_a, 'B': adjusted_b}).fillna(0.0)
            portfolio = weight_a * combined['A'] + weight_b * combined['B']
            
            equity = (1 + portfolio).cumprod() * self.config.initial_capital
            final_value = equity.iloc[-1]
            
            # Max DD
            rolling_max = equity.cummax()
            max_dd = ((equity - rolling_max) / rolling_max).min()
            
            results.append({
                'sim_corr': simulated_corr,
                'final_value': final_value,
                'max_dd': max_dd
            })
        
        df_results = pd.DataFrame(results)
        
        # Summary
        summary = {
            'mean_final': df_results['final_value'].mean(),
            'p5_final': df_results['final_value'].quantile(0.05),
            'p95_final': df_results['final_value'].quantile(0.95),
            'mean_dd': df_results['max_dd'].mean(),
            'p5_dd': df_results['max_dd'].quantile(0.05),  # Worst DD
            'p95_dd': df_results['max_dd'].quantile(0.95),
            'raw_results': df_results
        }
        
        print(f"\n[Monte Carlo Results ({self.config.mc_simulations} sims)]")
        print(f"  Final Value: ${summary['mean_final']:,.0f} "
              f"(5th: ${summary['p5_final']:,.0f}, 95th: ${summary['p95_final']:,.0f})")
        print(f"  Max Drawdown: {summary['mean_dd']*100:.2f}% "
              f"(5th: {summary['p5_dd']*100:.2f}%, 95th: {summary['p95_dd']*100:.2f}%)")
        
        return summary


if __name__ == "__main__":
    # Quick test
    config = ResearchConfig()
    
    print("=== Rolling Correlation Test ===")
    analyzer = RollingCorrelationAnalyzer(config)
    analyzer.load_family_a()
    analyzer.load_family_b(use_historical=True)
    analyzer.compute_rolling_correlation()
