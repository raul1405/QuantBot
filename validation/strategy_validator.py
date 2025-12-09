
"""
INSTITUTIONAL VALIDATION LAB
============================
Independent Risk & Validation Framework.
Treats the trading strategy as a Black Box.
Input: Trade Log & Equity Curve.
Output: Institutional Grade Report.

Metrics: SQN, Expectancy, Sharpe, Sortino, Calmar, Ulcer Index.
Tests: Bootstrap CI, Monte Carlo, Randomization (Luck Test), Regime Analysis.
"""

import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

class StrategyValidator:
    def __init__(self, trade_log_df, equity_curve_series, initial_balance=100000.0):
        """
        Initialize Validator.
        :param trade_log_df: DataFrame with cols [EntryTime, ExitTime, Symbol, Direction, Size, EntryPrice, ExitPrice, NetPnL, Risk, R, Regime_Vol, Regime_Trend]
        :param equity_curve_series: Series of Equity values (index=Datetime).
        """
        self.trades = trade_log_df.copy()
        self.equity = equity_curve_series.copy()
        self.initial_balance = initial_balance
        
        # Standardize Columns
        col_map = {
            'Entry Time': 'EntryTime',
            'Exit Time': 'ExitTime',
            'Entry Price': 'EntryPrice',
            'Exit Price': 'ExitPrice',
            'PnL': 'NetPnL',
            'Risk_Dollars': 'Risk'
        }
        self.trades.rename(columns=col_map, inplace=True)
        
        # Ensure Datetimes
        if not self.trades.empty:
            if 'EntryTime' in self.trades.columns:
                 self.trades['EntryTime'] = pd.to_datetime(self.trades['EntryTime'])
            if 'ExitTime' in self.trades.columns:
                 self.trades['ExitTime'] = pd.to_datetime(self.trades['ExitTime'])
            
        # Ensure R-Multiple exists
        if 'R' not in self.trades.columns:
            if 'R_Multiple' in self.trades.columns:
                 self.trades['R'] = self.trades['R_Multiple']
            elif 'NetPnL' in self.trades.columns and 'Risk' in self.trades.columns:
                 self.trades['R'] = self.trades['NetPnL'] / self.trades['Risk'].replace(0, 1)
            
    def compute_trade_metrics(self):
        """
        2.1 Trade-level edge metrics
        """
        if self.trades.empty: return {}
        
        r = self.trades['R']
        n = len(r)
        
        avg_r = r.mean()
        std_r = r.std()
        
        # SQN: sqrt(N) * Mean / Std
        sqn = np.sqrt(n) * (avg_r / std_r) if std_r > 0 else 0
        
        wins = r[r > 0]
        losses = r[r <= 0]
        
        win_rate = len(wins) / n
        avg_win = wins.mean() if not wins.empty else 0
        avg_loss = abs(losses.mean()) if not losses.empty else 0
        payoff = avg_win / avg_loss if avg_loss > 0 else 0
        
        # Skew / Kurtosis
        skew = r.skew()
        kurted = r.kurtosis()
        
        fat_tails = len(r[abs(r) > 3]) / n if not r.empty else 0
        
        return {
            "Total Trades": n,
            "Win Rate": win_rate,
            "Avg Win R": avg_win,
            "Avg Loss R": avg_loss,
            "Payoff Ratio": payoff,
            "Expectancy (Mean R)": avg_r,
            "SQN": sqn,
            "Skew": skew,
            "Fat Tails (>3R)": fat_tails
        }

    def compute_account_metrics(self):
        """
        2.2 Return & risk metrics (equity curve)
        """
        if self.equity.empty: return {}
        
        # Returns
        returns = self.equity.pct_change().dropna()
        if returns.empty: return {}
        
        # CAGR (Approx)
        days = (self.equity.index[-1] - self.equity.index[0]).days
        years = days / 365.25
        total_ret = (self.equity.iloc[-1] / self.equity.iloc[0]) - 1
        cagr = ((1 + total_ret) ** (1 / years)) - 1 if years > 0 else 0
        
        # Volatility (Annualized)
        ann_vol = returns.std() * np.sqrt(252) # Assuming Daily Steps? If H1, adjust.
        # Check granularity
        # If mean step is < 1 day, adjust.
        # Let's assume input is Daily Equity for standardization, or adjust.
        
        # Sharpe (Rf=0)
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
        
        # Sortino (Downside Dev)
        downside = returns[returns < 0]
        sortino = (returns.mean() / downside.std()) * np.sqrt(252) if not downside.empty and downside.std() > 0 else 0
        
        # Max Drawdown
        cummax = self.equity.cummax()
        dd = (self.equity - cummax) / cummax
        max_dd = dd.min()
        
        # Calmar
        calmar = cagr / abs(max_dd) if max_dd != 0 else 0
        
        # Ulcer Index
        ulcer = np.sqrt((dd**2).mean())
        
        return {
            "CAGR": cagr,
            "Ann Vol": ann_vol,
            "Sharpe": sharpe,
            "Sortino": sortino,
            "Max DD": max_dd,
            "Calmar": calmar,
            "Ulcer Index": ulcer
        }

    def run_statistical_tests(self):
        """
        3.1 Basic hypothesis tests (Null vs Edge)
        """
        if self.trades.empty: return {}
        
        r = self.trades['R'].dropna()
        
        # T-Test (Mean > 0)
        t_stat, p_val = stats.ttest_1samp(r, 0, alternative='greater')
        
        # Bootstrap CI (Mean R)
        resamples = 10000
        means = []
        for _ in range(resamples):
            sample = np.random.choice(r, size=len(r), replace=True)
            means.append(np.mean(sample))
            
        ci_lower = np.percentile(means, 5)
        ci_upper = np.percentile(means, 95)
        
        return {
            "T-Stat": t_stat,
            "P-Value": p_val,
            "Bootstrap Mean R (5%)": ci_lower,
            "Bootstrap Mean R (95%)": ci_upper,
            "Edge Significant?": "YES" if p_val < 0.05 else "NO"
        }

    def run_monte_carlo(self, sims=2000):
        """
        3.2 Monte Carlo & “prop-risk” stress tests
        """
        if self.trades.empty: return {}
        
        pnls = self.trades['NetPnL'].values
        start_eq = self.initial_balance
        
        pass_ftmo = 0
        fail_dd = 0
        
        # FTMO Rules
        MAX_DD_LIMIT = 0.10 # 10%
        DAILY_DD_LIMIT = 0.05 # 5% (Approximated as single trade hit? No, need daily sum. Hard with just trade list)
        # We will check Total DD.
        
        for _ in range(sims):
            # Shuffle order
            np.random.shuffle(pnls)
            curve = np.cumsum(np.concatenate(([start_eq], pnls)))
            
            # Check Max DD
            peak = np.maximum.accumulate(curve)
            dd = (curve - peak) / peak
            max_d_run = np.min(dd)
            
            # Check Profit > 10%
            final_ret = (curve[-1] - start_eq) / start_eq
            
            if max_d_run < -MAX_DD_LIMIT:
                fail_dd += 1
            elif final_ret >= 0.10:
                pass_ftmo += 1
                
        return {
            "Simulations": sims,
            "Pass Probability (>10% Profit, No Bust)": pass_ftmo / sims,
            "Bust Probability (max DD > 10%)": fail_dd / sims
        }

    def analyze_regimes(self):
        """
        4. Regime-conditioned diagnostics
        """
        if self.trades.empty: return {}
        
        results = {}
        
        # Volatility Regime
        if 'Regime_Vol' in self.trades.columns:
            for reg, group in self.trades.groupby('Regime_Vol'):
                results[f"Vol_{reg}_Count"] = len(group)
                results[f"Vol_{reg}_MeanR"] = group['R'].mean()
                results[f"Vol_{reg}_Sharpe"] = (group['R'].mean() / group['R'].std()) * np.sqrt(252/24) if len(group)>1 else 0 # Rough annualized
        
        # Trend Regime
        if 'Regime_Trend' in self.trades.columns:
            for reg, group in self.trades.groupby('Regime_Trend'):
                results[f"Trend_{reg}_Count"] = len(group)
                results[f"Trend_{reg}_MeanR"] = group['R'].mean()
                
        return results

    def generate_report(self):
        """
        5. Output Structured Report
        """
        t_metrics = self.compute_trade_metrics()
        a_metrics = self.compute_account_metrics()
        stats_res = self.run_statistical_tests()
        mc_res = self.run_monte_carlo()
        reg_res = self.analyze_regimes()
        
        report = []
        report.append("# 🏛️ INSTITUTIONAL VALIDATION REPORT")
        report.append(f"**Strategy Analysis** | Init Balance: ${self.initial_balance:,.0f}")
        report.append("")
        
        report.append("## 1. Trade Edge (Micro)")
        report.append(f"- **SQN Score**: `{t_metrics.get('SQN', 0):.2f}` (Target > 1.6)")
        report.append(f"- **Expectancy**: `{t_metrics.get('Expectancy (Mean R)', 0):.4f} R` per trade")
        report.append(f"- **Win Rate**: `{t_metrics.get('Win Rate', 0)*100:.1f}%`")
        report.append(f"- **Payoff Ratio**: `{t_metrics.get('Payoff Ratio', 0):.2f}`")
        report.append("")
        
        report.append("## 2. Account Health (Macro)")
        report.append(f"- **Sharpe Ratio**: `{a_metrics.get('Sharpe', 0):.2f}`")
        report.append(f"- **Sortino Ratio**: `{a_metrics.get('Sortino', 0):.2f}`")
        report.append(f"- **CAGR**: `{a_metrics.get('CAGR', 0)*100:.1f}%`")
        report.append(f"- **Max Drawdown**: `{a_metrics.get('Max DD', 0)*100:.2f}%`")
        report.append(f"- **Calmar Ratio**: `{a_metrics.get('Calmar', 0):.2f}`")
        report.append(f"- **Ulcer Index**: `{a_metrics.get('Ulcer Index', 0):.4f}`")
        report.append("")
        
        report.append("## 3. Statistical Proof (Null Hypothesis)")
        report.append(f"- **P-Value (Mean > 0)**: `{stats_res.get('P-Value', 1):.5f}`")
        report.append(f"- **Edge Significant?**: {stats_res.get('Edge Significant?', 'NO')}")
        report.append(f"- **95% CI Mean R**: `{stats_res.get('Bootstrap Mean R (5%)', 0):.4f}` to `{stats_res.get('Bootstrap Mean R (95%)', 0):.4f}`")
        report.append("")
        
        report.append("## 4. Monte Carlo (FTMO Simulation)")
        report.append(f"- **Pass Rate (>10%)**: `{mc_res.get('Pass Probability (>10% Profit, No Bust)', 0)*100:.1f}%`")
        report.append(f"- **Bust Rate (>10% DD)**: `{mc_res.get('Bust Probability (max DD > 10%)', 0)*100:.1f}%`")
        report.append("")
        
        report.append("## 5. Regime Analysis")
        for k, v in reg_res.items():
            report.append(f"- **{k}**: {v:.4f}")
            
        return "\n".join(report)

if __name__ == "__main__":
    # Test Stub
    print("StrategyValidator Module Loaded.")
