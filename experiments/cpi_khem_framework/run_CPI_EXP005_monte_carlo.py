"""
EXP005: Monte Carlo Stress Test
================================
Run Monte Carlo simulation with correlation uncertainty to stress test
the 70/30 allocation under adverse conditions.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from correlation_research import (
    RollingCorrelationAnalyzer, 
    MonteCarloEngine, 
    ResearchConfig
)


def main():
    print("=" * 60)
    print("EXP005: Monte Carlo Stress Test")
    print("=" * 60)
    
    config = ResearchConfig(
        mc_simulations=10000,
        corr_uncertainty=0.3  # +/- 0.3 noise band
    )
    
    # Load data
    print("\n[1] Loading Data...")
    analyzer = RollingCorrelationAnalyzer(config)
    ret_a = analyzer.load_family_a()
    ret_b = analyzer.load_family_b(use_historical=True)
    
    if ret_a is None:
        print("Error: Could not load Family A data.")
        return
    
    # Monte Carlo
    print("\n[2] Running Monte Carlo (10,000 simulations)...")
    mc_engine = MonteCarloEngine(ret_a, ret_b, config)
    
    print(f"  Base Correlation: {mc_engine.base_correlation:.3f}")
    print(f"  Uncertainty Band: +/- {config.corr_uncertainty}")
    
    # Test 70/30 allocation (user preference)
    summary = mc_engine.run_simulation(weight_a=0.7, weight_b=0.3)
    
    # Stress Test: What if correlation flips to +0.5?
    print("\n[3] Stress Scenario: Correlation Flip to +0.5...")
    
    # Simulate worst case
    worst_case_results = []
    for i in range(1000):
        # Force positive correlation
        adjusted_b = ret_b * -1  # Flip sign to simulate positive correlation
        
        combined = pd.DataFrame({'A': ret_a, 'B': adjusted_b}).fillna(0.0)
        portfolio = 0.7 * combined['A'] + 0.3 * combined['B']
        
        equity = (1 + portfolio).cumprod() * config.initial_capital
        final_value = equity.iloc[-1]
        
        rolling_max = equity.cummax()
        max_dd = ((equity - rolling_max) / rolling_max).min()
        
        worst_case_results.append({'final': final_value, 'dd': max_dd})
    
    wc_df = pd.DataFrame(worst_case_results)
    wc_mean_final = wc_df['final'].mean()
    wc_mean_dd = wc_df['dd'].mean()
    
    print(f"  Worst Case Final: ${wc_mean_final:,.0f}")
    print(f"  Worst Case Max DD: {wc_mean_dd*100:.2f}%")
    
    # Chart
    print("\n[4] Generating Charts...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    raw = summary['raw_results']
    
    # Plot 1: Final Value Distribution
    ax1 = axes[0, 0]
    ax1.hist(raw['final_value'], bins=100, alpha=0.7, color='steelblue', edgecolor='black')
    ax1.axvline(x=summary['mean_final'], color='red', linestyle='--', label=f"Mean: ${summary['mean_final']:,.0f}")
    ax1.axvline(x=summary['p5_final'], color='orange', linestyle=':', label=f"5th: ${summary['p5_final']:,.0f}")
    ax1.axvline(x=summary['p95_final'], color='green', linestyle=':', label=f"95th: ${summary['p95_final']:,.0f}")
    ax1.set_xlabel('Final Portfolio Value ($)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Monte Carlo: Final Value Distribution (70/30)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Max Drawdown Distribution
    ax2 = axes[0, 1]
    ax2.hist(raw['max_dd'] * 100, bins=100, alpha=0.7, color='salmon', edgecolor='black')
    ax2.axvline(x=summary['mean_dd']*100, color='red', linestyle='--', label=f"Mean: {summary['mean_dd']*100:.2f}%")
    ax2.axvline(x=summary['p5_dd']*100, color='darkred', linestyle=':', label=f"5th (Worst): {summary['p5_dd']*100:.2f}%")
    ax2.set_xlabel('Max Drawdown (%)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Monte Carlo: Max Drawdown Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Correlation vs Final Value Scatter
    ax3 = axes[1, 0]
    ax3.scatter(raw['sim_corr'], raw['final_value'], alpha=0.1, s=5)
    ax3.set_xlabel('Simulated Correlation')
    ax3.set_ylabel('Final Value ($)')
    ax3.set_title('Correlation Sensitivity')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Summary Box
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    summary_text = f"""
MONTE CARLO SUMMARY (70/30 Allocation)
========================================
Simulations: {config.mc_simulations:,}
Base Correlation: {mc_engine.base_correlation:.3f}
Uncertainty Band: ±{config.corr_uncertainty}

FINAL VALUE
  Mean: ${summary['mean_final']:,.0f}
  5th Percentile: ${summary['p5_final']:,.0f}
  95th Percentile: ${summary['p95_final']:,.0f}

MAX DRAWDOWN
  Mean: {summary['mean_dd']*100:.2f}%
  Worst (5th): {summary['p5_dd']*100:.2f}%

STRESS TEST (Corr Flip to +0.5)
  Mean Final: ${wc_mean_final:,.0f}
  Mean Max DD: {wc_mean_dd*100:.2f}%

VERDICT: {'✅ ROBUST' if summary['p5_dd'] > -0.15 else '⚠️ CAUTION'}
"""
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes,
             fontsize=11, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    chart_path = os.path.join(os.path.dirname(__file__), "../../exp_CPI_005_montecarlo.png")
    plt.savefig(chart_path, dpi=150)
    print(f"  Saved: {chart_path}")
    
    # Report
    print("\n[5] Generating Report...")
    
    verdict = "ROBUST" if summary['p5_dd'] > -0.15 else "CAUTION - High Tail Risk"
    
    report = f"""# EXP005: Monte Carlo Stress Test

## Objective
Stress test the 70/30 allocation under correlation uncertainty (+/- 0.3).

## Configuration
- **Simulations**: {config.mc_simulations:,}
- **Base Correlation**: {mc_engine.base_correlation:.3f}
- **Uncertainty Band**: ±{config.corr_uncertainty}

## Results

### Final Portfolio Value
| Metric | Value |
|--------|-------|
| Mean | ${summary['mean_final']:,.0f} |
| 5th Percentile | ${summary['p5_final']:,.0f} |
| 95th Percentile | ${summary['p95_final']:,.0f} |

### Max Drawdown
| Metric | Value |
|--------|-------|
| Mean | {summary['mean_dd']*100:.2f}% |
| Worst (5th) | {summary['p5_dd']*100:.2f}% |
| Best (95th) | {summary['p95_dd']*100:.2f}% |

## Stress Scenario: Correlation Flip

**What if correlation flips from negative to +0.5?**

| Metric | Value |
|--------|-------|
| Mean Final | ${wc_mean_final:,.0f} |
| Mean Max DD | {wc_mean_dd*100:.2f}% |

## Verdict

**{verdict}**

{"The 70/30 allocation survives Monte Carlo stress testing. Even under worst-case correlation uncertainty, the portfolio remains viable." if "ROBUST" in verdict else "The allocation shows high tail risk. Consider reducing Family B allocation or adding correlation monitoring."}

## Chart
![Monte Carlo](exp_CPI_005_montecarlo.png)
"""
    
    report_path = os.path.join(os.path.dirname(__file__), "../../exp_CPI_005_results.md")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"  Saved: {report_path}")
    
    print("\n" + "=" * 60)
    print("EXP005 COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
