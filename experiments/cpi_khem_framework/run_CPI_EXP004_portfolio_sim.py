"""
EXP004: Portfolio Allocation Simulation
=======================================
Test different allocation splits between Family A (ML) and Family B (CPI).
Compute combined Sharpe, Max Drawdown, and Calmar Ratio.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from correlation_research import (
    RollingCorrelationAnalyzer, 
    PortfolioSimulator, 
    ResearchConfig
)


def main():
    print("=" * 60)
    print("EXP004: Portfolio Allocation Simulation")
    print("=" * 60)
    
    config = ResearchConfig(
        allocation_scenarios=[
            (1.0, 0.0),   # 100% A (baseline)
            (0.9, 0.1),   # 90/10
            (0.8, 0.2),   # 80/20
            (0.7, 0.3),   # 70/30
            (0.6, 0.4),   # 60/40
            (0.5, 0.5),   # 50/50
        ]
    )
    
    # Load data
    print("\n[1] Loading Data...")
    analyzer = RollingCorrelationAnalyzer(config)
    ret_a = analyzer.load_family_a()
    ret_b = analyzer.load_family_b(use_historical=True)
    
    if ret_a is None:
        print("Error: Could not load Family A data.")
        return
    
    print(f"  Family A: {len(ret_a)} days, Total Return: {(1+ret_a).prod()-1:.2%}")
    print(f"  Family B: {len(ret_b)} days, Total Return: {(1+ret_b).prod()-1:.2%}")
    
    # Run simulations
    print("\n[2] Simulating Portfolio Allocations...")
    print("-" * 60)
    
    simulator = PortfolioSimulator(ret_a, ret_b, config)
    results = simulator.run_all_scenarios()
    
    # Find optimal
    print("\n[3] Finding Optimal Allocation...")
    
    # Best Sharpe
    best_sharpe = max(results, key=lambda x: x['sharpe'])
    print(f"  Best Sharpe: {best_sharpe['weights'][0]*100:.0f}/{best_sharpe['weights'][1]*100:.0f} "
          f"(Sharpe={best_sharpe['sharpe']:.3f})")
    
    # Best Calmar
    best_calmar = max(results, key=lambda x: x['calmar'])
    print(f"  Best Calmar: {best_calmar['weights'][0]*100:.0f}/{best_calmar['weights'][1]*100:.0f} "
          f"(Calmar={best_calmar['calmar']:.3f})")
    
    # Lowest Drawdown
    best_dd = max(results, key=lambda x: x['max_drawdown'])  # Max DD is negative, so max = least severe
    print(f"  Lowest DD: {best_dd['weights'][0]*100:.0f}/{best_dd['weights'][1]*100:.0f} "
          f"(MaxDD={best_dd['max_drawdown']*100:.2f}%)")
    
    # Chart
    print("\n[4] Generating Charts...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Equity Curves
    ax1 = axes[0, 0]
    colors = plt.cm.viridis(np.linspace(0, 1, len(results)))
    for i, res in enumerate(results):
        w_a, w_b = res['weights']
        ax1.plot(res['equity_curve'].index, res['equity_curve'].values,
                 label=f"{w_a*100:.0f}/{w_b*100:.0f}", color=colors[i], alpha=0.8)
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.set_title('Equity Curves by Allocation')
    ax1.legend(title='A/B %', loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Metrics Bar Chart
    ax2 = axes[0, 1]
    labels = [f"{r['weights'][0]*100:.0f}/{r['weights'][1]*100:.0f}" for r in results]
    sharpes = [r['sharpe'] for r in results]
    ax2.bar(labels, sharpes, color='steelblue')
    ax2.set_ylabel('Sharpe Ratio')
    ax2.set_xlabel('Allocation (A/B)')
    ax2.set_title('Sharpe by Allocation')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Total Return
    ax3 = axes[1, 0]
    returns = [r['total_return'] * 100 for r in results]
    ax3.bar(labels, returns, color='green', alpha=0.7)
    ax3.set_ylabel('Total Return (%)')
    ax3.set_xlabel('Allocation (A/B)')
    ax3.set_title('Total Return by Allocation')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Max Drawdown
    ax4 = axes[1, 1]
    drawdowns = [r['max_drawdown'] * 100 for r in results]
    ax4.bar(labels, drawdowns, color='red', alpha=0.7)
    ax4.set_ylabel('Max Drawdown (%)')
    ax4.set_xlabel('Allocation (A/B)')
    ax4.set_title('Max Drawdown by Allocation')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    chart_path = os.path.join(os.path.dirname(__file__), "../../exp_CPI_004_portfolio.png")
    plt.savefig(chart_path, dpi=150)
    print(f"  Saved: {chart_path}")
    
    # Report
    print("\n[5] Generating Report...")
    
    report = f"""# EXP004: Portfolio Allocation Simulation

## Objective
Test different allocation splits between Family A (ML) and Family B (CPI) to find the optimal hedge ratio.

## Data Summary
- **Family A**: {len(ret_a)} trading days
- **Family B**: {len(ret_b)} CPI event-triggered trades
- **Correlation**: Stable negative (see EXP003)

## Results

| Allocation | Total Return | Sharpe | Max DD | Calmar |
|------------|--------------|--------|--------|--------|
"""
    
    for r in results:
        report += f"| {r['weights'][0]*100:.0f}/{r['weights'][1]*100:.0f} | {r['total_return']*100:.2f}% | {r['sharpe']:.3f} | {r['max_drawdown']*100:.2f}% | {r['calmar']:.3f} |\n"
    
    report += f"""
## Optimal Allocations

| Criterion | Best Allocation | Value |
|-----------|-----------------|-------|
| **Sharpe** | {best_sharpe['weights'][0]*100:.0f}/{best_sharpe['weights'][1]*100:.0f} | {best_sharpe['sharpe']:.3f} |
| **Calmar** | {best_calmar['weights'][0]*100:.0f}/{best_calmar['weights'][1]*100:.0f} | {best_calmar['calmar']:.3f} |
| **Lowest DD** | {best_dd['weights'][0]*100:.0f}/{best_dd['weights'][1]*100:.0f} | {best_dd['max_drawdown']*100:.2f}% |

## Recommendation

Based on the user's preference for **70/30 allocation**:
- Sharpe: {[r for r in results if r['weights']==(0.7, 0.3)][0]['sharpe']:.3f}
- Max DD: {[r for r in results if r['weights']==(0.7, 0.3)][0]['max_drawdown']*100:.2f}%

## Chart
![Portfolio Allocation](exp_CPI_004_portfolio.png)
"""
    
    report_path = os.path.join(os.path.dirname(__file__), "../../exp_CPI_004_results.md")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"  Saved: {report_path}")
    
    print("\n" + "=" * 60)
    print("EXP004 COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
