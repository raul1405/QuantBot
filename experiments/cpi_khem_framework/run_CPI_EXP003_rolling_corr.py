"""
EXP003: Rolling Correlation Analysis
=====================================
Compute rolling correlation between Family A and Family B
over multiple time windows to assess stability.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from correlation_research import RollingCorrelationAnalyzer, ResearchConfig


def main():
    print("=" * 60)
    print("EXP003: Rolling Correlation Analysis")
    print("=" * 60)
    
    config = ResearchConfig(corr_windows=[60, 120, 252])
    analyzer = RollingCorrelationAnalyzer(config)
    
    # Load data
    print("\n[1] Loading Data...")
    ret_a = analyzer.load_family_a()
    ret_b = analyzer.load_family_b(use_historical=True)
    
    if ret_a is None:
        print("Error: Could not load Family A data.")
        return
    
    print(f"  Family A: {len(ret_a)} days")
    print(f"  Family B: {len(ret_b)} days (CPI events mapped to exit dates)")
    
    # Compute rolling correlations
    print("\n[2] Computing Rolling Correlations...")
    rolling_results = analyzer.compute_rolling_correlation()
    
    # Stability Analysis
    print("\n[3] Stability Analysis")
    print("-" * 40)
    
    stability_data = []
    for window, corr_series in rolling_results.items():
        mean_c = corr_series.mean()
        std_c = corr_series.std()
        pct_negative = (corr_series < 0).mean() * 100
        
        stability_data.append({
            'Window': f"{window}d",
            'Mean Corr': mean_c,
            'Std Corr': std_c,
            '% Negative': pct_negative,
            'Stable': 'YES' if std_c < 0.3 else 'NO'
        })
        
        print(f"  {window}d: Mean={mean_c:.3f}, Std={std_c:.3f}, "
              f"Negative={pct_negative:.1f}%, Stable={'YES' if std_c < 0.3 else 'NO'}")
    
    stability_df = pd.DataFrame(stability_data)
    
    # Chart
    print("\n[4] Generating Charts...")
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot 1: Rolling Correlations
    ax1 = axes[0]
    colors = ['blue', 'green', 'red']
    for i, (window, corr_series) in enumerate(rolling_results.items()):
        ax1.plot(corr_series.index, corr_series.values, 
                 label=f'{window}d Window', color=colors[i], alpha=0.7)
    
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    ax1.axhline(y=-0.63, color='purple', linestyle=':', label='Initial Corr (-0.63)')
    ax1.set_ylabel('Correlation')
    ax1.set_title('Rolling Correlation: Family A (ML) vs Family B (CPI)')
    ax1.legend(loc='upper right')
    ax1.set_ylim(-1, 1)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Histogram
    ax2 = axes[1]
    for i, (window, corr_series) in enumerate(rolling_results.items()):
        ax2.hist(corr_series.dropna(), bins=50, alpha=0.5, 
                 label=f'{window}d Window', color=colors[i])
    
    ax2.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax2.axvline(x=-0.63, color='purple', linestyle=':', linewidth=2, label='Initial (-0.63)')
    ax2.set_xlabel('Correlation')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Correlation Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    chart_path = os.path.join(os.path.dirname(__file__), "../../exp_CPI_003_rolling_corr.png")
    plt.savefig(chart_path, dpi=150)
    print(f"  Saved: {chart_path}")
    
    # Report
    print("\n[5] Generating Report...")
    
    overall_verdict = "STABLE" if stability_df['Stable'].str.contains('YES').all() else "UNSTABLE"
    mean_overall = np.mean([r['Mean Corr'] for r in stability_data])
    
    report = f"""# EXP003: Rolling Correlation Analysis

## Objective
Assess the stability of the -0.63 correlation between Family A (ML) and Family B (CPI) over time.

## Data Summary
- **Family A**: {len(ret_a)} trading days
- **Family B**: {len(ret_b)} CPI event-triggered trades

## Rolling Correlation Results

| Window | Mean Corr | Std | % Negative | Stable? |
|--------|-----------|-----|------------|---------|
"""
    for row in stability_data:
        report += f"| {row['Window']} | {row['Mean Corr']:.3f} | {row['Std Corr']:.3f} | {row['% Negative']:.1f}% | {row['Stable']} |\n"
    
    report += f"""
## Verdict

**Overall Stability**: {overall_verdict}

**Mean Correlation (Across Windows)**: {mean_overall:.3f}

### Interpretation
"""
    
    if overall_verdict == "STABLE":
        report += """- ✅ Correlation is relatively stable across time windows
- ✅ The negative correlation suggests genuine diversification benefit
- ✅ Family B qualifies as a hedge candidate
"""
    else:
        report += """- ⚠️ Correlation shows significant regime-dependence
- ⚠️ Hedging benefit may be unreliable in certain market conditions
- ⚠️ Consider shadow mode before capital allocation
"""
    
    report += f"""
## Chart
![Rolling Correlation](exp_CPI_003_rolling_corr.png)
"""
    
    report_path = os.path.join(os.path.dirname(__file__), "../../exp_CPI_003_results.md")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"  Saved: {report_path}")
    
    print("\n" + "=" * 60)
    print("EXP003 COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
