"""
Generate sample visualizations for README.md
These are illustrative plots demonstrating the system's analytical capabilities.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import os

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.facecolor'] = '#0d1117'
plt.rcParams['axes.facecolor'] = '#161b22'
plt.rcParams['axes.edgecolor'] = '#30363d'
plt.rcParams['axes.labelcolor'] = '#c9d1d9'
plt.rcParams['text.color'] = '#c9d1d9'
plt.rcParams['xtick.color'] = '#8b949e'
plt.rcParams['ytick.color'] = '#8b949e'
plt.rcParams['grid.color'] = '#21262d'
plt.rcParams['legend.facecolor'] = '#161b22'
plt.rcParams['legend.edgecolor'] = '#30363d'

# Create output directory
os.makedirs('docs/images', exist_ok=True)

# Watermark configuration
WATERMARK_TEXT = "© Raul Schalkhammer 2025"

def add_watermark(fig):
    """Add author watermark to figure."""
    fig.text(0.99, 0.01, WATERMARK_TEXT, 
             fontsize=9, color='#8b949e', alpha=0.7,
             ha='right', va='bottom',
             transform=fig.transFigure,
             style='italic')

# =============================================================================
# 1. EQUITY CURVE WITH DRAWDOWN
# =============================================================================
def plot_equity_drawdown():
    np.random.seed(42)
    
    # Generate realistic equity curve
    days = 365
    dates = pd.date_range(start='2024-01-01', periods=days, freq='D')
    
    # Simulate returns with regime changes
    returns = np.random.normal(0.0003, 0.008, days)
    
    # Add some crisis periods
    returns[60:75] = np.random.normal(-0.005, 0.015, 15)  # Q1 Correction
    returns[180:195] = np.random.normal(-0.008, 0.02, 15)  # Summer Vol
    returns[280:290] = np.random.normal(-0.003, 0.01, 10)  # Q4 Wobble
    
    # Build equity curve
    equity = 100000 * np.cumprod(1 + returns)
    
    # Calculate drawdown
    hwm = np.maximum.accumulate(equity)
    drawdown = (equity - hwm) / hwm
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), 
                                     gridspec_kw={'height_ratios': [2.5, 1]},
                                     sharex=True)
    
    # Equity curve
    ax1.plot(dates, equity, color='#58a6ff', linewidth=2, label='Strategy Equity')
    ax1.plot(dates, hwm, color='#238636', linewidth=1, linestyle='--', 
             alpha=0.7, label='High Watermark')
    ax1.fill_between(dates, equity, hwm, where=(equity < hwm), 
                     color='#f85149', alpha=0.15)
    
    ax1.set_ylabel('Account Value ($)', fontsize=12)
    ax1.set_title('Strategy Performance: Equity Curve & Drawdown Analysis', 
                  fontsize=14, fontweight='bold', pad=15)
    ax1.legend(loc='upper left', framealpha=0.9)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # Add annotations for crisis periods
    ax1.annotate('Q1 Correction', xy=(dates[67], equity[67]), 
                 xytext=(dates[40], equity[67]*1.03),
                 arrowprops=dict(arrowstyle='->', color='#f85149'),
                 fontsize=9, color='#f85149')
    
    # Drawdown
    ax2.fill_between(dates, drawdown * 100, 0, color='#f85149', alpha=0.4)
    ax2.plot(dates, drawdown * 100, color='#f85149', linewidth=1.5)
    ax2.axhline(y=-5, color='#ffa657', linestyle='--', linewidth=1, 
                label='5% Daily Limit')
    ax2.axhline(y=-10, color='#f85149', linestyle='--', linewidth=1, 
                label='10% Max DD')
    
    ax2.set_ylabel('Drawdown (%)', fontsize=12)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylim([min(drawdown * 100) * 1.3, 2])
    ax2.legend(loc='lower left', framealpha=0.9)
    
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    
    plt.tight_layout()
    add_watermark(fig)
    plt.savefig('docs/images/equity_drawdown.png', dpi=150, 
                bbox_inches='tight', facecolor='#0d1117')
    plt.close()
    print("✓ Generated: equity_drawdown.png")

# =============================================================================
# 2. MONTHLY RETURNS HEATMAP
# =============================================================================
def plot_monthly_heatmap():
    np.random.seed(42)
    
    # Generate monthly returns for 2 years
    years = [2023, 2024]
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    # Create realistic return pattern
    returns = np.array([
        [1.2, 0.8, -2.1, 1.5, 2.3, -0.5, 1.1, -1.8, 0.4, 2.1, 1.6, 0.9],  # 2023
        [2.1, 1.4, -3.2, 0.8, 1.9, 0.3, -0.7, 1.2, 1.8, 0.0, 0.0, 0.0]   # 2024
    ])
    
    # Mark future months as NaN
    returns[1, 10:] = np.nan
    
    fig, ax = plt.subplots(figsize=(12, 4))
    
    # Create heatmap
    cmap = plt.cm.RdYlGn
    cmap.set_bad(color='#21262d')
    
    im = ax.imshow(returns, cmap=cmap, aspect='auto', vmin=-5, vmax=5)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Return (%)', fontsize=11)
    
    # Configure axes
    ax.set_xticks(np.arange(len(months)))
    ax.set_yticks(np.arange(len(years)))
    ax.set_xticklabels(months)
    ax.set_yticklabels(years)
    
    # Add text annotations
    for i in range(len(years)):
        for j in range(len(months)):
            if not np.isnan(returns[i, j]):
                color = 'white' if abs(returns[i, j]) > 2 else '#c9d1d9'
                ax.text(j, i, f'{returns[i, j]:.1f}%',
                       ha='center', va='center', color=color, fontweight='bold')
    
    ax.set_title('Monthly Returns Heatmap: Calendar View', 
                 fontsize=14, fontweight='bold', pad=15)
    
    plt.tight_layout()
    add_watermark(fig)
    plt.savefig('docs/images/monthly_heatmap.png', dpi=150, 
                bbox_inches='tight', facecolor='#0d1117')
    plt.close()
    print("✓ Generated: monthly_heatmap.png")

# =============================================================================
# 3. REGIME CLASSIFICATION VISUALIZATION
# =============================================================================
def plot_regime_analysis():
    np.random.seed(42)
    
    days = 200
    dates = pd.date_range(start='2024-01-01', periods=days, freq='D')
    
    # Generate price data
    returns = np.random.normal(0.0005, 0.01, days)
    returns[50:70] = np.random.normal(-0.002, 0.025, 20)  # High vol period
    returns[120:140] = np.random.normal(0.001, 0.005, 20)  # Low vol period
    
    price = 100 * np.cumprod(1 + returns)
    
    # Calculate volatility
    vol = pd.Series(returns).rolling(20).std() * np.sqrt(252) * 100
    
    # Determine regimes
    vol_q25 = vol.quantile(0.25)
    vol_q75 = vol.quantile(0.75)
    
    regime = pd.Series('Normal', index=range(days))
    regime[vol > vol_q75] = 'High Vol'
    regime[vol < vol_q25] = 'Low Vol'
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), 
                                     gridspec_kw={'height_ratios': [2, 1]},
                                     sharex=True)
    
    # Price with regime shading
    ax1.plot(dates, price, color='#58a6ff', linewidth=2)
    
    # Shade regimes
    for i in range(len(dates)):
        if regime.iloc[i] == 'High Vol':
            ax1.axvspan(dates[i], dates[min(i+1, len(dates)-1)], 
                       alpha=0.2, color='#f85149')
        elif regime.iloc[i] == 'Low Vol':
            ax1.axvspan(dates[i], dates[min(i+1, len(dates)-1)], 
                       alpha=0.2, color='#238636')
    
    ax1.set_ylabel('Price', fontsize=12)
    ax1.set_title('Regime Classification: Volatility State Detection', 
                  fontsize=14, fontweight='bold', pad=15)
    
    # Add legend patches
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#f85149', alpha=0.3, label='High Volatility'),
        Patch(facecolor='#238636', alpha=0.3, label='Low Volatility'),
        Patch(facecolor='#0d1117', alpha=0.3, label='Normal')
    ]
    ax1.legend(handles=legend_elements, loc='upper left', framealpha=0.9)
    
    # Volatility
    ax2.plot(dates, vol, color='#a371f7', linewidth=2, label='Annualized Vol')
    ax2.axhline(y=vol_q75, color='#f85149', linestyle='--', alpha=0.7, 
                label=f'Q75 ({vol_q75:.1f}%)')
    ax2.axhline(y=vol_q25, color='#238636', linestyle='--', alpha=0.7, 
                label=f'Q25 ({vol_q25:.1f}%)')
    ax2.fill_between(dates, vol_q25, vol_q75, alpha=0.1, color='#8b949e')
    
    ax2.set_ylabel('Volatility (%)', fontsize=12)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.legend(loc='upper right', framealpha=0.9)
    
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator())
    
    plt.tight_layout()
    add_watermark(fig)
    plt.savefig('docs/images/regime_analysis.png', dpi=150, 
                bbox_inches='tight', facecolor='#0d1117')
    plt.close()
    print("✓ Generated: regime_analysis.png")

# =============================================================================
# 4. ENTROPY SIGNAL QUALITY
# =============================================================================
def plot_entropy_analysis():
    np.random.seed(42)
    
    days = 150
    dates = pd.date_range(start='2024-01-01', periods=days, freq='D')
    
    # Simulate entropy values
    entropy = 1.0 + 0.3 * np.sin(np.linspace(0, 6*np.pi, days)) + \
              np.random.normal(0, 0.1, days)
    entropy = np.clip(entropy, 0.3, 1.58)
    
    # Signal strength (inverse of entropy)
    signal_strength = 1.58 - entropy
    
    # Position sizing based on entropy
    position_mult = np.where(entropy < 1.0, 1.0, 
                            np.where(entropy < 1.3, 0.5, 0.0))
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    
    # Entropy
    ax1 = axes[0]
    ax1.plot(dates, entropy, color='#ffa657', linewidth=2)
    ax1.axhline(y=1.0, color='#238636', linestyle='--', alpha=0.7, 
                label='Low Entropy (High Confidence)')
    ax1.axhline(y=1.3, color='#f85149', linestyle='--', alpha=0.7, 
                label='High Entropy (Low Confidence)')
    ax1.fill_between(dates, entropy, 1.58, where=(entropy > 1.3), 
                     alpha=0.2, color='#f85149')
    ax1.fill_between(dates, entropy, 0, where=(entropy < 1.0), 
                     alpha=0.2, color='#238636')
    ax1.set_ylabel('Shannon Entropy (H)', fontsize=12)
    ax1.set_title('Uncertainty Quantification: Entropy-Based Position Governance', 
                  fontsize=14, fontweight='bold', pad=15)
    ax1.legend(loc='upper right', framealpha=0.9)
    ax1.set_ylim([0, 1.7])
    
    # Signal Strength
    ax2 = axes[1]
    colors = ['#238636' if e < 1.0 else '#ffa657' if e < 1.3 else '#f85149' 
              for e in entropy]
    ax2.bar(dates, signal_strength, color=colors, alpha=0.7, width=1)
    ax2.set_ylabel('Signal Clarity', fontsize=12)
    ax2.axhline(y=0.58, color='#58a6ff', linestyle='-', alpha=0.5)
    
    # Position Multiplier
    ax3 = axes[2]
    ax3.step(dates, position_mult, color='#58a6ff', linewidth=2, where='mid')
    ax3.fill_between(dates, position_mult, step='mid', alpha=0.3, color='#58a6ff')
    ax3.set_ylabel('Position Multiplier', fontsize=12)
    ax3.set_xlabel('Date', fontsize=12)
    ax3.set_ylim([-0.1, 1.2])
    ax3.set_yticks([0, 0.5, 1.0])
    ax3.set_yticklabels(['0% (Halted)', '50% (Reduced)', '100% (Full)'])
    
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax3.xaxis.set_major_locator(mdates.MonthLocator())
    
    plt.tight_layout()
    add_watermark(fig)
    plt.savefig('docs/images/entropy_analysis.png', dpi=150, 
                bbox_inches='tight', facecolor='#0d1117')
    plt.close()
    print("✓ Generated: entropy_analysis.png")

# =============================================================================
# 5. WALK-FORWARD OPTIMIZATION SCHEMA
# =============================================================================
def plot_wfo_schema():
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # WFO Folds
    n_folds = 5
    train_len = 6
    test_len = 2
    
    colors_train = ['#1f6feb', '#388bfd', '#58a6ff', '#79c0ff', '#a5d6ff']
    colors_test = ['#238636', '#2ea043', '#3fb950', '#56d364', '#7ee787']
    
    for i in range(n_folds):
        start = i * test_len
        
        # Training block
        ax.barh(i, train_len, left=start, height=0.6, 
                color=colors_train[i], alpha=0.8,
                label='Train' if i == 0 else '')
        
        # Test block
        ax.barh(i, test_len, left=start + train_len, height=0.6, 
                color=colors_test[i], alpha=0.8,
                label='Test (OOS)' if i == 0 else '')
        
        # Add labels
        ax.text(start + train_len/2, i, f'Train {i+1}', 
                ha='center', va='center', fontweight='bold', fontsize=10)
        ax.text(start + train_len + test_len/2, i, f'Test {i+1}', 
                ha='center', va='center', fontweight='bold', fontsize=10)
    
    # Add arrows showing the flow
    for i in range(n_folds):
        if i < n_folds - 1:
            ax.annotate('', xy=(i*test_len + train_len + test_len + 0.3, i - 0.5),
                       xytext=(i*test_len + train_len + test_len + 0.3, i + 0.5),
                       arrowprops=dict(arrowstyle='->', color='#8b949e', lw=1.5))
    
    ax.set_xlabel('Time (Months)', fontsize=12)
    ax.set_ylabel('Fold', fontsize=12)
    ax.set_title('Walk-Forward Optimization: Rolling Window Schema', 
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xlim([-0.5, n_folds * test_len + train_len + 0.5])
    ax.set_ylim([-0.5, n_folds - 0.5])
    ax.set_yticks(range(n_folds))
    ax.set_yticklabels([f'Fold {i+1}' for i in range(n_folds)])
    ax.legend(loc='upper right', framealpha=0.9)
    
    # Add annotation
    ax.text(6, -0.35, 'No Look-Ahead Bias: Each test set uses only past training data',
            fontsize=11, style='italic', color='#8b949e', ha='center')
    
    plt.tight_layout()
    add_watermark(fig)
    plt.savefig('docs/images/wfo_schema.png', dpi=150, 
                bbox_inches='tight', facecolor='#0d1117')
    plt.close()
    print("✓ Generated: wfo_schema.png")


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    print("\n" + "="*50)
    print("GENERATING README VISUALIZATIONS")
    print("="*50 + "\n")
    
    plot_equity_drawdown()
    plot_monthly_heatmap()
    plot_regime_analysis()
    plot_entropy_analysis()
    plot_wfo_schema()
    
    print("\n" + "="*50)
    print("All visualizations saved to docs/images/")
    print("="*50 + "\n")
