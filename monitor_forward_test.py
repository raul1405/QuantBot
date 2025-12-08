import pandas as pd
import numpy as np
import os
import sys

# CONFIG
LOG_FILE = "live_logs/FT_001_trades.csv"
INITIAL_BALANCE = 100000.0 # Adjust as needed
DD_LIMIT_PCT = 0.10 # 10%
KILL_SWITCH_WR = 0.30 # 30% WR
KILL_SWITCH_PF = 1.0

def load_trades():
    if not os.path.exists(LOG_FILE):
        print(f"[WARN] Log file {LOG_FILE} not found.")
        return None
    try:
        df = pd.read_csv(LOG_FILE)
        df['Entry_Time'] = pd.to_datetime(df['Entry_Time'])
        df['Exit_Time'] = pd.to_datetime(df['Exit_Time'])
        return df
    except Exception as e:
        print(f"[ERROR] Could not read log file: {e}")
        return None

def analyze_performance(df):
    if df is None or len(df) == 0:
        print("No trades found.")
        return

    # Sort
    df = df.sort_values('Exit_Time')
    
    # Equity Curve
    # Assuming PnL is in dollars
    df['Equity'] = INITIAL_BALANCE + df['PnL'].cumsum()
    current_equity = df['Equity'].iloc[-1]
    peak_equity = df['Equity'].cummax()
    drawdown_pct = (df['Equity'] - peak_equity) / peak_equity
    max_dd = drawdown_pct.min()
    
    # Stats
    n_trades = len(df)
    n_wins = len(df[df['PnL'] > 0])
    win_rate = n_wins / n_trades
    
    gross_profit = df[df['PnL'] > 0]['PnL'].sum()
    gross_loss = abs(df[df['PnL'] < 0]['PnL'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
    
    mean_r = df['R_Multiple'].mean()
    total_pnl = df['PnL'].sum()
    
    # REPORT
    print("\n" + "="*40)
    print(f"FORWARD TEST MONITOR (FT_001)")
    print("="*40)
    print(f"Trades:       {n_trades}")
    print(f"Win Rate:     {win_rate*100:.1f}%  (Limit: >{KILL_SWITCH_WR*100:.0f}%)")
    print(f"Profit Factor:{profit_factor:.2f}   (Limit: >{KILL_SWITCH_PF:.1f})")
    print(f"Mean R:       {mean_r:.3f}")
    print("-"*40)
    print(f"Total PnL:    ${total_pnl:,.2f}")
    print(f"Curr Equity:  ${current_equity:,.2f}")
    print(f"Max DD:       {max_dd*100:.2f}%    (Limit: -{DD_LIMIT_PCT*100:.0f}%)")
    print("="*40)
    
    # KILL SWITCH CHECKS
    # 1. Drawdown
    if max_dd < -DD_LIMIT_PCT:
        print("\n[CRITICAL] KILL SWITCH TRIGGERED: MAX DRAWDOWN EXCEEDED!")
        print(f"Current Max DD: {max_dd*100:.2f}% vs Limit: -{DD_LIMIT_PCT*100:.0f}%")
        
    # 2. Win Rate (Sample size check)
    if n_trades >= 20 and win_rate < KILL_SWITCH_WR:
        print("\n[CRITICAL] KILL SWITCH TRIGGERED: WIN RATE CRITICAL!")
    
    # 3. Profit Factor (Sample size check)
    if n_trades >= 20 and profit_factor < KILL_SWITCH_PF:
         print("\n[WARNING] Profit Factor below 1.0. Monitor closely.")

    # Recent Trend (Last 10 trades)
    if n_trades > 10:
        recent = df.tail(10)
        recent_wr = len(recent[recent['PnL'] > 0]) / 10
        print(f"\nLast 10 Trades WR: {recent_wr*100:.0f}%")
        
    # DRIFT MONITOR (Live vs Backtest)
    # Baseline from v2.1 Final Verification (MC Mode)
    BASELINE_MEAN_R = 0.12
    BASELINE_WR = 0.45
    DRIFT_TOLERANCE = 0.25 # 25% deviation allowed
    
    print("\n" + "="*40)
    print("DRIFT MONITOR (Live vs Frozen Spec v2.1)")
    print("="*40)
    
    # 1. R-Multiple Drift
    r_drift = (mean_r - BASELINE_MEAN_R) / BASELINE_MEAN_R
    print(f"Mean R:       {mean_r:.3f} vs {BASELINE_MEAN_R} (Drift: {r_drift*100:+.1f}%)")
    if r_drift < -DRIFT_TOLERANCE:
        print("  [WARNING] Trade Quality degrading significantly (>25% below baseline).")
        
    # 2. Win Rate Drift
    wr_drift = (win_rate - BASELINE_WR) / BASELINE_WR
    print(f"Win Rate:     {win_rate*100:.1f}% vs {BASELINE_WR*100:.1f}% (Drift: {wr_drift*100:+.1f}%)")
    
    # 3. ALERTS & REPORTING
    check_alerts(df, mean_r, max_dd, win_rate)
    
    if n_trades >= 50 and n_trades % 50 == 0:
        generate_drift_report(df, n_trades, mean_r, win_rate, max_dd)

def check_alerts(df, mean_r, max_dd, win_rate):
    """
    Checks for critical failure conditions.
    """
    alerts = []
    
    # Condition A: Quality Collapse
    # Check last 100 trades if possible, else all
    subset = df.tail(100)
    if len(subset) > 20: 
        # Recalculate R for subset
        subset_r = subset['R_Multiple'].mean()
        if subset_r < 0.09:
            alerts.append(f"CRITICAL: Low Trade Quality (Mean R {subset_r:.3f} < 0.09)")
            
    # Condition B: Drawdown Danger
    # FTMO Limit is 10% (0.10). Alert at 50% of limit (0.05).
    if max_dd < -0.05:
        alerts.append(f"CRITICAL: Max Drawdown ({max_dd*100:.1f}%) exceeded 50% of Limit.")
        
    # Condition C: Profit Factor Collapse
    gross_profit = df[df['PnL']>0]['PnL'].sum()
    gross_loss = abs(df[df['PnL']<0]['PnL'].sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else 999.0
    if pf < 1.1:
        alerts.append(f"CRITICAL: Profit Factor Collapse ({pf:.2f} < 1.1)")
        
    if alerts:
        print("\n" + "!"*40)
        print("RED ALERTS TRIGGERED")
        print("!"*40)
        for a in alerts:
            print(f"  - {a}")
            
        # Write Alert File
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        with open(f"notes/alerts/ALERT_{timestamp}.md", "w") as f:
            f.write(f"# 🚨 SYSTEM ALERT {timestamp}\n\n")
            for a in alerts:
                f.write(f"* {a}\n")

def generate_drift_report(df, n_trades, mean_r, win_rate, max_dd):
    timestamp = pd.Timestamp.now().strftime("%Y%m%d")
    filename = f"notes/drift_reports/FT_001_report_{n_trades}_{timestamp}.md"
    
    with open(filename, "w") as f:
        f.write(f"# Drift Report: Trade #{n_trades}\n")
        f.write(f"**Date**: {timestamp}\n\n")
        
        f.write("## 1. High-Level Health\n")
        f.write(f"| Metric | Live | Baseline (v2.1) | Status |\n")
        f.write(f"|---|---|---|---|\n")
        f.write(f"| **Mean R** | {mean_r:.4f} | 0.1200 | {'✅' if mean_r >= 0.09 else '❌'} |\n")
        f.write(f"| **Win Rate** | {win_rate*100:.1f}% | 45.0% | {'✅' if win_rate >= 0.35 else '⚠️'} |\n")
        f.write(f"| **Max DD** | {max_dd*100:.2f}% | < 10.0% | {'✅' if max_dd > -0.05 else '⚠️'} |\n")
        
        f.write("\n## 2. Recent Performance (Last 50)\n")
        recent = df.tail(50)
        rec_r = recent['R_Multiple'].mean()
        rec_wr = len(recent[recent['PnL'] > 0]) / 50
        f.write(f"| Mean R | Win Rate |\n")
        f.write(f"| {rec_r:.4f} | {rec_wr*100:.1f}% |\n")
    
    print(f"\n[REPORT] Generated {filename}")

if __name__ == "__main__":
    df = load_trades()
    if df is not None and not df.empty:
        analyze_performance(df)
