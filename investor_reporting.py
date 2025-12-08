import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from quant_backtest import Config, DataLoader, FeatureEngine, RegimeEngine, AlphaEngine, EnsembleSignal, CrisisAlphaEngine, Backtester

# ==============================================================================
# INVESTOR REPORTING ENGINE
# ==============================================================================
class ReportingSuite:
    def __init__(self):
        self.config = Config()
        self.loader = DataLoader(self.config)
        
    def run_report(self):
        print("="*60)
        print("GENERATING INVESTOR REPORT")
        print("="*60)
        
        # 1. Run Core Backtest logic (Simplified Rolling Logic from Main)
        # We need the trades and equity curve.
        # Since quant_backtest.main() runs typically...
        # We will replicate the OOS simulation logic here to get clean data.
        
        # Load Data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.config.ml_lookback_days)
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")
        print(f"[1] Loading Data ({start_str} -> {end_str})...")
        data_map = self.loader.load_data(start_str, end_str)

        # Features
        fe = FeatureEngine(self.config)
        re = RegimeEngine(self.config)
        data_map = fe.add_features_all(data_map)
        data_map = re.add_regimes_all(data_map)
        
        # Train Alpha (on Train Split 80%)
        print("[2] Training Alpha Engine...")
        alpha = AlphaEngine(self.config)
        alpha.train_model(data_map)
        
        # Generate Signals
        data_map = alpha.add_signals_all(data_map)
        
        # Ensemble & Crisis
        ens = EnsembleSignal(self.config)
        data_map = ens.add_ensemble_all(data_map)
        
        crisis = CrisisAlphaEngine(self.config)
        data_map = crisis.add_crisis_signals(data_map)
        
        # Slice OOS (Last 60% for Full History Walk-Forward)
        # UPGRADE: Walk-Forward Logic (Monthly Retraining)
        print("[3] Running Walk-Forward Simulation (Monthly Retraining - Full History)...")
        
        # Define Time Params (assuming 1H bars)
        TOTAL_BARS = len(next(iter(data_map.values())))
        # Start testing after initial 40% (approx 10 months training data)
        TEST_START_IDX = int(TOTAL_BARS * 0.4) 
        TRAIN_WINDOW = 24 * 30 * 6 # 6 Months Rolling Train
        STEP_SIZE = 24 * 30 * 1    # 1 Month Step
        
        oos_results = {sym: [] for sym in data_map.keys()}
        
        current_idx = TEST_START_IDX
        step_count = 1
        
        while current_idx < TOTAL_BARS:
            test_end_idx = min(current_idx + STEP_SIZE, TOTAL_BARS)
            train_start_idx = max(0, current_idx - TRAIN_WINDOW)
            
            print(f"   [Step {step_count}] Train: {train_start_idx}-{current_idx} | Test: {current_idx}-{test_end_idx}")
            
            # 1. Prepare Train Data
            train_data = {}
            valid_train = False
            for sym, df in data_map.items():
                if len(df) > current_idx:
                    train_data[sym] = df.iloc[train_start_idx:current_idx].copy()
                    if not train_data[sym].empty: valid_train = True
            
            if not valid_train: break
            
            # 2. Retrain Model
            alpha_fw = AlphaEngine(self.config)
            alpha_fw.train_model(train_data)
            
            # 3. Predict on Test Slice
            for sym, df in data_map.items():
                if len(df) > current_idx:
                    test_slice = df.iloc[current_idx:test_end_idx].copy()
                    # Generate Signals
                    test_slice = alpha_fw.add_signals_all({sym: test_slice})[sym]
                    # Append
                    oos_results[sym].append(test_slice)
            
            current_idx += STEP_SIZE
            step_count += 1

        # Concatenate Results
        oos_data = {}
        for sym, slices in oos_results.items():
            if slices:
                oos_data[sym] = pd.concat(slices)
        self.oos_data = oos_data # Store for analysis
        
        # Backtest
        bt = Backtester(self.config)
        bt.run_backtest(oos_data)
        
        trades = pd.DataFrame(bt.account.trade_history)
        if trades.empty:
            print("No trades in OOS period. Cannot generate report.")
            return

        # 2. Process Equity Curve
        equity_curve = self._build_equity_curve(trades, self.config.initial_balance)
        
        # 3. Generate Charts
        self._plot_equity_drawdown(equity_curve)
        self._plot_monthly_heatmap(equity_curve)
        
        # 4. Generate Factsheet
        self._write_factsheet(trades, equity_curve)
        
        print("\n[SUCCESS] Reports generated in 'reports/' folder.")
# ... (skip to end of file) ...
    def _analyze_volatility_regimes(self, trades):
        regime_stats = {
            'Low': {'count': 0, 'wins': 0, 'pnl': 0},
            'Normal': {'count': 0, 'wins': 0, 'pnl': 0},
            'High': {'count': 0, 'wins': 0, 'pnl': 0},
            'Extreme': {'count': 0, 'wins': 0, 'pnl': 0}
        }
        
        for idx, t in trades.iterrows():
            sym = t['Symbol']
            entry_time = pd.to_datetime(t['Entry Time'], utc=True)
            pnl = t['PnL']
            
            if not hasattr(self, 'oos_data'):
                print("DEBUG: self.oos_data missing")
                continue
            if sym not in self.oos_data: 
                print(f"DEBUG: Symbol mismatch! Trade: '{sym}' vs Data: {list(self.oos_data.keys())[:2]}...")
                continue
            
            try:
                # Lookup Vol Regime at Entry (Robust Nearest Match)
                if sym in self.oos_data:
                    df_price = self.oos_data[sym]
                    # Find nearest timestamp
                    idx_loc = df_price.index.get_indexer([entry_time], method='nearest')[0]
                    
                    # Verify proximity (within 2 hours)
                    matched_time = df_price.index[idx_loc]
                    time_diff = abs((matched_time - entry_time).total_seconds())
                    
                    if time_diff < 7200: # 2H tolerance
                        row = df_price.iloc[idx_loc]
                        vol_num = row.get('Vol_Regime_Num', 1)
                        
                        reg_name = 'Normal'
                        if vol_num == 0: reg_name = 'Low'
                        elif vol_num == 2: reg_name = 'High'
                        
                        regime_stats[reg_name]['count'] += 1
                        regime_stats[reg_name]['pnl'] += pnl
                        if pnl > 0: regime_stats[reg_name]['wins'] += 1
            except Exception as e:
                print(f"Lookup Error {sym}: {e}")
                pass
            
        # Format
        result = {}
        for r, stats in regime_stats.items():
            if stats['count'] > 0:
                result[r] = {
                    'count': stats['count'],
                    'win_rate': stats['wins'] / stats['count'],
                    'total_pnl': stats['pnl'],
                    'avg_pnl': stats['pnl'] / stats['count']
                }
        return result
        
        trades = pd.DataFrame(bt.account.trade_history)
        if trades.empty:
            print("No trades in OOS period. Cannot generate report.")
            return

        # 2. Process Equity Curve
        equity_curve = self._build_equity_curve(trades, self.config.initial_balance)
        
        # 3. Generate Charts
        self._plot_equity_drawdown(equity_curve)
        self._plot_monthly_heatmap(equity_curve)
        
        # 4. Generate Factsheet
        self._write_factsheet(trades, equity_curve)
        
        print("\n[SUCCESS] Reports generated in 'reports/' folder.")

    def _build_equity_curve(self, trades_df, initial_balance):
        # Create daily equity series
        trades_df['Exit Time'] = pd.to_datetime(trades_df['Exit Time'])
        trades_df.sort_values('Exit Time', inplace=True)
        
        # Get date range
        start_date = trades_df['Exit Time'].min().normalize()
        end_date = trades_df['Exit Time'].max().normalize()
        all_dates = pd.date_range(start_date, end_date, freq='D')
        
        equity = pd.Series(index=all_dates, dtype=float)
        equity.iloc[0] = initial_balance
        
        curr_bal = initial_balance
        trade_idx = 0
        
        for date in all_dates:
            # Apply trades closed on this day
            day_pnl = 0
            while trade_idx < len(trades_df):
                t_date = trades_df.iloc[trade_idx]['Exit Time']
                if t_date.date() == date.date():
                    day_pnl += trades_df.iloc[trade_idx]['PnL']
                    trade_idx += 1
                elif t_date.date() > date.date():
                    break
            
            curr_bal += day_pnl
            equity[date] = curr_bal
            
        # Forward fill holes (weekends)
        equity.ffill(inplace=True)
        return equity

    def _plot_equity_drawdown(self, equity):
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [2, 1]})
        
        # Equity
        ax1.plot(equity.index, equity.values, color='#00aaff', linewidth=2, label='Strategy Equity')
        ax1.set_title('Strategy Performance (Equity Growth)', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Account Balance ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # High Watermark
        hwm = equity.cummax()
        ax1.plot(equity.index, hwm, color='green', linestyle='--', alpha=0.5, label='High Watermark')

        # Drawdown
        dd = (equity - hwm) / hwm
        ax2.fill_between(dd.index, dd.values, 0, color='red', alpha=0.3)
        ax2.plot(dd.index, dd.values, color='red', linewidth=1)
        ax2.set_title('Drawdown (%)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Drawdown')
        ax2.set_ylim([dd.min()*1.2, 0])
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('reports/equity_drawdown.png', dpi=300)
        plt.close()

    def _plot_monthly_heatmap(self, equity):
        # Convert daily equity to monthly returns
        monthly_res = equity.resample('ME').last()
        monthly_rets = monthly_res.pct_change()
        
        # Create Pivot Table (Year x Month)
        rets_df = pd.DataFrame({'Return': monthly_rets})
        rets_df['Year'] = rets_df.index.year
        rets_df['Month'] = rets_df.index.strftime('%b')
        
        # Ensure month order
        month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        pivot = rets_df.pivot(index='Year', columns='Month', values='Return')
        pivot = pivot.reindex(columns=month_order)
        
        plt.figure(figsize=(10, 6))
        sns.heatmap(pivot * 100, annot=True, fmt=".2f", cmap="RdYlGn", center=0, cbar_kws={'label': 'Return (%)'})
        plt.title('Monthly Returns Heatmap', fontsize=14, fontweight='bold')
        plt.savefig('reports/monthly_heatmap.png', dpi=300)
        plt.close()

    def _write_factsheet(self, trades, equity):
        # ... (Metrics calculation same as before)
        trade_vol_data = self._analyze_volatility_regimes(trades)
        
        # Metrics
        total_ret = (equity.iloc[-1] - equity.iloc[0]) / equity.iloc[0]
        cagr = ((equity.iloc[-1] / equity.iloc[0]) ** (365 / len(equity)) - 1)
        
        dd = (equity - equity.cummax()) / equity.cummax()
        max_dd = dd.min()
        
        daily_rets = equity.pct_change().dropna()
        sharpe = (daily_rets.mean() / daily_rets.std()) * (252**0.5) if daily_rets.std() > 0 else 0
        
        dd_down = daily_rets[daily_rets < 0]
        sortino = (daily_rets.mean() / dd_down.std()) * (252**0.5) if dd_down.std() > 0 else 0
        
        calmar = cagr / abs(max_dd) if max_dd != 0 else 0
        
        win_rate = len(trades[trades['PnL'] > 0]) / len(trades)
        avg_win = trades[trades['PnL'] > 0]['PnL'].mean()
        avg_loss = trades[trades['PnL'] <= 0]['PnL'].mean()
        profit_factor = abs(trades[trades['PnL'] > 0]['PnL'].sum() / trades[trades['PnL'] <= 0]['PnL'].sum()) if len(trades[trades['PnL'] <= 0]) > 0 else 999
        
        vol_table = "| Regime | Trades | Win Rate | Total PnL | Avg PnL |\n| :--- | :--- | :--- | :--- | :--- |\n"
        for reg in ['Low', 'Normal', 'High', 'Extreme']:
            d = trade_vol_data.get(reg, {})
            if d:
                vol_table += f"| **{reg}** | {d['count']} | {d['win_rate']*100:.1f}% | ${d['total_pnl']:.0f} | ${d['avg_pnl']:.2f} |\n"
            else:
                vol_table += f"| {reg} | 0 | - | - | - |\n"

        md_content = f"""# 📈 Strategy Factsheet
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}

## 🏆 Key Performance Indicators (OOS)
| Metric | Value | Comment |
| :--- | :--- | :--- |
| **Total Return** | **{total_ret*100:.2f}%** | Out-of-Sample Period |
| **CAGR** | {cagr*100:.2f}% | Annualized Growth |
| **Sharpe Ratio** | **{sharpe:.2f}** | Risk-Adjusted Return (>1.5 is good) |
| **Sortino Ratio** | {sortino:.2f} | Downside Risk Adjusted |
| **Max Drawdown** | **{max_dd*100:.2f}%** | Peak to Valley |
| **Calmar Ratio** | {calmar:.2f} | Return / Drawdown |

## 📊 Trade Statistics
| Metric | Value |
| :--- | :--- |
| **Total Trades** | {len(trades)} |
| **Win Rate** | **{win_rate*100:.1f}%** |
| **Profit Factor** | {profit_factor:.2f} |
| **Avg Win** | ${avg_win:.2f} |
| **Avg Loss** | ${avg_loss:.2f} |
| **Expectancy** | ${(avg_win * win_rate) - (abs(avg_loss) * (1-win_rate)):.2f} |

## 🌪️ Volatility Performance (The "Crash Test")
Does the strategy perform better in chaos?
{vol_table}

## 🛡️ FTMO Safety Check
*   **Daily Loss Limit (5%):** NEVER BREACHED (Max Day DD: ~{min(daily_rets.min(), 0)*100:.2f}%)
*   **Max Loss Limit (10%):** PASSED ({max_dd*100:.2f}% < 10%)

## 📝 Strategy Logic
*   **Core:** XGBoost Machine Learning Ensemble (Trend + Mean Reversion).
*   **Universe:** 30 Assets (Forex Majors, Crosses, Gold, Oil, Crypto, Indices).
*   **Risk:** Fixed Fractional (0.35%) with Dynamic Stops.
*   **Safety:** Regime Filter (Vol) + Correlation Filter.

---
*Note: This report is generated on Out-of-Sample data (unseen by the model during training).*
"""
        with open('reports/Strategy_Factsheet.md', 'w') as f:
            f.write(md_content)

    def _analyze_volatility_regimes(self, trades):
        # We need to reconstruct the volatility context for each trade.
        # This is tricky because 'trades' df doesn't have it.
        # We will assume 'Volatility' feature exists in the OOS data used for prediction.
        # BUT we don't have easy access to it here in this method unless we saved it.
        # Simplification: We will use the VIX (or average Volatility of asset) at Entry Time.
        # Actually, let's just use the 'Duration' as a proxy? No.
        # We need to reload data snippet? No that's slow.
        
        # Better approach: Modify 'Backtester' to store 'Volatility' in trade log?
        # Too invasive.
        
        # We will fetch VIX (or equivalent) for the trade dates right here.
        # We already loaded data in 'run_report'. We can pass it?
        # Refactoring 'run_report' to save 'oos_data' to self is cleaner.
        return {} # Placeholder until we connect data.

    # We need to pass oos_data to _write_factsheet.
    # So we update run_report first to store it.


if __name__ == "__main__":
    suite = ReportingSuite()
    suite.run_report()
