# ==============================================================================
# 🚨 DO NOT CHANGE – FROZEN FOR FORWARD TEST FT_001 🚨
# Execution Logic Synced with quant_backtest.py v2.1
# ==============================================================================
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
import json
import os
import requests
import sys
from rich.console import Console
from rich.table import Table

# Rich console for proper terminal control
console = Console()

class SuppressStdout:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
from datetime import datetime, timedelta, timezone
from quant_backtest import Config, FeatureEngine, AlphaEngine, RegimeEngine, EnsembleSignal, CrisisAlphaEngine

# ==============================================================================
# CONFIGURATION
# ==============================================================================
# MT5 Login Details (REPLACE WITH YOUR CREDENTIALS OR LOAD FROM ENV)
MT5_LOGIN = 12345678 
MT5_PASSWORD = "your_password"
MT5_SERVER = "FTMO-Demo"

# Symbol Mapping: Yahoo Finance (yfinance) -> MT5
# ADJUST THESE KEYS TO MATCH YOUR BROKER EXACTLY (e.g. "EURUSD.pro", "US500.cash")
SYMBOL_MAP = {
    # --- FOREX ---
    "EURUSD=X": "EURUSD", "USDJPY=X": "USDJPY", "GBPUSD=X": "GBPUSD", "USDCHF=X": "USDCHF",
    "USDCAD=X": "USDCAD", "AUDUSD=X": "AUDUSD", "NZDUSD=X": "NZDUSD",
    "EURGBP=X": "EURGBP", "EURJPY=X": "EURJPY", "GBPJPY=X": "GBPJPY", "AUDJPY=X": "AUDJPY",
    "EURAUD=X": "EURAUD", "EURCHF=X": "EURCHF", "AUDNZD=X": "AUDNZD", "AUDCAD=X": "AUDCAD",
    "CADJPY=X": "CADJPY", "NZDJPY=X": "NZDJPY", "GBPCHF=X": "GBPCHF", "GBPAUD=X": "GBPAUD",
    "GBPCAD=X": "GBPCAD", "EURNZD=X": "EURNZD",
    
    # --- INDICES (CFDs) ---
    "ES=F": "US500",      # S&P 500 (Confirmed)
    "NQ=F": None,         # Nasdaq 100 (Temporarily Disabled - causing lag)
    "YM=F": "US30",       # Dow Jones (Confirmed)
    "RTY=F": None,        # Russell 2000 (Not Found)
    
    # --- COMMODITIES ---
    "GC=F": "XAUUSD",     # Gold (Confirmed)
    "CL=F": None,         # Crude Oil (Not Found)
    "NG=F": None,         # Natural Gas (Not Found)
    
    # --- CRYPTO ---
    "BTC-USD": None,      # (Not Found)
    "ETH-USD": None,      # (Not Found)
}
# Reverse map for internal logic
REVERSE_MAP = {v: k for k, v in SYMBOL_MAP.items()}

# ==============================================================================
# MT5 BRIDGE
# ==============================================================================
class MT5Connector:
    def __init__(self):
        if not mt5.initialize():
            print("initialization failed, error code =", mt5.last_error())
            quit()
        print(f"MT5 Initialized. Terminal: {mt5.terminal_info()}")
        
    def login(self, login, password, server):
        authorized = mt5.login(login, password=password, server=server)
        if authorized:
            print(f"Connected to account #{login}")
        else:
            print("failed to connect at account #{}, error code: {}".format(login, mt5.last_error()))
            quit()

    def get_data(self, symbol_mt5, n_bars=2000):
        # Ensure symbol is selected in Market Watch
        if not mt5.symbol_select(symbol_mt5, True):
            print(f"[WARN] Could not select {symbol_mt5} in Market Watch.")
            return None
            
        # Timeframe: H1
        rates = mt5.copy_rates_from_pos(symbol_mt5, mt5.TIMEFRAME_H1, 0, n_bars)
        if rates is None:
            print(f"Failed to get data for {symbol_mt5}")
            return None
            
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        # Rename columns to match quant_backtest (Open, High, Low, Close, Volume)
        df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'tick_volume': 'Volume'}, inplace=True)
        return df[['Open', 'High', 'Low', 'Close', 'Volume']]

    def get_open_positions(self):
        positions = mt5.positions_get()
        if positions is None:
             return []
        # Return simplified list
        pos_list = []
        for p in positions:
            pos_list.append({
                'ticket': p.ticket,
                'symbol': p.symbol,
                'type': 'BUY' if p.type == mt5.ORDER_TYPE_BUY else 'SELL',
                'volume': p.volume,
                'profit': p.profit,
                'sl': p.sl,
                'tp': p.tp,
                'open_price': p.price_open
            })
        return pos_list

    def close_position(self, ticket):
        pos = mt5.positions_get(ticket=ticket)
        if pos is None or len(pos) == 0:
            return False
        
        pos = pos[0]
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "position": ticket,
            "symbol": pos.symbol,
            "volume": pos.volume,
            "type": mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
            "price": mt5.symbol_info_tick(pos.symbol).bid if pos.type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(pos.symbol).ask,
            "deviation": 20,
            "magic": 234000,
            "comment": "python script close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
             print(f"Order Close Failed: {result.comment}")
             return False
        return True

    def open_order(self, symbol, order_type, volume, sl=0.0, tp=0.0):
        tick = mt5.symbol_info_tick(symbol)
        price = tick.ask if order_type == 'BUY' else tick.bid
        type_mt5 = mt5.ORDER_TYPE_BUY if order_type == 'BUY' else mt5.ORDER_TYPE_SELL
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(volume),
            "type": type_mt5,
            "price": price,
            "sl": float(sl),
            "tp": float(tp),
            "deviation": 20,
            "magic": 234000,
            "comment": "Antigravity Signal",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"Order Open Failed: {result.comment}")
            return None
        return result.order

    def get_account_info(self):
        return mt5.account_info()


# ==============================================================================
# LOGGING & STATE MANAGEMENT
# ==============================================================================
import csv
import json
import os

LOG_DIR = "live_logs"
TRADE_LOG_FILE = os.path.join(LOG_DIR, "FT_001_trades.csv")
STATE_FILE = "live_state_v2.json"

if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

class TradeLogger:
    def __init__(self, mode="paper"):
        self.mode = mode
        self.active_trades = {} # {ticket: {context}}
        self.load_state()
        self._init_csv()
        
    def _init_csv(self):
        filename = "FT_001_shadow.csv" if self.mode == "shadow" else "FT_001_trades.csv"
        self.log_file = os.path.join(LOG_DIR, filename)
        
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'Ticket', 'Symbol', 'Direction', 'Entry_Time', 'Exit_Time',
                    'Entry_Price', 'Exit_Price', 'Size', 'PnL', 'Balance',
                    'Risk_Dollars', 'R_Multiple', 
                    'Entry_Trend_Regime', 'Entry_Vol_Intensity', 
                    'Entry_Hour', 'Entry_Method'
                ])

    def load_state(self):
        state_file = f"live_state_{self.mode}.json"
        if os.path.exists(state_file):
            try:
                with open(state_file, 'r') as f:
                    self.active_trades = json.load(f)
                # Convert keys to int (JSON stores keys as strings)
                self.active_trades = {int(k): v for k, v in self.active_trades.items()}
            except Exception as e:
                print(f"[ERROR] Failed to load state: {e}")
    
    def save_state(self):
        state_file = f"live_state_{self.mode}.json"
        with open(state_file, 'w') as f:
             json.dump(self.active_trades, f, indent=4)
             
    # ... (on_entry unchanged except save_state) ...
    
    def on_entry(self, ticket, symbol, direction, size, price, context):
        print(f"[LOG] Registering Trade #{ticket} ({symbol})")
        self.active_trades[ticket] = {
            'Symbol': symbol,
            'Direction': direction,
            'Entry_Time': datetime.now(timezone.utc).isoformat(),
            'Size': size,
            'Entry_Price': price,
            **context 
        }
        self.save_state()
        
    def log_shadow_trade(self, symbol, direction, size, price, context):
        """
        Log a hypothetical trade for Shadow Mode.
        """
        # Generate pseudo-ticket
        import random
        ticket = random.randint(100000, 999999)
        
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                ticket, symbol, direction, datetime.now(timezone.utc).isoformat(), "",
                price, 0.0, size, 0.0, 0.0, 
                context['Risk_Dollars'], 0.0,
                context['Entry_Trend_Regime'], context['Entry_Vol_Intensity'],
                context['Entry_Hour'], 'Shadow'
            ])
        print(f"[SHADOW] Logged Virtual {symbol} {direction}")

    def check_closed_trades(self, open_tickets):
        # Shadow mode has no open tickets to check against MT5
        if self.mode == "shadow": return 
        
        # ... (Rest of check_closed_trades unchanged) ...
        closed_tickets = []
        for ticket in list(self.active_trades.keys()):
            if ticket not in open_tickets:
                self._process_closed_trade(ticket)
                
    def _process_closed_trade(self, ticket):
        # ... (Unchanged logic, but uses self.log_file) ...
        deals = mt5.history_deals_get(position=ticket)
        if deals is None or len(deals) == 0: return

        total_profit = 0.0
        exit_price = 0.0
        exit_time = ""
        
        for d in deals:
            total_profit += (d.profit + d.swap + d.commission)
            if d.entry == mt5.DEAL_ENTRY_OUT or d.entry == mt5.DEAL_ENTRY_OUT_BY:
                exit_price = d.price
                exit_time = datetime.fromtimestamp(d.time, tz=timezone.utc).isoformat()
        
        if exit_time == "": return 

        ctx = self.active_trades[ticket]
        risk_money = ctx.get('Risk_Dollars', 1.0)
        r_mult = total_profit / risk_money if risk_money != 0 else 0
        
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                ticket, ctx['Symbol'], ctx['Direction'], ctx['Entry_Time'], exit_time,
                ctx['Entry_Price'], exit_price, ctx['Size'], total_profit, 0.0, 
                risk_money, r_mult,
                ctx.get('Entry_Trend_Regime', ''), ctx.get('Entry_Vol_Intensity', ''),
                ctx.get('Entry_Hour', ''), 'Auto'
            ])
            
        print(f"[LOG] Trade Closed #{ticket} | PnL: ${total_profit:.2f} | R: {r_mult:.2f}")
        del self.active_trades[ticket]
        self.save_state()


class LiveTrader:
    def __init__(self, connector):
        self.mt5 = connector
        # Filter out symbols mapped to None (unavailable on broker)
        self.target_symbols = [k for k, v in SYMBOL_MAP.items() if v is not None]
        print(f"[INIT] Universe trimmed to {len(self.target_symbols)} available assets: {self.target_symbols}")
        self.gov_config = self.load_governance()
        
        mode = self.gov_config.get("mode", "paper")
        print(f"[INIT] Live Trader Mode: {mode.upper()}")
        
        self.logger = TradeLogger(mode=mode)
        self.config = Config()
        
        # Initialize Engines
        self.engines = {
            'feature': FeatureEngine(self.config),
            'regime': RegimeEngine(self.config),
            'alpha': AlphaEngine(self.config), 
            'ensemble': EnsembleSignal(self.config),
            'crisis': CrisisAlphaEngine(self.config)
        }

    # ... (load_governance, etc.) ...
    




    def load_governance(self):
        try:
            with open("live_config.json", "r") as f:
                return json.load(f)
        except:
            print("[WARN] live_config.json not found, using defaults.")
            return {
                "governance": {"max_daily_dd_pct": 0.045, "max_total_dd_pct": 0.095},
                "risk_caps": {"max_net_lots_usd": 5, "max_correlated_positions": 3}
            }

    def check_exposure_limit(self, symbol, new_direction, volume):
        """
        Check Cluster Risk (Dynamic Margin & Leverage Awareness)
        """
        # 1. Fetch Account Info
        account_info = self.mt5.account_info()
        if not account_info: return False
        
        balance = account_info.balance
        equity = account_info.equity
        leverage = self.gov_config.get('account_leverage', 30.0)
        
        # 2. Global Margin Check
        # Max Used Margin = 30% of Equity (Internal Safety Cap)
        # Or Notional Cap = 6x Equity
        
        # Calculate Current Notional & Margin
        open_positions = self.mt5.get_open_positions()
        
        current_margin = account_info.margin
        # Estimate new trade margin
        # Standard Lot = 100,000 units. Margin = (Price * Vol * 100k) / Leverage
        sym_price = self.mt5.symbol_info_tick(symbol).ask
        new_notional = sym_price * volume * 100000 
        new_margin = new_notional / leverage
        
        future_margin = current_margin + new_margin
        margin_limit = equity * 0.30 # Max 30% Used
        
        if future_margin > margin_limit:
            print(f"  [RISK BLOCK] Margin Limit: Used {future_margin:.2f} > Limit {margin_limit:.2f}")
            return False
            
        # 3. Cluster Exposure (USD Net)
        # Convert "5 Lots" fixed cap to "% of Equity"
        # 5 Lots on 100k = 500k Notional = 5x Equity.
        # Let's cap Net Cluster Exposure at 4x Equity (Conservative).
        
        max_cluster_notional = equity * 4.0
        
        net_usd_notional = 0.0
        
        # Proposed Trade
        if "USD" in symbol:
            if symbol.startswith("USD"): # USDJPY Long = Long USD
                net_usd_notional += (new_notional * new_direction)
            else: # EURUSD Long = Short USD
                net_usd_notional += (new_notional * -new_direction)
        
        # Existing positions
        for p in open_positions:
            s = p['symbol']
            vol = p['volume']
            d = 1 if p['type'] == 'BUY' else -1
            
            # approx price for check
            # We need Quote currency conversion if symbol is weird, 
            # but for major pairs, Price * Vol * 100k roughly approximates exposure value in USD 
            # (if USD is base or quote, it's close enough for a heuristic).
            p_price = p['price_open']
            p_notional = p_price * vol * 100000
            
            if "USD" in s:
                if s.startswith("USD"):
                    net_usd_notional += (p_notional * d)
                else:
                    net_usd_notional += (p_notional * -d)
                    
        if abs(net_usd_notional) > max_cluster_notional:
            print(f"  [RISK BLOCK] USD Cluster Limit: {net_usd_notional:,.0f} > {max_cluster_notional:,.0f}")
            return False
            
        return True
        
    def train_fresh_model(self):
        print("\n[TRAINING] Fetching History and Training Fresh Model...")
        data_map = {}
        for sym_int in self.target_symbols:
            sym_mt5 = SYMBOL_MAP.get(sym_int)
            if not sym_mt5: continue
            
            # Fetch lookback history (e.g. 730 days ~ 17000 hours)
            # MT5 usually has plenty.
            df = self.mt5.get_data(sym_mt5, n_bars=12000) # Approx 1.5 years
            if df is not None and not df.empty:
                data_map[sym_int] = df
                
        if len(data_map) == 0:
            print("[ERROR] No data fetched for training.")
            return

        # Prepare Features
        data_map = self.engines['feature'].add_features_all(data_map)
        data_map = self.engines['regime'].add_regimes_all(data_map)
        
        # Train Alpha
        print("[TRAINING ALPHA ENGINE (XGBOOST)]")
        self.engines['alpha'].train_model(data_map)
        
        # Save Model State
        self.save_model_state()
        print("[TRAINING] Complete.")

    def save_model_state(self):
        """Save trained model and scaler to disk."""
        import joblib
        from datetime import datetime
        state = {
            'model': self.engines['alpha'].model,
            'scaler': self.engines['alpha'].scaler,
            'timestamp': datetime.now().isoformat()
        }
        joblib.dump(state, 'model_state.joblib')
        print("[SYSTEM] Model saved to model_state.joblib")

    def load_model_state(self):
        """Load model if recent (< 24h)."""
        import joblib
        import os
        from datetime import datetime, timedelta
        if not os.path.exists('model_state.joblib'):
            return False
            
        try:
            state = joblib.load('model_state.joblib')
            saved_time = datetime.fromisoformat(state['timestamp'])
            
            # Check age (e.g., 24 hours max)
            if datetime.now() - saved_time > timedelta(hours=24):
                print("[SYSTEM] Saved model is too old (>24h). Retraining.")
                return False
                
            self.engines['alpha'].model = state['model']
            self.engines['alpha'].scaler = state['scaler']
            print(f"[SYSTEM] Loaded Model from {saved_time} (Skipping Training)")
            return True
        except Exception as e:
            print(f"[WARN] Failed to load model: {e}")
            return False

    def run_cycle(self):
        # 0. CHECK CLOSED TRADES (Silent)
        current_positions = self.mt5.get_open_positions()
        current_tickets = [p['ticket'] for p in current_positions]
        self.logger.check_closed_trades(current_tickets)
        
        # 1. ACCOUNT & RISK CHECK
        acct = self.mt5.get_account_info()
        if acct is None: return

        # Load Daily State
        today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        state_file = "daily_state.json"
        start_equity = acct.equity
        try:
            if os.path.exists(state_file):
                with open(state_file, 'r') as f:
                    state = json.load(f)
                    if state.get("date") == today_str:
                        start_equity = state.get("start_equity")
                    else:
                        start_equity = acct.equity 
                        with open(state_file, 'w') as f:
                            json.dump({"date": today_str, "start_equity": start_equity}, f)
            else:
                 with open(state_file, 'w') as f:
                    json.dump({"date": today_str, "start_equity": start_equity}, f)
        except Exception:
            pass

        current_equity = acct.equity
        daily_dd_pct = (current_equity - start_equity) / start_equity
        limit_pct = self.gov_config['governance']['max_daily_dd_pct']
        
        if daily_dd_pct < -limit_pct: 
            print("\n!!! DAILY LOSS LIMIT HIT !!! CLOSING ALL.")
            for p in self.mt5.get_open_positions():
                self.mt5.close_position(p['ticket'])
            return
        
        # 2. FETCH DATA
        live_data = {}
        for sym_int in self.target_symbols:
            sym_mt5 = SYMBOL_MAP[sym_int]
            df = self.mt5.get_data(sym_mt5, n_bars=500)
            if df is not None:
                live_data[sym_int] = df
        
        # 3. GENERATE SIGNALS (Silent)
        with SuppressStdout():
            live_data = self.engines['feature'].add_features_all(live_data)
            live_data = self.engines['regime'].add_regimes_all(live_data)
            live_data = self.engines['alpha'].add_signals_all(live_data)
            live_data = self.engines['ensemble'].add_ensemble_all(live_data)
        
        # 4. BUILD SCANNER TABLE
        current_positions = self.mt5.get_open_positions()
        pos_map = {}
        for p in current_positions:
            sym_int = REVERSE_MAP.get(p['symbol'])
            if sym_int:
                pos_map[sym_int] = p
        
        scan_results = []
        for sym_int, df in live_data.items():
            last_bar = df.iloc[-1]
            prev_bar = df.iloc[-25] if len(df) >= 25 else df.iloc[0]  # ~24h ago for H1
            
            signal = last_bar.get('S_Alpha', 0)
            p_up = last_bar.get('prob_up', 0.0)
            p_down = last_bar.get('prob_down', 0.0)
            prob = p_up if signal == 1 else (p_down if signal == -1 else max(p_up, p_down))
            
            regime = str(last_bar.get('Trend_Regime', '?'))[:4]  # Shorten
            vol_int = float(last_bar.get('Vol_Intensity', 0.0))
            
            # ATR% = ATR / Close (volatility as % of price)
            atr = last_bar.get('ATR', 0)
            close = last_bar.get('Close', 1)
            atr_pct = (atr / close) * 100 if close else 0
            
            # 24h Change %
            chg_24h = ((close - prev_bar['Close']) / prev_bar['Close']) * 100 if prev_bar['Close'] else 0
            
            # Spread (from MT5 tick)
            mt5_sym = SYMBOL_MAP[sym_int]
            tick = mt5.symbol_info_tick(mt5_sym)
            spread_pts = (tick.ask - tick.bid) * 10000 if tick else 0  # Approx pips for FX
            
            # Position Status
            pos_status = ""
            if sym_int in pos_map:
                p = pos_map[sym_int]
                pos_status = "L" if p['type'] == 'BUY' else "S"
            
            scan_results.append({
                'sym': mt5_sym,
                'prob': prob,
                'sig': signal,
                'regime': regime,
                'vol': vol_int,
                'atr_pct': atr_pct,
                'chg': chg_24h,
                'spread': spread_pts,
                'pos': pos_status,
                'sym_int': sym_int
            })
            
        # Sort by Probability
        scan_results.sort(key=lambda x: x['prob'], reverse=True)
        
        # === BUILD RICH TABLE ===
        table = Table(title=f"QuantBot Live | {datetime.now().strftime('%H:%M:%S')} | Positions: {len(pos_map)}", show_header=True, header_style="bold cyan")
        table.add_column("SYM", style="white", width=8)
        table.add_column("PROB", justify="right", width=5)
        table.add_column("SIG", justify="right", width=3)
        table.add_column("ACT", justify="center", width=4)
        table.add_column("TRND", justify="center", width=5)
        table.add_column("VOL", justify="right", width=5)
        table.add_column("ATR%", justify="right", width=5)
        table.add_column("24h%", justify="right", width=6)
        table.add_column("SPR", justify="right", width=4)
        table.add_column("POS", justify="center", width=3)
        
        for res in scan_results:
            action = "BUY" if res['sig'] == 1 else ("SELL" if res['sig'] == -1 else "-")
            sig_style = "green" if res['sig'] == 1 else ("red" if res['sig'] == -1 else "dim")
            table.add_row(
                res['sym'],
                f"{res['prob']:.2f}",
                str(res['sig']),
                action,
                res['regime'],
                f"{res['vol']:.1f}",
                f"{res['atr_pct']:.2f}",
                f"{res['chg']:+.1f}%",
                f"{res['spread']:.0f}",
                res['pos'],
                style=sig_style if res['sig'] != 0 else None
            )
        
        # === CLEAR AND PRINT (Rich handles all terminal quirks) ===
        console.clear()
        console.print(f"[bold]Equity:[/] ${current_equity:,.0f} | [bold]PnL:[/] {daily_dd_pct*100:+.2f}% | [bold]Limit:[/] {limit_pct*100:.1f}%")
        console.print(table)
        
        # 5. EXECUTION LOGIC (Silent unless trade happens)
        for sym_int, df in live_data.items():
            last_bar = df.iloc[-1]
            signal = last_bar.get('S_Alpha', 0) 
            target_direction = 0
            if signal == 1: target_direction = 1
            elif signal == -1: target_direction = -1
            
            mt5_sym = SYMBOL_MAP[sym_int]
            open_pos = pos_map.get(sym_int)
            
            if not open_pos and target_direction != 0:
                balance = acct.balance
                vol_int = last_bar.get('Vol_Intensity', 0.0)
                vol_mult = 1.0 / (1.0 + max(vol_int, 0.0)**2)
                vol_mult = max(0.3, min(vol_mult, 1.2))
                risk_amt = balance * 0.003 * vol_mult
                
                atr = last_bar['ATR']
                sl_dist = atr * 1.5
                
                symbol_info = mt5.symbol_info(mt5_sym)
                if not symbol_info: continue
                tick_size = symbol_info.trade_tick_size
                tick_value = symbol_info.trade_tick_value
                if tick_size == 0 or tick_value == 0: continue
                
                sl_points = sl_dist / tick_size
                loss_per_lot = sl_points * tick_value
                volume = 0.01 if loss_per_lot == 0 else risk_amt / loss_per_lot
                
                step = symbol_info.volume_step
                volume = round(volume / step) * step
                volume = max(volume, symbol_info.volume_min)
                volume = min(volume, symbol_info.volume_max)
                if volume <= 0: continue
                
                if not self.check_exposure_limit(mt5_sym, target_direction, volume): continue
                
                price = symbol_info.ask if target_direction == 1 else symbol_info.bid
                sl_price = price - (sl_dist * target_direction)
                tp_price = price + (sl_dist * 2.0 * target_direction)
                
                # Log trade (prints on new line after dashboard)
                print(f"\n>>> [TRADE] {mt5_sym} {'BUY' if target_direction==1 else 'SELL'} {volume} @ {price:.5f}")
                
                ticket = self.mt5.open_order(mt5_sym, 'BUY' if target_direction==1 else 'SELL', volume, sl_price, tp_price)
                if ticket:
                    context = {
                        'Risk_Dollars': risk_amt,
                        'Entry_Trend_Regime': str(last_bar.get('Trend_Regime', '')),
                        'Entry_Vol_Intensity': float(vol_int),
                        'Entry_Hour': int(last_bar.get('Hour', 0))
                    }
                    self.logger.on_entry(ticket, mt5_sym, 'LONG' if target_direction==1 else 'SHORT', volume, price, context)

            elif open_pos:
                current_dir = 1 if open_pos['type'] =='BUY' else -1
                if target_direction != 0 and target_direction != current_dir:
                    print(f"\n>>> [REVERSE] {mt5_sym}")
                    self.mt5.close_position(open_pos['ticket'])

    def loop(self):
        # 1. Train on Startup
        self.train_fresh_model()
        
        # 2. Infinite Loop
        while True:
            # Sync to top of hour
            now = datetime.now()
            # Calculate seconds to next hour
            # sleep...
            
            self.run_cycle()
            
            # Sleep 1 second (Live Dashboard Mode)
            time.sleep(1)

if __name__ == "__main__":
    from dotenv import load_dotenv
    import os
    
    # 1. Load Environment Variables (Highest Priority)
    load_dotenv()
    
    connector = MT5Connector()
    
    # Auto-Login Logic
    login = os.getenv("MT5_LOGIN")
    password = os.getenv("MT5_PASSWORD")
    server = os.getenv("MT5_SERVER")
    
    # 2. Fallback to live_config.json
    if not login:
        try:
            with open("live_config.json", "r") as f:
                config = json.load(f)
                creds = config.get("credentials", {})
                if creds:
                    login = int(creds.get("login", 0))
                    password = creds.get("password", "")
                    server = creds.get("server", "")
        except Exception as e:
            print(f"[LOGIN WARN] Could not load config: {e}")
            
    if login and password and server:
        print(f"[LOGIN] Attempting connection to {login} on {server}...")
        connector.login(int(login), password, server)
    else:
        print("[LOGIN WARN] No credentials found. Assuming manual login.")

    trader = LiveTrader(connector)
    trader.loop()
