"""
TELEGRAM DASHBOARD BOT
======================
Sends the live trading dashboard to Telegram on schedule or via /status command.
Run this alongside live_trader_mt5.py on your Windows server.

Commands:
  /status - Get current dashboard
  /positions - Get open positions
  /pnl - Get daily PnL summary
"""

import os
import sys
import time
import json
import threading
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv

# Telegram Bot
from telegram import Update, Bot
from telegram.ext import Application, CommandHandler, ContextTypes

# MT5 & Strategy Imports
import MetaTrader5 as mt5
import pandas as pd
import numpy as np

# Load Environment
load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# Symbol Map (Same as live_trader_mt5.py)
SYMBOL_MAP = {
    "EURUSD=X": "EURUSD", "USDJPY=X": "USDJPY", "GBPUSD=X": "GBPUSD",
    "USDCHF=X": "USDCHF", "USDCAD=X": "USDCAD", "AUDUSD=X": "AUDUSD",
    "NZDUSD=X": "NZDUSD", "EURGBP=X": "EURGBP", "EURJPY=X": "EURJPY",
    "GBPJPY=X": "GBPJPY", "AUDJPY=X": "AUDJPY", "EURAUD=X": "EURAUD",
    "EURCHF=X": "EURCHF"
}
REVERSE_MAP = {v: k for k, v in SYMBOL_MAP.items()}

# ==============================================================================
# MT5 CONNECTION
# ==============================================================================
def init_mt5():
    if not mt5.initialize():
        print("MT5 Init Failed")
        return False
    return True

def get_account_info():
    acct = mt5.account_info()
    if acct is None:
        return None
    return {
        'balance': acct.balance,
        'equity': acct.equity,
        'margin': acct.margin,
        'profit': acct.profit,
        'leverage': acct.leverage
    }

def get_open_positions():
    positions = mt5.positions_get()
    if positions is None:
        return []
    pos_list = []
    for p in positions:
        pos_list.append({
            'ticket': p.ticket,
            'symbol': p.symbol,
            'type': 'BUY' if p.type == mt5.ORDER_TYPE_BUY else 'SELL',
            'volume': p.volume,
            'profit': p.profit,
            'open_price': p.price_open,
            'current_price': p.price_current
        })
    return pos_list

def get_symbol_data(symbol_mt5, n_bars=50):
    if not mt5.symbol_select(symbol_mt5, True):
        return None
    rates = mt5.copy_rates_from_pos(symbol_mt5, mt5.TIMEFRAME_H1, 0, n_bars)
    if rates is None:
        return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close'}, inplace=True)
    return df

# ==============================================================================
# DASHBOARD GENERATION
# ==============================================================================
def generate_dashboard():
    acct = get_account_info()
    if acct is None:
        return "❌ MT5 Not Connected"
    
    # Load Daily State
    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    state_file = "daily_state.json"
    start_equity = acct['equity']
    try:
        if os.path.exists(state_file):
            with open(state_file, 'r') as f:
                state = json.load(f)
                if state.get("date") == today_str:
                    start_equity = state.get("start_equity", acct['equity'])
    except:
        pass
    
    daily_pnl = acct['equity'] - start_equity
    daily_pnl_pct = (daily_pnl / start_equity) * 100 if start_equity > 0 else 0
    
    # Build Dashboard
    lines = []
    lines.append("📊 **QUANTBOT DASHBOARD**")
    lines.append(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append(f"💰 **Equity**: ${acct['equity']:,.2f}")
    lines.append(f"📈 **Balance**: ${acct['balance']:,.2f}")
    lines.append(f"📉 **Daily PnL**: ${daily_pnl:+,.2f} ({daily_pnl_pct:+.2f}%)")
    lines.append(f"⚡ **Margin Used**: ${acct['margin']:,.2f}")
    lines.append(f"🔧 **Leverage**: {acct['leverage']}x")
    lines.append("")
    
    # Open Positions
    positions = get_open_positions()
    if positions:
        lines.append(f"📍 **Open Positions** ({len(positions)})")
        for p in positions:
            emoji = "🟢" if p['profit'] > 0 else "🔴"
            lines.append(f"  {emoji} {p['symbol']} {p['type']} {p['volume']} | ${p['profit']:+.2f}")
    else:
        lines.append("📍 **Open Positions**: None")
    
    lines.append("")
    
    # Symbol Scanner (Top 5 by price movement)
    lines.append("🔍 **Scanner (Top Movers)**")
    movers = []
    for sym_int, sym_mt5 in SYMBOL_MAP.items():
        if sym_mt5 is None:
            continue
        df = get_symbol_data(sym_mt5, n_bars=25)
        if df is None or len(df) < 25:
            continue
        close = df['Close'].iloc[-1]
        prev_close = df['Close'].iloc[0]
        chg = ((close - prev_close) / prev_close) * 100
        
        # Get Spread
        info = mt5.symbol_info(sym_mt5)
        spread = info.spread if info else 0
        
        movers.append({
            'sym': sym_mt5,
            'price': close,
            'chg': chg,
            'spread': spread
        })
    
    movers.sort(key=lambda x: abs(x['chg']), reverse=True)
    for m in movers[:5]:
        arrow = "🔼" if m['chg'] > 0 else "🔽"
        lines.append(f"  {m['sym']}: {m['price']:.5f} {arrow} {m['chg']:+.2f}% (SPD:{m['spread']})")
    
    return "\n".join(lines)

def generate_positions_report():
    positions = get_open_positions()
    if not positions:
        return "📍 No open positions."
    
    lines = ["📍 **Open Positions**"]
    total_pnl = 0
    for p in positions:
        total_pnl += p['profit']
        emoji = "🟢" if p['profit'] > 0 else "🔴"
        lines.append(f"{emoji} **{p['symbol']}** {p['type']}")
        lines.append(f"   Vol: {p['volume']} | Entry: {p['open_price']:.5f}")
        lines.append(f"   Current: {p['current_price']:.5f} | PnL: ${p['profit']:+.2f}")
        lines.append("")
    
    lines.append(f"**Total Floating PnL**: ${total_pnl:+.2f}")
    return "\n".join(lines)

def generate_pnl_report():
    acct = get_account_info()
    if acct is None:
        return "❌ MT5 Not Connected"
    
    # Load Daily State
    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    state_file = "daily_state.json"
    start_equity = acct['equity']
    try:
        if os.path.exists(state_file):
            with open(state_file, 'r') as f:
                state = json.load(f)
                if state.get("date") == today_str:
                    start_equity = state.get("start_equity", acct['equity'])
    except:
        pass
    
    daily_pnl = acct['equity'] - start_equity
    daily_pnl_pct = (daily_pnl / start_equity) * 100 if start_equity > 0 else 0
    
    # FTMO Limits
    ftmo_daily_limit = start_equity * 0.05  # 5%
    ftmo_total_limit = 100000 * 0.10  # Assuming 100k account
    
    lines = ["💵 **PnL Report**"]
    lines.append(f"📅 Date: {today_str}")
    lines.append("")
    lines.append(f"💰 Start Equity: ${start_equity:,.2f}")
    lines.append(f"💰 Current Equity: ${acct['equity']:,.2f}")
    lines.append(f"📈 Daily PnL: ${daily_pnl:+,.2f} ({daily_pnl_pct:+.2f}%)")
    lines.append("")
    lines.append("**FTMO Limits**")
    lines.append(f"  Daily DD Limit: -${ftmo_daily_limit:,.2f}")
    lines.append(f"  Buffer Remaining: ${ftmo_daily_limit + daily_pnl:,.2f}")
    
    if daily_pnl < 0:
        usage = abs(daily_pnl) / ftmo_daily_limit * 100
        lines.append(f"  ⚠️ Daily Limit Used: {usage:.1f}%")
    
    return "\n".join(lines)

# ==============================================================================
# TELEGRAM HANDLERS
# ==============================================================================
async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    dashboard = generate_dashboard()
    await update.message.reply_text(dashboard, parse_mode='Markdown')

async def cmd_positions(update: Update, context: ContextTypes.DEFAULT_TYPE):
    report = generate_positions_report()
    await update.message.reply_text(report, parse_mode='Markdown')

async def cmd_pnl(update: Update, context: ContextTypes.DEFAULT_TYPE):
    report = generate_pnl_report()
    await update.message.reply_text(report, parse_mode='Markdown')

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 **QuantBot Dashboard**\n\n"
        "Commands:\n"
        "/status - Full dashboard\n"
        "/positions - Open positions\n"
        "/pnl - Daily PnL report\n",
        parse_mode='Markdown'
    )

# ==============================================================================
# SCHEDULED BROADCASTS
# ==============================================================================
async def scheduled_broadcast(context: ContextTypes.DEFAULT_TYPE):
    """Send dashboard every hour."""
    dashboard = generate_dashboard()
    await context.bot.send_message(
        chat_id=TELEGRAM_CHAT_ID,
        text=dashboard,
        parse_mode='Markdown'
    )

# ==============================================================================
# MAIN
# ==============================================================================
def main():
    if not TELEGRAM_TOKEN:
        print("ERROR: TELEGRAM_BOT_TOKEN not set in .env")
        return
    
    if not init_mt5():
        print("ERROR: MT5 not available")
        return
    
    print(f"[BOT] Starting Telegram Dashboard Bot...")
    print(f"[BOT] Chat ID: {TELEGRAM_CHAT_ID}")
    
    # Create Application
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    
    # Add Handlers
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("positions", cmd_positions))
    app.add_handler(CommandHandler("pnl", cmd_pnl))
    
    # Schedule Hourly Broadcast
    job_queue = app.job_queue
    job_queue.run_repeating(scheduled_broadcast, interval=3600, first=60)  # Every hour
    
    print("[BOT] Bot is running. Press Ctrl+C to stop.")
    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
