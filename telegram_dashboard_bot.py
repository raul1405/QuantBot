"""
TELEGRAM DASHBOARD BOT v2
=========================
Full dashboard with auto-refresh, monospace table, and all metrics.

Commands:
  /live - Start live auto-refreshing dashboard (edits same message)
  /stop - Stop live dashboard
  /status - Get current snapshot
  /positions - Get open positions
  /pnl - Get daily PnL summary
"""

import os
import sys
import json
import asyncio
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv

# Telegram Bot
from telegram import Update, Bot
from telegram.ext import Application, CommandHandler, ContextTypes
from telegram.constants import ParseMode

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

# Live Dashboard State
live_sessions = {}  # chat_id -> {'message_id': id, 'active': bool}

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
            'type': 'L' if p.type == mt5.ORDER_TYPE_BUY else 'S',
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
# DASHBOARD GENERATION (MONOSPACE TABLE)
# ==============================================================================
def generate_full_dashboard():
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
    
    # Header
    lines = []
    lines.append(f"🤖 QuantBot v5 | {datetime.now().strftime('%H:%M:%S')}")
    lines.append(f"💰 Eq: ${acct['equity']:,.0f} | PnL: {daily_pnl_pct:+.2f}%")
    lines.append("")
    
    # Open Positions Section
    positions = get_open_positions()
    pos_map = {p['symbol']: p for p in positions}
    
    if positions:
        lines.append("📍 POSITIONS:")
        for p in positions:
            emoji = "🟢" if p['profit'] > 0 else "🔴"
            lines.append(f"  {emoji} {p['symbol']} {p['type']} {p['volume']} ${p['profit']:+.0f}")
        lines.append("")
    
    # Scanner Table (Monospace)
    lines.append("```")
    lines.append(f"{'SYM':<7}{'PRICE':>9}{'SPD':>4}{'24h':>7}{'POS':>4}")
    lines.append("-" * 31)
    
    scan_results = []
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
        
        # Position Status
        pos_status = ""
        if sym_mt5 in pos_map:
            pos_status = pos_map[sym_mt5]['type']
        
        scan_results.append({
            'sym': sym_mt5,
            'price': close,
            'chg': chg,
            'spread': spread,
            'pos': pos_status
        })
    
    # Sort by absolute change
    scan_results.sort(key=lambda x: abs(x['chg']), reverse=True)
    
    for r in scan_results:
        # Format price based on magnitude
        if r['price'] > 100:
            price_str = f"{r['price']:.2f}"
        elif r['price'] > 10:
            price_str = f"{r['price']:.3f}"
        else:
            price_str = f"{r['price']:.5f}"
        
        chg_str = f"{r['chg']:+.2f}%"
        pos_str = r['pos'] if r['pos'] else "-"
        
        lines.append(f"{r['sym']:<7}{price_str:>9}{r['spread']:>4}{chg_str:>7}{pos_str:>4}")
    
    lines.append("```")
    
    # FTMO Status Bar
    ftmo_daily_limit = start_equity * 0.05
    usage = abs(min(daily_pnl, 0)) / ftmo_daily_limit * 100
    
    lines.append("")
    if daily_pnl >= 0:
        lines.append(f"✅ FTMO Safe | Buffer: ${ftmo_daily_limit:,.0f}")
    else:
        lines.append(f"⚠️ DD: {usage:.1f}% of Daily Limit")
    
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
    ftmo_daily_limit = start_equity * 0.05
    
    lines = ["💵 **PnL Report**"]
    lines.append(f"📅 Date: {today_str}")
    lines.append("")
    lines.append(f"💰 Start: ${start_equity:,.2f}")
    lines.append(f"💰 Current: ${acct['equity']:,.2f}")
    lines.append(f"📈 Daily: ${daily_pnl:+,.2f} ({daily_pnl_pct:+.2f}%)")
    lines.append("")
    lines.append(f"🛡️ Daily DD Limit: -${ftmo_daily_limit:,.0f}")
    lines.append(f"🛡️ Buffer: ${ftmo_daily_limit + daily_pnl:,.0f}")
    
    if daily_pnl < 0:
        usage = abs(daily_pnl) / ftmo_daily_limit * 100
        lines.append(f"⚠️ Limit Used: {usage:.1f}%")
    
    return "\n".join(lines)

# ==============================================================================
# TELEGRAM HANDLERS
# ==============================================================================
async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    dashboard = generate_full_dashboard()
    await update.message.reply_text(dashboard, parse_mode=ParseMode.MARKDOWN)

async def cmd_positions(update: Update, context: ContextTypes.DEFAULT_TYPE):
    report = generate_positions_report()
    await update.message.reply_text(report, parse_mode=ParseMode.MARKDOWN)

async def cmd_pnl(update: Update, context: ContextTypes.DEFAULT_TYPE):
    report = generate_pnl_report()
    await update.message.reply_text(report, parse_mode=ParseMode.MARKDOWN)

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 **QuantBot Dashboard v2**\n\n"
        "Commands:\n"
        "/live - Start auto-refresh (15s)\n"
        "/stop - Stop auto-refresh\n"
        "/status - Snapshot\n"
        "/positions - Open trades\n"
        "/pnl - Daily PnL\n",
        parse_mode=ParseMode.MARKDOWN
    )

async def cmd_live(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    
    # Send initial message
    dashboard = generate_full_dashboard()
    msg = await update.message.reply_text(dashboard, parse_mode=ParseMode.MARKDOWN)
    
    # Store session
    live_sessions[chat_id] = {
        'message_id': msg.message_id,
        'active': True
    }
    
    # Start refresh loop
    asyncio.create_task(live_refresh_loop(context.bot, chat_id))

async def cmd_stop(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    if chat_id in live_sessions:
        live_sessions[chat_id]['active'] = False
        await update.message.reply_text("⏹️ Live dashboard stopped.")
    else:
        await update.message.reply_text("No active live dashboard.")

async def live_refresh_loop(bot: Bot, chat_id: int):
    """Refresh dashboard every 15 seconds by editing the message."""
    while chat_id in live_sessions and live_sessions[chat_id]['active']:
        await asyncio.sleep(15)
        
        if not live_sessions.get(chat_id, {}).get('active'):
            break
            
        try:
            dashboard = generate_full_dashboard()
            await bot.edit_message_text(
                chat_id=chat_id,
                message_id=live_sessions[chat_id]['message_id'],
                text=dashboard,
                parse_mode=ParseMode.MARKDOWN
            )
        except Exception as e:
            print(f"[REFRESH ERROR] {e}")
            # Message might be deleted, stop refreshing
            break
    
    # Cleanup
    if chat_id in live_sessions:
        del live_sessions[chat_id]

# ==============================================================================
# MAIN
# ==============================================================================
def main():
    if not TELEGRAM_TOKEN:
        print("ERROR: TELEGRAM_TOKEN not set in .env")
        return
    
    if not init_mt5():
        print("ERROR: MT5 not available")
        return
    
    print(f"[BOT] Starting Telegram Dashboard Bot v2...")
    print(f"[BOT] Chat ID: {TELEGRAM_CHAT_ID}")
    
    # Create Application
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    
    # Add Handlers
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("positions", cmd_positions))
    app.add_handler(CommandHandler("pnl", cmd_pnl))
    app.add_handler(CommandHandler("live", cmd_live))
    app.add_handler(CommandHandler("stop", cmd_stop))
    
    print("[BOT] Bot is running. Press Ctrl+C to stop.")
    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
