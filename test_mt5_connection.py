import MetaTrader5 as mt5
import os
import json
from dotenv import load_dotenv

# 1. Load Secrets (Try .env first, then config.json)
load_dotenv()

LOGIN = os.getenv("MT5_LOGIN")
PASSWORD = os.getenv("MT5_PASSWORD")
SERVER = os.getenv("MT5_SERVER")

if not LOGIN:
    # Try live_config.json fallback
    try:
        with open("live_config.json", "r") as f:
            data = json.load(f)
            creds = data.get("credentials", {})
            LOGIN = creds.get("login")
            PASSWORD = creds.get("password")
            SERVER = creds.get("server")
    except Exception as e:
        print(f"Config load failed: {e}")

if not LOGIN:
    print("❌ ERROR: No credentials found in .env or live_config.json")
    exit(1)

print(f"🔍 Attempting to connect to Account #{LOGIN} on {SERVER}...")

# 2. Initialize MT5
if not mt5.initialize():
    print("❌ initialize() failed, error code =", mt5.last_error())
    quit()

# 3. Login
authorized = mt5.login(int(LOGIN), password=PASSWORD, server=SERVER)
if authorized:
    print(f"✅ CONNECTED SUCCESSFULLY to Account #{LOGIN}")
    print(f"   Terminal: {mt5.terminal_info().path}")
    print(f"   Balance:  {mt5.account_info().balance}")
    print(f"   Equity:   {mt5.account_info().equity}")
else:
    print(f"❌ Login failed, error code: {mt5.last_error()}")

mt5.shutdown()
