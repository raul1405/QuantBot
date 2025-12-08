# 📱 How to Set Up Telegram Alerts

To get notifications on your phone, you need a **Bot Token** and your **Chat ID**.

## 1. Get a Bot Token (Expected Time: 30 seconds)
1. Open Telegram on your phone/PC.
2. Search for **@BotFather**.
3. Send the message `/newbot`.
4. Name it (e.g., `RaulQuantBot`).
5. Give it a username (must end in `bot`, e.g., `RaulQuantBot_v1`).
6. BotFather will give you a **TOKEN** (looks like `123456:ABC-DEF...`). **Copy this**.

## 2. Get Your Chat ID
1. Search for **@userinfobot** in Telegram.
2. Click Start.
3. It will reply with your `Id`. (e.g., `554433221`). **Copy this**.

## 3. Add to VPS
In your VPS PowerShell, update your `.env` file:

```powershell
Add-Content .env "TELEGRAM_TOKEN=your_token_here"
Add-Content .env "TELEGRAM_CHAT_ID=your_chat_id_here"
```

Then restart the bot (`Ctrl+C` -> `python live_trader_mt5.py`).
