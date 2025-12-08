# Windows VPS Deployment Checklist

## 1. Local Prep (Done ✅)
- [x] `requirements.txt` created (Dependencies pinned).
- [x] `.env.example` created (Secret template).
- [x] `live_trader_mt5.py` updated to support `.env`.
- [x] `test_mt5_connection.py` created (Sanity check).

## 2. VPS Setup (Do this on the Server)
1.  **Connect**: RDP into your VPS (Use Microsoft Remote Desktop).
2.  **Install Base Software**:
    *   **Python 3.10+**: Download from python.org. Check "Add to PATH".
    *   **Git**: Download Git for Windows.
    *   **MetaTrader 5**: Download from your Broker/FTMO.
3.  **Configure MT5**:
    *   Login to your account (`99832013`).
    *   Enable **Algo Trading** button.
    *   Tools -> Options -> Expert Advisors -> Allow WebRequest (if needed later).

## 3. Deploy Code (On VPS)
Open PowerShell and run:

```powershell
# 1. Clone Repo (or copy files)
git clone <your-repo> C:\QuantBot
cd C:\QuantBot

# 2. Virtual Env
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. Install Deps
pip install -r requirements.txt

# 4. Configure Secrets (Copy-Paste this Block)
# Run these commands in PowerShell to create your .env file automatically:

$secrets = @"
MT5_LOGIN=99832013
MT5_PASSWORD=1xPsWuD@
MT5_SERVER=MetaQuotes-Demo
"@

Set-Content .env $secrets
Write-Host "Secrets configured!"

# Verify it looks right:
Get-Content .env

```

## 4. Test Connectivity
```powershell
python test_mt5_connection.py
```
*   If it says `✅ CONNECTED SUCCESSFULLY`, you are gold.

## 5. Run Shadow Mode
```powershell
python live_trader_mt5.py
```

## 6. Persistence (Task Scheduler)
To make it auto-start on reboot:
1.  Open **Task Scheduler**.
2.  Create Task -> "QuantBot".
3.  Trigger: "At Startup".
4.  Action: Start Program -> `powershell.exe`.
5.  Arguments: `-ExecutionPolicy Bypass -Command "cd C:\QuantBot; .\.venv\Scripts\Activate.ps1; python live_trader_mt5.py >> logs.txt 2>&1"`
