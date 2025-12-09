# 📦 Alpha Hunt v3: Windows Deployment Instructions

## 1. Prerequisites
-   **Windows Server / VPS** (with Admin rights).
-   **MetaTrader 5 (MT5)** installed and logged into your FTMO account.
-   **Python 3.10 or newer** installed. [Download Python](https://www.python.org/downloads/windows/)
    -   *Important*: Check "Add Python to PATH" during installation.

## 2. Update Code (Git)
1.  Open **Command Prompt** (cmd) or PowerShell.
2.  Navigate to your bot folder:
    ```powershell
    cd C:\path\to\repo\FTMO Challenge
    ```
3.  **Pull Latest Changes**:
    ```powershell
    git pull
    ```

## 3. Install Dependencies
**Important**: We added `MetaTrader5` to the requirements.
```powershell
pip install -r requirements_windows.txt
```

## 4. Configuration
1.  Ensure you have a `.env` file in this folder. 
2.  If not, rename `.env.example` to `.env` and edit it.
    ```ini
    MT5_LOGIN=123456789
    MT5_PASSWORD=MySecretPassword
    MT5_SERVER=FTMO-Demo
    ```

## 4. MT5 Setup
1.  Open your **MetaTrader 5 Terminal**.
2.  Go to **Tools -> Options -> Expert Advisors**.
3.  ✅ Check **"Allow algorithmic trading"**.
4.  (Optional) You do NOT need to drag any EA onto the chart. The Python script controls MT5 externally.

## 5. Launch
Run the bot from your terminal:
```powershell
python live_trader_mt5.py
```

### Expected Output
-   The bot will connect to MT5.
-   It will download history for ~30 symbols.
-   It will train the Alpha Engine (takes ~1-2 mins).
-   It will show a Dashboard with live prices and ranks.

## 6. Troubleshooting
-   **"ModuleNotFoundError: No module named 'MetaTrader5'"**: You missed Step 2 (pip install).
-   **"IPC initialization failed"**: MT5 is not running or "Allow algorithmic trading" is disabled.
