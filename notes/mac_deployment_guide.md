# Mac User Deployment Guide

## 🚨 The Challenge
The `MetaTrader5` Python library is **Windows-Only**. It relies on `metatrader5.dll` and direct communication with the Windows terminal application. It **cannot** run natively on macOS, Linux, or Docker containers (unless using Wine, which is unstable).

## ✅ The Solution: Windows VPS
To run this strategy professionally (for FTMO), you should not be running it on your laptop anyway (due to sleep mode, wifi drops, etc.). You need a **Windows VPS**.

### Step 1: Get a Windows VPS
*   **Recommended**: FTMO often recommends providers, or use generic ones like Contabo, AWS (EC2 Windows), or Azure.
*   **Specs**: 2 vCPU, 4GB RAM is sufficient.

### Step 2: Install Required Software on VPS
1.  **MetaTrader 5 Terminal**: Download from FTMO or your broker. Install and login.
2.  **Aglo Trading**: Enable "Algo Trading" button in the toolbar.
3.  **Python**: Install Python 3.10 or 3.11 (Windows Installer). Check "Add to PATH".
4.  **Git**: Install Git for Windows.

### Step 3: Deploy the Code
Open PowerShell on the VPS:

```powershell
# 1. Clone your repo (or copy files via Remote Desktop)
git clone <your-repo-url>
cd <repo-folder>

# 2. Install Dependencies
pip install MetaTrader5 pandas numpy yfinance xgboost scikit-learn

# 3. Configure Credentials
# Edit live_config.json if needed (already configured for 99832013)
```

### Step 4: Run Shadow Mode
```powershell
python live_trader_mt5.py
```

## ⚠️ Alternative (Local Simulation)
If you just want to "test" the code logic on your Mac without connecting to MT5:
*   We would need to modify the code to use `yfinance` for live data and `mock` the execution.
*   This is useful for debugging but **invalid** for the Forward Test (FT_001) because it doesn't test the broker connection.

**Recommendation**: Go the VPS route. It is the only way to pass the "Infrastructure" requirement for a real fund.

## ❌ What NOT to use
*   **MQL5 Virtual Hosting**: This **DOES NOT SUPPORT** Python libraries (`pandas`, `scikit-learn`). It only runs compiled MQL5 code. Do not rent this.
*   **Docker**: MT5 is a GUI app. Dockerizing it (via Wine) is extremely unstable for financial operations.
*   **Mac "Crossover" / Wine**: Can run the terminal, but connecting Python to it is a nightmare of DLL pathing issues.

**Get a standard Windows VPS (Server 2019/2022).**
