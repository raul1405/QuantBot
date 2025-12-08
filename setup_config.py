import json

config = {
    "mode": "shadow",
    "account_leverage": 30,
    "credentials": {
        "login": 99832013,
        "password": "1xPsWuD@", # ACTUAL PASSWORD PPLIED
        "server": "MetaQuotes-Demo"
    },
    "governance": {
        "max_daily_dd_pct": 0.045,
        "max_total_dd_pct": 0.095,
        "max_drawdown_hard_stop": 0.08
    },
    "risk_caps": {
        "max_net_lots_usd": 5,
        "max_correlated_positions": 3,
        "max_margin_pct": 0.30,
        "max_notional_pct": 6.0
    }
}

with open("live_config.json", "w") as f:
    json.dump(config, f, indent=4)

print("✅ live_config.json created successfully.")
