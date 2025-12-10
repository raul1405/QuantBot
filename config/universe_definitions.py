
# ==============================================================================
# AUDIT UNIVERSE DEFINITIONS
# ==============================================================================
# Static lists for "Hostile Replication" to prevent selection bias.
# These lists are defined EX-ANTE based on liquidity and "Major" status.

# 1. THE MAJORS (Highest Liquidity, Lowest Spread)
UNIVERSE_MAJORS = [
    "EURUSD=X", 
    "GBPUSD=X", 
    "USDJPY=X", 
    "USDCHF=X", 
    "USDCAD=X", 
    "AUDUSD=X", 
    "NZDUSD=X"
]

# 2. THE EXTENDED LIQUIDS (Majors + Highest Volume Crosses)
# No exotic crosses (e.g. NOK, SEK, SGD) or historically "toxic" pairs allowed unless broad liquid.
UNIVERSE_LIQUID = UNIVERSE_MAJORS + [
    "EURGBP=X", 
    "EURJPY=X", 
    "GBPJPY=X", 
    "AUDJPY=X", 
    "EURAUD=X"
]

# 3. FULL AUDIT (Includes "Toxic" ones for control)
UNIVERSE_ALL = UNIVERSE_LIQUID + [
    "AUDNZD=X", "AUDCAD=X", "CADJPY=X", "NZDJPY=X", 
    "GBPCHF=X", "GBPAUD=X", "GBPCAD=X", "EURNZD=X"
]
