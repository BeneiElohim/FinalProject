# config.py
HOLD_DAYS       = 5
RET_TH_HIGH     = 0.02  
RET_TH_LOW      = -0.02  
PROB_LONG       = 0.70  
PROB_FLAT       = 0.30   

FEE_BPS         = 5      
INIT_CASH       = 100_000

# --- dynamic trading cost ----------------------------------------------------
def est_cost(atr14, close, k_spread: float = 0.5, k_impact: float = 0.1):
    """
    Estimate round-trip cost as:
        spread  = k_spread  * ATR14 / Close
        impact  = k_impact  * |ΔATR14|   (simple impact proxy)
    Returns fractional cost (e.g. 0.001 = 10 bp).
    """
    spr = k_spread * atr14 / close
    imp = k_impact * atr14.pct_change().abs().fillna(0)
    return spr + imp
