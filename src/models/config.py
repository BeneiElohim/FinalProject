"""
Global constants + **complete** cost model.
Everything that needs to be tuned lives here.
"""
from __future__ import annotations
import pandas as pd

# ----------------------- risk / label config --------------------------------
HOLD_DAYS       = 5
RET_TH_HIGH     = 0.02
RET_TH_LOW      = -0.02
PROB_LONG       = 0.70
PROB_FLAT       = 0.30

# ----------------------- portfolio & fees -----------------------------------
INIT_CASH       = 100_000
FIXED_COMM_BPS  = 5                # 5 bp round trip

def _bid_ask_spread(atr14: pd.Series, close: pd.Series) -> pd.Series:
    return 0.5 * atr14 / close        # 0.5 × ATR / Close

def _market_impact(delta_atr: pd.Series) -> pd.Series:
    return 0.1 * delta_atr.abs()      

def est_cost(atr14: pd.Series,
             close: pd.Series,
             k_spread: float = 0.5,
             k_impact: float = 0.1,
             fixed_comm_bps: float = FIXED_COMM_BPS) -> pd.Series:
    """
    Round‑trip fractional cost:
      total = fixed_commission + bid‑ask_spread + market_impact
    """
    fixed = fixed_comm_bps / 10_000
    spread = k_spread * _bid_ask_spread(atr14, close)
    impact = k_impact * _market_impact(atr14.pct_change())
    return (fixed + spread + impact).fillna(method="bfill").fillna(0)
