import numpy as np
import pandas as pd
import ta

def build_bigmove_target(close: pd.Series,
                         high: pd.Series,
                         low: pd.Series,
                         hold_days: int = 5,
                         vol_lookback: int = 20,
                         vol_mult: float = 1.0) -> pd.Series:
    """
    Builds a target variable based on volatility-adjusted thresholds.
    A positive label (1) is assigned if the price moves up by more than
    `vol_mult` times the average true range (ATR) within `hold_days`.
    A negative label (-1) is assigned for a similar move downwards.
    """
    fwd_ret = close.shift(-hold_days) / close - 1.0
    atr = ta.volatility.average_true_range(
        high=high,
        low=low,
        close=close,
        window=vol_lookback
    )
    pos_thres = (atr / close) * vol_mult
    neg_thres = -pos_thres

    lab = pd.Series(0, index=close.index, dtype='int8')
    lab.loc[fwd_ret >= pos_thres] = 1
    lab.loc[fwd_ret <= neg_thres] = -1
    return lab

__all__ = ["build_bigmove_target"]