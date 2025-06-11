import numpy as np
import pandas as pd

def build_bigmove_target(close: pd.Series,
                         hold_days: int = 5,
                         pos_thres: float = 0.02,
                         neg_thres: float = -0.02) -> pd.Series:

    fwd_ret = close.shift(-hold_days) / close - 1.0
    lab = pd.Series(0, index=close.index, dtype='int8')
    lab.loc[fwd_ret >=  pos_thres] = 1
    lab.loc[fwd_ret <=  neg_thres] = -1
    return lab
__all__ = ["build_bigmove_target"]