"""
Vectorised portfolio & trade-level performance metrics.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

TRADING_DAYS = 252


def sharpe_ratio(ret: pd.Series, rf: float = 0.0) -> float:
    ex = ret - rf / TRADING_DAYS
    vol = ex.std(ddof=0)
    if vol == 0 or np.isnan(vol):
        return 0.0
    return float(np.sqrt(TRADING_DAYS) * ex.mean() / vol)


def downside_deviation(ret: pd.Series, target: float = 0.0) -> float:
    neg = np.minimum(0, ret - target)
    return float(np.sqrt((neg**2).mean()) * np.sqrt(TRADING_DAYS))


def sortino_ratio(ret: pd.Series, target: float = 0.0) -> float:
    dd = downside_deviation(ret, target)
    return float("nan") if dd == 0 else float(
        (ret.mean() - target / TRADING_DAYS) * np.sqrt(TRADING_DAYS) / dd
    )


def calmar_ratio(cagr: float, max_dd: float) -> float:
    return float("nan") if max_dd == 0 else float(cagr / abs(max_dd))


def var_95(ret: pd.Series) -> float:
    return float(np.percentile(ret, 5))


def cagr(equity: pd.Series) -> float:
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1) if years else float("nan")


def max_drawdown(equity: pd.Series) -> float:
    roll_max = equity.cummax()
    dd = (equity / roll_max) - 1.0
    return float(dd.min())


def recovery_time_days(equity: pd.Series) -> float:
    peak = equity.cummax()
    underwater = equity < peak
    if not underwater.any():
        return 0.0
    rec_durs = []
    start = None
    for t, uw in underwater.items():  # .items avoids FutureWarning
        if uw and start is None:
            start = t
        elif not uw and start is not None:
            rec_durs.append((t - start).days)
            start = None
    return float(max(rec_durs)) if rec_durs else float("nan")


def consecutive_runs(x: pd.Series) -> tuple[int, int]:
    """Return (max_consecutive_true, max_consecutive_false)."""
    max_true = max_false = cur_true = cur_false = 0
    for v in x:
        if v:
            cur_true += 1
            cur_false = 0
        else:
            cur_false += 1
            cur_true = 0
        max_true = max(max_true, cur_true)
        max_false = max(max_false, cur_false)
    return max_true, max_false


# ---------------- Market regime helpers ---------------- #
def market_regime(benchmark_ret: pd.Series) -> pd.Series:
    """Label each day as bull / bear / sideways based on ±2% benchmark move."""
    cond = pd.Series("sideways", index=benchmark_ret.index)
    cond[benchmark_ret > 0.02] = "bull"
    cond[benchmark_ret < -0.02] = "bear"
    return cond


def regime_returns(ret: pd.Series, regime: pd.Series) -> dict[str, float]:
    """Aggregate strategy returns by regime labels."""
    d: dict[str, float] = {}
    for r in ("bull", "bear", "sideways"):
        filt = regime == r
        d[r] = ret[filt].add(1).prod() - 1 if filt.any() else np.nan
    return d
