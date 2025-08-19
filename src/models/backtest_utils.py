from __future__ import annotations
import pandas as pd
import vectorbt as vbt
from src.utils.metrics import (
    sharpe_ratio,
    sortino_ratio,
    calmar_ratio,
    var_95,
    downside_deviation,
    consecutive_runs,
    recovery_time_days,
    market_regime,
    regime_returns,
    max_drawdown as _max_dd,
)
import src.models.config as C


def _get_stat(stats, key):
    try:
        return stats.get(key)
    except Exception:
        return None


def run_backtest(symbol, price, entries, exits, fees, benchmark=None) -> dict:
    """Run a simple long/flat backtest using vectorbt and compute metrics."""
    pf = vbt.Portfolio.from_signals(
        price, entries, exits, fees=fees, init_cash=C.INIT_CASH, freq="D"
    )

    ret = pf.returns()           # daily returns (pd.Series)
    equity = pf.value()          # equity curve (pd.Series)
    stats = pf.stats()           # vectorbt summary dict-like

    m: dict[str, float | int | None] = {}

    # Returns (total & annualized)
    m["total_return"] = float(equity.iloc[-1] / equity.iloc[0] - 1)
    m["annual_return"] = float((1 + m["total_return"]) ** (252 / max(len(ret), 1)) - 1)

    # Risk-adjusted ratios
    m["sharpe_ratio"] = float(sharpe_ratio(ret))
    m["sortino_ratio"] = float(sortino_ratio(ret))

    # Drawdown (fallback if key missing in stats)
    md_pct = _get_stat(stats, "Max Drawdown [%]")
    m["max_drawdown"] = float(md_pct) / 100.0 if md_pct is not None else float(_max_dd(equity))

    # Risk metrics
    m["downside_deviation"] = float(downside_deviation(ret))
    m["calmar_ratio"] = float(calmar_ratio(m["annual_return"], m["max_drawdown"]))
    m["var_95"] = float(var_95(ret))

    # Trades
    tr = pf.trades.records
    m["num_trades"] = int(len(tr))
    m["avg_trade_duration"] = float(tr["exit_idx"].sub(tr["entry_idx"]).mean()) if len(tr) else 0.0
    w, l = consecutive_runs(tr["pnl"] > 0) if len(tr) else (0, 0)
    m["max_consecutive_wins"] = int(w)
    m["max_consecutive_losses"] = int(l)
    m["recovery_time_days"] = float(recovery_time_days(equity))

    # Extra summary stats useful for UI
    m["win_rate"] = float(((tr["pnl"] > 0).mean() * 100.0) if len(tr) else 0.0)
    if len(ret):
        m["best_day"] = float(ret.max())
        m["worst_day"] = float(ret.min())
        monthly = ret.resample("M").apply(lambda x: (1 + x).prod() - 1)
        m["best_month"] = float(monthly.max()) if len(monthly) else None
        m["worst_month"] = float(monthly.min()) if len(monthly) else None
    else:
        m["best_day"] = m["worst_day"] = m["best_month"] = m["worst_month"] = None

    # Regime performance (vs benchmark)
    if benchmark is not None:
        bench_ret = benchmark.pct_change().fillna(0.0)
        labels = market_regime(bench_ret)
        reg = regime_returns(ret, labels)
        m.update({f"{k}_market_return": float(v) if v is not None else None for k, v in reg.items()})

    # Turnover & costs (keys may not exist on older vectorbt)
    turnover_pct = _get_stat(stats, "Turnover [%]")
    m["turnover"] = float(turnover_pct) / 100.0 if turnover_pct is not None else None
    total_fees_paid = _get_stat(stats, "Total Fees Paid")
    m["total_slippage"] = float(total_fees_paid) if total_fees_paid is not None else None
    m["total_commission"] = None  # separated in your cost model; keep for schema symmetry

    return {
        "metrics": m,
        "equity_curve": equity.tolist(),
        "trade_log": tr.to_dict(orient="records"),
    }
