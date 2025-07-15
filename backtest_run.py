import json, os, textwrap, argparse
from typing import Tuple, List

import pandas as pd
import vectorbt as vbt
import joblib

import src.models.config as C
from src.models.config import est_cost

BASE   = os.path.dirname(__file__)
RAW    = f"{BASE}/data/processed"
FEAT   = f"{BASE}/data/features"
RULES  = f"{BASE}/data/strategies"
STRAT  = RULES   # alias

# --------------------------------------------------------------------------- #
def _load_price(sym: str) -> pd.Series:
    return pd.read_csv(f"{RAW}/{sym}.csv", index_col=0, parse_dates=True)["Close"]

def _load_feats(sym: str) -> pd.DataFrame:
    return pd.read_csv(f"{FEAT}/{sym}.csv", index_col=0, parse_dates=True)

def _pf_stats(pf: vbt.Portfolio) -> pd.Series:
    want = ["Total Return [%]", "CAGR", "Sharpe Ratio", "Max Drawdown [%]"]
    stats = pf.stats()
    cols  = [c for c in want if c in stats.index] or list(stats.index[:4])
    return stats[cols]

# ---------- helper that prefers calibrated pickle -------------------------- #
def _load_pickle(base_path_no_suffix: str):
    cal = base_path_no_suffix + "_cal.pkl"
    raw = base_path_no_suffix + ".pkl"
    return joblib.load(cal) if os.path.exists(cal) else joblib.load(raw)

# --------------------------------------------------------------------------- #
def _run_port(sym: str,
              price: pd.Series,
              entries: pd.Series,
              exits: pd.Series,
              k_spread: float,
              k_impact: float) -> pd.Series:
    atr = _load_feats(sym)["atr_14"]
    fees_ser = est_cost(atr, price, k_spread=k_spread, k_impact=k_impact)
    pf = vbt.Portfolio.from_signals(
        price, entries, exits,
        fees=fees_ser,
        init_cash=C.INIT_CASH,
        freq="D"
    )
    return _pf_stats(pf)

# ---------- engine wrappers ------------------------------------------------- #
def bt_dt(sym: str, k_spread: float, k_impact: float):
    mdl = _load_pickle(f"{STRAT}/{sym}_dt")
    feats  = _load_feats(sym)
    price  = _load_price(sym).reindex(feats.index).ffill()
    proba  = pd.Series(mdl.predict_proba(feats)[:, 1], index=feats.index)
    return _run_port(sym, price, proba >= 0.50, proba < 0.50,
                     k_spread, k_impact).rename(sym)

def bt_grid(sym: str, k_spread: float, k_impact: float):
    mdl   = _load_pickle(f"{STRAT}/{sym}_grid")
    with open(f"{STRAT}/{sym}_grid.json") as f:
        hi, lo = json.load(f)["prob_thresholds"] or (0.70, 0.30)
    feats  = _load_feats(sym)
    price  = _load_price(sym).reindex(feats.index).ffill()
    proba  = pd.Series(mdl.predict_proba(feats)[:, 1], index=feats.index)
    return _run_port(sym, price, proba >= hi, proba <= lo,
                     k_spread, k_impact).rename(sym)

def bt_rulefit(sym: str, k_spread: float, k_impact: float):
    feats = _load_feats(sym)
    price = _load_price(sym).reindex(feats.index).ffill()
    with open(f"{STRAT}/{sym}_rulefit.json") as f:
        rule = json.load(f)[-1]["top_rules"][0]["rule"]
    sig = pd.eval(rule, local_dict=feats).fillna(0).astype(bool)
    return _run_port(sym, price, sig, ~sig, k_spread, k_impact).rename(sym)

def bt_xgb(sym: str, k_spread: float, k_impact: float):
    pickles = [f for f in os.listdir(STRAT)
               if f.startswith(f"{sym}_") and f.endswith("_xgb.pkl")]
    if not pickles:
        raise FileNotFoundError(f"No XGB model for {sym}")
    mdl = _load_pickle(os.path.join(STRAT, pickles[-1][:-4]))  # strip '.pkl'
    feats = _load_feats(sym)
    price = _load_price(sym).reindex(feats.index).ffill()
    proba = pd.Series(mdl.predict_proba(feats)[:, 1], index=feats.index)
    return _run_port(sym, price, proba >= C.PROB_LONG, proba <= C.PROB_FLAT,
                     k_spread, k_impact).rename(sym)

# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Back-test DT, Grid-DT, RuleFit and XGB engines",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
          example:
            python backtest_run.py --syms MCD TMO COP PFE
            python backtest_run.py --syms MCD TMO --models dt rulefit
        """)
    )
    p.add_argument("--syms",   nargs="+", required=True)
    p.add_argument("--models", nargs="+",
                   choices=["dt", "grid", "rulefit", "xgb"],
                   default=["dt", "grid", "rulefit", "xgb"])
    p.add_argument("--k_spread", type=float, default=0.5)
    p.add_argument("--k_impact", type=float, default=0.1)
    args = p.parse_args()

    engines = {
        "dt": bt_dt,
        "grid": bt_grid,
        "rulefit": bt_rulefit,
        "xgb": bt_xgb
    }

    for tag in args.models:
        rows = [engines[tag](s, args.k_spread, args.k_impact) for s in args.syms]
        tbl  = pd.concat(rows, axis=1).T.round(2)
        print(f"\n=== {tag.upper()} BACK-TEST RESULTS ===\n")
        print(tbl)
