"""
Train XGBoost on big-move labels, calibrate probabilities, then fit
a shallow surrogate tree for interpretability.  Uses expanding
walk-forward windows (3y train, 1y val).
"""

import os, json, warnings, joblib, argparse
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from utils.windows import build_walkforward_windows
from target import build_bigmove_target
from calibrate import calibrate_prefit
import config as C

# --------------------------------------------------------------------------- #
BASE  = os.path.dirname(__file__) + "/../.."
FEAT  = f"{BASE}/data/features"
PROC  = f"{BASE}/data/processed"
STRAT = f"{BASE}/data/strategies"
os.makedirs(STRAT, exist_ok=True)
warnings.filterwarnings("ignore")

def _symbol_set(filter_syms=None):
    all_syms = [f[:-4] for f in os.listdir(FEAT) if f.endswith(".csv")]
    return [s for s in all_syms if (not filter_syms) or (s in filter_syms)]

# ----- hyper-param grid ----------------------------------------------------- #
PARAMS = {
    "n_estimators":      [300, 400, 500],
    "max_depth":         [3, 4, 5],
    "learning_rate":     [0.02, 0.05, 0.1],
    "subsample":         [0.7, 0.8, 1.0],
    "colsample_bytree":  [0.7, 0.8, 1.0],
}

def load_xy(sym: str):
    X = pd.read_csv(f"{FEAT}/{sym}.csv", index_col=0, parse_dates=True)
    close = pd.read_csv(f"{PROC}/{sym}.csv",
                        index_col=0, parse_dates=True)["Close"]
    y = build_bigmove_target(
        close,
        hold_days=C.HOLD_DAYS,
        pos_thres=C.RET_TH_HIGH,
        neg_thres=C.RET_TH_LOW
    ).reindex(X.index)
    y = y.dropna().astype(int).replace(-1, 0)
    return X.loc[y.index], y

# --------------------------------------------------------------------------- #
def process_symbol(sym: str):
    X, y = load_xy(sym)
    if len(X) < 500:
        print(f"× {sym}: too little data")
        return

    windows = build_walkforward_windows(X.index, 3, 1)
    if not windows:
        print(f"× {sym}: no windows")
        return

    summary: List[Dict] = []
    for t0, t1, v0, v1 in windows:
        tr = (X.index >= t0) & (X.index <= t1)
        va = (X.index >= v0) & (X.index <= v1)
        if va.sum() < 60:
            continue

        X_tr, y_tr = X.loc[tr], y.loc[tr]
        X_va, y_va = X.loc[va], y.loc[va]

        base = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1
        )
        search = RandomizedSearchCV(
            base, PARAMS, n_iter=30, scoring="roc_auc", cv=3,
            verbose=0, n_jobs=-1)
        search.fit(X_tr, y_tr)
        mdl = search.best_estimator_

        # --- probability calibration --------------------------------------- #
        cal_mdl = calibrate_prefit(mdl, X_va, y_va, method="isotonic")
        proba_va = cal_mdl.predict_proba(X_va)[:, 1]
        auc = roc_auc_score(y_va, proba_va)

        # --- surrogate tree ------------------------------------------------- #
        sur = DecisionTreeRegressor(max_depth=3, random_state=42)
        sur.fit(X_tr, mdl.predict_proba(X_tr)[:, 1])
        rules = export_text(sur, feature_names=list(X.columns))

        summary.append({
            "symbol": sym,
            "window": f"{t0.date()}–{v1.date()}",
            "auc": round(auc, 4),
            "params": search.best_params_,
            "prob_thresholds": [C.PROB_LONG, C.PROB_FLAT],
            "surrogate_rules": rules,
            "train_rows": int(tr.sum()),
            "val_rows":   int(va.sum()),
            "timestamp": datetime.utcnow().isoformat() + "Z"
        })

        joblib.dump(cal_mdl, f"{STRAT}/{sym}_{v1.year}_xgb_cal.pkl")

    if summary:
        json.dump(summary, open(f"{STRAT}/{sym}_xgb.json", "w"), indent=2)
        print(f"√ {sym}: {len(summary)} windows")
    else:
        print(f"× {sym}: nothing saved")

# --------------------------------------------------------------------------- #
def run_all(symbols=None):
    for s in _symbol_set(symbols):
        process_symbol(s)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--syms", nargs="+")
    run_all(parser.parse_args().syms)
