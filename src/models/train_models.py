import os, json
from datetime import datetime
from typing import List, Dict
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier
import joblib

from utils.windows import build_walkforward_windows
from calibrate import calibrate_prefit

# --------------------------------------------------------------------------- #
FEATURE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "features")
RAW_DIR     = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")
STRAT_DIR   = os.path.join(os.path.dirname(__file__), "..", "..", "data", "strategies")
os.makedirs(STRAT_DIR, exist_ok=True)

def _symbol_set(filter_syms=None):
    files = [f for f in os.listdir(FEATURE_DIR) if f.endswith(".csv")]
    all_syms = [f[:-4] for f in files]
    return [s for s in all_syms if (not filter_syms) or (s in filter_syms)]

# --------------------------------------------------------------------------- #
def build_target(close: pd.Series, hold_days: int = 1) -> pd.Series:
    future = close.shift(-hold_days)
    ret    = (future - close) / close
    return (ret > 0).astype(int)

def extract_tree_rules(clf: DecisionTreeClassifier, names: List[str]) -> str:
    return export_text(clf, feature_names=names)

# --------------------------------------------------------------------------- #
def process_symbol(symbol: str,
                   test_size: float = 0.2,
                   mode: str = "holdout"):
    feat_path = f"{FEATURE_DIR}/{symbol}.csv"
    if not os.path.exists(feat_path):
        print(f"→ no features for {symbol}")
        return

    feats = pd.read_csv(feat_path, index_col=0, parse_dates=True)
    close = pd.read_csv(f"{RAW_DIR}/{symbol}.csv",
                        index_col=0, parse_dates=True)["Close"]
    feats["Close"]  = close
    feats["target"] = build_target(close)

    X_all = feats.drop(columns=["Close", "target"]).replace([np.inf, -np.inf], np.nan)
    data  = pd.concat([X_all, feats["target"]], axis=1).dropna()
    X_all = data[X_all.columns]
    y_all = data["target"]

    if len(X_all) < 200 or y_all.nunique() < 2:
        print(f"→ skipping {symbol} (not enough data)")
        return

    out_json = []
    def train_window(X_tr, y_tr, X_va, y_va,
                     t0, t1, v0, v1):
        clf = DecisionTreeClassifier(max_depth=4, min_samples_leaf=50,
                                     random_state=42)
        clf.fit(X_tr, y_tr)

        # ---- calibrate ----------------------------------------------------- #
        cal_clf = calibrate_prefit(clf, X_va, y_va, method="isotonic")
        joblib.dump(cal_clf, f"{STRAT_DIR}/{symbol}_dt_cal.pkl")

        acc = cal_clf.score(X_va, y_va)
        rules = extract_tree_rules(clf, list(X_tr.columns))
        fi = dict(zip(X_tr.columns, clf.feature_importances_))

        rf = RandomForestClassifier(n_estimators=100, max_depth=6,
                                    random_state=42, n_jobs=-1)
        rf.fit(X_tr, y_tr)
        top_rf = sorted(zip(X_tr.columns, rf.feature_importances_),
                        key=lambda x: x[1], reverse=True)[:10]

        out_json.append({
            "symbol": symbol,
            "model_type": "DecisionTree",
            "window": f"{t0.date()}–{v1.date()}",
            "train_rows": len(X_tr),
            "val_rows":   len(X_va),
            "val_accuracy": round(acc, 4),
            "feature_importances": fi,
            "rules": rules,
            "rf_top_features": {f: round(w, 5) for f, w in top_rf},
            "timestamp": datetime.utcnow().isoformat() + "Z"
        })

    # ---------------- split strategy --------------------------------------- #
    if mode == "holdout":
        split = int(len(X_all) * (1 - test_size))
        train_window(X_all.iloc[:split], y_all.iloc[:split],
                     X_all.iloc[split:], y_all.iloc[split:],
                     X_all.index[0], X_all.index[split-1],
                     X_all.index[split], X_all.index[-1])
    else:
        for t0, t1, v0, v1 in build_walkforward_windows(X_all.index):
            tr = (X_all.index >= t0) & (X_all.index <= t1)
            va = (X_all.index >= v0) & (X_all.index <= v1)
            if va.sum() < 30 or tr.sum() < 100:
                continue
            train_window(X_all.loc[tr], y_all.loc[tr],
                         X_all.loc[va], y_all.loc[va],
                         t0, t1, v0, v1)

    with open(f"{STRAT_DIR}/{symbol}.json", "w") as f:
        json.dump(out_json, f, indent=2)
    print(f"√ {symbol}  windows={len(out_json)}")

# --------------------------------------------------------------------------- #
def run_all(symbols=None, **kwargs):
    for sym in _symbol_set(symbols):
        process_symbol(sym, **kwargs)

# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Train baseline DT models")
    p.add_argument("--mode", choices=["holdout", "walk"], default="holdout")
    p.add_argument("--syms", nargs="+")
    p.add_argument("--test_size", type=float, default=0.2)
    a = p.parse_args()
    run_all(a.syms, test_size=a.test_size, mode=a.mode)
