import os, json, warnings, time
from datetime import datetime
from multiprocessing import Pool, cpu_count
from typing import List, Dict

import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from imodels import RuleFitClassifier
import joblib
from tqdm import tqdm

from utils.windows import build_walkforward_windows
from calibrate import calibrate_prefit

# --------------------------------------------------------------------------- #
BASE      = os.path.dirname(__file__) + "/../.."
FEAT_DIR  = f"{BASE}/data/features"
PROC_DIR  = f"{BASE}/data/processed"
STRAT_DIR = f"{BASE}/data/strategies"
os.makedirs(STRAT_DIR, exist_ok=True)
warnings.filterwarnings("ignore")

def _symbol_set(filter_syms=None):
    all_syms = [f[:-4] for f in os.listdir(FEAT_DIR) if f.endswith(".csv")]
    return [s for s in all_syms if (not filter_syms) or (s in filter_syms)]

def build_target(close: pd.Series, hold: int = 1) -> pd.Series:
    fwd = close.shift(-hold) / close - 1
    return (fwd > 0).astype(int)

def load_xy(sym: str):
    X = pd.read_csv(f"{FEAT_DIR}/{sym}.csv", index_col=0, parse_dates=True)
    close = pd.read_csv(f"{PROC_DIR}/{sym}.csv",
                        index_col=0, parse_dates=True)["Close"]
    y = build_target(close).reindex(X.index).dropna()
    return X.loc[y.index], y

# --------------------------------------------------------------------------- #
def train_rulefit_for_symbol(sym: str, thresh: float = 0.5) -> str:
    X, y = load_xy(sym)
    if len(X) < 100 or y.nunique() < 2:
        return f"× {sym}: not enough data"

    windows = build_walkforward_windows(X.index, 3, 1)
    if not windows:
        return f"× {sym}: no windows"

    entries: List[Dict] = []
    for t0, t1, v0, v1 in windows:
        tr = (X.index >= t0) & (X.index <= t1)
        va = (X.index >= v0) & (X.index <= v1)
        if va.sum() < 30:     # tiny window guard
            continue

        X_tr, y_tr = X.loc[tr], y.loc[tr]
        X_va, y_va = X.loc[va], y.loc[va]

        rf_clf = RuleFitClassifier(
            tree_size=3,
            sample_fract=0.5,
            max_rules=500,
            n_estimators=50,
            memory_par=0.01,
            random_state=42,
            include_linear=False,
        )
        rf_clf.fit(X_tr.values, y_tr.values, feature_names=list(X.columns))

        # ---- calibration --------------------------------------------------- #
        cal_rf = calibrate_prefit(rf_clf, X_va.values, y_va.values,
                                  method="isotonic")
        prob_va = cal_rf.predict_proba(X_va.values)[:, 1]
        y_pred  = (prob_va >= thresh).astype(int)

        acc  = accuracy_score(y_va, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_va, y_pred, average="binary", zero_division=0)

        rules_df = pd.DataFrame({"rule": rf_clf.rules_, "coef": rf_clf.coef})
        rules_df = rules_df[rules_df["coef"] != 0]
        rules_df = rules_df.reindex(rules_df["coef"].abs()
                                    .sort_values(ascending=False).index)
        top_rules = [
            {"rule": str(r), "coef": round(float(c), 5)}
            for r, c in zip(rules_df["rule"][:5], rules_df["coef"][:5])
        ] or [{"rule": "No significant rules", "coef": 0.0}]

        entries.append({
            "symbol": sym,
            "model_type": "RuleFit",
            "window": f"{t0.date()}–{v1.date()}",
            "train_rows": int(tr.sum()),
            "val_rows":   int(va.sum()),
            "metrics": {
                "accuracy":  round(acc, 4),
                "precision": round(prec, 4),
                "recall":    round(rec, 4),
                "f1":        round(f1, 4)
            },
            "threshold": thresh,
            "total_rules": len(rules_df),
            "top_rules":   top_rules,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        })

    if not entries:
        return f"× {sym}: every window skipped"

    json.dump(entries, open(f"{STRAT_DIR}/{sym}_rulefit.json", "w"), indent=2)
    joblib.dump(cal_rf, f"{STRAT_DIR}/{sym}_rulefit_cal.pkl")
    return f"√ {sym}  windows={len(entries)}"

# --------------------------------------------------------------------------- #
def run_all_parallel(symbols=None):
    syms = _symbol_set(symbols)
    n = max(1, min(4, int(cpu_count() * 0.5)))
    with Pool(n) as pool:
        for res in tqdm(pool.imap(train_rulefit_for_symbol, syms),
                        total=len(syms), unit="sym"):
            print(res)

# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--syms", nargs="+")
    a = p.parse_args()
    run_all_parallel(a.syms)
