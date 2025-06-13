import os, json, warnings
import numpy as np, pandas as pd
from itertools import product
from datetime import datetime
from sklearn.tree    import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score
from sklearn.model_selection import TimeSeriesSplit
import argparse

BASE     = os.path.dirname(__file__) + "/../.."
FEAT_DIR = f"{BASE}/data/features"
PROC_DIR = f"{BASE}/data/processed"
STRATDIR = f"{BASE}/data/strategies"
os.makedirs(STRATDIR, exist_ok=True)
warnings.filterwarnings("ignore")

def _symbol_set(filter_syms):
    all_syms = [f[:-4] for f in os.listdir(FEAT_DIR) if f.endswith(".csv")]
    return [s for s in all_syms if (not filter_syms) or (s in filter_syms)]

PARAM_GRID = {
    "max_depth":       [3, 4, 5, 6],
    "min_samples_leaf":[10, 25, 50, 100],
}
PROB_THRESHOLDS = [(0.6, 0.4), (0.65, 0.35), (0.7, 0.3)]      # (long, short)

def build_target(close, hold=1):
    fwd_ret = close.shift(-hold).pct_change(periods=-hold)     # next-day %
    return (fwd_ret > 0).astype(int)

def load_xy(sym):
    X = pd.read_csv(f"{FEAT_DIR}/{sym}.csv", index_col=0, parse_dates=True)
    close = pd.read_csv(f"{PROC_DIR}/{sym}.csv", index_col=0, parse_dates=True)["Close"]
    y = build_target(close).reindex(X.index).dropna()
    X = X.loc[y.index]
    return X, y

def time_series_cv_score(model, X, y, splits=4):
    tscv = TimeSeriesSplit(n_splits=splits)
    scores = []
    for train_idx, val_idx in tscv.split(X):
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        y_pred = model.predict(X.iloc[val_idx])
        scores.append(accuracy_score(y.iloc[val_idx], y_pred))
    return np.mean(scores)

def best_tree_for_symbol(sym):
    X, y = load_xy(sym)
    best, best_params = -np.inf, None
    for depth, leaf in product(PARAM_GRID["max_depth"], PARAM_GRID["min_samples_leaf"]):
        clf = DecisionTreeClassifier(max_depth=depth,
                                     min_samples_leaf=leaf,
                                     random_state=42)
        score = time_series_cv_score(clf, X, y)
        if score > best:
            best, best_params = score, dict(max_depth=depth,
                                            min_samples_leaf=leaf)
    return best_params, best

def train_final_tree(sym, params):
    X, y = load_xy(sym)
    split = int(len(X) * 0.8)
    X_train, X_val = X.iloc[:split], X.iloc[split:]
    y_train, y_val = y.iloc[:split], y.iloc[split:]

    clf = DecisionTreeClassifier(**params, random_state=42)
    clf.fit(X_train, y_train)

    proba_val = clf.predict_proba(X_val)[:, 1]       # prob of class 1

    # choose threshold set that maximises accuracy after abstaining
    best_acc, best_thr = -np.inf, None
    for hi, lo in PROB_THRESHOLDS:
        preds = np.where(proba_val >= hi, 1,
                 np.where(proba_val <= lo, 0, -1))   # -1 = “no-trade”
        mask  = preds != -1
        if mask.any():
            acc = accuracy_score(y_val[mask], preds[mask])
            if acc > best_acc:
                best_acc, best_thr = acc, (hi, lo)

    rules_txt = export_text(clf, feature_names=list(X.columns))
    entry = {
        "symbol": sym,
        "model_type": "DecisionTree-Grid",
        "params": params,
        "cv_accuracy": round(best_acc, 4),
        "prob_thresholds": best_thr,
        "rules": rules_txt,
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }
    out = f"{STRATDIR}/{sym}_grid.json"
    with open(out, "w") as f:
        json.dump(entry, f, indent=2)
    print(f"√ {sym}  cv_acc={best_acc:.4f}  saved→{out}")

def run_all(symbols=None):
    for s in _symbol_set(symbols):
        try:
            params, cv = best_tree_for_symbol(s)
            train_final_tree(s, params)
        except Exception as e:
            print(f"× {s}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grid-search DT for subset")
    parser.add_argument("--syms", nargs="+")
    args  = parser.parse_args()
    run_all(args.syms)