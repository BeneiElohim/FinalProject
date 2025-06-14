import os
import json
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import argparse


def _symbol_set(filter_syms):
    files = [f for f in os.listdir(FEATURE_DIR) if f.endswith(".csv")]
    all_syms = [f[:-4] for f in files]
    return [s for s in all_syms if (not filter_syms) or (s in filter_syms)]

# TODO: Add walk-forward validation and hyperparameter tuning
def build_walkforward_windows(index, min_train_years=3, val_years=1):
    """
    Given a DatetimeIndex, return a list of
        (train_start, train_end, val_start, val_end)
    using expanding-window training and fixed-length validation.
    Example: first window -> train 2015-2017, validate 2018,
             next window  -> train 2015-2018, validate 2019, etc.
    """
    years = sorted({d.year for d in index})
    windows = []
    for i in range(min_train_years, len(years) - val_years + 1):
        train_start = pd.Timestamp(f"{years[0]}-01-01")
        train_end   = pd.Timestamp(f"{years[i-1]}-12-31")
        val_start   = pd.Timestamp(f"{years[i]}-01-01")
        val_end     = pd.Timestamp(f"{years[i]+val_years-1}-12-31")
        # ensure val_end is in index
        if val_end in index:
            windows.append((train_start, train_end, val_start, val_end))
    return windows

# PATHS
FEATURE_DIR    = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'features')
STRAT_DIR      = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'strategies')
os.makedirs(STRAT_DIR, exist_ok=True)

def build_target(df, hold_days=1, threshold=0.0):

    # Compute forward returns
    future_price = df['Close'].shift(-hold_days)
    returns       = (future_price - df['Close']) / df['Close']
    target        = (returns > threshold).astype(int)
    return target

# Extract human‐readable rules from a Decision Tree
def extract_tree_rules(clf: DecisionTreeClassifier, feature_names: list[str]) -> str:
    """
    Use sklearn.tree.export_text to get a human‐readable if–then description.
    """
    tree_text = export_text(clf, feature_names=feature_names)
    return tree_text

# Train for one symbol:
def process_symbol(symbol: str,
                   test_size: float = 0.2,
                   random_state: int = 42,
                   mode: str = "holdout"):            # mode: "holdout" | "walk"
    feat_path = os.path.join(FEATURE_DIR, f"{symbol}.csv")
    if not os.path.exists(feat_path):
        print(f"→ No features for {symbol}, skipping.")
        return

    df = pd.read_csv(feat_path, index_col=0, parse_dates=True)
    raw_path = os.path.join(os.path.dirname(__file__), '..', '..',
                            'data', 'processed', f"{symbol}.csv")
    price_df = pd.read_csv(raw_path, index_col=0, parse_dates=True)
    df['Close'] = price_df['Close']
    df['target'] = build_target(price_df)

    # clean NaNs/inf
    X_all = df.drop(columns=['Close', 'target']).replace([np.inf, -np.inf], np.nan)
    data  = pd.concat([X_all, df['target']], axis=1).dropna()
    X_all = data[X_all.columns]
    y_all = data['target']

    if len(X_all) < 200 or y_all.nunique() < 2:
        print(f"→ Not enough data for {symbol}, skipping.")
        return

    out_path = os.path.join(STRAT_DIR, f"{symbol}.json")
    existing = json.load(open(out_path)) if os.path.exists(out_path) else []

    def train_and_record(X_tr, y_tr, X_v, y_v,
                         tr_start, tr_end, va_start, va_end):
        clf = DecisionTreeClassifier(max_depth=4, min_samples_leaf=50,
                                     random_state=random_state)
        clf.fit(X_tr, y_tr)
        val_acc = clf.score(X_v, y_v)
        rules   = extract_tree_rules(clf, X_tr.columns.tolist())
        fi_dict = dict(zip(X_tr.columns, clf.feature_importances_.tolist()))

        rf = RandomForestClassifier(n_estimators=100, max_depth=6,
                                    random_state=random_state, n_jobs=-1)
        rf.fit(X_tr, y_tr)
        top_rf = sorted(zip(X_tr.columns, rf.feature_importances_),
                        key=lambda x: x[1], reverse=True)[:10]
        
        pkl_path = os.path.join(STRAT_DIR, f"{symbol}_dt.pkl")
        joblib.dump(clf, pkl_path)

        existing.append({
            'symbol': symbol,
            'model_type': 'DecisionTree',
            'train_period_start': str(tr_start.date()),
            'train_period_end':   str(tr_end.date()),
            'val_period_start':   str(va_start.date()),
            'val_period_end':     str(va_end.date()),
            'validation_accuracy': round(val_acc, 4),
            'feature_importances': fi_dict,
            'rules': rules,
            'rf_top_features': {f: round(w, 5) for f, w in top_rf},
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        })

    if mode == "holdout":
        split_idx = int(len(X_all) * (1 - test_size))
        X_tr, y_tr = X_all.iloc[:split_idx], y_all.iloc[:split_idx]
        X_v,  y_v  = X_all.iloc[split_idx:], y_all.iloc[split_idx:]
        train_and_record(X_tr, y_tr, X_v, y_v,
                         X_tr.index.min(), X_tr.index.max(),
                         X_v.index.min(),  X_v.index.max())
    else:  # walk-forward
        for t0, t1, v0, v1 in build_walkforward_windows(X_all.index):
            mask_tr = (X_all.index >= t0) & (X_all.index <= t1)
            mask_va = (X_all.index >= v0) & (X_all.index <= v1)
            X_tr, y_tr = X_all.loc[mask_tr], y_all.loc[mask_tr]
            X_v,  y_v  = X_all.loc[mask_va], y_all.loc[mask_va]
            if len(X_v) < 30 or len(X_tr) < 100:
                continue
            train_and_record(X_tr, y_tr, X_v, y_v, t0, t1, v0, v1)

    with open(out_path, 'w') as f:
        json.dump(existing, f, indent=2)
    print(f"→ Saved {symbol}.json ({len(existing)} total entries)")

# Run for all symbols in the features directory
def run_all(test_size: float = 0.2, mode: str = "holdout", symbols=None):
    for sym in _symbol_set(symbols):
        process_symbol(sym, test_size=test_size, mode=mode)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train baseline DT per symbol")
    parser.add_argument("--mode", choices=["holdout", "walk"], default="holdout")
    parser.add_argument("--syms", nargs="+", help="Optional subset of tickers")
    parser.add_argument("--test_size", type=float, default=0.2)
    args = parser.parse_args()

    run_all(test_size=args.test_size, mode=args.mode, symbols=args.syms)