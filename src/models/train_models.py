import os
import json
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# TODO: Add walk-forward validation and hyperparameter tuning
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
def process_symbol(symbol: str, test_size: float = 0.2, random_state: int = 42):
    """
    1. Load features for `symbol`
    2. Build target: next‐day up/down
    3. Drop last rows with NaN target
    4. Split into train/validation
    5. Train Decision Tree, extract rules, record performance
    6. Optionally train Random Forest + simple rule‐simplification
    7. Save candidate strategies to JSON in data/strategies/{symbol}.json
    """
    feat_path = os.path.join(FEATURE_DIR, f"{symbol}.csv")
    if not os.path.exists(feat_path):
        print(f"→ No features for {symbol}, skipping.")
        return

    df = pd.read_csv(feat_path, index_col=0, parse_dates=True)
    # We need the 'Close' column for target—so re‐load raw price
    raw_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'processed', f"{symbol}.csv")
    price_df = pd.read_csv(raw_path, index_col=0, parse_dates=True)
    df['Close'] = price_df['Close']

    # 1. Build target (next‐day direction: 1 if tomorrow's return > 0, else 0)
    df['target'] = build_target(price_df, hold_days=1, threshold=0.0)
    df.dropna(subset=['target'], inplace=True)

    # 2. Align features & target; separate X / y
    X = df.drop(columns=['Close', 'target'])
    y = df['target']
    # 2.2 Remove infinities and NaNs
    X = X.replace([np.inf, -np.inf], np.nan)
    data = pd.concat([X, y], axis=1).dropna()
    X = data[X.columns]
    y = data['target']

    if X.shape[0] < 10 or y.nunique() < 2:
        print(f"→ Not enough clean data for {symbol} (rows={X.shape[0]}, classes={y.nunique()}), skipping.")
        return

    # Chronological split index
    split_idx = int(len(X) * (1 - test_size))
    # Skip if split would produce empty train or val
    if split_idx < 1 or split_idx >= len(X):
        print(f"→ Train/val split invalid for {symbol} (split_idx={split_idx}, total={len(X)}), skipping.")
        return

    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

    if X_train.empty or X_val.empty:
        print(f"→ Empty train or val for {symbol}, skipping.")
        return
    # 4. Train Decision Tree
    clf = DecisionTreeClassifier(max_depth=4, min_samples_leaf=50, random_state=random_state)
    clf.fit(X_train, y_train)

    # 5. Evaluate on validation set
    val_acc = clf.score(X_val, y_val)
    # You can compute more metrics (precision, recall) if desired

    # 6. Extract tree rules
    feature_names = X.columns.tolist()
    tree_rules    = extract_tree_rules(clf, feature_names)

    # 7. Record feature importances
    fi = dict(zip(feature_names, clf.feature_importances_.tolist()))

    # 8. Prepare candidate strategy entry
    strategy_entry = {
        'symbol': symbol,
        'model_type': 'DecisionTree',
        'train_period_start': str(X_train.index.min().date()),
        'train_period_end':   str(X_train.index.max().date()),
        'val_period_start':   str(X_val.index.min().date()),
        'val_period_end':     str(X_val.index.max().date()),
        'validation_accuracy': round(val_acc, 4),
        'feature_importances': fi,
        'rules': tree_rules,
        'timestamp': datetime.utcnow().isoformat() + 'Z'
    }

    # 9. Train a Random Forest and save feature importances only
    rf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=random_state, n_jobs=-1)
    rf.fit(X_train, y_train)
    # We won’t extract all RF rules (too many), but record top‐features
    top_feats_rf = sorted(zip(feature_names, rf.feature_importances_), key=lambda x: x[1], reverse=True)[:10]
    strategy_entry['rf_top_features'] = {f: round(fi, 5) for f, fi in top_feats_rf}

    # 10. Save as JSON (if file exists, append to list; else create new list)
    out_path = os.path.join(STRAT_DIR, f"{symbol}.json")
    existing = []
    if os.path.exists(out_path):
        with open(out_path, 'r') as f:
            existing = json.load(f)
    existing.append(strategy_entry)
    with open(out_path, 'w') as f:
        json.dump(existing, f, indent=2)

    print(f"→ Saved {symbol}.json ({len(existing)} strategy entries)")

# Run for all symbols in the features directory
def run_all(test_size: float = 0.2):
    symbols = [fname.replace('.csv','') for fname in os.listdir(FEATURE_DIR) if fname.endswith('.csv')]
    for sym in symbols:
        process_symbol(sym, test_size=test_size)

if __name__ == "__main__":
    run_all()
