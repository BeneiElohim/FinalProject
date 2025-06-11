import os, json, warnings
import pandas as pd
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial
import time
import threading

from imodels import RuleFitClassifier        

BASE      = os.path.dirname(__file__) + "/../.."
FEAT_DIR  = f"{BASE}/data/features"
PROC_DIR  = f"{BASE}/data/processed"
STRAT_DIR = f"{BASE}/data/strategies"
os.makedirs(STRAT_DIR, exist_ok=True)
warnings.filterwarnings("ignore")


def build_target(close, hold=1):
    fwd_ret = close.shift(-hold) / close - 1
    return (fwd_ret > 0).astype(int)

def load_xy(sym):
    X = pd.read_csv(f"{FEAT_DIR}/{sym}.csv", index_col=0, parse_dates=True)
    close = pd.read_csv(f"{PROC_DIR}/{sym}.csv", index_col=0, parse_dates=True)["Close"]
    y = build_target(close).reindex(X.index).dropna()
    X = X.loc[y.index]
    return X, y


def train_rulefit_for_symbol(sym, thresh=0.5):
    try:
        start_time = time.time()
        
        X, y = load_xy(sym)
        if len(X) < 100:
            return f"× {sym}: not enough data for training (got {len(X)} rows)"
        
        # Subsample large datasets for speed
        if len(X) > 3000:  # Reduced threshold for even faster processing
            sample_size = 3000
            idx = X.sample(n=sample_size, random_state=42).index
            X, y = X.loc[idx], y.loc[idx]
        
        split = int(len(X)*0.8)
        X_train, X_val = X.iloc[:split], X.iloc[split:]
        y_train, y_val = y.iloc[:split], y.iloc[split:]


        rf_clf = RuleFitClassifier(
            tree_size      = 3,
            sample_fract   = 0.5,      
            max_rules      = 500,       
            n_estimators   = 50,       
            memory_par     = 0.01,
            random_state   = 42,
            include_linear = False,
        )
        rf_clf.fit(X_train.values, y_train.values, feature_names=list(X.columns))
        # Predict probabilities on validation
        prob_val = rf_clf.predict_proba(X_val.values)[:,1]
        y_pred   = (prob_val >= thresh).astype(int)
        acc      = accuracy_score(y_val, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_val, y_pred, average='binary')

        # Extract learned rules and coefficients
        rules = rf_clf.rules_
        all_coefs = rf_clf.coef
        
        # Extract rule coefficients
        if rf_clf.include_linear:
            rule_coefs = all_coefs[len(X.columns):]
        else:
            rule_coefs = all_coefs
        
        # Ensure arrays have same length
        min_len = min(len(rules), len(rule_coefs))
        rules = rules[:min_len]
        rule_coefs = rule_coefs[:min_len]
        
        # Create DataFrame for easier handling
        rules_df = pd.DataFrame({
            "rule": rules,
            "coef": rule_coefs
        })
        
        # Filter out zero coefficients and sort by absolute coefficient value
        rules_df = rules_df[rules_df['coef'] != 0].copy()
        if len(rules_df) > 0:
            rules_df = rules_df.reindex(rules_df['coef'].abs().sort_values(ascending=False).index)

        # Format top-N rules into plain text
        TOP_N = min(5, len(rules_df))  
        selected = rules_df.head(TOP_N)
        rule_texts = []
        
        for _, r in selected.iterrows():
            rule_texts.append({
                "rule": str(r['rule']),
                "coef": round(float(r['coef']), 5)
            })


        if len(rule_texts) == 0:
            rule_texts.append({
                "rule": "No significant rules found",
                "coef": 0.0
            })


        entry = {
            "symbol": sym,
            "model_type": "RuleFit",
            "train_rows": len(X_train),
            "val_rows":   len(X_val),
            "metrics": {
                "accuracy": round(acc,4),
                "precision":round(prec,4),
                "recall":   round(rec,4),
                "f1":       round(f1,4)
            },
            "threshold": thresh,
            "total_rules": len(rules_df),
            "top_rules": rule_texts,
            "processing_time": round(time.time() - start_time, 2),
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
        out = f"{STRAT_DIR}/{sym}_rulefit.json"
        with open(out,"w") as f:
            json.dump(entry, f, indent=2)
        
        return f"√ {sym}  acc={acc:.3f}  rules={len(rules_df)}  time={entry['processing_time']}s"
        
    except Exception as e:
        return f"× {sym}: {str(e)[:100]}..."  # Truncate long error messages

def train_single_symbol_wrapper(sym):
    """Wrapper for multiprocessing"""
    return train_rulefit_for_symbol(sym)

def run_all_parallel():
    syms = [f[:-4] for f in os.listdir(FEAT_DIR) if f.endswith(".csv")]
    
    n_cores = max(1, min(4, int(cpu_count() * 0.5)))  # Max 4 cores, 50% usage
    print(f"Using {n_cores} cores for parallel processing...")
    
    with Pool(n_cores) as pool:
        results = list(tqdm(
            pool.imap(train_single_symbol_wrapper, syms),
            total=len(syms),
            desc="Training RuleFit models",
            unit="symbol"
        ))
    
    # Print results
    for result in results:
        print(result)

def run_all_sequential():
    """Fallback sequential version"""
    syms = [f[:-4] for f in os.listdir(FEAT_DIR) if f.endswith(".csv")]
    results = []
    for s in tqdm(syms, desc="Training RuleFit models", unit="symbol"):
        result = train_rulefit_for_symbol(s)
        results.append(result)
        print(result)  # Print immediately for progress tracking
        
    return results

if __name__ == "__main__":
    print("Starting RuleFit training...")
    
    run_all_parallel()
    
    # Use sequential processing
   # run_all_sequential()