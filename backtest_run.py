import json, os
import pandas as pd
import vectorbt as vbt
import joblib
import src.models.config as C

BASE = os.path.dirname(__file__)
RAW   = f"{BASE}/data/processed"     
FEAT  = f"{BASE}/data/features"      
RULES = f"{BASE}/data/strategies"


def latest_xgb_model(sym):
    """
    Locate the most recent pickle produced by xgb_surrogate.py.
    File naming pattern:  data/strategies/{sym}_{VAL_END_YEAR}_xgb.pkl
    """
    files = [f for f in os.listdir(RULES) if f.startswith(f"{sym}_") and f.endswith("_xgb.pkl")]
    if not files:
        raise FileNotFoundError(f"No XGB model pickle found for {sym}")
    # sort by VAL_END_YEAR extracted from filename
    files.sort(key=lambda f: int(f.split("_")[1]))     # ['TMO_2024_xgb.pkl', ...]
    return os.path.join(RULES, files[-1])              # newest

def xgb_probability_series(sym, model_path):
    """
    Return a pd.Series of P(long) from the XGB model lined up with feature dates.
    """
    feats = load_feats(sym)
    mdl   = joblib.load(model_path)
    proba = pd.Series(
        mdl.predict_proba(feats.values)[:, 1],
        index=feats.index,
        name="prob_long")
    return proba

def prob_to_signals(prob_ser, p_long=C.PROB_LONG, p_flat=C.PROB_FLAT):
    """
    Turn probability series into long/flat boolean masks.
    """
    long_mask = prob_ser >= p_long
    flat_mask = prob_ser <= p_flat
    return long_mask, flat_mask

def backtest_xgb(sym):
    """
    One-liner interface mirroring backtest() for RuleFit.
    """
    price = load_close(sym)
    model_path = latest_xgb_model(sym)
    proba = xgb_probability_series(sym, model_path)

    # Align price to proba index & forward-fill
    price = price.reindex(proba.index).ffill()
    price.index = pd.DatetimeIndex(price.index, freq=None)

    long_sig, flat_sig = prob_to_signals(proba)

    pf = vbt.Portfolio.from_signals(
        price,
        entries=long_sig,
        exits=flat_sig,
        fees=C.FEE_BPS / 10_000,
        init_cash=C.INIT_CASH,
        freq="D"
    )
    preferred = {
        "Total Return":      "Total Return [%]",
        "Annual Return":     "Ann. Return [%]",
        "CAGR":              "CAGR",
        "Sharpe":            "Sharpe Ratio",
        "Max Drawdown":      "Max. Drawdown [%]"
    }

    stats = pf.stats()
    chosen = []
    for human, raw in preferred.items():
        # some versions omit brackets or use different wording
        for cand in (raw, raw.replace(" [%]", ""), raw.replace("Ann. ", "Annual ")):
            if cand in stats.index:
                chosen.append(cand)
                break

    # if nothing found, just take the first four available metrics
    if not chosen:
        chosen = list(stats.index[:4])

    return stats[chosen].rename(sym)

def load_close(sym):
    return pd.read_csv(f"{RAW}/{sym}.csv", index_col=0, parse_dates=True)['Close']

def load_feats(sym):
    return pd.read_csv(f"{FEAT}/{sym}.csv", index_col=0, parse_dates=True)

def top_rule(sym):
    path = f"{RULES}/{sym}_rulefit.json"
    with open(path) as f:
        data = json.load(f)
    return data['top_rules'][0]['rule']        # string

def rule_to_signal(rule_str, feats):
    """
    Pandas eval → 1/0 sinyali (uzun/flat).
    Örnek kural: "(adx_14 > 30) & (vol_20d < 0.02)"
    """
    sig = pd.eval(rule_str, local_dict=feats).astype(int)
    # forward-fill boşluklar:
    return sig.reindex(feats.index).fillna(0)

def backtest(sym):
    price = load_close(sym)
    feats = load_feats(sym)

    price = price.reindex(feats.index).ffill()
    price.index = pd.DatetimeIndex(price.index, freq=None)
    
    rule = top_rule(sym)
    signal = rule_to_signal(rule, feats)

    pf = vbt.Portfolio.from_signals(
        price,
        entries=signal == 1,
        exits=signal == 0,
        fees=0.0005,
        init_cash=100_000,
        freq='D'
    )
    
    # Get all available stats first
    all_stats = pf.stats()
    
    # Define mapping of desired stats to actual available names
    desired_stats = []
    stat_mapping = {
        'Total Return [%]': 'Total Return',
        'Total Return': 'Total Return', 
        'Ann. Return [%]': 'CAGR',
        'CAGR': 'CAGR',
        'Sharpe Ratio': 'Sharpe Ratio',
        'Max. Drawdown [%]': 'Max Drawdown',
        'Max Drawdown': 'Max Drawdown'
    }
    
    # Find which stats are actually available
    for actual_name, desired_name in stat_mapping.items():
        if actual_name in all_stats.index:
            desired_stats.append(actual_name)
            break
    
    # If no specific stats found, just return first few available stats
    if not desired_stats:
        print(f"Available stats for {sym}: {list(all_stats.index)}")
        desired_stats = all_stats.index[:4].tolist()
    
    stats = all_stats[desired_stats]
    return stats.rename(sym)

if __name__ == "__main__":
    with open("candidates.txt") as f:
        syms = [s.strip() for s in f if s.strip()]

    results = pd.concat([backtest(s) for s in syms], axis=1).T
    print("\n=== BACKTEST RESULTS ===\n")
    print(results.round(2))

    SYMS_XGB = ["TMO", "MCD"]          # or read from a file
    res_xgb = pd.concat([backtest_xgb(s) for s in SYMS_XGB], axis=1).T
    print("\n=== XGB BACKTEST RESULTS ===\n")
    print(res_xgb.round(2))