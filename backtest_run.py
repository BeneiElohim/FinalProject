import json, os
import pandas as pd
import vectorbt as vbt
import joblib
import src.models.config as C
import argparse, textwrap

BASE = os.path.dirname(__file__)
RAW   = f"{BASE}/data/processed"     
FEAT  = f"{BASE}/data/features"      
RULES = f"{BASE}/data/strategies"
STRAT = f"{BASE}/data/strategies"  


def _load_price(sym):
    return pd.read_csv(f"{RAW}/{sym}.csv", index_col=0, parse_dates=True)["Close"]

def _load_feats(sym):
    return pd.read_csv(f"{FEAT}/{sym}.csv", index_col=0, parse_dates=True)

def _pf_stats(pf):
    want = ["Total Return [%]", "CAGR", "Sharpe Ratio", "Max Drawdown [%]"]
    s = pf.stats()
    cols = [c for c in want if c in s.index]           
    if not cols:                                      
        cols = list(s.index[:4])
    return s[cols]

def _run_port(price, entries, exits):
    pf = vbt.Portfolio.from_signals(
        price, entries, exits,
        fees=C.FEE_BPS / 10_000,
        init_cash=C.INIT_CASH,
        freq="D")
    return _pf_stats(pf)

def bt_dt(sym):
    mdl = joblib.load(f"{STRAT}/{sym}_dt.pkl")
    feats  = _load_feats(sym) ; price = _load_price(sym).reindex(feats.index).ffill()
    prob   = pd.Series(mdl.predict_proba(feats)[:,1], index=feats.index)
    return _run_port(price, prob >= 0.50, prob < 0.50).rename(sym)

def bt_grid(sym):
    mdl  = joblib.load(f"{STRAT}/{sym}_grid.pkl")
    with open(f"{STRAT}/{sym}_grid.json") as f:
        hi, lo = json.load(f)["prob_thresholds"] or (0.70, 0.30)
    feats  = _load_feats(sym) ; price = _load_price(sym).reindex(feats.index).ffill()
    prob   = pd.Series(mdl.predict_proba(feats)[:,1], index=feats.index)
    return _run_port(price, prob >= hi, prob <= lo).rename(sym)

def bt_rulefit(sym):
    feats  = _load_feats(sym) ; price = _load_price(sym).reindex(feats.index).ffill()
    with open(f"{STRAT}/{sym}_rulefit.json") as f:
        rule = json.load(f)["top_rules"][0]["rule"]
    sig = pd.eval(rule, local_dict=feats).fillna(0).astype(bool)
    return _run_port(price, sig, ~sig).rename(sym)

def bt_xgb(sym):
    pkl = max([f for f in os.listdir(STRAT) if f.startswith(f"{sym}_") and f.endswith("_xgb.pkl")])
    mdl  = joblib.load(f"{STRAT}/{pkl}")
    feats  = _load_feats(sym) ; price = _load_price(sym).reindex(feats.index).ffill()
    prob   = pd.Series(mdl.predict_proba(feats)[:,1], index=feats.index)
    return _run_port(price, prob >= C.PROB_LONG, prob <= C.PROB_FLAT).rename(sym)

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

def print_rules(sym, model_tag, max_lines=12):
    """Pretty-print rules for dt | grid | rulefit"""
    if model_tag == "dt":
        path = f"{STRAT}/{sym}.json"
        meta = json.load(open(path))[-1] if isinstance(json.load(open(path)), list) else json.load(open(path))
        text = meta["rules"]
        head = "\n".join(text.splitlines()[:max_lines])
        print(f"\n{sym} – Decision Tree rules (first {max_lines} lines):\n{head}")
    elif model_tag == "grid":
        meta = json.load(open(f"{STRAT}/{sym}_grid.json"))
        head = "\n".join(meta["rules"].splitlines()[:max_lines])
        print(f"\n{sym} – Grid-DT rules (first {max_lines} lines):\n{head}")
    elif model_tag == "rulefit":
        meta = json.load(open(f"{STRAT}/{sym}_rulefit.json"))
        print(f"\n{sym} – RuleFit top-rules (coef | rule):")
        for r in meta["top_rules"]:
            print(f"{r['coef']:>7.4f} │ {r['rule']}")
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Back-test DT, Grid-DT, RuleFit and XGB engines",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            example:
              python backtest_run.py --syms MCD TMO COP PFE
              python backtest_run.py --syms MCD TMO --models dt rulefit
        """)
    )
    parser.add_argument("--syms",   nargs="+", required=True)
    parser.add_argument("--models", nargs="+",
                        choices=["dt", "grid", "rulefit", "xgb"],
                        default=["dt", "grid", "rulefit", "xgb"])
    args = parser.parse_args()

    engines = {
        "dt": bt_dt,
        "grid": bt_grid,
        "rulefit": bt_rulefit,
        "xgb": bt_xgb
    }

    for tag in args.models:
        tbl = pd.concat([engines[tag](s) for s in args.syms], axis=1).T.round(2)
        print(f"\n=== {tag.upper()} BACK-TEST RESULTS ===\n")
        print(tbl)

        if tag in ("dt", "grid", "rulefit"):
            for sym in args.syms:
                print_rules(sym, tag)