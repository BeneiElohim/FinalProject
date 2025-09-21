import pandas as pd
import numpy as np
import json
from sqlalchemy import create_engine, text
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import time
from src.models.calibrate import calibrate_prefit
from src.models.target import build_bigmove_target
from src.utils.windows import build_walkforward_windows

from sqlalchemy import select, table, column
# --- Setup ---
# Adjust path to import from the project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from signal_engine.paths import DATA, ROOT
from src.ingestion import fetch_data, clean_data
from src.models.backtest_utils import run_backtest
from src.models.config import est_cost
from train import MODEL_MAP # To re-run for degradation test

DB_PATH = DATA / "strategies.db"
PROCESSED_DIR = DATA / "processed"
EVAL_DIR = DATA / "evaluation"
EVAL_DIR.mkdir(exist_ok=True)
engine = create_engine(f"sqlite:///{DB_PATH}")

# --- Helper Functions ---

def get_price_data(symbol: str) -> pd.DataFrame | None:
    """Robustly loads or creates processed price data for a given symbol."""
    processed_path = PROCESSED_DIR / f"{symbol}.csv"
    if processed_path.exists():
        return pd.read_csv(processed_path, index_col=0, parse_dates=True)
    
    print(f"  Processed file for {symbol} not found. Generating...")
    raw_df = fetch_data.fetch_single(symbol)
    if raw_df is None: return None
    cleaned_df = clean_data.clean_dataframe(raw_df)
    cleaned_df.to_csv(processed_path)
    return cleaned_df

# --- Analysis Functions ---

def analyze_interpretability():
    """1. Generates rule complexity metrics and extracts example rules."""
    print("\n--- 1. Analyzing Interpretability ---")
    query = text("SELECT s.symbol, s.model_type, s.rules, s.hyperparameters, b.sharpe_ratio FROM strategies s JOIN backtests b ON s.id = b.strategy_id")
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)

    results = []
    for model_type, group in df.groupby('model_type'):
        avg_depth_val = "N/A"
        if model_type in ['dt', 'xgb'] and not group['hyperparameters'].empty:
            depths = [int(json.loads(hp).get('max_depth', 0)) for hp in group['hyperparameters'] if hp]
            if depths:
                avg_depth_val = np.mean(depths)
        
        rule_lengths = [len(json.loads(r).get('text', '').split('\n')) for r in group['rules'] if r]
        avg_rules = np.mean(rule_lengths) if rule_lengths else 0
        
        results.append({
            "Model": model_type, 
            "Avg Rules/Lines": avg_rules, 
            "Avg Tree Depth": avg_depth_val
        })

    interp_df = pd.DataFrame(results).set_index("Model")
    print("\nRule Complexity Metrics:")
    print(interp_df.round(2))
    interp_df.to_csv(EVAL_DIR / "4_interpretability_metrics.csv")
    
    # Extract example rules from the best simple DT strategies
    dt_df = df[df['model_type'] == 'dt'].sort_values('sharpe_ratio', ascending=False)
    print("\nExample Rules (from top Decision Tree strategies):")
    for i, row in enumerate(dt_df.head(2).itertuples()):
        print(f"\n--- Example Rule #{i+1} (Symbol: {row.symbol}, Sharpe: {row.sharpe_ratio:.2f}) ---")
        rules_text = json.loads(row.rules).get('text', 'No text found.')
        print(rules_text)

def analyze_walk_forward_degradation():
    """2. Analyzes average in-sample vs. out-of-sample degradation across multiple top strategies."""
    print("\n--- 2. Analyzing Walk-Forward Degradation (Aggregated) ---")
    print("NOTE: This re-runs the pipeline for multiple top strategies and will be slow.")

    symbols_to_test = ['AAPL', 'JPM', 'BTC-USD', 'NVDA']
    s = table('strategies', column('id'), column('symbol'), column('model_type'), column('strategic_params'), column('hyperparameters'))
    b = table('backtests', column('strategy_id'), column('sharpe_ratio'))
    query = (select(s, b.c.sharpe_ratio).join(b, s.c.id == b.c.strategy_id).where(s.c.symbol.in_(symbols_to_test)))

    with engine.connect() as conn:
        all_strats_df = pd.read_sql(query, conn)
    
    if all_strats_df.empty:
        print("No strategies found for the selected symbols."); return

    top_strategies = all_strats_df.loc[all_strats_df.groupby(['symbol', 'model_type'])['sharpe_ratio'].idxmax()]
    
    print(f"Found {len(top_strategies)} top strategies to analyze...")
    degradation_results = []

    for strat in top_strategies.itertuples():
        print(f"  Analyzing {strat.symbol} - {strat.model_type}...")
        
        clean_df = get_price_data(strat.symbol)
        if clean_df is None: continue

        feature_cache_path = DATA / "features" / f"{strat.symbol}.feather"
        if not feature_cache_path.exists(): continue
        features_df = pd.read_feather(feature_cache_path).set_index('Date')
        price = clean_df['Close']

        model_class = MODEL_MAP[strat.model_type]
        pipeline_runner = model_class(strat.symbol, features_df, price, pd.Series(0, index=features_df.index))
        windows = build_walkforward_windows(features_df.index)
        
        strat_params = json.loads(strat.strategic_params)
        target = build_bigmove_target(
            close=price, high=clean_df["High"], low=clean_df["Low"], **strat_params
        ).reindex(features_df.index).dropna().astype(int).replace(-1, 0)
        
        aligned_features, aligned_target = features_df.align(target, join='inner', axis=0)

        for t0, t1, v0, v1, test_start, test_end in windows:
            train_val_mask = (aligned_features.index >= t0) & (aligned_features.index <= v1)
            X_train_val, y_train_val = aligned_features.loc[train_val_mask], aligned_target.loc[train_val_mask]
            
            val_mask = (aligned_features.index >= v0) & (aligned_features.index <= v1)
            X_val, y_val = aligned_features.loc[val_mask], aligned_target.loc[val_mask]
            
            # Handle the GP model differently from standard sklearn models
            if strat.model_type == 'gp':
                # GP model has its own integrated training method
                model, artifacts = pipeline_runner.train_window(X_train_val, y_train_val)
                calibrated_model = model 
                artifacts['prob_thresholds'] = (0.5, 0.5) 
            else:
                model_instance = pipeline_runner._get_estimator_and_grid()[0]
                hyperparams_str = json.loads(strat.hyperparameters)
                
                sanitized_hyperparams = {}
                for key, value in hyperparams_str.items():
                    try:
                        sanitized_hyperparams[key] = int(value)
                    except (ValueError, TypeError):
                        try:
                            sanitized_hyperparams[key] = float(value)
                        except (ValueError, TypeError):
                            sanitized_hyperparams[key] = value
                
                model_instance.set_params(**sanitized_hyperparams)
                model = model_instance.fit(X_train_val, y_train_val)
                calibrated_model = calibrate_prefit(model, X_val, y_val)
                artifacts = {'prob_thresholds': pipeline_runner._find_best_thresholds(calibrated_model, X_val, y_val)}

            if model is None: continue
            
            in_sample_entries, in_sample_exits = pipeline_runner.predict_signals(calibrated_model, X_train_val, artifacts)
            in_sample_res = run_backtest("in-sample", price.loc[X_train_val.index], in_sample_entries, in_sample_exits, fees=0.0005)
            in_sample_sharpe = in_sample_res['metrics'].get('sharpe_ratio', 0)
            
            test_mask = (aligned_features.index >= test_start) & (aligned_features.index <= test_end)
            X_test = aligned_features.loc[test_mask]
            if X_test.empty: continue
            
            oos_entries, oos_exits = pipeline_runner.predict_signals(calibrated_model, X_test, artifacts)
            oos_res = run_backtest("out-of-sample", price.loc[X_test.index], oos_entries, oos_exits, fees=0.0005)
            oos_sharpe = oos_res['metrics'].get('sharpe_ratio', 0)
            
            degradation_results.append({
                "Model": strat.model_type,
                "In-Sample Sharpe": in_sample_sharpe,
                "Out-of-Sample Sharpe": oos_sharpe,
            })

    if not degradation_results:
        print("Could not generate degradation results."); return

    degradation_df = pd.DataFrame(degradation_results)
    summary = degradation_df.groupby('Model').mean()
    summary['Degradation'] = ((summary['Out-of-Sample Sharpe'] - summary['In-Sample Sharpe']) / abs(summary['In-Sample Sharpe']).replace(0, np.nan)).apply(lambda x: f"{x:.1%}" if pd.notna(x) else "N/A")

    print("\nAverage Walk-Forward Performance Degradation by Model Type:")
    print(summary)
    summary.to_csv(EVAL_DIR / "5_walk_forward_degradation_summary.csv")

def generate_descriptive_stats():
    """3. Generates the exact descriptive statistics for the report."""
    print("\n--- 3. Generating Descriptive Statistics ---")
    with engine.connect() as conn:
        strategy_count = conn.execute(text("SELECT COUNT(*) FROM strategies")).scalar()
    
    symbols = (ROOT / "candidates.txt").read_text().splitlines()
    symbol_count = len(symbols)
    
    first_date = get_price_data(symbols[0]).index.min().year
    last_date = get_price_data(symbols[0]).index.max().year
    
    print(f"Assets Processed: {symbol_count} unique symbols.")
    print(f"Time Period: {first_date} to {last_date} (approx. {last_date - first_date} years).")
    print(f"Strategies Generated: {strategy_count} unique and valid strategies.")

def analyze_cost_sensitivity():
    """4. Generates data for parameter sensitivity to transaction costs."""
    print("\n--- 4. Analyzing Parameter Sensitivity (Transaction Costs) ---")
    query = text("SELECT s.id, s.symbol, s.model_type, b.sharpe_ratio, b.trade_log FROM strategies s JOIN backtests b ON s.id = b.strategy_id ORDER BY b.sharpe_ratio DESC LIMIT 5")
    with engine.connect() as conn:
        top_strategies_df = pd.read_sql(query, conn)
    
    sensitivity_results = []
    cost_multipliers = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0] # From zero cost to 10x

    for strat in top_strategies_df.itertuples():
        price_df = get_price_data(strat.symbol)
        if price_df is None: continue
        price = price_df['Close']
        trade_log = pd.DataFrame(json.loads(strat.trade_log))
        if trade_log.empty: continue
        
        # Get the integer positions
        entry_indices = trade_log['entry_idx'].astype(int)
        exit_indices = trade_log['exit_idx'].astype(int)
        
        # Use the integer positions to look up the ACTUAL timestamps from the price index
        entry_timestamps = price.index[entry_indices]
        exit_timestamps = price.index[exit_indices]

        entries = pd.Series(False, index=price.index)
        exits = pd.Series(False, index=price.index)
        
        # Use the correct timestamps to set the signals
        entries.loc[entry_timestamps] = True
        exits.loc[exit_timestamps] = True
        
        for mult in cost_multipliers:
            # Re-calculate costs with the new multiplier
            fees = est_cost(price.rolling(14).std().bfill(), price, k_spread=0.5*mult, k_impact=0.1*mult, fixed_comm_bps=5*mult)
            res = run_backtest(strat.symbol, price, entries, exits, fees=fees)
            sharpe = res['metrics'].get('sharpe_ratio', 0)
            sensitivity_results.append({
                "Strategy": f"{strat.symbol}-{strat.model_type}",
                "Cost Multiplier": f"{mult}x",
                "Sharpe Ratio": sharpe
            })

    if not sensitivity_results:
        print("Could not generate sensitivity results. No valid trades found in top strategies.")
        return

    sensitivity_df = pd.DataFrame(sensitivity_results).pivot_table(index='Cost Multiplier', columns='Strategy', values='Sharpe Ratio')
    # Sort index correctly
    cat_index = pd.CategoricalIndex([f"{m}x" for m in cost_multipliers], categories=[f"{m}x" for m in cost_multipliers], ordered=True)
    sensitivity_df = sensitivity_df.reindex(cat_index)

    print("\nCost Sensitivity of Top 5 Strategies:")
    print(sensitivity_df)
    sensitivity_df.to_csv(EVAL_DIR / "6_cost_sensitivity.csv")

    # Plotting
    plt.figure(figsize=(10, 6))
    sensitivity_df.plot(kind='line', marker='o', ax=plt.gca(), title="Strategy Sharpe Ratio vs. Transaction Cost Multiplier")
    plt.ylabel("Annualized Sharpe Ratio"); plt.xlabel("Cost Multiplier (1.0x = Default)"); plt.grid(True, linestyle='--'); plt.tight_layout()
    plt.savefig(EVAL_DIR / "6_cost_sensitivity.png"); plt.close()
    print(f"Saved plot to {EVAL_DIR / '6_cost_sensitivity.png'}")

def main():
    """Runs all the final evaluation scripts."""
    start_time = time.time()
    # 1. Interpretability
    analyze_interpretability()

    # 2. Walk-Forward (this is slow, run only when needed for the report)
    analyze_walk_forward_degradation() 
    # 3. Descriptive Stats
    generate_descriptive_stats()
    # 4. Cost Sensitivity
    analyze_cost_sensitivity()
    
    print(f"\n--- Final Suite Finished in {time.time() - start_time:.2f} seconds ---")
    print(f"All new artifacts saved to: {EVAL_DIR}")


if __name__ == "__main__":
    main()