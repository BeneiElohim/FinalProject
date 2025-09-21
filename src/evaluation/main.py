import pandas as pd
import numpy as np
import json
from sqlalchemy import create_engine, text
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# --- Setup ---
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
# Adjust path to import from the project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from signal_engine.paths import DATA
from src.ingestion import fetch_data, clean_data
from src.models.backtest_utils import run_backtest
from src.models.config import est_cost

DB_PATH = DATA / "strategies.db"
PROCESSED_DIR = DATA / "processed"
EVAL_DIR = DATA / "evaluation"
PROCESSED_DIR.mkdir(exist_ok=True)
EVAL_DIR.mkdir(exist_ok=True)
engine = create_engine(f"sqlite:///{DB_PATH}")

# --- Helper Functions ---

def get_price_data(symbol: str) -> pd.DataFrame | None:
    """Robustly loads or creates processed price data for a given symbol."""
    processed_path = PROCESSED_DIR / f"{symbol}.csv"
    if processed_path.exists():
        return pd.read_csv(processed_path, index_col=0, parse_dates=True)
    
    print(f"  Processed file for {symbol} not found. Generating on-the-fly...")
    raw_df = fetch_data.fetch_single(symbol)
    if raw_df is None:
        return None
    
    cleaned_df = clean_data.clean_dataframe(raw_df)
    cleaned_df.to_csv(processed_path)
    return cleaned_df

def bootstrap_sharpe(returns: np.ndarray, num_resamples: int = 1000, seed: int = 42) -> tuple[float, float, float]:
    """Calculates mean Sharpe and a 95% confidence interval using a seeded RNG."""
    rng = np.random.RandomState(seed)
    if len(returns) < 30: return np.nan, np.nan, np.nan
    sharpe_ratios = []
    for _ in range(num_resamples):
        resampled_returns = rng.choice(returns, size=len(returns), replace=True)
        if np.std(resampled_returns, ddof=1) == 0: continue
        sharpe = (np.mean(resampled_returns) / np.std(resampled_returns, ddof=1)) * np.sqrt(252)
        sharpe_ratios.append(sharpe)
    if not sharpe_ratios: return np.nan, np.nan, np.nan
    return np.mean(sharpe_ratios), np.percentile(sharpe_ratios, 2.5), np.percentile(sharpe_ratios, 97.5)

# --- Analysis Functions ---

def evaluate_model_performance(df: pd.DataFrame) -> pd.DataFrame:
    """Task 1: Generate the main model performance summary with bootstrapped CIs."""
    print("\n--- 1. Evaluating Model Performance Summary ---")
    summary_results = []
    for model_type, group in df.groupby('model_type'):
        all_returns = [ret for ec_json in group['equity_curve'] if ec_json for ret in pd.read_json(ec_json, typ='series').pct_change().dropna().values]
        
        if all_returns:
            mean_s, lb, ub = bootstrap_sharpe(np.array(all_returns))
            summary_results.append({
                "Model": model_type,
                "Mean Sharpe": mean_s,
                "Sharpe CI Lower": lb,
                "Sharpe CI Upper": ub,
                "Strategy Count": len(group)
            })
    
    summary_df = pd.DataFrame(summary_results).set_index("Model").sort_values("Mean Sharpe", ascending=False)
    summary_df.to_csv(EVAL_DIR / "1_model_performance_summary.csv")
    print(summary_df)
    
    # Plotting
    summary_df['error'] = (summary_df['Sharpe CI Upper'] - summary_df['Sharpe CI Lower']) / 2
    plt.figure(figsize=(10, 6))
    summary_df['Mean Sharpe'].plot(kind='bar', yerr=summary_df['error'], title='Model Comparison: Mean Sharpe Ratio (95% CI)', capsize=4, colormap='viridis', legend=False)
    plt.ylabel("Annualized Sharpe Ratio"); plt.xticks(rotation=45, ha='right'); plt.grid(axis='y', linestyle='--', alpha=0.7); plt.tight_layout()
    plt.savefig(EVAL_DIR / "1_model_comparison_sharpe.png"); plt.close()
    print(f"Saved plot to {EVAL_DIR / '1_model_comparison_sharpe.png'}")
    return summary_df

def evaluate_baselines(df: pd.DataFrame, summary_df: pd.DataFrame) -> pd.DataFrame:
    """Task 2: Compare model performance against Buy & Hold and SMA Crossover baselines."""
    print("\n--- 2. Evaluating Baselines ---")
    symbols = df['symbol'].unique()
    
    # Buy & Hold
    buy_hold_sharpes = [((p := get_price_data(s)) is not None and (r := p['Close'].pct_change().dropna()).std() > 0 and (r.mean() / r.std()) * np.sqrt(252)) for s in symbols]
    avg_bh_sharpe = np.nanmean([s for s in buy_hold_sharpes if s is not False])
    print(f"Average Buy-and-Hold Sharpe: {avg_bh_sharpe:.3f}")

    # SMA Crossover
    sma_sharpes = []
    for s in symbols:
        price_df = get_price_data(s)
        if price_df is None: continue
        price = price_df['Close']
        entries = (price.rolling(50).mean() > price.rolling(200).mean()) & (price.rolling(50).mean().shift(1) <= price.rolling(200).mean().shift(1))
        exits = (price.rolling(50).mean() < price.rolling(200).mean()) & (price.rolling(50).mean().shift(1) >= price.rolling(200).mean().shift(1))
        res = run_backtest(s, price, entries, exits, fees=0.0005)
        sma_sharpes.append(res['metrics'].get('sharpe_ratio', 0))
    avg_sma_sharpe = np.mean(sma_sharpes)
    print(f"Average SMA Crossover Sharpe: {avg_sma_sharpe:.3f}")

    # Combine and Save
    summary_df['Vs. Buy & Hold'] = summary_df['Mean Sharpe'] - avg_bh_sharpe
    summary_df['Vs. SMA Crossover'] = summary_df['Mean Sharpe'] - avg_sma_sharpe
    summary_df.to_csv(EVAL_DIR / "2_model_vs_baselines.csv")
    print(summary_df[['Mean Sharpe', 'Vs. Buy & Hold', 'Vs. SMA Crossover']])
    return summary_df

def evaluate_market_regimes(df: pd.DataFrame):
    """Task 3: Analyze strategy performance in bull vs. bear markets."""
    print("\n--- 3. Evaluating Market Regimes ---")
    spy_data = get_price_data("SPY")
    if spy_data is None:
        print("Could not perform regime analysis: SPY data is missing.")
        return
        
    spy_sma = spy_data['Close'].rolling(window=200).mean()
    regime_results = []
    for model_type, group in df.groupby('model_type'):
        bull_returns, bear_returns = [], []
        for _, row in group.iterrows():
            if not row['equity_curve']: continue
            ec = pd.read_json(row['equity_curve'], typ='series')
            if len(ec) > len(spy_data): continue
            ec.index = spy_data.index[-len(ec):]
            returns = ec.pct_change().dropna()
            
            regime = pd.Series('bull', index=returns.index)
            regime.loc[returns.index.isin(spy_sma.index[spy_data['Close'] < spy_sma])] = 'bear'
            
            bull_returns.extend(returns[regime == 'bull'])
            bear_returns.extend(returns[regime == 'bear'])

        bull_sharpe = (np.mean(bull_returns) / np.std(bull_returns, ddof=1)) * np.sqrt(252) if len(bull_returns) > 1 else 0
        bear_sharpe = (np.mean(bear_returns) / np.std(bear_returns, ddof=1)) * np.sqrt(252) if len(bear_returns) > 1 else 0
        regime_results.append({"Model": model_type, "Bull Market Sharpe": bull_sharpe, "Bear Market Sharpe": bear_sharpe})

    regime_df = pd.DataFrame(regime_results).set_index("Model")
    regime_df.to_csv(EVAL_DIR / "3_regime_performance.csv")
    print(regime_df)
    
    # Plotting
    regime_df.plot(kind='bar', title='Performance in Market Regimes (SPY vs 200-day SMA)', figsize=(10, 6), colormap='coolwarm')
    plt.ylabel("Annualized Sharpe Ratio"); plt.xticks(rotation=45, ha='right'); plt.grid(axis='y', linestyle='--', alpha=0.7); plt.tight_layout()
    plt.savefig(EVAL_DIR / "3_regime_performance.png"); plt.close()
    print(f"Saved plot to {EVAL_DIR / '3_regime_performance.png'}")

def main():
    """Main function to run all evaluation tasks and generate artifacts."""
    print("--- Starting Full Evaluation Suite ---")
    
    query = text("SELECT s.symbol, s.model_type, b.sharpe_ratio, b.equity_curve, b.trade_log FROM strategies s JOIN backtests b ON s.id = b.strategy_id")
    try:
        with engine.connect() as connection:
            df = pd.read_sql(query, connection)
    except Exception as e:
        print(f"FATAL: Failed to read from database: {e}"); return

    if df.empty:
        print("FATAL: No data in database to evaluate."); return

    # Run all evaluation tasks
    summary_df = evaluate_model_performance(df)
    evaluate_baselines(df, summary_df)
    evaluate_market_regimes(df)
    print("\n--- Evaluation Suite Complete ---")
    print(f"All CSVs and plots saved to: {EVAL_DIR}")


if __name__ == "__main__":
    main()