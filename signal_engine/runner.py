import os
import sys
from pathlib import Path
from sqlalchemy.orm import Session

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ingestion import fetch_data, clean_data
from src.features import compute_features
from src.models.target import build_bigmove_target
from train import MODEL_MAP
from src.models.parameter_grids import STRATEGIC_PARAMS, FEATURE_PARAMS

def run_for_symbol(symbol: str, model_key: str, db: Session):
    """
    Runs the entire pipeline for a single symbol and model, iterating through
    all strategic and feature parameter combinations.
    """
    try:
        # 1. Ingestion (once per symbol)
        raw_df = fetch_data.fetch_single(symbol)
        if raw_df is None or raw_df.empty:
            print(f"[SKIP] No data fetched for {symbol}.")
            return
        clean_df = clean_data.clean_dataframe(raw_df)

        # 2. Iterate through strategic parameter sets
        for strat_params in STRATEGIC_PARAMS:
            print(f"\nRunning with strategic params: {strat_params}")
            
            # 3. Feature Engineering (dynamic based on params)
            print("  Generating features...")
            features_df = compute_features.compute_features(clean_df, params=FEATURE_PARAMS)
            
            # 4. Target Generation (dynamic based on params)
            price = clean_df["Close"]
            target = build_bigmove_target(
                close=price,
                high=clean_df["High"],
                low=clean_df["Low"],
                hold_days=strat_params.get("hold_days", 5),
                vol_mult=strat_params.get("vol_mult", 1.0),
            ).reindex(features_df.index).dropna().astype(int).replace(-1, 0)
            
            # Align data
            aligned_features, aligned_price = features_df.align(price, join='inner', axis=0)
            aligned_features, aligned_target = aligned_features.align(target, join='inner', axis=0)

            # 5. Get Model Class and Instantiate
            model_class = MODEL_MAP.get(model_key)
            if not model_class:
                print(f"  [WARN] Unknown model '{model_key}'. Skipping.")
                continue

            # 6. Run the self-tuning pipeline
            pipeline_runner = model_class(
                symbol=symbol,
                features=aligned_features,
                price=aligned_price,
                target=aligned_target
            )
            pipeline_runner.run_pipeline(db)

    except Exception as e:
        print(f"[FATAL] Unhandled exception for {symbol}/{model_key}: {e}")
        import traceback
        traceback.print_exc()