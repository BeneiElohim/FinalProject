import os
import sys
from pathlib import Path
from sqlalchemy.orm import Session
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ingestion import fetch_data, clean_data
from src.features import compute_features
from src.models.target import build_bigmove_target
from train import MODEL_MAP
from src.models.parameter_grids import STRATEGIC_PARAMS, FEATURE_PARAMS
from signal_engine.paths import FEAT_DIR # 

def run_for_symbol(symbol: str, model_key: str, db: Session):
    """
    Runs the entire pipeline for a single symbol and model, with intelligent caching.
    """
    try:
        feature_cache_path = FEAT_DIR / f"{symbol}.feather"
        
        if feature_cache_path.exists():
            print(f"  Loading cached features for {symbol}...")
            features_df = pd.read_feather(feature_cache_path)
            features_df = features_df.set_index('Date')
        else:
            print(f"  No feature cache found for {symbol}. Generating from scratch...")
            raw_df = fetch_data.fetch_single(symbol)
            if raw_df is None or raw_df.empty:
                print(f"[SKIP] No data fetched for {symbol}.")
                return
            clean_df = clean_data.clean_dataframe(raw_df)
            features_df = compute_features.compute_features(clean_df, params=FEATURE_PARAMS)
            features_df.reset_index().to_feather(feature_cache_path)

        raw_df = fetch_data.fetch_single(symbol)
        clean_df = clean_data.clean_dataframe(raw_df)

        for strat_params in STRATEGIC_PARAMS:
            print(f"\nRunning model '{model_key}' with strategic params: {strat_params}")
            
            price = clean_df["Close"]
            target = build_bigmove_target(
                close=price,
                high=clean_df["High"],
                low=clean_df["Low"],
                hold_days=strat_params.get("hold_days", 5),
                vol_mult=strat_params.get("vol_mult", 1.0),
            ).reindex(features_df.index).dropna().astype(int).replace(-1, 0)
            
            aligned_features, aligned_price = features_df.align(price, join='inner', axis=0)
            aligned_features, aligned_target = aligned_features.align(target, join='inner', axis=0)

            model_class = MODEL_MAP.get(model_key)
            if not model_class:
                print(f"  [WARN] Unknown model '{model_key}'. Skipping.")
                continue

            pipeline_runner = model_class(
                symbol=symbol,
                features=aligned_features,
                price=aligned_price,
                target=aligned_target
            )
            pipeline_runner.run_pipeline(db, strategic_params=strat_params)

    except Exception as e:
        print(f"[FATAL] Unhandled exception for {symbol}/{model_key}: {e}")
        import traceback
        traceback.print_exc()