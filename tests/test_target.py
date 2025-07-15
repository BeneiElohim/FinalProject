import pandas as pd, os
from src.models.target import build_bigmove_target

PROC_DIR = "data/processed"

def test_target_values():
    for f in os.listdir(PROC_DIR):
        if f.endswith(".csv"):
            close = pd.read_csv(f"{PROC_DIR}/{f}", index_col=0, parse_dates=True)["Close"]
            y = build_bigmove_target(close)
            assert set(y.dropna().unique()) <= {-1, 0, 1}
