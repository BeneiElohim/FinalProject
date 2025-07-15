import pandas as pd, os

FEATURE_DIR = "data/features"

def test_no_nan():
    for f in os.listdir(FEATURE_DIR):
        if f.endswith(".csv"):
            df = pd.read_csv(f"{FEATURE_DIR}/{f}", index_col=0, parse_dates=True)
            assert not df.isna().any().any(), f"{f} has NaNs"
