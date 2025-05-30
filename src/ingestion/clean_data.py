import os
import pandas as pd

RAW_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'raw')
PROC_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'processed')
os.makedirs(PROC_DIR, exist_ok=True)

def clean_file(path):
    df = pd.read_csv(path, parse_dates=True, index_col=0)
    df = df[~df.index.duplicated(keep='first')]             # drop dupes
    df = df.asfreq('B', method='ffill')                   # business‐day calendar
    df.fillna(method='ffill', inplace=True)                # propagate last good
    return df

def run():
    for fname in os.listdir(RAW_DIR):
        in_path = os.path.join(RAW_DIR, fname)
        out_path = os.path.join(PROC_DIR, fname)
        df = clean_file(in_path)
        df.to_csv(out_path)
        print(f"Cleaned → {out_path} ({len(df)} rows)")

if __name__ == "__main__":
    run()
