import os
import pandas as pd
import argparse

RAW_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'raw')
PROC_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'processed')
os.makedirs(PROC_DIR, exist_ok=True)

def _file_list(filter_syms):
    files = [f for f in os.listdir(RAW_DIR) if f.endswith(".csv")]
    if filter_syms:
        files = [f"{s}.csv" for s in filter_syms if f"{s}.csv" in files]
    return files

def clean_file(path):
    df = pd.read_csv(path, parse_dates=True, index_col=0)
    df = df[~df.index.duplicated(keep='first')]             # drop dupes
    df = df.asfreq('B', method='ffill')                   # business‐day calendar
    df.fillna(method='ffill', inplace=True)                # propagate last good
    return df

def run(symbols=None):
    for fname in _file_list(symbols):
        in_path  = os.path.join(RAW_DIR, fname)
        out_path = os.path.join(PROC_DIR, fname)
        df = clean_file(in_path)
        df.to_csv(out_path)
        print(f"Cleaned → {out_path} ({len(df)} rows)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Forward-fill & business-day align raw CSVs")
    parser.add_argument("--syms", nargs="+",
                        help="Tickers to clean, e.g. --syms MCD TMO PFE")
    args = parser.parse_args()
    run(args.syms)