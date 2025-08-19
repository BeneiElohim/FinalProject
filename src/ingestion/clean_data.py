import os
import pandas as pd
import argparse

RAW_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'raw')
PROC_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'processed')
os.makedirs(PROC_DIR, exist_ok=True)

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df_copy = df.copy()
    if not isinstance(df_copy.index, pd.DatetimeIndex):
        df_copy.index = pd.to_datetime(df_copy.index)
        
    # Drop duplicates, align to business day calendar, and forward-fill missing values
    df_copy = df_copy[~df_copy.index.duplicated(keep='first')]
    df_copy = df_copy.asfreq('B', method='ffill')
    df_copy.fillna(method='ffill', inplace=True)
    return df_copy

def run(symbols=None):
    files_to_process = [f for f in os.listdir(RAW_DIR) if f.endswith(".csv")]
    if symbols:
        files_to_process = [f"{s}.csv" for s in symbols if f"{s}.csv" in files_to_process]

    if not files_to_process:
        print("No raw data files found to clean.")
        return

    for fname in files_to_process:
        in_path = os.path.join(RAW_DIR, fname)
        out_path = os.path.join(PROC_DIR, fname)
        
        try:
            raw_df = pd.read_csv(in_path, index_col=0, parse_dates=True)
            cleaned_df = clean_dataframe(raw_df)
            cleaned_df.to_csv(out_path)
            print(f"Cleaned -> {out_path} ({len(cleaned_df)} rows)")
        except Exception as e:
            print(f"Failed to clean {fname}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Forward-fill & business-day align raw CSVs"
    )
    parser.add_argument("--syms", nargs="+",
                        help="Tickers to clean, e.g. --syms MCD TMO PFE")
    args = parser.parse_args()
    run(args.syms)