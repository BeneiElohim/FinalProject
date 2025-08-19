import os
import certifi
from datetime import datetime
import pandas as pd
import yfinance as yf
import argparse

os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

RAW_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'raw')
os.makedirs(RAW_DIR, exist_ok=True)

def get_sp100_tickers():
    """Fetch the S&P 100 tickers from Wikipedia."""
    url = 'https://en.wikipedia.org/wiki/S%26P_100'
    tables = pd.read_html(url)
    sp100_df = tables[2]
    return sp100_df['Symbol'].tolist()

def fetch_single(ticker: str, start="2015-01-01", end=None) -> pd.DataFrame | None:
    """
    Fetches data for a single ticker and handles the MultiIndex column format.
    """
    if end is None:
        end = datetime.now().strftime('%Y-%m-%d')

    data = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False, timeout=10)

    if data.empty:
        print(f"[WARN] yfinance returned an empty DataFrame for {ticker}. Skipping.")
        return None

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(1)
    expected_cols = {'Open', 'High', 'Low', 'Close', 'Volume'}
    if not expected_cols.issubset(data.columns):
        print(f"[WARN] yfinance did not return valid OHLCV data for {ticker}. Skipping.")
        return None

    data.index = pd.to_datetime(data.index)
    data.dropna(subset=['Open', 'High', 'Low', 'Close'], how='all', inplace=True)

    return data if not data.empty else None

def run(tickers=None, start="2015-01-01", end=None):
    """
    Standalone script to fetch data and save to CSV files.
    """
    if end is None:
        end = datetime.now().strftime('%Y-%m-%d')
        
    if not tickers:
        print("No tickers provided, fetching S&P 100 + major crypto...")
        tickers = get_sp100_tickers() + ['BTC-USD', 'ETH-USD', 'BNB-USD', 'XRP-USD']
    
    print(f"Fetching data for {len(tickers)} tickers from {start} to {end}...")
    
    for ticker in tickers:
        df = fetch_single(ticker, start=start, end=end)
        if df is not None:
            df.to_csv(os.path.join(RAW_DIR, f'{ticker}.csv'))
            print(f"Saved data for {ticker}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download OHLCV for a list of tickers")
    parser.add_argument("--tickers", nargs="+", help="Space-separated tickers, e.g. --tickers AAPL MSFT")
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default=datetime.now().strftime('%Y-%m-%d'))
    args = parser.parse_args()
    
    run(tickers=args.tickers, start=args.start, end=args.end)