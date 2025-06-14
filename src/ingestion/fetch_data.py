import os, certifi 
from datetime import datetime
import pandas as pd
import yfinance as yf
import argparse


os.environ['REQUESTS_CA_BUNDLE']=certifi.where()

RAW_DIR = os.path.join(os.path.dirname(__file__), '..','..','data','raw')
os.makedirs(RAW_DIR, exist_ok=True)

def get_sp100_tickers():
    """Fetch the S&P 100 tickers from Wikipedia."""
    url = 'https://en.wikipedia.org/wiki/S%26P_100'
    tables = pd.read_html(url)
    sp100_df = tables[2]
    return sp100_df['Symbol'].tolist()

def fetch_and_save_data(tickers,start,end):
    """Fetch historical stock data for the given tickers and save to CSV."""
    data = yf.download(tickers, start=start, end=end, group_by='ticker', auto_adjust=True)
    for ticker in tickers:
        df = data[ticker].copy()
        df.index = pd.to_datetime(df.index)
        df.to_csv(os.path.join(RAW_DIR, f'{ticker}.csv'))
        print(f"Saved data for {ticker} to {RAW_DIR}/{ticker}.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download OHLCV for a custom list of tickers")
    parser.add_argument("--tickers", nargs="+",
                        help="Space-separated tickers, e.g. --tickers MCD TMO COP PFE")
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end",   default=datetime.now().strftime('%Y-%m-%d'))
    args = parser.parse_args()
    if args.tickers:
        tickers = args.tickers
    else:
        tickers = get_sp100_tickers() + ['BTC-USD', 'ETH-USD', 'BNB-USD', 'XRP-USD']
    fetch_and_save_data(tickers, start='2015-01-01', end=datetime.now().strftime('%Y-%m-%d'))
