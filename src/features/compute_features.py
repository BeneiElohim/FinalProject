import os
import pandas as pd
import ta

# Paths
PROC_DIR    = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'processed')
FEATURE_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'features')
os.makedirs(FEATURE_DIR, exist_ok=True)

def compute_features(df):
    feat = pd.DataFrame(index=df.index)
    close = df['Close']
    high  = df['High']
    low   = df['Low']
    vol   = df['Volume']

    # Returns
    feat['ret_1d']   = close.pct_change(1)
    feat['ret_5d']   = close.pct_change(5)
    feat['ret_10d']  = close.pct_change(10)

    # Moving Averages
    feat['sma_5']    = ta.trend.sma_indicator(close, window=5)
    feat['sma_10']   = ta.trend.sma_indicator(close, window=10)
    feat['ema_20']   = ta.trend.ema_indicator(close, window=20)
    feat['ema_50']   = ta.trend.ema_indicator(close, window=50)

    # Momentum & Oscillators
    feat['roc_10']   = ta.momentum.roc(close, window=10)
    feat['rsi_14']   = ta.momentum.rsi(close, window=14)
    macd = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
    feat['macd']     = macd.macd()
    feat['macd_sig'] = macd.macd_signal()
    feat['macd_hist']= macd.macd_diff()

    # Volatility
    feat['atr_14']   = ta.volatility.average_true_range(high, low, close, window=14)
    feat['vol_20d']  = close.pct_change().rolling(20).std()

    # Mean Reversion (Bollinger)
    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    feat['bb_width'] = (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg()
    feat['pct_b']    = bb.bollinger_pband()

    # Volume-based
    feat['obv']      = ta.volume.on_balance_volume(close, vol)
    feat['vol_chg']  = vol.pct_change()

    # Trend Strength
    feat['adx_14']   = ta.trend.adx(high, low, close, window=14)

    # Drop initial NaNs
    return feat.dropna()

def run():
    for fname in os.listdir(PROC_DIR):
        if not fname.endswith('.csv'): 
            continue
        sym = fname.replace('.csv','')
        df  = pd.read_csv(os.path.join(PROC_DIR, fname), index_col=0, parse_dates=True)
        feats = compute_features(df)
        out_path = os.path.join(FEATURE_DIR, f"{sym}.csv")
        feats.to_csv(out_path)
        print(f"→ Features for {sym}: {feats.shape[0]} rows, {feats.shape[1]} cols")

if __name__ == "__main__":
    run()
