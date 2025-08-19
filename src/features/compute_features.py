import pandas as pd
import numpy as np
import ta
from ta.utils import dropna
import yfinance as yf

_spy_data_cache = None

def get_spy_data():
    """Fetches and caches S&P 500 ETF (SPY) data, ensuring simple columns."""
    global _spy_data_cache
    if _spy_data_cache is None:
        print("    Fetching SPY data for market context...")
        data = yf.download("SPY", start="2014-01-01", auto_adjust=True, progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)
            
        _spy_data_cache = data
    return _spy_data_cache

def compute_features(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Computes a wide range of technical analysis features based on the
    provided parameter dictionary, including market-relative features.
    """
    feat = pd.DataFrame(index=df.index)
    close = df['Close']
    high = df['High']
    low = df['Low']
    vol = df['Volume'].replace(0, 1) # Avoid division by zero in volume calcs

    # --- 1. Market Context Features ---
    spy = get_spy_data()
    spy_close = spy['Close'].reindex(df.index, method='ffill')
    
    feat['relative_strength_spy'] = (close / spy_close)
    for w in params.get("ret_windows", []):
        feat[f'spy_ret_{w}d'] = spy_close.pct_change(w)

    # Long-term trend filter
    for w in params.get("regime_windows", []):
        feat[f'price_vs_sma_{w}'] = close / ta.trend.sma_indicator(close, window=w)

    # --- 2. Standard Technical Indicators ---
    # Returns
    for w in params.get("ret_windows", []):
        feat[f'ret_{w}d'] = close.pct_change(w)

    # Moving Averages
    for w in params.get("sma_windows", []):
        feat[f'sma_{w}'] = ta.trend.sma_indicator(close, window=w)
    for w in params.get("ema_windows", []):
        feat[f'ema_{w}'] = ta.trend.ema_indicator(close, window=w)

    # Momentum & Oscillators
    for w in params.get("roc_windows", []):
        feat[f'roc_{w}'] = ta.momentum.roc(close, window=w)
    for w in params.get("rsi_windows", []):
        feat[f'rsi_{w}'] = ta.momentum.rsi(close, window=w)
    for w in params.get("stoch_windows", []):
        feat[f'stoch_k_{w}'] = ta.momentum.stoch(high, low, close, window=w)
    
    macd = ta.trend.MACD(close)
    feat['macd'] = macd.macd_diff()

    # Volatility
    for w in params.get("atr_windows", []):
        feat[f'atr_{w}'] = ta.volatility.average_true_range(high, low, close, window=w)
    for w in params.get("vol_windows", []):
        feat[f'vol_{w}d'] = close.pct_change().rolling(w).std()

    # Mean Reversion (Bollinger)
    for w in params.get("bb_windows", []):
        bb = ta.volatility.BollingerBands(close, window=w, window_dev=2)
        feat[f'bb_width_{w}'] = bb.bollinger_wband()
        feat[f'pct_b_{w}'] = bb.bollinger_pband()

    # Volume-based
    for w in params.get("cmf_windows", []):
        feat[f'cmf_{w}'] = ta.volume.chaikin_money_flow(high, low, close, vol, window=w)
    feat['obv_ema_ratio'] = ta.volume.on_balance_volume(close, vol) / ta.trend.ema_indicator(ta.volume.on_balance_volume(close, vol), window=20)

    # Trend Strength
    for w in params.get("adx_windows", []):
        feat[f'adx_{w}'] = ta.trend.adx(high, low, close, window=w)
    for w in params.get("vortex_windows", []):
        vortex = ta.trend.VortexIndicator(high, low, close, window=w)
        feat[f'vortex_diff_{w}'] = vortex.vortex_indicator_diff()

    # Clean up and return
    feat = feat.replace([np.inf, -np.inf], np.nan)
    final_feat = feat.dropna(axis=0, how='any')
    
    return final_feat
