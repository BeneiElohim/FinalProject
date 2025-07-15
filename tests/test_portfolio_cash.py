import pandas as pd, os
import vectorbt as vbt
from src.models.config import est_cost, INIT_CASH

RAW = "data/processed"
FEAT = "data/features"

def test_cash_nonnegative():
    any_sym = next(f[:-4] for f in os.listdir(RAW) if f.endswith(".csv"))
    price = pd.read_csv(f"{RAW}/{any_sym}.csv", index_col=0, parse_dates=True)["Close"]
    feats = pd.read_csv(f"{FEAT}/{any_sym}.csv", index_col=0, parse_dates=True)
    price = price.reindex(feats.index).ffill()
    atr   = feats["atr_14"]
    fees  = est_cost(atr, price)
    # dummy signals: always flat
    pf = vbt.Portfolio.from_signals(price, entries=False, exits=False,
                                    fees=fees, init_cash=INIT_CASH, freq="D")
    assert (pf.cash() >= 0).all()
