import pandas as pd

def build_walkforward_windows(index,
                              min_train_years: int = 3,
                              val_years: int = 1):
    """
    Generate expanding-train / fixed-validation windows.

    Returns a list of tuples:
        (train_start, train_end, val_start, val_end)
    Example if min_train_years=3, val_years=1:
        2015-2017 train → 2018 val
        2015-2018 train → 2019 val
        ...
    """
    years = sorted({d.year for d in index})
    windows = []
    for i in range(min_train_years, len(years) - val_years + 1):
        t0 = pd.Timestamp(f"{years[0]}-01-01")
        t1 = pd.Timestamp(f"{years[i-1]}-12-31")
        v0 = pd.Timestamp(f"{years[i]}-01-01")
        v1 = pd.Timestamp(f"{years[i] + val_years - 1}-12-31")
        if v1 in index:
            windows.append((t0, t1, v0, v1))
    return windows

__all__ = ["build_walkforward_windows"]
