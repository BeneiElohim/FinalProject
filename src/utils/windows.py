import pandas as pd
from pandas.tseries.offsets import Day

def build_walkforward_windows(index,
                              min_train_years: int = 3,
                              val_years: int = 1,
                              test_years: int = 1):
    """
    Generate expanding-train / fixed-validation/test windows.

    Returns a list of 6-element tuples:
        (train_start, train_end, val_start, val_end, test_start, test_end)
    Example if min_train_years=3, val_years=1, test_years=1:
        2015-2017 train => 2018 val => 2019 test
        2015-2018 train => 2019 val => 2020 test
        ...
    """
    if not isinstance(index, pd.DatetimeIndex):
        raise ValueError("Input 'index' must be a pandas DatetimeIndex.")
        
    years = sorted(list(set(index.year)))
    if len(years) < min_train_years + val_years + test_years:
        return [] # Not enough data to create a single window

    windows = []
    # We stop when there are not enough future years for a full validation + test set
    for i in range(min_train_years, len(years) - val_years - test_years + 1):
        train_start_year = years[0]
        train_end_year = years[i-1]
        
        val_start_year = years[i]
        val_end_year = years[i + val_years - 1]

        test_start_year = years[i + val_years]
        test_end_year = years[i + val_years + test_years - 1]

        # Define precise start and end dates for each period
        t0 = pd.Timestamp(f"{train_start_year}-01-01")
        t1 = pd.Timestamp(f"{train_end_year}-12-31")
        
        v0 = pd.Timestamp(f"{val_start_year}-01-01")
        v1 = pd.Timestamp(f"{val_end_year}-12-31")
        
        test_start = pd.Timestamp(f"{test_start_year}-01-01")
        test_end = pd.Timestamp(f"{test_end_year}-12-31")
        
        # Ensure the periods are within the data's range
        if test_end <= index.max():
            windows.append((t0, t1, v0, v1, test_start, test_end))
            
    return windows

__all__ = ["build_walkforward_windows"]