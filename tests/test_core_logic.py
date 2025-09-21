import pandas as pd
import pytest
from datetime import datetime
import sys
from pathlib import Path

# Add project root to sys.path to allow imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.windows import build_walkforward_windows
from src.models.target import build_bigmove_target
from src.models.db import generate_strategy_id
from src.models.config import est_cost

@pytest.fixture
def sample_dates():
    """Provides a sample DatetimeIndex for testing."""
    return pd.to_datetime(pd.date_range(start='2015-01-01', end='2023-12-31', freq='B'))

def test_walkforward_windows_no_overlap(sample_dates):
    """Ensures validation and test sets are always sequential and non-overlapping."""
    windows = build_walkforward_windows(sample_dates, min_train_years=3, val_years=1, test_years=1)
    
    assert len(windows) > 0, "Should generate at least one window for the given date range"
    
    for t0, t1, v0, v1, test_start, test_end in windows:
        assert t1 < v0, "Train period must end before validation starts"
        assert v1 < test_start, "Validation period must end before test starts"
    
    # Check that the number of windows is correct
    # 2015-2023 is 9 years. Train(3)+Val(1)+Test(1) = 5 years.
    # Windows: (15-17, 18, 19), (15-18, 19, 20), (15-19, 20, 21), (15-20, 21, 22), (15-21, 22, 23)
    assert len(windows) == 5

def test_bigmove_target_labeling():
    """Tests if the target function correctly labels known significant price moves."""
    dates = pd.to_datetime(pd.date_range(start='2023-01-01', periods=10, freq='B'))
    
    # Test a clear positive move
    close_up = pd.Series([100, 101, 102, 103, 104, 105, 120, 121, 122, 123], index=dates)
    high_up, low_up = close_up + 1, close_up - 1
    target_up = build_bigmove_target(close_up, high_up, low_up, hold_days=5, vol_lookback=3, vol_mult=1.0)
    # Compares index 0 (100) with index 5 (105). This move is 5%, may not trigger.
    # Compares index 1 (101) with index 6 (120). This move is ~18.8%, should trigger.
    assert target_up.iloc[1] == 1, "Should label a clear upward move as 1"

    # Test a clear negative move
    close_down = pd.Series([100, 99, 98, 97, 96, 95, 80, 79, 78, 77], index=dates)
    high_down, low_down = close_down + 1, close_down - 1
    target_down = build_bigmove_target(close_down, high_down, low_down, hold_days=5, vol_lookback=3, vol_mult=1.0)
    assert target_down.iloc[1] == -1, "Should label a clear downward move as -1"

    # Test sideways move
    close_side = pd.Series([100] * 10, index=dates)
    high_side, low_side = close_side + 0.1, close_side - 0.1
    target_side = build_bigmove_target(close_side, high_side, low_side, hold_days=5, vol_lookback=3, vol_mult=1.0)
    assert target_side.iloc[1] == 0, "Should label a sideways move as 0"

def test_strategy_id_is_sensitive_to_all_params():
    """Ensures changing any part of the strategy definition results in a new, unique ID."""
    base_params = {
        "symbol": "TEST",
        "model_type": "dt",
        "rules": {"text": "rsi < 30"},
        "hyperparameters": {"max_depth": 3},
        "strategic_params": {"hold_days": 5, "vol_mult": 1.0}
    }
    
    id1 = generate_strategy_id(**base_params)
    
    # Change hyperparameters
    params2 = base_params.copy(); params2["hyperparameters"] = {"max_depth": 4}
    id2 = generate_strategy_id(**params2)
    assert id1 != id2, "Changing hyperparameters should produce a new ID"

    # Change strategic_params
    params3 = base_params.copy(); params3["strategic_params"] = {"hold_days": 10}
    id3 = generate_strategy_id(**params3)
    assert id1 != id3, "Changing strategic params should produce a new ID"

    # Change rules
    params4 = base_params.copy(); params4["rules"] = {"text": "rsi > 70"}
    id4 = generate_strategy_id(**params4)
    assert id1 != id4, "Changing rules should produce a new ID"

    # Identical params
    id5 = generate_strategy_id(**base_params)
    assert id1 == id5, "Identical parameters should produce the same ID"

def test_cost_model():
    """Tests the est_cost function with simple inputs."""
    close = pd.Series([100.0, 101.0, 102.0])
    atr14 = pd.Series([1.0, 1.1, 1.2]) # ~1% of price
    
    # Default params: 5bp fixed, 0.5 spread, 0.1 impact
    costs = est_cost(atr14, close)
    
    # Expected cost at index 1:
    # fixed = 0.0005
    # spread = 0.5 * (1.1 / 101.0) = 0.00544
    # impact = 0.1 * abs((1.1 - 1.0) / 1.0) = 0.01
    # total = 0.0005 + 0.00544 + 0.01 = 0.01594
    assert costs.iloc[1] == pytest.approx(0.01594, abs=1e-4)