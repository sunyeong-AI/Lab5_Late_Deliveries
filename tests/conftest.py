"""Shared order data for feature and split tests."""

import pandas as pd
import pytest

from latedeliveries.data import make_orders


@pytest.fixture
def orders() -> pd.DataFrame:
    """Return the unchanged instructor generator's default orders."""
    return make_orders()


@pytest.fixture
def synthetic_orders() -> tuple[pd.DataFrame, pd.Series]:
    """Return six invented training rows; these are not held-out lab orders."""
    frame = pd.DataFrame(
        {
            "distance_km": [1.0, 3.0, float("nan"), 7.0, 9.0, 11.0],
            "items": [1, 2, 3, 4, 5, 6],
            "hour": [10, 12, 14, 16, 18, 20],
            "weather": ["clear", "rain", "snow", "clear", "rain", "snow"],
            "day_type": ["weekday", "weekend"] * 3,
            "prep_minutes": [12.0] * 6,
        }
    )
    return frame, pd.Series([0, 0, 1, 0, 1, 1], name="late")
