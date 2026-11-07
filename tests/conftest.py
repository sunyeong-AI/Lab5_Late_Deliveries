"""Shared order data for feature and split tests."""

import pandas as pd
import pytest

from latedeliveries.data import make_orders


@pytest.fixture
def orders() -> pd.DataFrame:
    """Return the unchanged instructor generator's default orders."""
    return make_orders()
