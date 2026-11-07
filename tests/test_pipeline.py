"""Protect approved feature timing and held-out split integrity."""

import pandas as pd
import pytest

from latedeliveries.pipeline import CATEGORICAL, NUMERIC, split_orders


def test_approved_feature_lists():
    """Only approved order-time attributes are selected, without duplicates."""
    assert NUMERIC == ["distance_km", "items", "hour"]
    assert CATEGORICAL == ["weather", "day_type"]
    assert len(set(NUMERIC + CATEGORICAL)) == len(NUMERIC + CATEGORICAL)


@pytest.mark.parametrize(
    "excluded",
    [
        "order_id",
        "prep_minutes",
        "customer_rating",
        "vehicle",
        "courier_months",
        "late",
    ],
)
def test_excluded_columns_never_reach_features(orders, excluded):
    """Identifiers, future/uncertain attributes and target never enter X."""
    X_train, X_test, _, _ = split_orders(orders)
    assert excluded not in NUMERIC + CATEGORICAL
    assert excluded not in X_train.columns
    assert excluded not in X_test.columns


def test_default_data_and_split_sizes(orders):
    """The 3,000 default orders are split exactly 75/25."""
    X_train, X_test, y_train, y_test = split_orders(orders)
    assert orders.shape == (3000, 11)
    assert len(X_train) == len(y_train) == 2250
    assert len(X_test) == len(y_test) == 750


def test_split_preserves_rows_and_target_alignment(orders):
    """No row overlaps, disappears, changes values or loses its target pairing."""
    X_train, X_test, y_train, y_test = split_orders(orders)
    assert X_train.index.intersection(X_test.index).empty
    assert set(X_train.index) | set(X_test.index) == set(orders.index)
    for X, y in [(X_train, y_train), (X_test, y_test)]:
        assert list(X.columns) == NUMERIC + CATEGORICAL
        pd.testing.assert_frame_equal(X, orders.loc[X.index, NUMERIC + CATEGORICAL])
        pd.testing.assert_series_equal(y, orders.loc[X.index, "late"])


@pytest.mark.parametrize("label", [0, 1])
def test_split_preserves_class_proportions(orders, label):
    """Each partition's class counts differ from proportional counts by <=1."""
    X_train, X_test, y_train, y_test = split_orders(orders)
    prevalence = (orders["late"] == label).mean()
    for X, y in [(X_train, y_train), (X_test, y_test)]:
        assert abs((y == label).sum() - len(X) * prevalence) <= 1


def test_split_is_repeatable(orders):
    """Repeated calls keep the same row identities and order in both parts."""
    first = split_orders(orders)
    second = split_orders(orders)
    for left, right in zip(first, second, strict=True):
        if isinstance(left, pd.DataFrame):
            pd.testing.assert_frame_equal(left, right)
        else:
            pd.testing.assert_series_equal(left, right)


def test_split_does_not_modify_input(orders):
    """Selecting columns and splitting leave the raw order table untouched."""
    original = orders.copy(deep=True)
    split_orders(orders)
    pd.testing.assert_frame_equal(orders, original)
