"""Hand-worked cost and threshold checks before any real-data application."""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from latedeliveries import evaluation
from latedeliveries.pipeline import NUMERIC, SEED


@pytest.mark.parametrize(
    "y,flagged,expected",
    [
        ([0, 1], [1, 1], 8),  # Both vouchers cost 4, including a true positive.
        ([1, 1, 0], [0, 0, 0], 30),  # Two missed delays cost 2*15.
        ([0, 0], [0, 0], 0),
        ([1, 0, 1, 0], [1, 0, 0, 1], 23),  # 2*4 + 1*15.
    ],
)
def test_hand_calculated_order_cost(y, flagged, expected):
    """Charge every voucher and only unflagged delays, never missed on-time rows."""
    assert evaluation.order_cost(y, flagged) == expected


@pytest.mark.parametrize(
    "y,probability,threshold,flagged,missed,cost",
    [
        ([0, 0], [0.10, 0.20], 0.25, 0, 0, 0),  # Nobody: all vouchers waste 4.
        ([1, 1], [0.80, 0.90], 0.05, 2, 0, 8),  # Everybody: 8 vs 19 vs 30.
        ([0, 1, 1], [0.10, 0.40, 0.80], 0.15, 2, 0, 8),  # 12 vs 8 vs 19 vs 30.
        ([0, 0], [0.0, 0.0], 0.05, 0, 0, 0),  # All 19 tie: choose 0.05.
    ],
)
def test_hand_calculated_threshold_selection(
    y, probability, threshold, flagged, missed, cost
):
    """Pick the first minimum, including nobody/everybody and an interior optimum."""
    chosen, table = evaluation.choose_threshold(y, probability)
    assert chosen == threshold
    best = table.loc[table.threshold == chosen].iloc[0]
    assert best.flagged_orders == flagged
    assert best.missed_late_orders == missed
    assert best.total_cost == cost
    for row in table.itertuples():
        decisions = [p >= row.threshold for p in probability]
        vouchers = sum(decisions)
        missed_delays = sum(
            actual == 1 and not flag for actual, flag in zip(y, decisions, strict=True)
        )
        assert row.total_cost == vouchers * 4 + missed_delays * 15


def test_exact_nineteen_thresholds_and_inclusive_boundary():
    """Decimal boundaries are generated without cumulative addition errors."""
    _, table = evaluation.choose_threshold([1], [0.15])
    expected = [i / 100 for i in range(5, 100, 5)]
    assert table.threshold.tolist() == expected
    assert len(table) == 19
    assert table.loc[table.threshold == 0.15, "flagged_orders"].item() == 1
    assert table.loc[table.threshold == 0.20, "flagged_orders"].item() == 0


@pytest.mark.parametrize("probability", [[np.nan], [-0.1], [1.1]])
def test_invalid_probabilities_are_rejected(probability):
    """Invalid OOF values cannot silently choose a policy."""
    with pytest.raises(ValueError, match="finite values between 0 and 1"):
        evaluation.choose_threshold([1], probability)


def test_mismatched_cost_inputs_are_rejected():
    """Avoid broadcasting a wrong-length decision vector into a cost."""
    with pytest.raises(ValueError, match="same length"):
        evaluation.order_cost([1, 0], [True])


@pytest.fixture
def synthetic_training():
    """Invent sixty rows with sufficient examples of each class for five folds."""
    row = np.arange(60)
    X = pd.DataFrame(
        {
            "distance_km": row / 5 + 0.5,
            "items": row % 8 + 1,
            "hour": row % 13 + 10,
            "weather": np.array(["clear", "rain", "snow"])[row % 3],
            "day_type": np.where(row % 2, "weekend", "weekday"),
        }
    )
    X.loc[::7, "distance_km"] = np.nan
    return X, pd.Series((row % 4 == 0).astype(int), name="late")


def test_oof_uses_pipeline_correct_model_and_fold_local_fits(
    synthetic_training, monkeypatch
):
    """Audit real cross_val_predict and independently reconstruct class-1 OOF."""
    X, y = synthetic_training
    real_predict = evaluation.cross_val_predict
    original_fit = SimpleImputer.fit
    fitted_rows = []
    captured = []

    def audit_fit(imputer, rows, target=None):
        fitted = original_fit(imputer, rows, target)
        fitted_rows.append((rows.copy(), fitted.statistics_.copy()))
        return fitted

    def audit_predict(pipe, features, target, **kwargs):
        assert features is X and target is y
        assert isinstance(pipe, Pipeline)
        model = pipe.named_steps["model"]
        assert model.C == 0.1 and model.class_weight is None
        assert model.solver == "lbfgs" and model.max_iter == 1000
        assert model.random_state == SEED
        assert kwargs["method"] == "predict_proba"
        assert isinstance(kwargs["cv"], StratifiedKFold)
        assert kwargs["cv"].n_splits == 5 and kwargs["cv"].shuffle
        assert kwargs["cv"].random_state == SEED
        probabilities = real_predict(pipe, features, target, **kwargs)
        captured.append((pipe, kwargs["cv"], probabilities))
        return probabilities

    monkeypatch.setattr(SimpleImputer, "fit", audit_fit)
    monkeypatch.setattr(evaluation, "cross_val_predict", audit_predict)
    probability = evaluation.training_oof_probabilities(X, y)
    pipe, cv, matrix = captured[0]
    assert len(fitted_rows) == 5
    for (train, _), (rows, medians) in zip(cv.split(X, y), fitted_rows, strict=True):
        pd.testing.assert_frame_equal(rows, X.iloc[train][NUMERIC])
        np.testing.assert_allclose(medians, rows.median())
    assert probability.shape == (len(X),)
    assert np.isfinite(probability).all()
    assert ((probability >= 0) & (probability <= 1)).all()
    np.testing.assert_allclose(probability, matrix[:, 1])
    np.testing.assert_allclose(matrix.sum(axis=1), 1)
    expected = np.empty(len(X))
    for train, valid in cv.split(X, y):
        fitted = clone(pipe).fit(X.iloc[train], y.iloc[train])
        expected[valid] = fitted.predict_proba(X.iloc[valid])[:, 1]
    np.testing.assert_allclose(probability, expected)


def test_same_seed_reproduces_oof_and_threshold(synthetic_training):
    """Same data and seed reproduce probabilities and all policy counts/costs."""
    X, y = synthetic_training
    first = evaluation.training_oof_probabilities(X, y)
    second = evaluation.training_oof_probabilities(X, y)
    np.testing.assert_array_equal(first, second)
    threshold_a, table_a = evaluation.choose_threshold(y, first)
    threshold_b, table_b = evaluation.choose_threshold(y, second)
    assert threshold_a == threshold_b
    pd.testing.assert_frame_equal(table_a, table_b)


def test_entry_point_never_passes_held_out_data_to_selection(monkeypatch, tmp_path):
    """Only training identities reach OOF and threshold selection; CSV is exact."""
    raw, training_X, training_y, held_out = object(), object(), object(), object()
    monkeypatch.setattr(evaluation, "make_orders", lambda: raw)

    def split(received):
        assert received is raw
        return training_X, held_out, training_y, held_out

    probability = np.array([0.1, 0.9])
    expected_table = pd.DataFrame({"threshold": [0.25], "total_cost": [8]})

    def oof(features, target):
        assert features is training_X and target is training_y
        return probability

    def select(target, received_probability):
        assert target is training_y and received_probability is probability
        return 0.25, expected_table

    monkeypatch.setattr(evaluation, "split_orders", split)
    monkeypatch.setattr(evaluation, "training_oof_probabilities", oof)
    monkeypatch.setattr(evaluation, "choose_threshold", select)
    path = tmp_path / "reports" / "threshold_results.csv"
    chosen, table, returned_probability = evaluation.run_threshold_analysis(path)
    assert chosen == 0.25 and returned_probability is probability
    pd.testing.assert_frame_equal(table, expected_table)
    pd.testing.assert_frame_equal(pd.read_csv(path), expected_table)
