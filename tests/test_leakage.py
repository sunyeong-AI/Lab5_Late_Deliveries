"""Keep intentionally leaky experimental preprocessing isolated and train-only."""

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from latedeliveries import leakage
from latedeliveries.pipeline import CATEGORICAL, NUMERIC, SEED, build_pipeline


def test_experimental_builder_adds_only_rating_and_preserves_production_lists():
    """Only the fresh experimental column selector contains customer_rating."""
    numeric_before, categories_before = NUMERIC.copy(), CATEGORICAL.copy()
    experiment = leakage.build_leakage_pipeline(LogisticRegression())
    ordinary = build_pipeline(LogisticRegression())
    assert isinstance(experiment, Pipeline)
    assert experiment.named_steps["prep"].transformers[0][2] == NUMERIC + [
        "customer_rating"
    ]
    assert ordinary.named_steps["prep"].transformers[0][2] == NUMERIC
    assert experiment.named_steps["prep"].transformers[1][2] == CATEGORICAL
    assert NUMERIC == numeric_before and CATEGORICAL == categories_before
    assert not hasattr(experiment.named_steps["prep"], "transformers_")


def test_paired_cv_pipeline_fold_statistics_and_aggregates(monkeypatch):
    """Audit real clean/leaky CV, full pipelines, shared folds and rating medians."""
    row = np.arange(60)
    X = pd.DataFrame(
        {
            "distance_km": row / 5 + 0.5,
            "items": row % 8 + 1,
            "hour": row % 13 + 10,
            "weather": np.array(["clear", "rain", "snow"])[row % 3],
            "day_type": np.where(row % 2, "weekend", "weekday"),
            "customer_rating": 1.0 + row % 5,
        }
    )
    X.loc[::3, "customer_rating"] = np.nan
    y = pd.Series((row % 4 == 0).astype(int))
    original = leakage.cross_validate
    calls = []

    def audit(pipe, features, target, **kwargs):
        assert features is X and target is y
        assert isinstance(pipe, Pipeline)
        result = original(pipe, features, target, **kwargs, return_estimator=True)
        calls.append((pipe, kwargs, result))
        return result

    monkeypatch.setattr(leakage, "cross_validate", audit)
    summary, scores = leakage.compare_leakage(X, y)
    assert len(calls) == 8 and summary.shape == (4, 10) and scores.shape == (40, 4)
    reference = calls[0][1]["cv"]
    assert len(reference) == 5
    expected = list(
        leakage.StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y)
    )
    for (train, valid), (expected_train, expected_valid) in zip(
        reference, expected, strict=True
    ):
        np.testing.assert_array_equal(train, expected_train)
        np.testing.assert_array_equal(valid, expected_valid)
    for index, (pipe, kwargs, result) in enumerate(calls):
        assert kwargs["cv"] is reference and kwargs["scoring"] == "roc_auc"
        columns = NUMERIC if index % 2 == 0 else NUMERIC + ["customer_rating"]
        assert pipe.named_steps["prep"].transformers[0][2] == columns
        for (train, _), fitted in zip(reference, result["estimator"], strict=True):
            numeric = fitted.named_steps["prep"].named_transformers_["num"]
            expected_median = X.iloc[train][columns].median()
            assert numeric.named_steps["imputer"].strategy == "median"
            np.testing.assert_allclose(
                numeric.named_steps["imputer"].statistics_, expected_median
            )
            np.testing.assert_allclose(
                numeric.named_steps["scaler"].mean_,
                X.iloc[train][columns].fillna(expected_median).mean(),
            )
    for report in summary.itertuples():
        for variant in ["clean", "leakage"]:
            values = scores.loc[
                (scores.model == report.model) & (scores.variant == variant), "roc_auc"
            ]
            assert getattr(report, variant + "_roc_auc_mean") == pytest.approx(
                values.mean()
            )
            assert getattr(report, variant + "_roc_auc_std") == pytest.approx(
                values.std(ddof=0)
            )
        assert report.roc_auc_difference == pytest.approx(
            report.leakage_roc_auc_mean - report.clean_roc_auc_mean
        )
    dummy = summary.loc[summary.model == "dummy"].iloc[0]
    assert dummy.roc_auc_difference == 0
    assert isinstance(calls[0][0].named_steps["model"], DummyClassifier)
    assert NUMERIC == ["distance_km", "items", "hour"]


def test_entry_point_attaches_only_training_ratings(monkeypatch, tmp_path):
    """Attach ratings by training indices without accessing held-out rows."""
    raw = pd.DataFrame({"customer_rating": [1.0, 5.0, 3.0]}, index=[12, 34, 56])
    train = pd.DataFrame({"distance_km": [2.0, 4.0]}, index=[56, 12])
    target = pd.Series([1, 0], index=[56, 12])
    held_out = object()
    monkeypatch.setattr(leakage, "make_orders", lambda: raw)

    def split(received):
        assert received is raw
        return train, held_out, target, held_out

    expected = pd.DataFrame({"model": ["dummy"], "roc_auc_difference": [0.0]})

    def compare(features, y):
        assert features.index.tolist() == [56, 12] and y is target
        assert features.columns.tolist() == ["distance_km", "customer_rating"]
        assert features.customer_rating.tolist() == [3.0, 1.0]
        return expected, pd.DataFrame()

    monkeypatch.setattr(leakage, "split_orders", split)
    monkeypatch.setattr(leakage, "compare_leakage", compare)
    path = tmp_path / "reports" / "leakage_results.csv"
    table, _ = leakage.run_leakage_experiment(path)
    pd.testing.assert_frame_equal(table, expected)
    pd.testing.assert_frame_equal(pd.read_csv(path), expected)
