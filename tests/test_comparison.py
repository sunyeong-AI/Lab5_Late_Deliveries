"""Verify training-only CV, fold-local preparation and reproducible summaries."""

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from latedeliveries import comparison
from latedeliveries.pipeline import NUMERIC, SEED


@pytest.fixture
def audited_comparison(monkeypatch):
    """Run real CV on invented training rows and capture fitted fold pipelines."""
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
    y = pd.Series((row % 4 == 0).astype(int), name="late")
    original = comparison.cross_validate
    calls = []

    def audit(pipe, features, target, **kwargs):
        assert features is X and target is y
        result = original(pipe, features, target, **kwargs, return_estimator=True)
        calls.append((pipe, kwargs, result))
        return result

    monkeypatch.setattr(comparison, "cross_validate", audit)
    summary, folds = comparison.compare_models(X, y)
    return X, y, summary, folds, calls


def test_approved_candidate_roster():
    """Compare exactly the four approved families, including balanced tree."""
    candidates = comparison.candidate_models()
    assert list(candidates) == [
        "dummy",
        "logistic_regression",
        "random_forest",
        "balanced_decision_tree",
    ]
    assert candidates["dummy"].strategy == "most_frequent"
    assert candidates["balanced_decision_tree"].class_weight == "balanced"
    for name in ["logistic_regression", "random_forest", "balanced_decision_tree"]:
        assert candidates[name].random_state == SEED


def test_all_models_share_five_stratified_training_folds(audited_comparison):
    """Each supplied training row validates once and all models share indices."""
    X, y, summary, scores, calls = audited_comparison
    required = {"model", "training_rows", "cv_folds", "seed", "std_ddof"} | {
        f"{metric}_{stat}" for metric in comparison.METRICS for stat in ["mean", "std"]
    }
    assert set(summary.columns) == required
    assert summary.shape == (4, 15)
    assert scores.shape == (20, 7)
    assert (summary.training_rows == len(X)).all()
    assert (summary.cv_folds == 5).all()
    assert (summary.std_ddof == 0).all()
    assert len(calls) == 4
    reference = calls[0][1]["cv"]
    assert len(reference) == 5
    seen = []
    for train, valid in reference:
        assert set(train).isdisjoint(valid)
        assert set(train) | set(valid) == set(range(len(X)))
        seen.extend(valid)
        for indices in [train, valid]:
            assert abs(y.iloc[indices].sum() - len(indices) * y.mean()) <= 1
    assert sorted(seen) == list(range(len(X)))
    for _, kwargs, _ in calls:
        assert kwargs["cv"] is reference
        assert set(kwargs["scoring"]) == set(comparison.METRICS)
    expected = list(
        comparison.StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(
            X, y
        )
    )
    for (train, valid), (expected_train, expected_valid) in zip(
        reference, expected, strict=True
    ):
        np.testing.assert_array_equal(train, expected_train)
        np.testing.assert_array_equal(valid, expected_valid)


def test_entire_pipeline_is_fitted_inside_each_fold(audited_comparison):
    """Imputer/scaler statistics come from that fold's fit rows, not all rows."""
    X, _, _, _, calls = audited_comparison
    for pipe, kwargs, result in calls:
        assert isinstance(pipe, Pipeline)
        assert isinstance(pipe.named_steps["prep"], ColumnTransformer)
        assert not hasattr(pipe.named_steps["prep"], "transformers_")
        for (train, _), fitted in zip(kwargs["cv"], result["estimator"], strict=True):
            numeric = fitted.named_steps["prep"].named_transformers_["num"]
            rows = X.iloc[train][NUMERIC]
            medians = rows.median()
            np.testing.assert_allclose(
                numeric.named_steps["imputer"].statistics_, medians
            )
            np.testing.assert_allclose(
                numeric.named_steps["scaler"].mean_, rows.fillna(medians).mean()
            )


@pytest.mark.parametrize("metric", ["accuracy", "precision", "recall", "f1", "roc_auc"])
def test_fold_metrics_and_mean_std_are_correct(audited_comparison, metric):
    """Recompute validation scores and aggregate independently from the report."""
    X, y, summary, folds, calls = audited_comparison
    scorers = {
        "accuracy": accuracy_score,
        "precision": lambda y, p: precision_score(y, p, zero_division=0),
        "recall": lambda y, p: recall_score(y, p, zero_division=0),
        "f1": lambda y, p: f1_score(y, p, zero_division=0),
        "roc_auc": roc_auc_score,
    }
    for name, (_, kwargs, result) in zip(summary["model"], calls, strict=True):
        values = []
        for (_, valid), fitted in zip(kwargs["cv"], result["estimator"], strict=True):
            prediction = (
                fitted.predict_proba(X.iloc[valid])[:, 1]
                if metric == "roc_auc"
                else fitted.predict(X.iloc[valid])
            )
            values.append(scorers[metric](y.iloc[valid], prediction))
        np.testing.assert_allclose(folds.loc[folds.model == name, metric], values)
        row = summary.loc[summary.model == name].iloc[0]
        assert row[f"{metric}_mean"] == pytest.approx(np.mean(values))
        assert row[f"{metric}_std"] == pytest.approx(np.std(values, ddof=0))


def test_entry_point_passes_only_training_partition_and_writes_csv(
    monkeypatch, tmp_path
):
    """The orchestration passes no held-out object to CV and preserves all fields."""
    raw, train_X, train_y, forbidden_test = object(), object(), object(), object()
    monkeypatch.setattr(comparison, "make_orders", lambda: raw)

    def split(received):
        assert received is raw
        return train_X, forbidden_test, train_y, forbidden_test

    expected = pd.DataFrame(
        [
            {
                "model": "dummy",
                "training_rows": 60,
                "cv_folds": 5,
                "seed": SEED,
                "std_ddof": 0,
                **{
                    f"{m}_{stat}": 0.5
                    for m in comparison.METRICS
                    for stat in ["mean", "std"]
                },
            }
        ]
    )

    def compare(features, target):
        assert features is train_X and target is train_y
        return expected, pd.DataFrame()

    monkeypatch.setattr(comparison, "split_orders", split)
    monkeypatch.setattr(comparison, "compare_models", compare)
    path = tmp_path / "reports" / "cv_results.csv"
    summary, _ = comparison.run_comparison(path)
    pd.testing.assert_frame_equal(summary, expected)
    pd.testing.assert_frame_equal(pd.read_csv(path), expected)
