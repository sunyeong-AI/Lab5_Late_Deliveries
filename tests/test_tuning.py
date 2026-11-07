"""Protect approved tuning settings, fold isolation and score calculations."""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import ParameterGrid, StratifiedKFold
from sklearn.pipeline import Pipeline

from latedeliveries import tuning
from latedeliveries.pipeline import NUMERIC, SEED


def test_search_uses_approved_grid_pipeline_and_scoring():
    """Exactly two parameters and eight combinations use the unfitted recipe."""
    search = tuning.build_search()
    assert isinstance(search.estimator, Pipeline)
    assert not hasattr(search.estimator.named_steps["prep"], "transformers_")
    assert search.param_grid == {
        "model__C": [0.01, 0.1, 1.0, 10.0],
        "model__class_weight": [None, "balanced"],
    }
    assert len(ParameterGrid(search.param_grid)) == 8
    assert search.scoring == "roc_auc"
    assert search.refit is True
    model = search.estimator.named_steps["model"]
    assert model.max_iter == 1000
    assert model.solver == "lbfgs"
    assert model.random_state == SEED
    assert isinstance(search.cv, StratifiedKFold)
    assert search.cv.n_splits == 5
    assert search.cv.shuffle is True
    assert search.cv.random_state == SEED


@pytest.fixture
def synthetic_training():
    """Invent sixty training rows, not held-out delivery data."""
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


@pytest.fixture
def audited_tuning(synthetic_training, monkeypatch):
    """Capture actual fold-local imputer fits and the final training refit."""
    X, y = synthetic_training
    calls = []
    original = SimpleImputer.fit

    def audit(imputer, rows, target=None):
        fitted = original(imputer, rows, target)
        calls.append((rows.copy(), fitted.statistics_.copy()))
        return fitted

    monkeypatch.setattr(SimpleImputer, "fit", audit)
    search, table = tuning.tune_model(X, y)
    return X, y, search, table, calls


def test_training_fold_isolation_and_refit(audited_tuning):
    """All 40 fold fits learn from 48 rows; only final refit sees all 60."""
    X, y, search, _, calls = audited_tuning
    expected = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y))
    actual = list(search.cv.split(X, y))
    assert len(calls) == 41
    for (train, valid), (other_train, other_valid) in zip(
        expected, actual, strict=True
    ):
        np.testing.assert_array_equal(train, other_train)
        np.testing.assert_array_equal(valid, other_valid)
        assert set(train).isdisjoint(valid)
    for index, (rows, statistics) in enumerate(calls[:-1]):
        train, _ = expected[index % 5]
        pd.testing.assert_frame_equal(rows, X.iloc[train][NUMERIC])
        np.testing.assert_allclose(statistics, rows.median())
    pd.testing.assert_frame_equal(calls[-1][0], X[NUMERIC])
    np.testing.assert_allclose(calls[-1][1], X[NUMERIC].median())


@pytest.mark.parametrize("combination", range(8))
def test_each_combination_scores_match_manual_cv(audited_tuning, combination):
    """Recompute all five AUCs and population mean/SD outside GridSearchCV."""
    X, y, search, table, _ = audited_tuning
    params = search.cv_results_["params"][combination]
    scores = []
    for train, valid in search.cv.split(X, y):
        pipeline = clone(search.estimator).set_params(**params)
        pipeline.fit(X.iloc[train], y.iloc[train])
        scores.append(
            roc_auc_score(y.iloc[valid], pipeline.predict_proba(X.iloc[valid])[:, 1])
        )
    row = table.iloc[combination]
    np.testing.assert_allclose([row[f"split{i}_roc_auc"] for i in range(5)], scores)
    assert row.roc_auc_mean == pytest.approx(np.mean(scores))
    assert row.roc_auc_std == pytest.approx(np.std(scores, ddof=0))


def test_best_parameters_and_score_match_results(audited_tuning):
    """The refitted best pipeline corresponds to the highest CV mean AUC."""
    _, _, search, table, _ = audited_tuning
    best_index = int(table.roc_auc_mean.to_numpy().argmax())
    assert search.best_index_ == best_index
    assert search.best_score_ == pytest.approx(table.roc_auc_mean.max())
    assert table.iloc[best_index]["rank"] == 1
    assert search.best_params_ == search.cv_results_["params"][best_index]
    assert (
        search.best_estimator_.named_steps["model"].C == search.best_params_["model__C"]
    )
    assert (
        search.best_estimator_.named_steps["model"].class_weight
        == search.best_params_["model__class_weight"]
    )


def test_run_tuning_routes_training_only_and_saves_csv(monkeypatch, tmp_path):
    """No held-out object reaches tuning; the complete table is saved."""
    raw, training_X, training_y, held_out = object(), object(), object(), object()
    monkeypatch.setattr(tuning, "make_orders", lambda: raw)

    def split(received):
        assert received is raw
        return training_X, held_out, training_y, held_out

    table = pd.DataFrame({"C": [0.01], "roc_auc_mean": [0.75]})
    sentinel_search = object()

    def tune(features, target):
        assert features is training_X and target is training_y
        return sentinel_search, table

    monkeypatch.setattr(tuning, "split_orders", split)
    monkeypatch.setattr(tuning, "tune_model", tune)
    path = tmp_path / "reports" / "tuning_results.csv"
    search, results = tuning.run_tuning(path)
    assert search is sentinel_search
    pd.testing.assert_frame_equal(results, table)
    pd.testing.assert_frame_equal(pd.read_csv(path), table)
