"""Verify real full-pipeline persistence using synthetic rows and tmp_path only."""

import json

import joblib
import numpy as np
import sklearn
from sklearn.pipeline import Pipeline

from latedeliveries.evaluation import evaluate_probabilities
from latedeliveries.pipeline import CATEGORICAL, NUMERIC, SEED
from latedeliveries.train import build_final_pipeline, save_final_artifacts


def test_saved_pipeline_and_json_round_trip(synthetic_orders, tmp_path):
    """Raw-row probabilities, labels and policy decisions survive loading."""
    X, y = synthetic_orders
    pipe = build_final_pipeline().fit(X, y)
    raw = X.copy()
    raw.loc[0, "items"] = np.nan
    raw.loc[0, "weather"] = "hail"
    raw.loc[0, "prep_minutes"] = np.nan
    before = pipe.predict_proba(raw)
    metrics = evaluate_probabilities(y, before[:, 1], 0.25)
    save_final_artifacts(pipe, metrics, tmp_path)
    loaded = joblib.load(tmp_path / "models" / "late_orders.joblib")
    assert isinstance(loaded, Pipeline)
    assert "prep" in loaded.named_steps and "model" in loaded.named_steps
    np.testing.assert_array_equal(loaded.predict_proba(raw), before)
    np.testing.assert_array_equal(loaded.predict(raw), pipe.predict(raw))
    np.testing.assert_array_equal(
        loaded.predict_proba(raw)[:, 1] >= 0.25, before[:, 1] >= 0.25
    )
    info = json.loads((tmp_path / "models" / "model_info.json").read_text())
    report = json.loads((tmp_path / "reports" / "metrics.json").read_text())
    assert report == metrics == info["test_results"]
    assert info["scikit_learn_version"] == sklearn.__version__
    assert info["seed"] == SEED and info["threshold"] == 0.25
    assert info["numeric_features"] == NUMERIC
    assert info["categorical_features"] == CATEGORICAL
    assert info["best_parameters"] == {"model__C": 0.1, "model__class_weight": None}
    assert info["model_parameters"]["solver"] == "lbfgs"
    assert info["model_parameters"]["max_iter"] == 1000
    assert set(report) == {
        "threshold",
        "n_orders",
        "confusion_matrix",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "flagged_orders",
        "missed_late_orders",
        "policy_costs",
    }
