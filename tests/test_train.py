"""Audit orchestration order with synthetic training data and opaque test sentinels."""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from latedeliveries import train
from latedeliveries.pipeline import CATEGORICAL, NUMERIC


@pytest.fixture
def mock_workflow(monkeypatch, synthetic_orders):
    """Replace every real data/evaluation call so no lab test data can be touched."""
    raw, y = synthetic_orders
    raw = raw.assign(customer_rating=3.0)
    X = raw[NUMERIC + CATEGORICAL]
    held_X, held_y = object(), object()
    events = []
    evidence = pd.DataFrame({"synthetic_evidence": [1]})
    state = SimpleNamespace(
        best_params=dict(train.APPROVED_BEST_PARAMS), threshold=0.25
    )
    metrics = {"synthetic_test": True, "threshold": 0.25}
    monkeypatch.setattr(train, "make_orders", lambda: raw)
    monkeypatch.setattr(train, "split_orders", lambda received: (X, held_X, y, held_y))

    def check_training(features, target, event):
        assert features is X and target is y
        events.append(event)

    def compare(features, target):
        check_training(features, target, "compare")
        return evidence, pd.DataFrame()

    def tune(features, target):
        check_training(features, target, "tune")
        return SimpleNamespace(best_params_=state.best_params), evidence

    probability = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

    def oof(features, target):
        check_training(features, target, "oof")
        return probability

    def choose(target, received):
        assert target is y and received is probability
        events.append("threshold")
        return state.threshold, evidence

    def leakage(features, target):
        assert features.index.equals(X.index) and target is y
        pd.testing.assert_series_equal(features.customer_rating, raw.customer_rating)
        events.append("leakage")
        return evidence, pd.DataFrame()

    class FinalPipeline:
        def fit(self, features, target):
            check_training(features, target, "fit")
            return self

    pipe = FinalPipeline()

    def evaluate(received, features, target):
        assert events == ["compare", "tune", "oof", "threshold", "leakage", "fit"]
        assert received is pipe and features is held_X and target is held_y
        events.append("evaluate")
        return metrics

    def save(received, result, output_dir):
        assert received is pipe and result is metrics
        assert events[-1] == "evaluate"
        events.append("save")

    for name, value in [
        ("compare_models", compare),
        ("tune_model", tune),
        ("training_oof_probabilities", oof),
        ("choose_threshold", choose),
        ("compare_leakage", leakage),
        ("build_final_pipeline", lambda: pipe),
        ("evaluate_final", evaluate),
        ("save_final_artifacts", save),
    ]:
        monkeypatch.setattr(train, name, value)
    return events, state, metrics


def test_only_final_step_receives_test_objects_and_rebuilds_reports(
    mock_workflow, tmp_path
):
    """All selection is training-only and frozen before the sole evaluation call."""
    events, _, metrics = mock_workflow
    assert train.run_training(tmp_path) is metrics
    assert events == [
        "compare",
        "tune",
        "oof",
        "threshold",
        "leakage",
        "fit",
        "evaluate",
        "save",
    ]
    for name in [
        "cv_results.csv",
        "tuning_results.csv",
        "threshold_results.csv",
        "leakage_results.csv",
    ]:
        pd.testing.assert_frame_equal(
            pd.read_csv(tmp_path / "reports" / name),
            pd.DataFrame({"synthetic_evidence": [1]}),
        )


@pytest.mark.parametrize("mismatch", ["parameters", "threshold"])
def test_changed_selection_aborts_before_test_use(mock_workflow, tmp_path, mismatch):
    """Fresh-run disagreement must stop, not silently adopt another final policy."""
    events, state, _ = mock_workflow
    if mismatch == "parameters":
        state.best_params = {"model__C": 1.0, "model__class_weight": None}
    else:
        state.threshold = 0.30
    with pytest.raises(RuntimeError, match="stop before test evaluation"):
        train.run_training(tmp_path)
    assert "fit" not in events and "evaluate" not in events and "save" not in events
    assert not (tmp_path / "models").exists()


def test_main_delegates_without_extra_predictions(monkeypatch, capsys):
    """The command wrapper delegates once; this test replaces the full real run."""
    calls = []
    monkeypatch.setattr(
        train, "run_training", lambda: calls.append("run") or {"synthetic": True}
    )
    train.main()
    assert calls == ["run"]
    assert json.loads(capsys.readouterr().out) == {"synthetic": True}
