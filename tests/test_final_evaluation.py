"""Check fixed-policy final metrics on hand-calculated synthetic outcomes."""

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from latedeliveries.evaluation import evaluate_probabilities
from latedeliveries.train import (
    APPROVED_THRESHOLD,
    build_final_pipeline,
    evaluate_final,
)


@pytest.fixture
def hand_example():
    """At 0.25: TN=1, FP=2, FN=1, TP=2; seven of nine positive/negative pairs win."""
    return [0, 0, 0, 1, 1, 1], [0.10, 0.25, 0.70, 0.20, 0.80, 0.90]


def test_approved_final_recipe_is_unfitted_and_unchanged():
    """Final model uses only the previously approved settings and feature recipe."""
    pipe = build_final_pipeline()
    model = pipe.named_steps["model"]
    assert isinstance(model, LogisticRegression)
    assert model.C == 0.1 and model.class_weight is None
    assert model.solver == "lbfgs" and model.max_iter == 1000
    assert model.random_state == 7
    assert APPROVED_THRESHOLD == 0.25
    assert not hasattr(pipe.named_steps["prep"], "transformers_")


def test_hand_calculated_metrics_and_all_policy_costs(hand_example):
    """F1=4/7, probability AUC=7/9; model costs 4*4+15=31, not the cheapest policy."""
    actual, probability = hand_example
    metrics = evaluate_probabilities(actual, probability, 0.25)
    assert metrics["confusion_matrix"] == [[1, 2], [1, 2]]
    assert metrics["precision"] == pytest.approx(1 / 2)
    assert metrics["recall"] == pytest.approx(2 / 3)
    assert metrics["f1"] == pytest.approx(4 / 7)
    assert metrics["roc_auc"] == pytest.approx(7 / 9)
    assert metrics["roc_auc"] != pytest.approx(0.5)  # AUC of these binary labels.
    assert metrics["policy_costs"] == {
        "flag_nobody": 45,
        "flag_everybody": 24,
        "model": 31,
    }
    assert metrics["flagged_orders"] == 4 and metrics["missed_late_orders"] == 1
    assert metrics["threshold"] == 0.25 and metrics["n_orders"] == 6


@pytest.mark.parametrize("classes", [[0, 1], [1, 0]])
def test_single_probability_call_uses_class_one_and_fixed_threshold(
    hand_example, classes
):
    """Class order is respected and evaluation never calls fit/predict or reselects."""
    actual, probability = hand_example
    sentinel = object()

    class Predictor:
        classes_ = np.array(classes)
        calls = 0

        def predict_proba(self, received):
            assert received is sentinel
            self.calls += 1
            p = np.array(probability)
            return np.column_stack([p if label == 1 else 1 - p for label in classes])

    pipe = Predictor()
    metrics = evaluate_final(pipe, sentinel, actual)
    assert pipe.calls == 1
    assert metrics == evaluate_probabilities(actual, probability, 0.25)


@pytest.mark.parametrize("probability", [[np.nan, 0.8], [-0.1, 0.8], [0.1, 1.2]])
def test_invalid_probabilities_cannot_create_final_results(probability):
    """Reject nonfinite and out-of-range scores instead of writing misleading JSON."""
    with pytest.raises(ValueError, match="finite values between 0 and 1"):
        evaluate_probabilities([0, 1], probability, 0.25)
