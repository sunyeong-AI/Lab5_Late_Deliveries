"""Compute voucher costs and propose a threshold from training OOF predictions."""

from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from latedeliveries.data import make_orders
from latedeliveries.pipeline import SEED, build_pipeline, split_orders

VOUCHER_COST = 4
MISSED_LATE_COST = 15


def _binary_vector(values: ArrayLike, name: str) -> np.ndarray:
    """Require a nonempty binary vector to prevent silent cost miscounts."""
    vector = np.asarray(values)
    if vector.ndim != 1 or not vector.size or not np.isin(vector, [0, 1]).all():
        raise ValueError(f"{name} must be a nonempty one-dimensional binary vector")
    return vector.astype(bool)


def order_cost(y_true: ArrayLike, flagged: ArrayLike) -> int:
    """Charge $4 per flagged order plus $15 per unflagged actual late order."""
    actual = _binary_vector(y_true, "y_true")
    decisions = _binary_vector(flagged, "flagged")
    if actual.shape != decisions.shape:
        raise ValueError("y_true and flagged must have the same length")
    return int(
        VOUCHER_COST * decisions.sum() + MISSED_LATE_COST * (actual & ~decisions).sum()
    )


def choose_threshold(
    y_true: ArrayLike,
    probability: ArrayLike,
) -> tuple[float, pd.DataFrame]:
    """Return the lowest minimum-cost grid threshold and all 19 policy costs.

    Thresholds are 0.05..0.95 by 0.05. Flag p >= threshold, including equality.
    The ascending grid and first minimum implement the preapproved tie rule.
    Returned threshold is a proposal for review, not a final policy approval.
    """
    actual = _binary_vector(y_true, "y_true")
    probability = np.asarray(probability, dtype=float)
    if probability.shape != actual.shape:
        raise ValueError("y_true and probability must have the same length and shape")
    if (
        not np.isfinite(probability).all()
        or ((probability < 0) | (probability > 1)).any()
    ):
        raise ValueError("probability must contain finite values between 0 and 1")
    rows = []
    for step in range(1, 20):
        threshold = step / 20
        flagged = probability >= threshold
        rows.append(
            {
                "threshold": threshold,
                "flagged_orders": int(flagged.sum()),
                "missed_late_orders": int((actual & ~flagged).sum()),
                "total_cost": order_cost(actual, flagged),
            }
        )
    table = pd.DataFrame(rows)
    best_index = table.total_cost.idxmin()
    return float(table.loc[best_index, "threshold"]), table


def training_oof_probabilities(X_train: pd.DataFrame, y_train: pd.Series) -> np.ndarray:
    """Predict class-1 probabilities once per training row using fold-local fits."""
    if not np.array_equal(np.unique(y_train), [0, 1]):
        raise ValueError("y_train must contain both binary classes 0 and 1")
    model = LogisticRegression(
        C=0.1,
        class_weight=None,
        solver="lbfgs",
        max_iter=1000,
        random_state=SEED,
    )
    probability = cross_val_predict(
        build_pipeline(model),
        X_train,
        y_train,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED),
        method="predict_proba",
        n_jobs=1,
    )
    return probability[:, 1]


def run_threshold_analysis(
    output_path: Path = Path("reports/threshold_results.csv"),
) -> tuple[float, pd.DataFrame, np.ndarray]:
    """Use only the training partition, save its cost curve, and return evidence."""
    X_train, _, y_train, _ = split_orders(make_orders())
    probability = training_oof_probabilities(X_train, y_train)
    threshold, table = choose_threshold(y_train, probability)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_path, index=False)
    return threshold, table, probability


def evaluate_probabilities(
    y_true: ArrayLike,
    probability: ArrayLike,
    threshold: float,
) -> dict:
    """Compute fixed-threshold metrics and policy costs from one probability vector.

    No estimator is fitted, threshold searched or policy selected here. AUC uses
    the original probabilities; the confusion matrix has actual/predicted order
    [0, 1], i.e. [[TN, FP], [FN, TP]]. Both actual classes are needed for AUC.
    """
    actual = _binary_vector(y_true, "y_true")
    probability = np.asarray(probability, dtype=float)
    if probability.shape != actual.shape:
        raise ValueError("y_true and probability must have the same length and shape")
    if (
        not np.isfinite(probability).all()
        or ((probability < 0) | (probability > 1)).any()
    ):
        raise ValueError("probability must contain finite values between 0 and 1")
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be finite and between 0 and 1")
    if np.unique(actual).size != 2:
        raise ValueError("y_true must contain both classes for ROC-AUC")
    flagged = probability >= threshold
    return {
        "threshold": float(threshold),
        "n_orders": len(actual),
        "confusion_matrix": confusion_matrix(actual, flagged, labels=[0, 1]).tolist(),
        "precision": float(precision_score(actual, flagged, zero_division=0)),
        "recall": float(recall_score(actual, flagged, zero_division=0)),
        "f1": float(f1_score(actual, flagged, zero_division=0)),
        "roc_auc": float(roc_auc_score(actual, probability)),
        "flagged_orders": int(flagged.sum()),
        "missed_late_orders": int((actual & ~flagged).sum()),
        "policy_costs": {
            "flag_nobody": order_cost(actual, np.zeros(len(actual), dtype=bool)),
            "flag_everybody": order_cost(actual, np.ones(len(actual), dtype=bool)),
            "model": order_cost(actual, flagged),
        },
    }


if __name__ == "__main__":
    threshold, table, _ = run_threshold_analysis()
    print(table.to_string(index=False))
    print("Proposed threshold (requires approval):", threshold)
