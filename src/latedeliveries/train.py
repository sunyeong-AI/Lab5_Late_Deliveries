"""Recreate training evidence, evaluate the frozen policy once, and save it.

Running python -m latedeliveries.train performs the final held-out evaluation.
Do not run this command until that evaluation has been explicitly approved.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.pipeline import Pipeline

from latedeliveries.comparison import compare_models
from latedeliveries.data import make_orders
from latedeliveries.evaluation import (
    choose_threshold,
    evaluate_probabilities,
    training_oof_probabilities,
)
from latedeliveries.leakage import compare_leakage
from latedeliveries.pipeline import CATEGORICAL, NUMERIC, SEED, split_orders
from latedeliveries.tuning import build_search, tune_model

APPROVED_BEST_PARAMS = {"model__C": 0.1, "model__class_weight": None}
APPROVED_THRESHOLD = 0.25


def build_final_pipeline() -> Pipeline:
    """Create the approved logistic recipe without fitting or selecting anything."""
    return build_search().estimator.set_params(**APPROVED_BEST_PARAMS)


def evaluate_final(pipe: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Use exactly one held-out predict_proba call with the frozen threshold."""
    positive = np.flatnonzero(np.asarray(pipe.classes_) == 1)
    if len(positive) != 1:
        raise ValueError("The fitted pipeline must contain late class 1")
    probability = pipe.predict_proba(X_test)[:, int(positive[0])]
    return evaluate_probabilities(y_test, probability, APPROVED_THRESHOLD)


def save_final_artifacts(pipe: Pipeline, metrics: dict, output_dir: Path) -> None:
    """Save the whole fitted pipeline and matching JSON results/version metadata."""
    models = output_dir / "models"
    reports = output_dir / "reports"
    models.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    info = {
        "scikit_learn_version": sklearn.__version__,
        "seed": SEED,
        "threshold": APPROVED_THRESHOLD,
        "numeric_features": NUMERIC.copy(),
        "categorical_features": CATEGORICAL.copy(),
        "best_parameters": {
            key: pipe.get_params()[key] for key in APPROVED_BEST_PARAMS
        },
        "model_parameters": pipe.named_steps["model"].get_params(),
        "test_results": metrics,
    }
    metrics_json = json.dumps(metrics, indent=2, allow_nan=False) + "\n"
    info_json = json.dumps(info, indent=2, allow_nan=False) + "\n"
    joblib.dump(pipe, models / "late_orders.joblib")
    (reports / "metrics.json").write_text(metrics_json, encoding="utf-8")
    (models / "model_info.json").write_text(info_json, encoding="utf-8")


def run_training(output_dir: Path = Path(".")) -> dict:
    """Rebuild artifacts with training-only decisions before one final evaluation.

    Reproduced settings/threshold must match prior approval. A mismatch aborts
    before touching held-out predictions; it never silently adopts a new policy.
    Test results cannot feed any model, parameter or threshold decision.
    """
    orders = make_orders()
    X_train, X_test, y_train, y_test = split_orders(orders)
    comparison, _ = compare_models(X_train, y_train)
    search, tuning = tune_model(X_train, y_train)
    if search.best_params_ != APPROVED_BEST_PARAMS:
        raise RuntimeError(
            "Tuning differs from approved parameters; stop before test evaluation"
        )
    probability = training_oof_probabilities(X_train, y_train)
    threshold, threshold_table = choose_threshold(y_train, probability)
    if threshold != APPROVED_THRESHOLD:
        raise RuntimeError(
            "OOF threshold differs from approved policy; stop before test evaluation"
        )
    leakage, _ = compare_leakage(
        X_train.assign(customer_rating=orders.loc[X_train.index, "customer_rating"]),
        y_train,
    )
    reports = output_dir / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    for name, frame in [
        ("cv_results.csv", comparison),
        ("tuning_results.csv", tuning),
        ("threshold_results.csv", threshold_table),
        ("leakage_results.csv", leakage),
    ]:
        frame.to_csv(reports / name, index=False)
    pipe = build_final_pipeline()
    pipe.fit(X_train, y_train)
    metrics = evaluate_final(pipe, X_test, y_test)
    save_final_artifacts(pipe, metrics, output_dir)
    return metrics


def main() -> None:
    """Run the complete approved workflow from the repository root."""
    metrics = run_training()
    print(json.dumps(metrics, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
