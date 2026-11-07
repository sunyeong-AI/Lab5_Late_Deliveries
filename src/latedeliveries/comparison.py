"""Compare complete pipelines with training-only stratified cross-validation."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.tree import DecisionTreeClassifier

from latedeliveries.data import make_orders
from latedeliveries.pipeline import SEED, build_pipeline, split_orders

METRICS = ("accuracy", "precision", "recall", "f1", "roc_auc")


def candidate_models() -> dict[str, BaseEstimator]:
    """Return the approved untuned candidates without choosing a winner."""
    return {
        "dummy": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=SEED),
        "random_forest": RandomForestClassifier(random_state=SEED),
        "balanced_decision_tree": DecisionTreeClassifier(
            class_weight="balanced", random_state=SEED
        ),
    }


def compare_models(
    X_train: pd.DataFrame, y_train: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return summary and fold scores using only the supplied training rows.

    All candidates share the same five seeded stratified index splits. Each
    cross_validate call clones and fits the entire unfitted pipeline per fold.
    Precision/recall/F1 describe class 1 at each estimator's default rule;
    undefined scores are zero. ROC-AUC uses probability/ranking scores.
    Summary SD is the population SD of the five folds (ddof=0), not a confidence
    interval. This function neither chooses a model nor evaluates a held-out set.
    """
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    folds = list(splitter.split(X_train, y_train))
    scoring = {
        "accuracy": "accuracy",
        "precision": make_scorer(precision_score, zero_division=0),
        "recall": make_scorer(recall_score, zero_division=0),
        "f1": make_scorer(f1_score, zero_division=0),
        "roc_auc": "roc_auc",
    }
    summaries, fold_rows = [], []
    for name, model in candidate_models().items():
        result = cross_validate(
            build_pipeline(model),
            X_train,
            y_train,
            cv=folds,
            scoring=scoring,
            error_score="raise",
            n_jobs=1,
        )
        summary = {
            "model": name,
            "training_rows": len(X_train),
            "cv_folds": len(folds),
            "seed": SEED,
            "std_ddof": 0,
        }
        for metric in METRICS:
            values = result[f"test_{metric}"]
            summary[f"{metric}_mean"] = float(np.mean(values))
            summary[f"{metric}_std"] = float(np.std(values, ddof=0))
        summaries.append(summary)
        for fold in range(len(folds)):
            fold_rows.append(
                {
                    "model": name,
                    "fold": fold + 1,
                    **{
                        metric: float(result[f"test_{metric}"][fold])
                        for metric in METRICS
                    },
                }
            )
    return pd.DataFrame(summaries), pd.DataFrame(fold_rows)


def run_comparison(
    output_path: Path = Path("reports/cv_results.csv"),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate/split orders, compare training rows, and save the summary CSV."""
    X_train, _, y_train, _ = split_orders(make_orders())
    summary, folds = compare_models(X_train, y_train)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)
    return summary, folds


if __name__ == "__main__":
    summary, _ = run_comparison()
    print(summary.to_string(index=False))
