"""Demonstrate post-outcome feature leakage without changing approved models."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from latedeliveries.comparison import candidate_models
from latedeliveries.data import make_orders
from latedeliveries.pipeline import NUMERIC, SEED, build_pipeline, split_orders


def build_leakage_pipeline(model: BaseEstimator) -> Pipeline:
    """Add only customer_rating to a fresh experimental numeric transformer.

    This intentionally invalid order-time feature is confined to this experiment.
    Its imputer/scaler are still fitted inside CV folds; approved lists and the
    production builder remain unchanged.
    """
    pipe = build_pipeline(model)
    prep = pipe.named_steps["prep"]
    prep.set_params(
        transformers=[
            (
                name,
                transformer,
                NUMERIC + ["customer_rating"] if name == "num" else columns,
            )
            for name, transformer, columns in prep.transformers
        ]
    )
    return pipe


def compare_leakage(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare Stage 4 candidates with/without rating using identical train folds."""
    folds = list(
        StratifiedKFold(5, shuffle=True, random_state=SEED).split(X_train, y_train)
    )
    summaries, fold_rows = [], []
    for name, model in candidate_models().items():
        summary = {
            "model": name,
            "training_rows": len(X_train),
            "cv_folds": 5,
            "seed": SEED,
            "std_ddof": 0,
        }
        for variant, builder in [
            ("clean", build_pipeline),
            ("leakage", build_leakage_pipeline),
        ]:
            scores = cross_validate(
                builder(model),
                X_train,
                y_train,
                cv=folds,
                scoring="roc_auc",
                error_score="raise",
                n_jobs=1,
            )["test_score"]
            summary[f"{variant}_roc_auc_mean"] = float(np.mean(scores))
            summary[f"{variant}_roc_auc_std"] = float(np.std(scores, ddof=0))
            fold_rows.extend(
                {
                    "model": name,
                    "variant": variant,
                    "fold": i + 1,
                    "roc_auc": float(score),
                }
                for i, score in enumerate(scores)
            )
        summary["roc_auc_difference"] = (
            summary["leakage_roc_auc_mean"] - summary["clean_roc_auc_mean"]
        )
        summaries.append(summary)
    return pd.DataFrame(summaries), pd.DataFrame(fold_rows)


def run_leakage_experiment(
    output_path: Path = Path("reports/leakage_results.csv"),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach only training-indexed ratings and save a separate demonstration."""
    orders = make_orders()
    X_train, _, y_train, _ = split_orders(orders)
    experimental = X_train.assign(
        customer_rating=orders.loc[X_train.index, "customer_rating"]
    )
    summary, folds = compare_leakage(experimental, y_train)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)
    return summary, folds


if __name__ == "__main__":
    summary, _ = run_leakage_experiment()
    print(summary.to_string(index=False))
