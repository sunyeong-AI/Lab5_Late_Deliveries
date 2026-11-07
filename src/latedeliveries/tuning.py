"""Tune the approved logistic pipeline using training-only ROC-AUC CV."""

from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from latedeliveries.data import make_orders
from latedeliveries.pipeline import SEED, build_pipeline, split_orders


def build_search() -> GridSearchCV:
    """Create the approved eight-combination search without fitting anything."""
    model = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=SEED)
    return GridSearchCV(
        build_pipeline(model),
        param_grid={
            "model__C": [0.01, 0.1, 1.0, 10.0],
            "model__class_weight": [None, "balanced"],
        },
        scoring="roc_auc",
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED),
        refit=True,
        error_score="raise",
        n_jobs=1,
    )


def tune_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[GridSearchCV, pd.DataFrame]:
    """Fit training folds and return the search and its auditable score table.

    GridSearchCV fits full pipeline clones inside each fold and refits its best
    parameters on the supplied training rows. Its best CV score is selection
    evidence, not an unbiased final performance estimate or a business cost.
    """
    search = build_search()
    search.fit(X_train, y_train)
    rows = []
    for index, params in enumerate(search.cv_results_["params"]):
        rows.append(
            {
                "C": params["model__C"],
                "class_weight": "None"
                if params["model__class_weight"] is None
                else params["model__class_weight"],
                "training_rows": len(X_train),
                "cv_folds": 5,
                "seed": SEED,
                "scoring": search.scoring,
                "std_ddof": 0,
                **{
                    f"split{fold}_roc_auc": float(
                        search.cv_results_[f"split{fold}_test_score"][index]
                    )
                    for fold in range(5)
                },
                "roc_auc_mean": float(search.cv_results_["mean_test_score"][index]),
                "roc_auc_std": float(search.cv_results_["std_test_score"][index]),
                "rank": int(search.cv_results_["rank_test_score"][index]),
            }
        )
    return search, pd.DataFrame(rows)


def run_tuning(
    output_path: Path = Path("reports/tuning_results.csv"),
) -> tuple[GridSearchCV, pd.DataFrame]:
    """Tune only the training partition and save all eight CV configurations."""
    X_train, _, y_train, _ = split_orders(make_orders())
    search, results = tune_model(X_train, y_train)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    return search, results


if __name__ == "__main__":
    search, results = run_tuning()
    print(results.to_string(index=False))
    print("Best parameters:", search.best_params_)
    print("Best CV ROC-AUC:", search.best_score_)
