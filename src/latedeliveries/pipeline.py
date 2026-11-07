"""Define approved order-time features and a reproducible held-out split."""

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 7
NUMERIC = ["distance_km", "items", "hour"]
CATEGORICAL = ["weather", "day_type"]


def split_orders(
    orders: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Return X_train, X_test, y_train, y_test without learning from the data.

    Only approved order-time attributes enter X. Original row indices are kept
    so callers can verify disjoint partitions and target alignment. The held-out
    quarter is reserved for the single final evaluation after all choices freeze.
    """
    X = orders.loc[:, NUMERIC + CATEGORICAL]
    y = orders["late"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=SEED
    )
    return X_train, X_test, y_train, y_test


def build_pipeline(model: BaseEstimator) -> Pipeline:
    """Wrap an estimator with unfitted preprocessing for approved raw columns.

    Imputation, scaling and encoding learn only when the returned pipeline is
    fitted, so callers can fit the complete recipe inside each training fold.
    Extra raw columns, including prep_minutes, are dropped rather than imputed.
    """
    numeric = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    preprocess = ColumnTransformer(
        [
            ("num", numeric, NUMERIC),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL,
            ),
        ],
        remainder="drop",
    )
    return Pipeline([("prep", preprocess), ("model", model)])
