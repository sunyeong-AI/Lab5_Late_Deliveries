"""Check preprocessing behavior with hand-verifiable synthetic training data."""

import numpy as np
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from latedeliveries.pipeline import CATEGORICAL, NUMERIC, SEED, build_pipeline


@pytest.fixture
def fitted_pipeline(synthetic_orders):
    """Fit one pipeline on the six invented rows, never on held-out lab data."""
    X, y = synthetic_orders
    return build_pipeline(LogisticRegression()).fit(X, y)


@pytest.mark.parametrize(
    "model",
    [
        LogisticRegression(),
        DecisionTreeClassifier(random_state=SEED),
        LinearRegression(),
    ],
    ids=["logistic", "tree", "regression"],
)
def test_pipeline_accepts_different_estimators(model, synthetic_orders):
    """The builder wraps the supplied estimator without selecting its family."""
    pipe = build_pipeline(model)
    assert isinstance(pipe, Pipeline)
    assert isinstance(pipe.named_steps["prep"], ColumnTransformer)
    assert pipe.named_steps["model"] is model
    X, y = synthetic_orders
    assert pipe.fit(X, y).predict(X).shape == (len(X),)


def test_builder_does_not_fit_or_share_preprocessors():
    """Construction learns nothing; separate calls have separate transformers."""
    first = build_pipeline(LogisticRegression())
    second = build_pipeline(LogisticRegression())
    prep = first.named_steps["prep"]
    assert not hasattr(prep, "transformers_")
    numeric = prep.transformers[0][1]
    assert not hasattr(numeric.named_steps["imputer"], "statistics_")
    assert not hasattr(numeric.named_steps["scaler"], "mean_")
    assert prep is not second.named_steps["prep"]


def test_numeric_medians_are_learned_from_fit_rows(fitted_pipeline, synthetic_orders):
    """Distance's observed median is 7; item/hour medians are 3.5 and 15."""
    X, _ = synthetic_orders
    numeric = fitted_pipeline.named_steps["prep"].named_transformers_["num"]
    imputer = numeric.named_steps["imputer"]
    assert imputer.strategy == "median"
    np.testing.assert_allclose(imputer.statistics_, [7.0, 3.5, 15.0])
    np.testing.assert_allclose(imputer.transform(X[NUMERIC])[:, 0], [1, 3, 7, 7, 9, 11])


def test_numeric_scaling_uses_imputed_training_values(
    fitted_pipeline, synthetic_orders
):
    """Scaling matches hand-specified imputed rows and population variance."""
    X, _ = synthetic_orders
    numeric = fitted_pipeline.named_steps["prep"].named_transformers_["num"]
    imputed = np.array(
        [[1, 1, 10], [3, 2, 12], [7, 3, 14], [7, 4, 16], [9, 5, 18], [11, 6, 20]],
        dtype=float,
    )
    expected = (imputed - imputed.mean(axis=0)) / imputed.std(axis=0)
    np.testing.assert_allclose(numeric.transform(X[NUMERIC]), expected)
    np.testing.assert_allclose(
        numeric.named_steps["scaler"].mean_, imputed.mean(axis=0)
    )
    np.testing.assert_allclose(
        numeric.transform(X[NUMERIC]).mean(axis=0), 0, atol=1e-12
    )
    np.testing.assert_allclose(numeric.transform(X[NUMERIC]).std(axis=0), 1)


def test_categories_are_one_hot_encoded(fitted_pipeline, synthetic_orders):
    """Known weather and day categories occupy the expected indicator columns."""
    X, _ = synthetic_orders
    encoder = fitted_pipeline.named_steps["prep"].named_transformers_["cat"]
    assert encoder.handle_unknown == "ignore"
    assert [list(c) for c in encoder.categories_] == [
        ["clear", "rain", "snow"],
        ["weekday", "weekend"],
    ]
    np.testing.assert_array_equal(
        encoder.transform(X[CATEGORICAL]),
        [
            [1, 0, 0, 1, 0],
            [0, 1, 0, 0, 1],
            [0, 0, 1, 1, 0],
            [1, 0, 0, 0, 1],
            [0, 1, 0, 1, 0],
            [0, 0, 1, 0, 1],
        ],
    )
    names = fitted_pipeline.named_steps["prep"].get_feature_names_out()
    assert len(names) == 8
    assert not any("prep_minutes" in name for name in names)


def test_unseen_weather_is_zero_encoded(fitted_pipeline, synthetic_orders):
    """Unknown weather is ignored while the known day indicator is preserved."""
    X, _ = synthetic_orders
    new = X.iloc[[0]].copy()
    new["weather"] = "hail"
    encoder = fitted_pipeline.named_steps["prep"].named_transformers_["cat"]
    np.testing.assert_array_equal(
        encoder.transform(new[CATEGORICAL]), [[0, 0, 0, 1, 0]]
    )
    assert fitted_pipeline.predict(new).shape == (1,)


def test_missing_prep_and_unseen_weather_in_raw_order(
    fitted_pipeline, synthetic_orders
):
    """Missing excluded prep is ignored, independently of numeric imputation."""
    X, _ = synthetic_orders
    raw = X.iloc[[0]].copy()
    raw["prep_minutes"] = np.nan
    raw["weather"] = "hail"
    raw["order_id"] = 999999
    raw["customer_rating"] = 1.0
    raw["vehicle"] = "car"
    raw["courier_months"] = 0
    assert fitted_pipeline.predict(raw).shape == (1,)
    np.testing.assert_allclose(
        fitted_pipeline.predict_proba(raw),
        fitted_pipeline.predict_proba(raw[NUMERIC + CATEGORICAL]),
    )
    changed = raw.copy()
    changed["prep_minutes"] = 9999.0
    np.testing.assert_allclose(
        fitted_pipeline.predict_proba(raw), fitted_pipeline.predict_proba(changed)
    )


@pytest.mark.parametrize(
    "feature,median", [("distance_km", 7), ("items", 3.5), ("hour", 15)]
)
def test_missing_used_numeric_feature_is_imputed(
    fitted_pipeline, synthetic_orders, feature, median
):
    """Missing retained inputs transform exactly like their fitted medians."""
    X, _ = synthetic_orders
    missing = X.iloc[[0]].copy()
    missing[feature] = np.nan
    expected = missing.copy()
    expected[feature] = median
    prep = fitted_pipeline.named_steps["prep"]
    assert np.isfinite(prep.transform(missing)).all()
    np.testing.assert_allclose(prep.transform(missing), prep.transform(expected))
    np.testing.assert_allclose(
        fitted_pipeline.predict_proba(missing), fitted_pipeline.predict_proba(expected)
    )


def test_probabilities_are_valid(fitted_pipeline, synthetic_orders):
    """Probabilities are finite, within [0,1], and sum to one for each row."""
    X, _ = synthetic_orders
    probability = fitted_pipeline.predict_proba(X)
    assert probability.shape == (6, 2)
    assert np.isfinite(probability).all()
    assert ((probability >= 0) & (probability <= 1)).all()
    np.testing.assert_allclose(probability.sum(axis=1), 1)


def test_prediction_does_not_relearn_preprocessing(fitted_pipeline, synthetic_orders):
    """Extreme new input cannot change medians, scaling or learned categories."""
    X, _ = synthetic_orders
    prep = fitted_pipeline.named_steps["prep"]
    numeric = prep.named_transformers_["num"]
    imputer = numeric.named_steps["imputer"]
    scaler = numeric.named_steps["scaler"]
    encoder = prep.named_transformers_["cat"]
    medians, means, variances = (
        imputer.statistics_.copy(),
        scaler.mean_.copy(),
        scaler.var_.copy(),
    )
    categories = [c.copy() for c in encoder.categories_]
    new = X.iloc[[0]].copy()
    new["distance_km"] = 1e9
    new["weather"] = "hail"
    fitted_pipeline.predict_proba(new)
    np.testing.assert_array_equal(imputer.statistics_, medians)
    np.testing.assert_array_equal(scaler.mean_, means)
    np.testing.assert_array_equal(scaler.var_, variances)
    for before, after in zip(categories, encoder.categories_, strict=True):
        np.testing.assert_array_equal(before, after)
