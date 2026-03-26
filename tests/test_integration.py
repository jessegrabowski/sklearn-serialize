import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.datasets import make_classification, make_regression
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    PolynomialFeatures,
    RobustScaler,
    StandardScaler,
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from sklearn_serialize.serialize import data_to_json, json_to_data, trust_module

trust_module("tests")


class ClampTransformer(BaseEstimator, TransformerMixin):
    """Clips values to [min_val, max_val], learned from training data."""

    def __init__(self, quantile: float = 0.05):
        self.quantile = quantile

    def fit(self, X, y=None):
        self.min_ = np.quantile(X, self.quantile, axis=0)
        self.max_ = np.quantile(X, 1 - self.quantile, axis=0)
        return self

    def transform(self, X):
        return np.clip(X, self.min_, self.max_)


def roundtrip(estimator):
    return json_to_data(data_to_json(estimator))


def assert_estimator_roundtrip(estimator, X, y=None):
    if y is not None:
        estimator.fit(X, y)
    else:
        estimator.fit(X)

    restored = roundtrip(estimator)

    if hasattr(estimator, "transform"):
        np.testing.assert_allclose(
            restored.transform(X),
            estimator.transform(X),
            rtol=1e-10,
        )

    if hasattr(estimator, "predict"):
        np.testing.assert_allclose(
            restored.predict(X),
            estimator.predict(X),
            rtol=1e-10,
        )

    if hasattr(estimator, "predict_proba"):
        np.testing.assert_allclose(
            restored.predict_proba(X),
            estimator.predict_proba(X),
            rtol=1e-10,
        )

    return restored


@pytest.fixture(scope="module")
def regression_Xy():
    return make_regression(n_samples=100, n_features=8, noise=0.1, random_state=42)


@pytest.fixture(scope="module")
def classification_Xy():
    return make_classification(n_samples=100, n_features=8, n_classes=3, n_informative=4, random_state=42)


@pytest.fixture(scope="module")
def numeric_X():
    X, _ = make_regression(n_samples=100, n_features=8, noise=0.0, random_state=42)
    return X


@pytest.fixture(scope="module")
def categorical_X():
    rng = np.random.default_rng(42)
    return np.column_stack(
        [
            rng.choice(["cat", "dog", "bird"], 100),
            rng.choice(["red", "blue", "green", "yellow"], 100),
        ]
    )


@pytest.fixture(scope="module")
def mixed_df():
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "age": rng.uniform(20, 80, 100),
            "income": rng.uniform(20_000, 120_000, 100),
            "score": rng.uniform(0, 1, 100),
            "category": rng.choice(["A", "B", "C"], 100),
            "region": rng.choice(["north", "south", "east", "west"], 100),
        }
    )


@pytest.mark.parametrize(
    "estimator",
    [
        StandardScaler(),
        StandardScaler(with_mean=False),
        MinMaxScaler(),
        MinMaxScaler(feature_range=(-1, 1)),
        RobustScaler(),
        PolynomialFeatures(degree=2, include_bias=False),
        PCA(n_components=3, random_state=42),
    ],
    ids=[
        "StandardScaler",
        "StandardScaler(no_mean)",
        "MinMaxScaler",
        "MinMaxScaler(-1,1)",
        "RobustScaler",
        "PolynomialFeatures(d=2)",
        "PCA(k=3)",
    ],
)
def test_numeric_transformer(estimator, numeric_X):
    assert_estimator_roundtrip(estimator, numeric_X)


@pytest.mark.parametrize(
    "estimator",
    [
        OneHotEncoder(sparse_output=False),
        OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
        OrdinalEncoder(),
    ],
    ids=[
        "OneHotEncoder",
        "OneHotEncoder(handle_unknown=ignore)",
        "OrdinalEncoder",
    ],
)
def test_categorical_transformer(estimator, categorical_X):
    assert_estimator_roundtrip(estimator, categorical_X)


@pytest.mark.parametrize(
    "estimator",
    [
        LinearRegression(),
        Ridge(alpha=1.0),
        Lasso(alpha=0.01),
        DecisionTreeRegressor(max_depth=3, random_state=42),
        RandomForestRegressor(n_estimators=10, random_state=42),
        GradientBoostingRegressor(n_estimators=10, random_state=42),
    ],
    ids=[
        "LinearRegression",
        "Ridge",
        "Lasso",
        "DecisionTreeRegressor",
        "RandomForestRegressor",
        "GradientBoostingRegressor",
    ],
)
def test_regressor(estimator, regression_Xy):
    X, y = regression_Xy
    assert_estimator_roundtrip(estimator, X, y)


@pytest.mark.parametrize(
    "estimator",
    [
        LogisticRegression(max_iter=500, random_state=42),
        DecisionTreeClassifier(max_depth=3, random_state=42),
        RandomForestClassifier(n_estimators=10, random_state=42),
    ],
    ids=[
        "LogisticRegression",
        "DecisionTreeClassifier",
        "RandomForestClassifier",
    ],
)
def test_classifier(estimator, classification_Xy):
    X, y = classification_Xy
    assert_estimator_roundtrip(estimator, X, y)


@pytest.mark.parametrize(
    "pipeline",
    [
        Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]),
        Pipeline([("poly", PolynomialFeatures(degree=2, include_bias=False)), ("model", Ridge(alpha=1.0))]),
        Pipeline([("pca", PCA(n_components=3, random_state=42)), ("model", Lasso(alpha=0.01))]),
    ],
    ids=[
        "StandardScaler+LinearRegression",
        "PolynomialFeatures+Ridge",
        "PCA+Lasso",
    ],
)
def test_regression_pipeline(pipeline, regression_Xy):
    X, y = regression_Xy
    assert_estimator_roundtrip(pipeline, X, y)


@pytest.mark.parametrize(
    "pipeline",
    [
        Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=500, random_state=42))]),
        Pipeline(
            [
                ("pca", PCA(n_components=4, random_state=42)),
                ("model", LogisticRegression(max_iter=500, random_state=42)),
            ]
        ),
    ],
    ids=[
        "StandardScaler+LogisticRegression",
        "PCA+LogisticRegression",
    ],
)
def test_classification_pipeline(pipeline, classification_Xy):
    X, y = classification_Xy
    assert_estimator_roundtrip(pipeline, X, y)


def test_column_transformer_mixed_types(mixed_df):
    ct = ColumnTransformer(
        [
            ("num", StandardScaler(), ["age", "income", "score"]),
            ("cat", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), ["category", "region"]),
        ]
    )
    assert_estimator_roundtrip(ct, mixed_df)


def test_column_transformer_with_remainder(mixed_df):
    ct = ColumnTransformer(
        [
            ("cat", OneHotEncoder(sparse_output=False), ["category", "region"]),
        ],
        remainder="passthrough",
    )
    assert_estimator_roundtrip(ct, mixed_df)


def test_column_transformer_make_column_selector(mixed_df):
    ct = ColumnTransformer(
        [
            ("num", StandardScaler(), make_column_selector(dtype_include=np.number)),
            (
                "cat",
                OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
                make_column_selector(dtype_include=object),
            ),
        ]
    )
    assert_estimator_roundtrip(ct, mixed_df)


def test_column_transformer_in_pipeline(mixed_df, regression_Xy):
    preprocessor = ColumnTransformer(
        [
            ("num", StandardScaler(), ["age", "income", "score"]),
            ("cat", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), ["category", "region"]),
        ]
    )
    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", Ridge(alpha=1.0)),
        ]
    )
    _, y = regression_Xy
    assert_estimator_roundtrip(pipeline, mixed_df, y)


def test_feature_union_parallel_transforms(numeric_X):
    fu = FeatureUnion(
        [
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=2, random_state=42)),
        ]
    )
    assert_estimator_roundtrip(fu, numeric_X)


def test_custom_estimator_standalone(numeric_X):
    assert_estimator_roundtrip(ClampTransformer(quantile=0.1), numeric_X)


def test_custom_estimator_in_pipeline(numeric_X, regression_Xy):
    _, y = regression_Xy
    pipeline = Pipeline([("clamp", ClampTransformer(quantile=0.1)), ("model", Ridge(alpha=1.0))])
    assert_estimator_roundtrip(pipeline, numeric_X, y)


def test_feature_union_in_pipeline(numeric_X, regression_Xy):
    _, y = regression_Xy
    pipeline = Pipeline(
        [
            (
                "features",
                FeatureUnion(
                    [
                        ("scaled", StandardScaler()),
                        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
                    ]
                ),
            ),
            ("model", Ridge(alpha=1.0)),
        ]
    )
    assert_estimator_roundtrip(pipeline, numeric_X, y)
