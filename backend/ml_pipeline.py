"""
Training and inference pipelines for crop yield regression.

Each model is a self-contained sklearn ``Pipeline`` (preprocessing + estimator)
so artifacts are reproducible and deployment-friendly.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from backend import config
from utils.evaluation import regression_metrics
from utils.feature_engineering import add_derived_features
from utils.preprocessing import build_column_transformer, coerce_types


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    """
    Engineer features and return the design matrix with resolved column lists.

    Derived columns are appended when base inputs exist; numeric feature list
    is expanded accordingly for the preprocessor.
    """
    out = df.copy()
    for col in config.DEFAULT_NUMERIC_FEATURES:
        if col not in out.columns:
            out[col] = np.nan
    for col in config.DEFAULT_CATEGORICAL_FEATURES:
        if col not in out.columns:
            out[col] = "unknown"
    out = add_derived_features(
        coerce_types(out, config.DEFAULT_NUMERIC_FEATURES, config.DEFAULT_CATEGORICAL_FEATURES)
    )
    numeric = list(config.DEFAULT_NUMERIC_FEATURES)
    for col in config.DERIVED_FEATURES:
        if col in out.columns:
            numeric.append(col)
    categorical = [c for c in config.DEFAULT_CATEGORICAL_FEATURES if c in out.columns]
    return out, numeric, categorical


def build_estimator_pipelines(
    numeric_features: list[str],
    categorical_features: list[str],
    random_state: int | None = None,
) -> dict[str, Pipeline]:
    """
    Construct three baseline regressors with identical preprocessing branches.

    Returns
    -------
    dict
        Keys: ``linear_regression``, ``random_forest``, ``xgboost``.
    """
    rs = random_state if random_state is not None else config.RANDOM_STATE
    pre = build_column_transformer(numeric_features, categorical_features)

    linear = Pipeline(
        steps=[
            ("preprocessor", pre),
            ("estimator", LinearRegression()),
        ]
    )
    rf = Pipeline(
        steps=[
            ("preprocessor", clone(pre)),
            (
                "estimator",
                RandomForestRegressor(
                    n_estimators=400,
                    max_depth=None,
                    min_samples_leaf=2,
                    random_state=rs,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    xgb = Pipeline(
        steps=[
            ("preprocessor", clone(pre)),
            (
                "estimator",
                XGBRegressor(
                    n_estimators=600,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    objective="reg:squarederror",
                    random_state=rs,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    return {
        "linear_regression": linear,
        "random_forest": rf,
        "xgboost": xgb,
    }


def train_models(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int | None = None,
) -> dict[str, Any]:
    """
    Fit all models, evaluate on a holdout split, and return payloads + test frame.

    Parameters
    ----------
    df:
        Training table containing ``config.TARGET_COL`` and feature columns.
    test_size:
        Fraction used for validation metrics.
    random_state:
        RNG seed for the shuffle split.

    Returns
    -------
    dict
        ``models``, ``metrics``, ``comparison``, ``X_test``, ``y_test``,
        ``feature_meta`` (numeric/categorical names used).
    """
    from sklearn.model_selection import train_test_split

    rs = random_state if random_state is not None else config.RANDOM_STATE
    if config.TARGET_COL not in df.columns:
        raise ValueError(f"Training data must include target column '{config.TARGET_COL}'.")

    X_raw = df.drop(columns=[config.TARGET_COL], errors="ignore")
    y = df[config.TARGET_COL].astype(float).values

    X_eng, numeric, categorical = prepare_features(X_raw)
    X_train, X_test, y_train, y_test = train_test_split(
        X_eng, y, test_size=test_size, random_state=rs, shuffle=True
    )

    pipelines = build_estimator_pipelines(numeric, categorical, rs)
    results: dict[str, Any] = {}
    fitted: dict[str, Pipeline] = {}

    for name, pipe in pipelines.items():
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        metrics = regression_metrics(y_test, preds)
        fitted[name] = pipe
        results[name] = {"metrics": metrics, "pipeline": pipe}

    comparison_rows = []
    for name, payload in results.items():
        m = payload["metrics"]
        comparison_rows.append({"model": name, **m})
    comparison = pd.DataFrame(comparison_rows).sort_values("rmse")

    return {
        "models": fitted,
        "raw_results": results,
        "comparison": comparison,
        "X_test": X_test,
        "y_test": y_test,
        "feature_meta": {"numeric": numeric, "categorical": categorical},
    }


def predict_yield(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Run inference on raw features (engineering applied internally)."""
    X_eng, _, _ = prepare_features(X)
    return np.asarray(model.predict(X_eng), dtype=float)
