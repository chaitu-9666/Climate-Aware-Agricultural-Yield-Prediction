"""
Data preprocessing for tabular climate/soil/yield datasets.

Handles missing values, scaling, and column-wise transforms suitable for
sklearn pipelines and downstream ML models.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_numeric_preprocessor(columns: Sequence[str]) -> Pipeline:
    """
    Build a sklearn Pipeline for numeric columns: median imputation + scaling.

    Parameters
    ----------
    columns:
        Names of numeric columns this branch will operate on (for documentation;
        the caller attaches this via ColumnTransformer).

    Returns
    -------
    sklearn.pipeline.Pipeline
        Imputer + scaler suitable for tree and linear models when combined with
        one-hot encoded categoricals.
    """
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )


def build_categorical_preprocessor() -> Pipeline:
    """Impute missing categories then one-hot encode."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )


def build_column_transformer(
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
) -> ColumnTransformer:
    """
    Compose preprocessing for mixed numeric / categorical agronomy tables.

    Parameters
    ----------
    numeric_features:
        Continuous covariates (e.g. rainfall, temperature, soil metrics).
    categorical_features:
        Discrete fields (e.g. crop type, season).

    Returns
    -------
    ColumnTransformer
        Ready to be placed as the first stage of an sklearn estimator Pipeline.
    """
    transformers = []
    if numeric_features:
        transformers.append(
            ("num", build_numeric_preprocessor(numeric_features), list(numeric_features))
        )
    if categorical_features:
        transformers.append(
            ("cat", build_categorical_preprocessor(), list(categorical_features))
        )
    return ColumnTransformer(transformers=transformers)


def coerce_types(
    df: pd.DataFrame,
    numeric_features: Iterable[str],
    categorical_features: Iterable[str],
) -> pd.DataFrame:
    """
    Best-effort casting for uploaded CSVs with inconsistent dtypes.

    Numeric columns are coerced with errors coerced to NaN (then imputed later).
    Categorical columns are cast to string to stabilize encoding.
    """
    out = df.copy()
    for col in numeric_features:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    for col in categorical_features:
        if col in out.columns:
            out[col] = out[col].astype("string").fillna("unknown")
    return out


def validate_required_columns(df: pd.DataFrame, required: Sequence[str]) -> None:
    """Raise ValueError if any required column is absent."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
