"""Regression metrics for model comparison (RMSE, MAE, R²)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """
    Compute RMSE, MAE, and R² for regression models.

    Parameters
    ----------
    y_true:
        Ground-truth target vector.
    y_pred:
        Model predictions of the same shape.

    Returns
    -------
    dict
        Keys: ``rmse``, ``mae``, ``r2``.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"rmse": rmse, "mae": mae, "r2": r2}


def metrics_table(results: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """Render nested metric dicts as a tidy dataframe for Streamlit/Plotly."""
    rows = []
    for model_name, payload in results.items():
        m = payload.get("metrics", {})
        rows.append(
            {
                "model": model_name,
                "rmse": m.get("rmse"),
                "mae": m.get("mae"),
                "r2": m.get("r2"),
            }
        )
    return pd.DataFrame(rows)
