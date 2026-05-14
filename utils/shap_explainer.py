"""
SHAP-based explainability for tree ensembles and linear baselines.

Provides fallbacks when SHAP is unavailable or sample size is too small.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def compute_shap_values(
    model: Any,
    X_sample: pd.DataFrame,
    max_samples: int = 400,
) -> tuple[np.ndarray | None, list[str] | None, str | None]:
    """
    Compute SHAP values for supported sklearn / XGBoost models.

    Parameters
    ----------
    model:
        Fitted sklearn Pipeline whose final step exposes ``predict`` and is
        compatible with ``shap`` explainers (TreeExplainer / LinearExplainer).
    X_sample:
        Feature matrix in *raw* pre-transform space (same columns as training
        before engineering). Target column should be omitted.
    max_samples:
        Row cap for performance on large uploads.

    Returns
    -------
    shap_values, feature_names, error_message
        ``shap_values`` is a 2D numpy array aligned with ``feature_names`` when
        successful; otherwise ``None`` with a human-readable error string.
    """
    try:
        import shap
    except ImportError as e:  # pragma: no cover
        return None, None, f"SHAP not installed: {e}"

    if X_sample.empty:
        return None, None, "Empty sample for SHAP."

    try:
        from backend.ml_pipeline import prepare_features
    except Exception as e:  # pragma: no cover
        return None, None, f"Could not import prepare_features: {e}"

    Xs = X_sample.copy()
    if len(Xs) > max_samples:
        Xs = Xs.sample(max_samples, random_state=42)

    try:
        X_eng, _, _ = prepare_features(Xs)
    except Exception as exc:
        return None, None, f"Feature preparation failed: {exc}"

    try:
        if hasattr(model, "named_steps") and "preprocessor" in model.named_steps:
            pre = model.named_steps["preprocessor"]
            est = model.named_steps["estimator"]
            X_trans = pre.transform(X_eng)
            fnames = _transformed_feature_names(pre, X_eng.columns.tolist())
            if _is_tree_estimator(est):
                explainer = shap.TreeExplainer(est)
                sv = explainer.shap_values(X_trans)
            else:
                explainer = shap.LinearExplainer(est, X_trans)
                sv = explainer.shap_values(X_trans)
            if isinstance(sv, list):  # multiclass edge case
                sv = sv[0]
            return np.asarray(sv), fnames, None
        return None, None, "Unsupported model type for SHAP helper."
    except Exception as exc:  # pragma: no cover - shap backend specific
        return None, None, f"SHAP computation failed: {exc}"


def mean_abs_shap(sv: np.ndarray, feature_names: list[str]) -> pd.DataFrame:
    """Aggregate mean(|SHAP|) per feature for bar charts."""
    ma = np.mean(np.abs(sv), axis=0)
    return pd.DataFrame({"feature": feature_names, "mean_abs_shap": ma}).sort_values(
        "mean_abs_shap", ascending=False
    )


def _is_tree_estimator(est: Any) -> bool:
    try:
        from sklearn.ensemble import RandomForestRegressor
        from xgboost import XGBRegressor

        return isinstance(est, (RandomForestRegressor, XGBRegressor))
    except Exception:
        return False


def _transformed_feature_names(pre: Any, original: list[str]) -> list[str]:
    """Best-effort feature names after ColumnTransformer."""
    try:
        names = pre.get_feature_names_out()
        return [str(n) for n in names]
    except Exception:
        return original
