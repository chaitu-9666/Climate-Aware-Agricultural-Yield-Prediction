"""Load persisted bundles and run batch inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from backend import config
from backend.ml_pipeline import predict_yield


def load_bundle(path: Path | None = None) -> dict[str, Any]:
    """Load the joblib artifact written by ``backend.train``."""
    p = path or (config.MODEL_DIR / config.ARTIFACT_NAME)
    if not p.exists():
        raise FileNotFoundError(
            f"No model bundle at {p}. Train first: python -m backend.train --data <csv>"
        )
    return joblib.load(p)


def predict_from_bundle(X: pd.DataFrame, bundle: dict[str, Any] | None = None) -> np.ndarray:
    """Predict using the best estimator from a bundle."""
    b = bundle or load_bundle()
    model = b.get("best_model")
    if model is None:
        raise KeyError("Bundle missing 'best_model'")
    return predict_yield(model, X)
