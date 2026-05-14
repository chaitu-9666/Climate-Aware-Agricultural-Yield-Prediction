"""
Central configuration for the yield modeling stack.

Paths resolve relative to the project root so the same code runs locally,
inside Docker, and on AWS (override with environment variables).
"""

from __future__ import annotations

import os
from pathlib import Path

# Project root: parent of ``backend/``
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = Path(os.environ.get("AGRI_DATA_DIR", PROJECT_ROOT / "data" / "raw"))
MODEL_DIR = Path(os.environ.get("AGRI_MODEL_DIR", PROJECT_ROOT / "models"))
ARTIFACT_NAME = os.environ.get("AGRI_ARTIFACT_NAME", "yield_model_bundle.joblib")

TARGET_COL = "yield_kg_ha"
ID_COLS: list[str] = ["region_id", "year"]

DEFAULT_NUMERIC_FEATURES: list[str] = [
    "rainfall_mm",
    "temp_mean_c",
    "humidity_pct",
    "soil_ph",
    "soil_organic_carbon_pct",
    "nitrogen_kg_ha",
    "fertilizer_kg_ha",
    "irrigation_score",
    "historical_yield_lag1",
]

DEFAULT_CATEGORICAL_FEATURES: list[str] = ["crop_type", "season"]

DERIVED_FEATURES: list[str] = [
    "growing_degree_days_proxy",
    "rainfall_temp_ratio",
    "soil_quality_index",
]

RANDOM_STATE = int(os.environ.get("AGRI_RANDOM_STATE", "42"))
