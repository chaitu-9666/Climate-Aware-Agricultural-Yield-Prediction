"""
FastAPI service for health checks, metadata, and yield prediction.

Designed for container deployment behind an ALB on AWS; state is read-only
from the local model artifact path (swap for S3 in production).
"""

from __future__ import annotations

from io import BytesIO
from typing import Any

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from backend import config
from backend.predict import predict_from_bundle

app = FastAPI(
    title="Agri Yield Pro API",
    version="1.0.0",
    description="Climate-soil aware crop yield inference service.",
)


class PredictRequest(BaseModel):
    """Row-oriented prediction payload."""

    rows: list[dict[str, Any]] = Field(
        ...,
        description="List of records sharing the same schema as training features.",
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/config")
def get_config() -> dict[str, Any]:
    return {
        "target": config.TARGET_COL,
        "default_numeric": config.DEFAULT_NUMERIC_FEATURES,
        "default_categorical": config.DEFAULT_CATEGORICAL_FEATURES,
        "model_path": str(config.MODEL_DIR / config.ARTIFACT_NAME),
    }


@app.post("/predict")
def predict(req: PredictRequest) -> dict[str, Any]:
    if not req.rows:
        raise HTTPException(status_code=400, detail="rows must be non-empty")
    df = pd.DataFrame(req.rows)
    try:
        preds = predict_from_bundle(df)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    return {"predictions": preds.tolist()}


@app.post("/predict_csv")
async def predict_csv(file: UploadFile = File(...)) -> dict[str, Any]:
    content = await file.read()
    try:
        df = pd.read_csv(BytesIO(content))
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=400, detail=f"Invalid CSV: {exc}") from exc
    try:
        preds = predict_from_bundle(df)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    out = df.copy()
    out["predicted_yield_kg_ha"] = preds
    return {"rows": out.to_dict(orient="records")}
