"""
Climate risk analytics: drought heuristics, rainfall deficiency, yield impact.

These functions implement transparent, audit-friendly rules that complement
ML predictions. They are suitable for dashboards and reporting layers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ClimateRiskReport:
    """Structured summary for a single site-year or aggregate panel."""

    drought_risk_score: float
    rainfall_deficit_pct: float
    estimated_yield_impact_pct: float
    notes: str


def _safe_pct(num: float, denom: float) -> float:
    if denom == 0 or np.isnan(denom):
        return 0.0
    return float(100.0 * (1.0 - num / denom))


def drought_risk_score(
    rainfall_mm: float,
    humidity_pct: float,
    temp_mean_c: float,
    long_term_median_rain: float,
) -> float:
    """
    Map key hydro-climate indicators to a 0-1 drought risk score.

    The score increases with rainfall shortfall vs. climatology, low humidity,
    and high temperature (simple stress composite).
    """
    rain_ratio = rainfall_mm / (long_term_median_rain + 1e-6)
    rain_stress = np.clip(1.0 - rain_ratio, 0.0, 1.0)
    humidity_stress = np.clip((50.0 - humidity_pct) / 50.0, 0.0, 1.0)
    temp_stress = np.clip((temp_mean_c - 22.0) / 15.0, 0.0, 1.0)
    score = 0.5 * rain_stress + 0.25 * humidity_stress + 0.25 * temp_stress
    return float(np.clip(score, 0.0, 1.0))


def rainfall_deficiency_pct(actual_mm: float, expected_mm: float) -> float:
    """Percent shortfall relative to expected (LTA or seasonal norm)."""
    if expected_mm <= 0:
        return 0.0
    return float(max(0.0, 100.0 * (expected_mm - actual_mm) / expected_mm))


def climate_yield_impact_pct(drought_score: float, rainfall_deficit_pct: float) -> float:
    """
    Heuristic mapping from climate stress to potential yield impact.

    This is a *scenario* estimate for communication, not a crop simulation model.
    """
    # Weighted blend, capped for UI stability
    raw = 0.55 * drought_score * 35.0 + 0.45 * np.clip(rainfall_deficit_pct, 0.0, 60.0) * 0.35
    return float(np.clip(raw, 0.0, 45.0))


def summarize_row_risk(
    row: pd.Series,
    long_term_median_rain: float,
    expected_seasonal_rain: float,
) -> ClimateRiskReport:
    """Build a ``ClimateRiskReport`` for one dataframe row."""
    rain = float(row.get("rainfall_mm", np.nan))
    hum = float(row.get("humidity_pct", np.nan))
    temp = float(row.get("temp_mean_c", np.nan))
    dscore = drought_risk_score(rain, hum, temp, long_term_median_rain)
    rdef = rainfall_deficiency_pct(rain, expected_seasonal_rain)
    impact = climate_yield_impact_pct(dscore, rdef)
    notes = (
        "Heuristic risk index from rainfall vs. climatology, humidity, and temperature. "
        "Calibrate thresholds with regional agronomy for operational use."
    )
    return ClimateRiskReport(
        drought_risk_score=dscore,
        rainfall_deficit_pct=rdef,
        estimated_yield_impact_pct=impact,
        notes=notes,
    )


def panel_risk_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vectorized climate risk table for dashboard analytics.

    Expects columns ``rainfall_mm``, ``humidity_pct``, ``temp_mean_c``.
    Uses dataset median rainfall as a pragmatic climatology proxy when
    long-term normals are not supplied.
    """
    if df.empty:
        return pd.DataFrame()
    lta = float(df["rainfall_mm"].median()) if "rainfall_mm" in df.columns else 800.0
    expected = lta
    rows = []
    for _, r in df.iterrows():
        rep = summarize_row_risk(r, lta, expected)
        rows.append(
            {
                "drought_risk_score": rep.drought_risk_score,
                "rainfall_deficit_pct": rep.rainfall_deficit_pct,
                "climate_yield_impact_pct": rep.estimated_yield_impact_pct,
            }
        )
    return pd.concat([df.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
