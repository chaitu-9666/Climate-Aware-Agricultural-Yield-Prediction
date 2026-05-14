"""
Feature engineering helpers for climate-soil-yield modeling.

Derives agronomically interpretable signals used by both ML models and
rule-based climate risk heuristics.
"""

from __future__ import annotations

import pandas as pd


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived features in-place on a copy of the input frame.

    Features
    --------
    - ``growing_degree_days_proxy``: simplified thermal accumulation proxy.
    - ``rainfall_temp_ratio``: moisture availability vs. evaporative demand proxy.
    - ``soil_quality_index``: weighted composite of pH proximity to 6.5 and organic carbon.

    Notes
    -----
    These proxies are not a substitute for crop-specific phenology models; they
    provide compact, interpretable inputs for tabular ML baselines.
    """
    out = df.copy()
    if {"temp_mean_c", "rainfall_mm"}.issubset(out.columns):
        # Simple GDD-style proxy (only positive degrees above 10°C)
        base = 10.0
        out["growing_degree_days_proxy"] = (out["temp_mean_c"] - base).clip(lower=0) * 30.0
        denom = (out["temp_mean_c"].abs() + 1e-6)
        out["rainfall_temp_ratio"] = out["rainfall_mm"] / denom
    if {"soil_ph", "soil_organic_carbon_pct"}.issubset(out.columns):
        ph_score = 1.0 - (out["soil_ph"] - 6.5).abs() / 2.5
        ph_score = ph_score.clip(0, 1)
        oc_norm = (out["soil_organic_carbon_pct"] / (out["soil_organic_carbon_pct"].max() + 1e-6)).clip(
            0, 1
        )
        out["soil_quality_index"] = 0.55 * ph_score + 0.45 * oc_norm
    return out


def add_lag_features(
    df: pd.DataFrame,
    group_cols: list[str],
    value_col: str,
    lags: list[int],
    sort_col: str | None = None,
) -> pd.DataFrame:
    """
    Optional grouped lag features when panel / time-ordered data exist.

    If ``sort_col`` is provided, rows are sorted within each group before lagging.
    """
    out = df.copy()
    if value_col not in out.columns or not group_cols:
        return out
    keys = [c for c in group_cols if c in out.columns]
    if not keys:
        return out
    g = out.sort_values(sort_col) if sort_col and sort_col in out.columns else out
    g = g.groupby(keys, sort=False)[value_col]
    for lag in lags:
        out[f"{value_col}_lag{lag}"] = g.shift(lag).reindex(out.index)
    return out
