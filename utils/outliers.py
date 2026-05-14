"""
Outlier detection utilities for agronomic tabular data.

Uses robust z-score (MAD-based) and optional IQR flagging suitable for
exploratory workflows and soft filtering (winsorization).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def modified_zscore(series: pd.Series, threshold: float = 3.5) -> pd.Series:
    """
    Return a boolean mask for outliers using the modified Z-score (MAD).

    References
    ----------
    Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
    Handle Outliers", ASQC Quality Press.
    """
    s = series.astype(float)
    med = s.median()
    mad = (s - med).abs().median()
    if mad == 0 or np.isnan(mad):
        return pd.Series(False, index=s.index)
    mz = 0.6745 * (s - med) / mad
    return mz.abs() > threshold


def iqr_mask(series: pd.Series, k: float = 1.5) -> pd.Series:
    """Classic Tukey IQR fence mask."""
    s = series.astype(float)
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    low, high = q1 - k * iqr, q3 + k * iqr
    return (s < low) | (s > high)


def flag_numeric_outliers(
    df: pd.DataFrame,
    numeric_cols: list[str],
    method: str = "modified_zscore",
) -> pd.DataFrame:
    """
    Add ``*_outlier`` boolean columns for each numeric field.

    Parameters
    ----------
    df:
        Input dataframe.
    numeric_cols:
        Columns to scan.
    method:
        ``modified_zscore`` or ``iqr``.
    """
    out = df.copy()
    for col in numeric_cols:
        if col not in out.columns:
            continue
        if method == "modified_zscore":
            mask = modified_zscore(out[col])
        elif method == "iqr":
            mask = iqr_mask(out[col])
        else:
            raise ValueError("method must be 'modified_zscore' or 'iqr'")
        out[f"{col}_outlier"] = mask
    return out


def winsorize_series(series: pd.Series, lower_q: float = 0.01, upper_q: float = 0.99) -> pd.Series:
    """Clip a numeric series to given quantiles (robust tail shrinkage)."""
    s = series.astype(float)
    lo, hi = s.quantile(lower_q), s.quantile(upper_q)
    return s.clip(lo, hi)
