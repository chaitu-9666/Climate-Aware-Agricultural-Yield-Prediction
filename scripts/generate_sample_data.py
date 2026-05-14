"""
Generate a synthetic yet structurally realistic crop yield dataset.

The schema mirrors public agronomy tabular releases (e.g. district-year crop
statistics merged with reanalysis climate and soil gridded products). Replace
this file with curated open data for publication-grade studies.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

CROPS = ["rice", "wheat", "maize", "cotton", "soybean", "sugarcane"]
SEASONS = ["kharif", "rabi", "zaid"]


def generate(n_rows: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    crop = rng.choice(CROPS, size=n_rows)
    season = rng.choice(SEASONS, size=n_rows)
    region_id = rng.integers(1, 80, size=n_rows)
    year = rng.integers(2005, 2024, size=n_rows)

    rainfall_mm = rng.normal(950, 220, size=n_rows).clip(150, 2200)
    temp_mean_c = rng.normal(26, 3.5, size=n_rows).clip(12, 40)
    humidity_pct = rng.normal(62, 12, size=n_rows).clip(20, 95)
    soil_ph = rng.normal(6.6, 0.55, size=n_rows).clip(4.5, 8.5)
    soil_oc = rng.lognormal(mean=np.log(0.55), sigma=0.35, size=n_rows).clip(0.1, 2.5)
    nitrogen_kg_ha = rng.normal(95, 25, size=n_rows).clip(0, 220)
    fertilizer_kg_ha = rng.normal(140, 45, size=n_rows).clip(0, 400)
    irrigation_score = rng.uniform(0, 1, size=n_rows)

    # Crop-specific water/temperature preferences (toy but interpretable)
    water_need = np.where(np.isin(crop, ["rice", "sugarcane"]), 1.15, 1.0)
    water_need = np.where(crop == "wheat", 0.85, water_need)
    temp_stress = np.abs(temp_mean_c - np.where(crop == "wheat", 22.0, 27.0))

    base = (
        1850.0
        + 0.55 * (rainfall_mm / water_need)
        - 18.0 * temp_stress
        + 6.5 * (humidity_pct / 100.0)
        + 120.0 * (soil_ph - 6.5).clip(-1.5, 1.5)
        + 260.0 * soil_oc
        + 3.8 * nitrogen_kg_ha
        + 1.1 * fertilizer_kg_ha
        + 420.0 * irrigation_score
    )

    season_k = np.select(
        [season == "kharif", season == "rabi"],
        [90.0, -40.0],
        default=20.0,
    )
    crop_k = np.select(
        [
            crop == "rice",
            crop == "wheat",
            crop == "maize",
            crop == "cotton",
            crop == "soybean",
            crop == "sugarcane",
        ],
        [220.0, 140.0, 110.0, 80.0, 130.0, 150.0],
        default=0.0,
    )

    noise = rng.normal(0, 160, size=n_rows)
    lag = rng.normal(0, 120, size=n_rows)
    historical_yield_lag1 = np.clip(base + lag, 200, 9000)

    yield_kg_ha = np.clip(base + season_k + crop_k + noise, 200, 12000)

    return pd.DataFrame(
        {
            "region_id": region_id,
            "year": year,
            "crop_type": crop,
            "season": season,
            "rainfall_mm": rainfall_mm,
            "temp_mean_c": temp_mean_c,
            "humidity_pct": humidity_pct,
            "soil_ph": soil_ph,
            "soil_organic_carbon_pct": soil_oc,
            "nitrogen_kg_ha": nitrogen_kg_ha,
            "fertilizer_kg_ha": fertilizer_kg_ha,
            "irrigation_score": irrigation_score,
            "historical_yield_lag1": historical_yield_lag1,
            "yield_kg_ha": yield_kg_ha,
        }
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--rows", type=int, default=4000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "raw" / "sample_crop_yields.csv",
    )
    args = p.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df = generate(args.rows, args.seed)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} rows to {args.out}")


if __name__ == "__main__":
    main()
