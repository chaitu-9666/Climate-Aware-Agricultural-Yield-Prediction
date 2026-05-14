# Raw datasets

## Expected schema

The training pipeline expects a CSV with (at minimum) the following columns:

| Column | Role |
| --- | --- |
| `yield_kg_ha` | Target |
| `rainfall_mm`, `temp_mean_c`, `humidity_pct` | Climate drivers |
| `soil_ph`, `soil_organic_carbon_pct` | Soil indicators |
| `nitrogen_kg_ha`, `fertilizer_kg_ha`, `irrigation_score` | Management |
| `historical_yield_lag1` | Optional lag feature |
| `crop_type`, `season` | Categorical agronomy context |
| `region_id`, `year` | Optional identifiers for dashboards |

## Real-world open data sources

These public resources align with the schema above and can be fused for research-grade work:

- **FAOSTAT** — national and sub-national crop production and yield statistics  
  https://www.fao.org/faostat/

- **ICAR-INDIASTAT / data.gov.in** — district-level Indian crop statistics suitable for merging with IMD rainfall products  
  https://data.gov.in/

- **NASA POWER / CHIRPS** — daily-to-monthly rainfall and temperature for agricultural climate analytics  
  https://power.larc.nasa.gov/ and https://www.chc.ucsb.edu/data/chirps

- **SoilGrids / ISRIC** — global gridded soil property layers (pH, organic carbon)  
  https://www.isric.org/explore/soilgrids

- **Kaggle crop datasets** — convenient baselines (always document provenance and bias)

## Synthetic demo file

Run:

```bash
python scripts/generate_sample_data.py --rows 5000
```

This writes `data/raw/sample_crop_yields.csv` for reproducible local experiments without external downloads.
