"""
Training entrypoint: load CSV, train models, persist bundle + metrics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from backend import config
from backend.ml_pipeline import train_models


def main() -> None:
    parser = argparse.ArgumentParser(description="Train yield prediction models.")
    parser.add_argument(
        "--data",
        type=Path,
        default=config.DATA_DIR / "sample_crop_yields.csv",
        help="Path to training CSV.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=config.MODEL_DIR / config.ARTIFACT_NAME,
        help="Output joblib path for the model bundle.",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    bundle = train_models(df, test_size=args.test_size)

    best_name = bundle["comparison"].iloc[0]["model"]
    best_model = bundle["models"][best_name]

    payload = {
        "best_model_name": best_name,
        "comparison": bundle["comparison"].to_dict(orient="records"),
        "feature_meta": bundle["feature_meta"],
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "models": bundle["models"],
            "best_model_name": best_name,
            "best_model": best_model,
            "metrics": {k: v["metrics"] for k, v in bundle["raw_results"].items()},
            "feature_meta": bundle["feature_meta"],
        },
        args.out,
    )

    metrics_path = args.out.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved bundle to {args.out}")
    print(f"Saved metrics summary to {metrics_path}")
    print(bundle["comparison"])


if __name__ == "__main__":
    main()
