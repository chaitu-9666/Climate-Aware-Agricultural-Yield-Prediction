"""
Agri Yield Pro — interactive research dashboard.

Run from the repository root:

    streamlit run frontend/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend import config  # noqa: E402
from backend.ml_pipeline import predict_yield, train_models  # noqa: E402
from backend.predict import load_bundle  # noqa: E402
from utils.climate_risk import panel_risk_summary  # noqa: E402
from utils.evaluation import regression_metrics  # noqa: E402
from utils.outliers import flag_numeric_outliers  # noqa: E402
from utils.shap_explainer import compute_shap_values, mean_abs_shap  # noqa: E402


st.set_page_config(
    page_title="Agri Yield Pro",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _default_data_path() -> Path:
    return ROOT / "data" / "raw" / "sample_crop_yields.csv"


def sidebar() -> None:
    st.sidebar.title("Agri Yield Pro")
    st.sidebar.markdown(
        "Climate-soil aware **yield prediction** with transparent **risk** "
        "analytics and **SHAP** explanations."
    )
    st.sidebar.markdown("---")
    st.sidebar.caption("Configure paths for AWS by setting `AGRI_MODEL_DIR` and `AGRI_DATA_DIR`.")


def load_csv(uploaded) -> pd.DataFrame | None:
    if uploaded is None:
        return None
    return pd.read_csv(uploaded)


def main() -> None:
    sidebar()
    st.title("Agricultural Yield Prediction & Climate Intelligence")
    st.caption(
        "Modular Python stack: preprocessing → ML baselines → climate risk → explainability. "
        "Designed for reproducible research and portfolio-grade engineering."
    )

    tabs = st.tabs(
        [
            "Data & QC",
            "Train & Evaluate",
            "Climate Risk",
            "Explainability (SHAP)",
            "Inference",
        ]
    )

    with tabs[0]:
        st.subheader("Upload or load tabular agronomy data")
        default_path = _default_data_path()
        use_default = st.checkbox("Load bundled sample CSV", value=default_path.exists())
        uploaded = st.file_uploader("CSV upload", type=["csv"])
        if uploaded is not None:
            df = load_csv(uploaded)
        elif use_default and default_path.exists():
            df = pd.read_csv(default_path)
        else:
            st.info("Upload a CSV or generate `data/raw/sample_crop_yields.csv` via `scripts/generate_sample_data.py`.")
            df = None

        if df is not None:
            st.session_state["df"] = df
            st.dataframe(df.head(50), use_container_width=True)
            st.write(f"Shape: `{df.shape}`")

            numeric_cols = [c for c in config.DEFAULT_NUMERIC_FEATURES if c in df.columns]
            if numeric_cols:
                flags = flag_numeric_outliers(df, numeric_cols, method="modified_zscore")
                out_cols = [c for c in flags.columns if c.endswith("_outlier")]
                rate = flags[out_cols].mean().sort_values(ascending=False) if out_cols else None
                if rate is not None:
                    st.markdown("**Outlier prevalence (modified Z-score)**")
                    st.dataframe(rate.to_frame("fraction_flagged"), use_container_width=True)

    with tabs[1]:
        st.subheader("Train and compare models")
        if "df" not in st.session_state:
            st.warning("Load data in the first tab first.")
        else:
            df = st.session_state["df"]
            if config.TARGET_COL not in df.columns:
                st.error(f"Dataset must include target column `{config.TARGET_COL}`.")
            else:
                test_size = st.slider("Holdout fraction", 0.1, 0.4, 0.2, 0.05)
                if st.button("Train models", type="primary"):
                    with st.spinner("Fitting Linear Regression, Random Forest, and XGBoost…"):
                        bundle = train_models(df, test_size=test_size)
                        st.session_state["train_bundle"] = bundle
                if "train_bundle" in st.session_state:
                    bundle = st.session_state["train_bundle"]
                    comp = bundle["comparison"]
                    st.success("Training complete. Metrics below are RMSE / MAE / R² on holdout data.")
                    c1, c2 = st.columns(2)
                    with c1:
                        fig = px.bar(
                            comp,
                            x="model",
                            y="rmse",
                            color="model",
                            title="RMSE (lower is better)",
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    with c2:
                        fig2 = px.bar(
                            comp,
                            x="model",
                            y="r2",
                            color="model",
                            title="R² (higher is better)",
                        )
                        st.plotly_chart(fig2, use_container_width=True)
                    st.dataframe(comp, use_container_width=True)

                    best = comp.iloc[0]["model"]
                    model = bundle["models"][best]
                    preds = model.predict(bundle["X_test"])
                    metrics = regression_metrics(bundle["y_test"], preds)
                    st.markdown(f"**Best model:** `{best}` — RMSE `{metrics['rmse']:.2f}`, MAE `{metrics['mae']:.2f}`, R² `{metrics['r2']:.3f}`")

                    scat = pd.DataFrame({"actual": bundle["y_test"], "predicted": preds})
                    fig3 = px.scatter(
                        scat,
                        x="actual",
                        y="predicted",
                        title="Holdout: actual vs predicted yield",
                        trendline="ols",
                    )
                    st.plotly_chart(fig3, use_container_width=True)

                    if "year" in df.columns and config.TARGET_COL in df.columns:
                        trend = (
                            df.groupby("year", as_index=False)[config.TARGET_COL]
                            .mean()
                            .sort_values("year")
                        )
                        fig4 = px.line(
                            trend,
                            x="year",
                            y=config.TARGET_COL,
                            markers=True,
                            title="Mean observed yield by year (uploaded dataset)",
                        )
                        st.plotly_chart(fig4, use_container_width=True)

    with tabs[2]:
        st.subheader("Climate risk analytics")
        if "df" not in st.session_state:
            st.warning("Load data first.")
        else:
            df = st.session_state["df"]
            needed = {"rainfall_mm", "humidity_pct", "temp_mean_c"}
            if not needed.issubset(set(df.columns)):
                st.error(f"Climate risk tab requires columns: {sorted(needed)}")
            else:
                enriched = panel_risk_summary(df)
                st.dataframe(enriched.head(100), use_container_width=True)
                fig = px.histogram(enriched, x="drought_risk_score", nbins=30, title="Drought risk score distribution")
                st.plotly_chart(fig, use_container_width=True)
                fig2 = px.scatter(
                    enriched,
                    x="rainfall_mm",
                    y="climate_yield_impact_pct",
                    color="drought_risk_score",
                    title="Rainfall vs heuristic climate yield impact",
                )
                st.plotly_chart(fig2, use_container_width=True)

    with tabs[3]:
        st.subheader("SHAP feature influence")
        if "train_bundle" not in st.session_state:
            st.warning("Train models first to enable SHAP on the holdout matrix.")
        else:
            bundle = st.session_state["train_bundle"]
            comp = bundle["comparison"]
            best = comp.iloc[0]["model"]
            model = bundle["models"][best]
            Xs = bundle["X_test"]
            st.caption(f"Explaining `{best}` on up to 400 holdout rows for responsiveness.")
            if st.button("Compute SHAP", type="primary"):
                sv, fnames, err = compute_shap_values(model, Xs, max_samples=400)
                if err:
                    st.warning(err)
                elif sv is not None and fnames is not None:
                    st.session_state["shap"] = (sv, fnames)
            if "shap" in st.session_state:
                sv, fnames = st.session_state["shap"]
                importance = mean_abs_shap(sv, fnames).head(25)
                fig = px.bar(
                    importance.iloc[::-1],
                    x="mean_abs_shap",
                    y="feature",
                    orientation="h",
                    title="Mean |SHAP| (top features)",
                )
                st.plotly_chart(fig, use_container_width=True)

    with tabs[4]:
        st.subheader("Batch inference")
        st.markdown(
            "Score new rows with the **best estimator from the last in-app training**. "
            "For HTTP access, run `uvicorn backend.api.main:app --reload`."
        )
        inf = st.file_uploader("Inference CSV (without target)", type=["csv"], key="inf")
        if "train_bundle" not in st.session_state:
            st.info("Train models in the *Train & Evaluate* tab to populate a session model.")
        elif inf is not None:
            new_df = pd.read_csv(inf)
            bundle = st.session_state["train_bundle"]
            best = bundle["comparison"].iloc[0]["model"]
            model = bundle["models"][best]
            preds = predict_yield(model, new_df)
            out = new_df.copy()
            out["predicted_yield_kg_ha"] = preds
            st.dataframe(out.head(100), use_container_width=True)
            csv_bytes = out.to_csv(index=False).encode("utf-8")
            st.download_button("Download predictions CSV", data=csv_bytes, file_name="predictions.csv")

        st.markdown("---")
        st.markdown("**Persisted bundle (optional)**")
        st.caption(str(config.MODEL_DIR / config.ARTIFACT_NAME))
        if st.button("Try load joblib bundle from disk"):
            try:
                b = load_bundle()
                st.success(f"Loaded bundle; best model: `{b.get('best_model_name')}`")
            except FileNotFoundError as e:
                st.warning(str(e))


if __name__ == "__main__":
    main()
