from __future__ import annotations

import io
from pathlib import Path
from typing import Tuple

import pandas as pd
import plotly.express as px
import streamlit as st

from core.pipeline import run_quality_assessment


# ---- Paths (repo root) ----
FORMULAS = "configs/formulas.yaml"
PROMPTS = "configs/prompts.yaml"

DEFAULT_MODELS = [
    "google/flan-t5-small",
    "google/flan-t5-base",
]


def _read_table(uploaded_file) -> Tuple[pd.DataFrame, str]:
    """
    Read CSV/XLSX. Returns (df, ext).
    Tries a few encodings for CSV to handle Estonian portal downloads.
    """
    name = uploaded_file.name
    ext = Path(name).suffix.lower().lstrip(".")

    if ext in ("xlsx", "xls"):
        df = pd.read_excel(uploaded_file)
        return df, ext

    raw = uploaded_file.getvalue()
    sep = ","  # default
    if raw.count(b"\t") > raw.count(b","):
        sep = "\t"

    encodings = ["utf-8-sig", "utf-8", "cp1252", "latin1"]
    last_err = None
    for enc in encodings:
        try:
            df = pd.read_csv(io.BytesIO(raw), encoding=enc, sep=sep)
            return df, ext
        except Exception as e:
            last_err = e
            continue
    raise last_err  # type: ignore


def _nice_metric_order(df: pd.DataFrame) -> pd.DataFrame:
    dim_order = ["traceability", "currentness", "completeness", "compliance", "understandability", "accuracy"]
    df = df.copy()
    df["dimension"] = pd.Categorical(df["dimension"], categories=dim_order, ordered=True)
    return df.sort_values(["dimension", "metric_id"])


st.set_page_config(page_title="Open Data Quality Assessment", layout="wide")

st.title("Open Data Quality Assessment (Vetrò-style + optional AI)")
st.write(
    "Upload a dataset file (CSV/XLSX) and compute open data quality metrics. "
    "The tool combines rule-based checks with optional Hugging Face models."
)

uploaded = st.file_uploader("Dataset file", type=["csv", "tsv", "xlsx", "xls"])

colA, colB, colC = st.columns([1.2, 1.0, 1.2], vertical_alignment="top")

with colA:
    max_rows = st.number_input(
        "Max rows to process (0 = all rows)",
        min_value=0,
        value=0,
        step=10000,
        help="For very large files, using all rows can be slow on Streamlit Cloud.",
    )
with colB:
    use_llm = st.checkbox(
        "Use AI for metadata-related checks",
        value=False,
        help="AI helps with text/metadata symbols. For best results, paste portal metadata below.",
    )
    weight_by_conf = st.checkbox(
        "Weight AI symbols by confidence",
        value=False,
        help="If enabled, AI-derived numeric symbols are multiplied by an estimated confidence (0..1).",
    )
with colC:
    hf_model_name = st.selectbox("Hugging Face model", DEFAULT_MODELS, index=1)
    hf_custom = st.text_input("Custom model (optional)", value="")
    if hf_custom.strip():
        hf_model_name = hf_custom.strip()

dataset_description = st.text_area(
    "Portal metadata / dataset description (optional but recommended)",
    height=160,
    placeholder="Paste the dataset description, license, publisher, update info, etc. (copy from andmed.eesti.ee).",
)

run_btn = st.button("Run assessment", type="primary", disabled=(uploaded is None))

# Persist results across reruns (so download button won't “wipe” output)
if "last_result" not in st.session_state:
    st.session_state["last_result"] = None

if run_btn and uploaded is not None:
    try:
        df, ext = _read_table(uploaded)

        if max_rows and max_rows > 0 and len(df) > max_rows:
            df = df.head(int(max_rows)).copy()

        with st.spinner("Computing metrics..."):
            _, metrics_df, details = run_quality_assessment(
                df=df,
                formulas_yaml_path=FORMULAS,
                prompts_yaml_path=PROMPTS,
                use_llm=use_llm,
                hf_model_name=hf_model_name,
                dataset_description=dataset_description,
                file_name=uploaded.name,
                file_ext=ext,
                weight_by_confidence=weight_by_conf,
            )

        st.session_state["last_result"] = (metrics_df, details, uploaded.name)

    except Exception as e:
        st.error(f"Failed to run assessment: {e}")
        st.stop()

if st.session_state["last_result"] is not None:
    metrics_df, details, fname = st.session_state["last_result"]

    st.subheader("Results")
    st.caption(f"File: {fname}")

    metrics_df = _nice_metric_order(metrics_df)

    plot_df = metrics_df.dropna(subset=["value"]).copy()
    if not plot_df.empty:
        fig = px.bar(
            plot_df,
            x="metric_label",
            y="value",
            color="dimension",
            hover_data=["metric_id", "description"],
            title="Metric scores (0..1 where applicable)",
        )
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("No metric values were computed (all values are empty). Check symbols/debug below.")

    st.dataframe(metrics_df, width="stretch")

    csv_bytes = metrics_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download metrics (CSV)",
        data=csv_bytes,
        file_name="open_data_quality_metrics.csv",
        mime="text/csv",
    )

    st.subheader("Debug (symbols)")
    symbols_df = pd.DataFrame(details.get("symbols", []))
    if not symbols_df.empty:
        st.dataframe(symbols_df, width="stretch")
        st.download_button(
            "Download symbols debug (CSV)",
            data=symbols_df.to_csv(index=False).encode("utf-8"),
            file_name="open_data_quality_symbols_debug.csv",
            mime="text/csv",
        )

    with st.expander("Auto inputs (derived from data)"):
        st.json(details.get("auto_inputs", {}))

    if use_llm:
        with st.expander("LLM raw outputs"):
            st.json(details.get("llm_raw", {}))
