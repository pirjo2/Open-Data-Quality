from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import plotly.express as px
import streamlit as st

from core.pipeline import run_quality_assessment


# -------------------------------------------------------------------
# Basic configuration
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Open Data Quality Assessment",
    layout="wide",
)

REPO_ROOT = Path(__file__).parent
CONFIG_DIR = REPO_ROOT / "configs"
FORMULAS = CONFIG_DIR / "formulas.yaml"
PROMPTS = CONFIG_DIR / "prompts.yaml"


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def _load_table(file, ext: str, max_rows: int) -> pd.DataFrame:
    ext = ext.lower()
    if ext in ("csv", "txt"):
        # Try comma first, then fall back to semicolon
        try:
            df = pd.read_csv(file)
        except Exception:
            df = pd.read_csv(file, sep=";")
    elif ext in ("tsv",):
        df = pd.read_csv(file, sep="\t")
    elif ext in ("xls", "xlsx"):
        df = pd.read_excel(file)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

    if max_rows > 0 and len(df) > max_rows:
        df = df.head(max_rows)
    return df


def _safe_metrics_value(v) -> Optional[float]:
    try:
        f = float(v)
        if pd.isna(f):
            return None
        return f
    except Exception:
        return None


# -------------------------------------------------------------------
# UI
# -------------------------------------------------------------------
st.title("Open Data Quality Assessment")

st.markdown(
    """
Upload an open data file (CSV, TSV, or Excel) and get an automatic quality profile
based on the Vetrò et al. methodology (traceability, currentness, completeness,
compliance, understandability, accuracy).

The app combines **direct computations** from the data with **AI-assisted heuristics**
for metadata fields that are not present in the file itself.
"""
)

# --- File upload + basic info -------------------------------------------------
with st.container():
    uploaded = st.file_uploader(
        "Upload a tabular dataset (CSV, TSV, XLSX)",
        type=["csv", "tsv", "txt", "xls", "xlsx"],
    )

    dataset_description = st.text_area(
        "Optional dataset description / metadata (copy from portal if available)",
        help="This text is passed to the AI model together with column names and basic profiling.",
        height=120,
    )

    col_params = st.columns(3)
    with col_params[0]:
        max_rows = st.number_input(
            "Maximum rows to analyse (0 = all rows)",
            min_value=0,
            value=50000,
            step=1000,
        )
    with col_params[1]:
        use_llm = st.checkbox(
            "Use AI-assisted metadata estimation",
            value=True,
            help="If disabled, only metrics that can be derived directly from the data are computed.",
        )
    with col_params[2]:
        model_choice = st.selectbox(
            "Hugging Face model",
            options=[
                "google/flan-t5-base",
                "google/flan-t5-small",
                "t5-small",
                "custom...",
            ],
            index=0,
        )
    if model_choice == "custom...":
        hf_model_name = st.text_input(
            "Custom HF model name",
            value="google/flan-t5-base",
            help="Any Seq2Seq or causal text model from HuggingFace Hub (must fit into available memory).",
        )
    else:
        hf_model_name = model_choice

run_button = st.button("Run assessment", type="primary", disabled=uploaded is None)

if uploaded is None:
    st.info("Upload a dataset to get started.")
    st.stop()

# -------------------------------------------------------------------
# Load data
# -------------------------------------------------------------------
file_ext = Path(uploaded.name).suffix.lstrip(".") or "csv"

try:
    df = _load_table(uploaded, file_ext, max_rows=max_rows)
except Exception as e:
    st.error(f"Failed to read file: {e}")
    st.stop()

st.subheader("Dataset preview")
st.write(f"{len(df)} rows × {len(df.columns)} columns used in the analysis.")
st.dataframe(df.head(30), use_container_width=True)


# -------------------------------------------------------------------
# Run assessment
# -------------------------------------------------------------------
if run_button:
    if not FORMULAS.exists() or not PROMPTS.exists():
        st.error(
            f"Configuration files not found. Expected:\n- {FORMULAS}\n- {PROMPTS}"
        )
        st.stop()

    with st.spinner("Computing quality metrics... this may take a bit if the model is large."):
        try:
            _, metrics_df, details = run_quality_assessment(
                df=df,
                formulas_yaml_path=str(FORMULAS),
                prompts_yaml_path=str(PROMPTS),
                use_llm=use_llm,
                hf_model_name=hf_model_name,
                dataset_description=dataset_description,
                file_name=uploaded.name,
                file_ext=file_ext,
            )
        except Exception as e:
            st.error(f"Failed to run assessment: {e}")
            st.stop()

    # -------------------------------------------------------------------
    # Metrics summary
    # -------------------------------------------------------------------
    st.subheader("Quality metrics")

    if metrics_df.empty:
        st.warning("No metrics could be computed.")
    else:
        # Clean up NaNs for display
        metrics_df_display = metrics_df.copy()
        metrics_df_display["value_display"] = metrics_df_display["value"].apply(_safe_metrics_value)

        # Order by dimension for nicer layout
        metrics_df_display = metrics_df_display.sort_values(
            by=["dimension", "metric"]
        )

        st.dataframe(
            metrics_df_display[["dimension", "metric", "metric_id", "value_display", "description"]],
            use_container_width=True,
            hide_index=True,
        )

        # Simple bar chart of non-NaN values
        chart_df = metrics_df_display.dropna(subset=["value_display"]).copy()
        if not chart_df.empty:
            chart_df["metric_full"] = chart_df["dimension"] + " · " + chart_df["metric"]
            fig = px.bar(
                chart_df,
                x="metric_full",
                y="value_display",
                color="dimension",
                title="Normalised metric values (0–1)",
            )
            fig.update_layout(
                xaxis_title="Metric",
                yaxis_title="Value (0–1)",
                xaxis_tickangle=-45,
            )
            st.plotly_chart(fig, use_container_width=True)

        # Download CSV of metrics
        csv_bytes = metrics_df_display.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download metrics as CSV",
            data=csv_bytes,
            file_name="open_data_quality_metrics.csv",
            mime="text/csv",
        )

    # -------------------------------------------------------------------
    # Debug / diagnostics
    # -------------------------------------------------------------------
    with st.expander("Debug: inputs and AI inferences"):
        st.markdown("**Auto-derived inputs** (from the dataframe only):")
        st.json(details.get("auto_inputs", {}), expanded=False)

        # Per-symbol table: use string values to avoid Arrow type issues
        sym_vals = details.get("symbol_values", {})
        sym_src = details.get("symbol_source", {})
        llm_conf = details.get("llm_confidence", {})
        llm_raw = details.get("llm_raw", {})
        llm_ev = details.get("llm_evidence", {})

        rows = []
        for sym in sorted(details.get("required_symbols", sym_vals.keys())):
            val = sym_vals.get(sym, None)
            if val is None:
                v_str = ""
            else:
                v_str = str(val)
            rows.append(
                {
                    "symbol": sym,
                    "value (string)": v_str,
                    "source": sym_src.get(sym, ""),
                    "confidence": llm_conf.get(sym, ""),
                    "evidence": llm_ev.get(sym, ""),
                    "raw": llm_raw.get(sym, ""),
                }
            )
        if rows:
            debug_df = pd.DataFrame(rows)
            st.dataframe(debug_df, use_container_width=True, hide_index=True)
        else:
            st.write("No symbol debug information available.")
