from __future__ import annotations

import os
from typing import Optional, Tuple, Dict, Any

import pandas as pd
import plotly.express as px
import streamlit as st

from core.pipeline import run_quality_assessment

# --- Paths --- #
FORMULAS_YAML = "configs/formulas.yaml"
PROMPTS_YAML = "configs/prompts.yaml"

DEFAULT_MODEL = "google/flan-t5-base"
MODEL_OPTIONS = [
    "google/flan-t5-base",
    "google/flan-t5-small",
]

# --- Page config --- #
st.set_page_config(
    page_title="Open Data Quality Assessment",
    layout="wide",
)

st.title("Open Data Quality Assessment (Vetrò et al. 2016)")

st.markdown(
    """
Upload an open data table (CSV or Excel) **or query a Trino database**, and this
tool will approximate data quality metrics following Vetrò et al.'s framework:
traceability, currentness, completeness, compliance, understandability and accuracy.

The AI assistance is used only for metadata-like signals (e.g., publisher, language, coverage),
while numeric indicators are derived directly from the data.
"""
)

# -------------------------------------------------------------------
# 1. Data source selection: file vs Trino
# -------------------------------------------------------------------
st.subheader("1. Choose data source")

data_source = st.radio(
    "Data source",
    options=["Upload file", "Trino SQL query (beta)"],
    index=0,
    horizontal=True,
)

# --- Common settings (rows, LLM) --- #
col_settings1, col_settings2, col_settings3 = st.columns(3)
with col_settings1:
    row_limit = st.number_input(
        "Row limit (0 = all rows)",
        min_value=0,
        value=500_000,
        step=10_000,
        help="For file uploads only. Set 0 to use all rows from the file.",
    )
with col_settings2:
    use_llm = st.checkbox("Use AI assistance for metadata (beta)", value=True)
with col_settings3:
    hf_model_name = st.selectbox(
        "Hugging Face model",
        options=MODEL_OPTIONS,
        index=0,
        disabled=not use_llm,
    )

# -------------------------------------------------------------------
# 2A. File upload UI
# -------------------------------------------------------------------
uploaded_file = None
if data_source == "Upload file":
    uploaded_file = st.file_uploader(
        "Upload a CSV or Excel file",
        type=["csv", "tsv", "txt", "xls", "xlsx"],
    )

# -------------------------------------------------------------------
# 2B. Trino DB UI
# -------------------------------------------------------------------
trino_host = trino_port = trino_catalog = trino_schema = trino_user = trino_password = ""
trino_sql = ""

if data_source == "Trino SQL query (beta)":
    st.markdown(
        """
Connect to a Trino endpoint and run a SQL query. The result table will be used as
input for the quality assessment.

**Note:** Credentials are used only in this session and are not stored by the app.
"""
    )
    col_conn1, col_conn2 = st.columns(2)
    with col_conn1:
        trino_host = st.text_input("Trino host", value="trino.avaandmeait.ee")
        trino_port = st.number_input("Trino port", min_value=1, max_value=65535, value=443)
    with col_conn2:
        trino_catalog = st.text_input("Catalog (optional)", value="")
        trino_schema = st.text_input("Schema (optional)", value="")

    col_auth1, col_auth2 = st.columns(2)
    with col_auth1:
        trino_user = st.text_input("Trino username", value="")
    with col_auth2:
        trino_password = st.text_input("Trino password", value="", type="password")

    trino_sql = st.text_area(
        "SQL query",
        height=180,
        placeholder="SELECT * FROM some_table LIMIT 100000",
        help="Use LIMIT in your query if the table is very large.",
    )

    st.caption(
        "JDBC-style URL (for reference): "
        "jdbc:trino://trino.avaandmeait.ee:443?SSL=true&SSLVerification=NONE"
    )

# -------------------------------------------------------------------
# 3. Run button
# -------------------------------------------------------------------
run_btn = st.button(
    "Run assessment",
    type="primary",
)

# -------------------------------------------------------------------
# 4. Main execution logic: load DataFrame, then compute metrics
# -------------------------------------------------------------------
if run_btn:
    try:
        df: Optional[pd.DataFrame] = None
        ext: Optional[str] = None

        # ------------------------------------------------------------
        # A) File path
        # ------------------------------------------------------------
        if data_source == "Upload file":
            if uploaded_file is None:
                st.error("Please upload a CSV/Excel file first.")
                st.stop()

            name = uploaded_file.name
            ext = os.path.splitext(name)[1].lower()

            # Load dataframe
            if ext in [".csv", ".tsv", ".txt"]:
                df = pd.read_csv(uploaded_file, sep=None, engine="python")
            elif ext in [".xls", ".xlsx"]:
                df = pd.read_excel(uploaded_file)
            else:
                st.error(f"Unsupported file type: {ext}")
                st.stop()

            if row_limit and row_limit > 0:
                df = df.head(row_limit)

        # ------------------------------------------------------------
        # B) Trino path
        # ------------------------------------------------------------
        elif data_source == "Trino SQL query (beta)":
            if not trino_sql.strip():
                st.error("Please enter a SQL query for Trino.")
                st.stop()
            if not trino_host.strip():
                st.error("Please enter Trino host.")
                st.stop()
            if not trino_user.strip():
                st.error("Please enter Trino username.")
                st.stop()

            try:
                import trino
                from trino.dbapi import connect as trino_connect
                from trino.auth import BasicAuthentication
            except Exception as e:
                st.error(
                    "The 'trino' Python package is required for DB mode. "
                    "Please ensure it is installed.\n\n"
                    f"Import error: {e}"
                )
                st.stop()

            # Build connection
            conn_kwargs: Dict[str, Any] = {
                "host": trino_host.strip(),
                "port": int(trino_port),
                "user": trino_user.strip(),
                "http_scheme": "https",
            }
            if trino_catalog.strip():
                conn_kwargs["catalog"] = trino_catalog.strip()
            if trino_schema.strip():
                conn_kwargs["schema"] = trino_schema.strip()

            # Auth: Basic if password given, otherwise no auth object
            if trino_password:
                conn_kwargs["auth"] = BasicAuthentication(
                    trino_user.strip(), trino_password
                )

            try:
                conn = trino_connect(**conn_kwargs)
                df = pd.read_sql(trino_sql, conn)
                ext = ".sql"
            except Exception as e:
                st.error(f"Failed to execute Trino query: {e}")
                st.stop()

        # ------------------------------------------------------------
        # Sanity check: df must exist
        # ------------------------------------------------------------
        if df is None:
            st.error("No data could be loaded from the selected data source.")
            st.stop()

        # ------------------------------------------------------------
        # Preview
        # ------------------------------------------------------------
        st.subheader("Preview of the dataset")
        st.dataframe(df.head(20), width="stretch")
        st.caption(f"{df.shape[0]} rows × {df.shape[1]} columns used for metrics.")

        # If extension is still None (should not happen), fall back
        if ext is None:
            ext = ".table"

        # ------------------------------------------------------------
        # Compute metrics
        # ------------------------------------------------------------
        with st.spinner("Computing quality metrics..."):
            metrics_df, details = run_quality_assessment(
                df=df,
                formulas_yaml_path=FORMULAS_YAML,
                prompts_yaml_path=PROMPTS_YAML,
                use_llm=use_llm,
                hf_model_name=hf_model_name,
                file_ext=ext,
            )

        st.subheader("Quality metrics")

        if metrics_df.empty or metrics_df["value"].dropna().empty:
            st.info("No metrics could be computed.")
        else:
            metrics_non_null = metrics_df.dropna(subset=["value"]).copy()
            metrics_non_null["value_clamped"] = metrics_non_null["value"].clip(0.0, 1.0)

            fig = px.bar(
                metrics_non_null,
                x="metric_label",
                y="value_clamped",
                color="dimension",
                range_y=[0, 1],
                labels={
                    "metric_label": "Metric",
                    "value_clamped": "Normalised value (0–1)",
                    "dimension": "Dimension",
                },
            )
            fig.update_layout(xaxis_tickangle=-35)
            st.plotly_chart(fig, width="stretch")

            st.dataframe(
                metrics_non_null[
                    ["dimension", "metric_label", "value", "metric_id"]
                ].sort_values(["dimension", "metric_id"]),
                width="stretch",
            )

        # ------------------------------------------------------------
        # Debug / explanations
        # ------------------------------------------------------------
        with st.expander("Debug: auto-derived inputs and AI inferences"):
            st.markdown("**Auto-derived inputs (from the table only):**")
            st.json(details.get("auto_inputs", {}))

            symbol_values = details.get("symbol_values", {})
            if not symbol_values:
                st.write("No symbol debug information available.")
            else:
                src = details.get("symbol_source", {})
                conf = details.get("llm_confidence", {})
                evid = details.get("llm_evidence", {})
                rows = []
                for sym in sorted(symbol_values.keys()):
                    rows.append(
                        {
                            "symbol": sym,
                            "value": symbol_values.get(sym),
                            "source": src.get(sym, ""),
                            "llm_confidence": conf.get(sym),
                            "llm_evidence": evid.get(sym, ""),
                        }
                    )
                st.dataframe(pd.DataFrame(rows), width="stretch")

    except Exception as e:
        st.exception(e)
