from __future__ import annotations

import os
from typing import Optional, Tuple, Dict, Any

import pandas as pd
import plotly.express as px
import streamlit as st

from core.utils import make_arrow_safe
from core.pipeline import run_quality_assessment

# --- Paths --- #
FORMULAS_YAML = "configs/formulas.yaml"
PROMPTS_YAML = "configs/prompts.yaml"

DEFAULT_MODEL = "google/flan-t5-base"
MODEL_OPTIONS = [
    "google/flan-t5-base",
    "google/flan-t5-small",
]


def parse_kv_metadata(text: str) -> Dict[str, Any]:
    """
    Parse lines like:
      pb: 1
      publisher: Siseministeerium
      metadata_created: 2023-01-01
    """
    meta: Dict[str, Any] = {}
    for line in (text or "").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        k = k.strip()
        v = v.strip()
        if not k:
            continue
        # try numeric
        try:
            meta[k] = float(v)
        except Exception:
            meta[k] = v
    return meta


def normalize_metadata_to_symbols(meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Accept either:
      - direct symbols (pb, t, d, dc, cv, l, id, s, etc.)
      - or common field names (publisher, title, description, metadata_created, coverage, language, identifier, source)
    Output is a dict that can be used as symbol values.
    """
    out: Dict[str, Any] = {}

    # keep direct symbol keys if present
    for k, v in meta.items():
        if k in {"pb", "t", "d", "dc", "cv", "l", "id", "s", "dp", "sd", "edp", "ed", "cd"}:
            out[k] = v

    def _present(x: Any) -> bool:
        if x is None:
            return False
        if isinstance(x, float) and pd.isna(x):
            return False
        if isinstance(x, str) and not x.strip():
            return False
        return True

    # map common names -> symbols (presence flags)
    title = meta.get("title")
    if "t" not in out and _present(title):
        out["t"] = 1.0

    desc = meta.get("description") or meta.get("notes")
    if "d" not in out and _present(desc):
        out["d"] = 1.0

    publisher = meta.get("publisher") or meta.get("organization") or meta.get("org_name")
    if "pb" not in out and _present(publisher):
        out["pb"] = 1.0
        out["s"] = 1.0 

    # --- Date of creation mapping ---
    created = (
        meta.get("metadata_created")
        or meta.get("issued")
        or meta.get("created")
        or meta.get("date_of_creation")
    )

    if _present(created):
        out["dc"] = 1.0

    coverage = meta.get("coverage")
    if "cv" not in out and _present(coverage):
        out["cv"] = 1.0

    language = meta.get("language") or meta.get("lang")
    if "l" not in out and _present(language):
        out["l"] = 1.0

    identifier = meta.get("identifier") or meta.get("dataset_id") or meta.get("id")
    if "id" not in out and _present(identifier):
        out["id"] = 1.0

    source = meta.get("source")
    if "s" not in out and _present(source):
        out["s"] = 1.0

    # If metadata includes explicit publication date, pass dp (date-like string)
    dp = meta.get("date_of_publication") or meta.get("metadata_modified") or meta.get("modified")
    if "dp" not in out and _present(dp):
        out["dp"] = dp

    return out


# --- Page config --- #
st.set_page_config(page_title="Open Data Quality Assessment", layout="wide")
st.title("Open Data Quality Assessment (Vetrò et al. 2016)")

st.markdown(
    """
Upload an open data table (CSV / Excel) **or query a Trino database**, and this tool will approximate
data quality metrics following Vetrò et al.'s framework.

Priority for inputs:
1) auto-derived from table,
2) Trino metadata (if provided),
3) manual metadata textbox,
4) LLM fallback (optional).
"""
)

# -------------------------------------------------------------------
# 1. Data source selection
# -------------------------------------------------------------------
st.subheader("1) Choose data source")
data_source = st.radio(
    "Data source",
    options=["Upload file", "Trino SQL query (beta)"],
    index=0,
    horizontal=True,
)

# --- Common settings ---
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
trino_host = trino_port = trino_catalog = trino_schema = trino_user = trino_password = ""
trino_sql = ""
trino_meta_sql = ""
trino_metadata_raw: Dict[str, Any] = {}

if data_source == "Upload file":
    uploaded_file = st.file_uploader(
        "Upload a CSV or Excel file",
        type=["csv", "tsv", "txt", "xls", "xlsx"],
    )

# -------------------------------------------------------------------
# 2B. Trino DB UI
# -------------------------------------------------------------------
if data_source == "Trino SQL query (beta)":
    st.markdown(
        """
Connect to a Trino endpoint and run a SQL query. The result table will be used as input.
Optionally, you can also run a **metadata query** (one-row result) to provide portal metadata.
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
        "Data SQL query",
        height=160,
        placeholder="SELECT * FROM some_table LIMIT 100000",
        help="Use LIMIT if the table is very large.",
    )

    trino_meta_sql = st.text_area(
        "Metadata SQL query (optional, should return 1 row)",
        height=160,
        placeholder=(
            "Example:\n"
            "SELECT\n"
            "  title,\n"
            "  notes AS description,\n"
            "  metadata_created,\n"
            "  metadata_modified,\n"
            "  organization.name AS publisher\n"
            "FROM landing.avaandmete_portaal.dataset_metadata\n"
            "WHERE lower(title) LIKE '%abiel%'\n"
            "LIMIT 1"
        ),
        help="If provided, this should return 1 row with columns like title/description/publisher/metadata_created etc.",
    )

# -------------------------------------------------------------------
# 2C. Manual metadata textbox (both modes)
# -------------------------------------------------------------------
st.subheader("2) Optional manual metadata")
manual_meta_text = st.text_area(
    "Manual metadata (key: value per line). Used if auto/Trino doesn't provide it.",
    height=140,
    help=(
        "You can provide either symbols or common names.\n\n"
        "Symbols example:\n"
        "pb: 1\n"
        "dc: 1\n"
        "t: 1\n\n"
        "Common names example:\n"
        "publisher: Siseministeerium\n"
        "title: Abielud maakonna ja aasta järgi\n"
        "metadata_created: 2018-01-15\n"
    ),
)

# -------------------------------------------------------------------
# 3. Run button
# -------------------------------------------------------------------
run_btn = st.button("Run assessment", type="primary")

# -------------------------------------------------------------------
# 4. Main logic
# -------------------------------------------------------------------
if run_btn:
    try:
        df: Optional[pd.DataFrame] = None
        ext: Optional[str] = None
        trino_metadata: Dict[str, Any] = {}
        manual_metadata_raw = parse_kv_metadata(manual_meta_text)
        manual_metadata = normalize_metadata_to_symbols(manual_metadata_raw)
        if not trino_metadata_raw:
            trino_metadata_raw = manual_metadata_raw

        conn = None

        # -----------------------
        # A) File mode
        # -----------------------
        if data_source == "Upload file":
            if uploaded_file is None:
                st.error("Please upload a CSV/Excel file first.")
                st.stop()

            name = uploaded_file.name
            ext = os.path.splitext(name)[1].lower()

            if ext in [".csv", ".tsv", ".txt"]:
                df = pd.read_csv(uploaded_file, sep=None, engine="python")
            elif ext in [".xls", ".xlsx"]:
                df = pd.read_excel(uploaded_file)
            else:
                st.error(f"Unsupported file type: {ext}")
                st.stop()

            if row_limit and row_limit > 0:
                df = df.head(row_limit)

        # -----------------------
        # B) Trino mode
        # -----------------------
        else:
            if not trino_sql.strip():
                st.error("Please enter a Data SQL query for Trino.")
                st.stop()
            if not trino_host.strip():
                st.error("Please enter Trino host.")
                st.stop()
            if not trino_user.strip():
                st.error("Please enter Trino username.")
                st.stop()

            try:
                from trino.dbapi import connect as trino_connect
                from trino.auth import BasicAuthentication
            except Exception as e:
                st.error(
                    "The 'trino' Python package is required for DB mode. "
                    f"Import error: {e}"
                )
                st.stop()

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
            if trino_password:
                conn_kwargs["auth"] = BasicAuthentication(trino_user.strip(), trino_password)

            try:
                conn = trino_connect(**conn_kwargs)
                df = pd.read_sql(trino_sql, conn)
                ext = ".sql"
            except Exception as e:
                st.error(f"Failed to execute Trino data query: {e}")
                st.stop()

            # Optional metadata query
            if trino_meta_sql.strip():
                try:
                    meta_df = pd.read_sql(trino_meta_sql, conn)
                    if not meta_df.empty:
                        meta_row = meta_df.iloc[0].to_dict()
                        trino_metadata_raw = meta_row
                        trino_metadata = normalize_metadata_to_symbols(trino_metadata_raw)
                except Exception as e:
                    st.warning(f"Metadata query failed (continuing without it): {e}")

        # -----------------------
        # Sanity check
        # -----------------------
        if df is None:
            st.error("No data could be loaded from the selected data source.")
            st.stop()

        df = make_arrow_safe(df)

        # Preview
        st.subheader("Preview of the dataset")
        # Fix nested object columns for Arrow compatibility
        for col in df.columns:
            if df[col].dtype == "object":
                df[col] = df[col].apply(
                    lambda x: str(x) if isinstance(x, (dict, list, tuple)) else x
                )
        st.dataframe(df.head(20), width="stretch")
        st.caption(f"{df.shape[0]} rows × {df.shape[1]} columns used for metrics.")

        if ext is None:
            ext = ".table"

        # Compute
        with st.spinner("Computing quality metrics..."):
            metrics_df, details = run_quality_assessment(
                df=df,
                formulas_yaml_path=FORMULAS_YAML,
                prompts_yaml_path=PROMPTS_YAML,
                use_llm=use_llm,
                hf_model_name=hf_model_name,
                file_ext=ext,
                manual_metadata=manual_metadata,
                trino_metadata={},            
                trino_metadata_raw=trino_metadata_raw,
)

        metrics_df["value"] = metrics_df["value"].apply(
            lambda x: float(x) if isinstance(x, (int, float)) else None
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
                metrics_non_null[["dimension", "metric_label", "value", "metric_id"]]
                .sort_values(["dimension", "metric_id"]),
                width="stretch",
            )

        # Debug
        with st.expander("Debug: auto-derived inputs and AI/metadata inferences"):
            st.markdown("**Auto-derived inputs (from the table only):**")
            st.json(details.get("auto_inputs", {}))

            st.markdown("**Trino metadata (normalised to symbols):**")
            st.json(trino_metadata)

            st.markdown("**Manual metadata (normalised to symbols):**")
            st.json(manual_metadata)

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

                debug_df = pd.DataFrame(rows)

                # Fix Arrow dtype issues
                debug_df["value"] = debug_df["value"].astype(str)
                debug_df["llm_confidence"] = pd.to_numeric(debug_df["llm_confidence"], errors="coerce")
                debug_df["llm_evidence"] = debug_df["llm_evidence"].astype(str)

                st.dataframe(debug_df, width="stretch")

    except Exception as e:
        st.exception(e)