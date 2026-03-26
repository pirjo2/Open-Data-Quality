from __future__ import annotations

import os
from typing import Optional, Dict, Any, List, Tuple

import pandas as pd
import plotly.express as px
import streamlit as st

from core.utils import make_arrow_safe
from core.pipeline import run_quality_assessment
from core.llm import get_llm_runner, infer_manual_metadata_symbols

import re
import yaml

# --- Paths --- #
FORMULAS_YAML = "configs/formulas.yaml"
PROMPTS_YAML = "configs/prompts.yaml"

# --- LLM options --- #
HF_MODEL_OPTIONS = [
    "google/flan-t5-small",
    "google/flan-t5-base",
    "google/flan-t5-large",
]

# These are example OpenAI model IDs you can expose in the UI.
# User can also type a custom model name below.
OPENAI_MODEL_OPTIONS = [
    "gpt-4.1-mini",
    "gpt-4.1",
    "gpt-5-mini",
    "gpt-5",
]

UPLOAD_MODE = "Upload file"
TRINO_MODE = "Trino SQL query (advanced)"

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
        try:
            meta[k] = float(v)
        except Exception:
            meta[k] = v
    return meta

def extract_symbols_from_realistic_text(text: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    low = (text or "").lower()

    # ---- test 2: track of updates ----
    if re.search(r"updates\s*:\s*\[\s*\]", low):
        out["lu"] = 0.0
        out["du"] = 0.0

    if "update history" in low or "updated regularly" in low:
        out["lu"] = 1.0

    if re.search(r"updated on\s+\d{4}-\d{2}-\d{2}", low):
        out["du"] = 1.0

    # ---- test 4: delay in publication ----
    m = re.search(
        r"covers the period from\s+(20\d{2}-\d{2}-\d{2})\s+to\s+(20\d{2}-\d{2}-\d{2})",
        low
    )
    if m:
        out["sd"] = m.group(1)
        out["edp"] = m.group(2)

    m = re.search(r"published on\s+(20\d{2}-\d{2}-\d{2})", low)
    if m:
        out["dp"] = m.group(1)

    # ---- test 5: delay after expiration ----
    m = re.search(r"expired on\s+(20\d{2}-\d{2}-\d{2})", low)
    if m:
        out["ed"] = m.group(1)

    m = re.search(r"became available on\s+(20\d{2}-\d{2}-\d{2})", low)
    if m:
        out["cd"] = m.group(1)

    # ---- test 9: eGMS ----
    if re.search(r"^title\s*:", text, re.I | re.M):
        out["t"] = 1.0
    elif "title is missing" in low:
        out["t"] = 0.0

    if re.search(r"^description\s*:", text, re.I | re.M):
        out["d"] = 1.0
    elif "description is missing" in low:
        out["d"] = 0.0

    if re.search(r"^identifier\s*:", text, re.I | re.M):
        out["id"] = 1.0
    elif "no identifier" in low or "identifier is missing" in low:
        out["id"] = 0.0

    if re.search(r"^publisher\s*:", text, re.I | re.M):
        out["pb"] = 1.0
    elif "publisher is missing" in low:
        out["pb"] = 0.0

    if re.search(r"^coverage\s*:", text, re.I | re.M):
        out["cv"] = 1.0
    elif "no coverage information" in low:
        out["cv"] = 0.0

    if re.search(r"^language\s*:", text, re.I | re.M):
        out["l"] = 1.0
    elif "language is missing" in low:
        out["l"] = 0.0

    if re.search(r"^source\s*:", text, re.I | re.M):
        out["s"] = 1.0
    elif "no source information" in low:
        out["s"] = 0.0

    if re.search(r"^date of creation\s*:\s*(20\d{2}-\d{2}-\d{2})", text, re.I | re.M):
        out["dc"] = 1.0
    elif "no creation date" in low or "creation date is missing" in low:
        out["dc"] = 0.0

    if re.search(r"^category\s*:", text, re.I | re.M):
        out["c"] = 1.0
    elif "no category" in low or "category is missing" in low:
        out["c"] = 0.0

    return out

def normalize_metadata_to_symbols(meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    AI-friendly normalizer:
    - explicit key:value overrides are kept
    - missing key != explicit 0
    - 0 is used only when the value explicitly indicates absence
    """
    out: Dict[str, Any] = {}

    # lowercase keys first
    meta_l = {str(k).strip().lower(): v for k, v in (meta or {}).items()}

    direct_symbols = {
        "pb", "t", "d", "dc", "cv", "l", "id", "s", "c",
        "dp", "sd", "edp", "ed", "cd",
        "lu", "du",
    }

    for k, v in meta_l.items():
        if k in direct_symbols:
            out[k] = v

    def _is_explicit_missing(x: Any) -> bool:
        if x is None:
            return True
        if isinstance(x, float) and pd.isna(x):
            return True
        if isinstance(x, str):
            s = x.strip().lower()
            return s in {
                "", "none", "null", "missing", "n/a", "na", "[]",
                "not provided", "not available"
            }
        return False

    def _presence_from_key(meta_dict: Dict[str, Any], *keys: str) -> Optional[float]:
        for key in keys:
            if key in meta_dict:
                return 0.0 if _is_explicit_missing(meta_dict[key]) else 1.0
        return None  # key absent -> let AI decide

    # semantic presence fields
    if "t" not in out:
        v = _presence_from_key(meta_l, "title")
        if v is not None:
            out["t"] = v

    if "d" not in out:
        v = _presence_from_key(meta_l, "description", "notes")
        if v is not None:
            out["d"] = v

    if "pb" not in out:
        v = _presence_from_key(meta_l, "publisher", "organization", "org_name")
        if v is not None:
            out["pb"] = v

    if "s" not in out:
        v = _presence_from_key(meta_l, "source")
        if v is not None:
            out["s"] = v

    if "dc" not in out:
        v = _presence_from_key(meta_l, "metadata_created", "issued", "created", "date_of_creation")
        if v is not None:
            out["dc"] = v

    if "cv" not in out:
        v = _presence_from_key(meta_l, "coverage")
        if v is not None:
            out["cv"] = v

    if "l" not in out:
        v = _presence_from_key(meta_l, "language", "lang")
        if v is not None:
            out["l"] = v

    if "id" not in out:
        v = _presence_from_key(meta_l, "identifier", "dataset_id", "id")
        if v is not None:
            out["id"] = v

    if "c" not in out:
        v = _presence_from_key(meta_l, "category", "theme")
        if v is not None:
            out["c"] = v

    # date-like fields: only pass through if key exists
    if "dp" not in out and "date_of_publication" in meta_l:
        out["dp"] = str(meta_l["date_of_publication"])
    if "dp" not in out and "metadata_modified" in meta_l:
        out["dp"] = str(meta_l["metadata_modified"])
    if "dp" not in out and "modified" in meta_l:
        out["dp"] = str(meta_l["modified"])

    if "sd" not in out and "sd" in meta_l:
        out["sd"] = str(meta_l["sd"])
    if "edp" not in out and "edp" in meta_l:
        out["edp"] = str(meta_l["edp"])
    if "ed" not in out and "ed" in meta_l:
        out["ed"] = str(meta_l["ed"])
    if "cd" not in out and "cd" in meta_l:
        out["cd"] = str(meta_l["cd"])

    # updates: only set if explicit structured key exists
    if "lu" not in out and "updates" in meta_l:
        out["lu"] = 0.0 if _is_explicit_missing(meta_l["updates"]) else 1.0

    if "du" not in out and "updates" in meta_l:
        # explicit update list/history key alone does NOT guarantee dated updates
        out["du"] = 0.0 if _is_explicit_missing(meta_l["updates"]) else None

    return {k: v for k, v in out.items() if v is not None}

def parse_uploaded_metadata_file(uploaded) -> Tuple[Dict[str, Any], str]:
    """
    Parse optional metadata files in TXT / JSON / YAML format.
    Returns:
      - parsed raw metadata dict
      - original text content for optional LLM interpretation
    """
    if uploaded is None:
        return {}, ""

    raw_bytes = uploaded.getvalue()
    text_content = raw_bytes.decode("utf-8", errors="ignore")
    suffix = os.path.splitext(uploaded.name)[1].lower()

    if suffix in {".json", ".yaml", ".yml"}:
        try:
            parsed = yaml.safe_load(text_content) or {}
        except Exception:
            parsed = {}

        if isinstance(parsed, dict):
            return parsed, text_content
        return {}, text_content

    return parse_kv_metadata(text_content), text_content


def build_quality_recommendations(metrics_df: pd.DataFrame) -> List[str]:
    if metrics_df.empty:
        return []

    guidance = {
        "traceability.track_of_creation": "Add clear source information and the dataset or metadata creation date.",
        "traceability.track_of_updates": "Document update history and include explicit update dates or a regular update schedule.",
        "currentness.percentage_of_current_rows": "Review the time-related columns and remove or flag rows that refer to outdated periods.",
        "currentness.delay_in_publication": "Publish data closer to the end of the reference period and document the publication date consistently.",
        "currentness.delay_after_expiration": "Refresh or archive expired datasets faster so they do not remain outdated for long.",
        "completeness.percentage_of_complete_cells": "Reduce missing values in key columns and use consistent placeholders only when they are truly needed.",
        "completeness.percentage_of_complete_rows": "Focus on row-level completeness for the most important records before publication.",
        "compliance.percentage_of_standardized_columns": "Use standardised formats, codes, and controlled vocabularies where possible.",
        "compliance.egms_compliance": "Improve metadata completeness by adding title, description, publisher, identifier, language, category, source, and coverage.",
        "compliance.five_stars_open_data": "Publish the dataset in a machine-readable non-proprietary format and add persistent identifiers or links where relevant.",
        "understandability.percentage_of_columns_with_metadata": "Add a column-level data dictionary so every field has a clear description.",
        "understandability.percentage_of_columns_in_comprehensible_format": "Rename cryptic columns, explain abbreviations, and use human-readable value formats.",
        "accuracy.percentage_of_syntactically_accurate_cells": "Validate dates, numeric fields, and code formats before publication to reduce syntax errors.",
        "accuracy.accuracy_in_aggregation": "Cross-check totals and aggregates against row-level values to ensure calculations are internally consistent.",
    }

    low_metrics = (
        metrics_df.dropna(subset=["value_clamped"])
        .sort_values("value_clamped", ascending=True)
        .head(4)
    )

    recommendations: List[str] = []
    for _, row in low_metrics.iterrows():
        metric_id = str(row.get("metric_id", "")).strip()
        score = float(row.get("value_clamped", 0.0))
        message = guidance.get(metric_id)
        if message:
            recommendations.append(f"{row['metric_label']} ({score:.2f}): {message}")

    return recommendations


def build_ai_recommendation_prompt(metrics_df: pd.DataFrame, data_source: str) -> str:
    weakest = (
        metrics_df.sort_values("value_clamped", ascending=True)
        .head(5)[["dimension", "metric_label", "metric_id", "value_clamped"]]
        .to_dict(orient="records")
    )

    return f"""
You are reviewing open data quality results.

Dataset source mode: {data_source}

Weakest metrics:
{yaml.safe_dump(weakest, sort_keys=False, allow_unicode=True)}

Write exactly 4 short markdown bullet points.
Each bullet must:
- explain what to improve
- be concrete and actionable
- mention metadata and/or content quality when relevant
- stay under 28 words

Do not mention confidence.
Do not mention that you are an AI.
"""

# --- Page config --- #
# --- Page config --- #
st.set_page_config(page_title="Open Data Quality Assessment", layout="wide")

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .hero-card {
        padding: 1.4rem 1.6rem;
        border: 1px solid rgba(49, 51, 63, 0.16);
        border-radius: 20px;
        background: linear-gradient(135deg, rgba(244,246,255,0.95), rgba(250,250,250,0.98));
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero-card">
        <div style="font-size: 0.95rem; font-weight: 600; opacity: 0.8;">
            Bachelor's thesis prototype
        </div>
        <div style="font-size: 2.1rem; font-weight: 800; margin-top: 0.15rem;">
            Open Data Quality Assessment
        </div>
        <div style="font-size: 1.03rem; margin-top: 0.65rem; line-height: 1.55;">
            This tool was developed in the context of a bachelor's thesis and evaluates open datasets
            using the Vetrò et al. (2016) quality framework adapted for an AI-assisted workflow.
            In practice, the app combines table-based checks, optional metadata, and AI-assisted
            inference for metadata signals that cannot be derived reliably from structure alone.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.subheader("1) Choose data source")
data_source = st.radio(
    "How would you like to assess the dataset?",
    options=[UPLOAD_MODE, TRINO_MODE],
    index=0,
    horizontal=True,
)

source_col1, source_col2 = st.columns(2)
with source_col1:
    st.markdown("#### File upload")
    st.caption("Recommended for most users. Upload a CSV or Excel dataset and optionally enrich it with metadata.")
with source_col2:
    st.markdown("#### Trino SQL query")
    st.caption("Advanced mode for users who already have Trino access and want to query both data and metadata.")

if data_source == UPLOAD_MODE:
    st.info(
        "Logic: upload a table → derive structural signals → merge optional metadata → optionally use AI for semantic gaps → compute Vetrò-based scores."
    )

    col_settings1, col_settings2, col_settings3 = st.columns(3)
    with col_settings1:
        row_limit = st.number_input(
            "Row limit (0 = all rows)",
            min_value=0,
            value=500_000,
            step=10_000,
            help="Used only in file mode.",
        )
    with col_settings2:
        use_llm = st.checkbox("Use AI to infer missing metadata", value=True)
    with col_settings3:
        llm_provider = st.selectbox(
            "AI provider",
            options=["huggingface", "openai"],
            index=0,
            disabled=not use_llm,
        )
else:
    st.info(
        "Logic: run a Trino data query → optionally fetch metadata with a second query → merge signals → optionally use AI for semantic gaps → compute Vetrò-based scores."
    )

    row_limit = 0
    col_settings1, col_settings2 = st.columns(2)
    with col_settings1:
        use_llm = st.checkbox("Use AI to infer missing metadata", value=True)
    with col_settings2:
        llm_provider = st.selectbox(
            "AI provider",
            options=["huggingface", "openai"],
            index=0,
            disabled=not use_llm,
        )

# Extra provider settings
llm_model_name = ""
openai_api_key: Optional[str] = None
prompt_regime = "zero_shot"

if use_llm:
    if llm_provider == "huggingface":
        llm_model_name = st.selectbox(
            "Hugging Face model",
            options=HF_MODEL_OPTIONS,
            index=0,
        )

    elif llm_provider == "openai":
        col_openai1, col_openai2 = st.columns(2)

        with col_openai1:
            openai_model_preset = st.selectbox(
                "OpenAI model",
                options=OPENAI_MODEL_OPTIONS,
                index=0,
                help="Choose a preset or override it with a custom model name below.",
            )

        with col_openai2:
            custom_openai_model = st.text_input(
                "Custom OpenAI model name (optional)",
                value="",
                placeholder="e.g. gpt-5 or another model ID",
            )

        llm_model_name = custom_openai_model.strip() or openai_model_preset

        prompt_regime = st.selectbox(
            "Prompting strategy",
            options=["zero_shot", "few_shot", "reasoning"],
            index=0,
            help="Controls which prompt template family is used when a YAML prompt is available.",
        )

        openai_api_key = st.text_input(
            "OpenAI API key",
            type="password",
            value="",
            help="Used only for this session unless you store it in Streamlit secrets or environment variables.",
        )

        if not openai_api_key:
            try:
                openai_api_key = st.secrets.get("OPENAI_API_KEY", "")
            except Exception:
                openai_api_key = ""

        if not openai_api_key:
            openai_api_key = os.getenv("OPENAI_API_KEY", "")

        if openai_api_key:
            st.caption("OpenAI API key detected.")
        else:
            st.caption("No OpenAI API key detected yet.")

# -------------------------------------------------------------------
# 2A. File upload UI
# -------------------------------------------------------------------
st.subheader("2) Dataset input")

uploaded_file = None
manual_meta_file = None
manual_meta_file_raw: Dict[str, Any] = {}
manual_meta_file_text = ""

trino_host = trino_port = trino_catalog = trino_schema = trino_user = trino_password = ""
trino_sql = ""
trino_meta_sql = ""
trino_metadata_raw: Dict[str, Any] = {}

if data_source == UPLOAD_MODE:
    uploaded_file = st.file_uploader(
        "Upload a dataset file",
        type=["csv", "tsv", "txt", "xls", "xlsx"],
        help="CSV and Excel are supported.",
    )

if data_source == TRINO_MODE:
    st.markdown(
        """
Connect to a Trino endpoint and run a data query.
Optionally, you can also run a separate metadata query that returns one row.
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
        help="Should return one metadata row with columns like title, description, publisher, metadata_created, etc.",
    )

st.subheader("3) Optional metadata")

meta_col1, meta_col2 = st.columns([1.7, 1])

with meta_col1:
    manual_meta_text = st.text_area(
        "Manual metadata (key: value per line)",
        height=160,
        help=(
            "You can provide either Vetrò symbols or common names.\n\n"
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

with meta_col2:
    manual_meta_file = st.file_uploader(
        "Metadata file (optional)",
        type=["txt", "json", "yaml", "yml"],
        help="Upload metadata as TXT, JSON, or YAML.",
    )

    if manual_meta_file is not None:
        manual_meta_file_raw, manual_meta_file_text = parse_uploaded_metadata_file(manual_meta_file)
        st.caption(f"Loaded metadata file: {manual_meta_file.name}")

# -------------------------------------------------------------------
# 3. Run button
# -------------------------------------------------------------------
st.subheader("4) Run assessment")

action_col1, action_col2 = st.columns([1, 1])

with action_col1:
    run_btn = st.button("Run assessment", type="primary")

with action_col2:
    with st.expander("Test AI connection", expanded=False):
        if st.button("Run connection test"):
            try:
                runner = get_llm_runner(
                    provider=llm_provider,
                    model_name=llm_model_name,
                    api_key=openai_api_key,
                )
                raw = runner("Return exactly 3 lines:\nanswer: 1\nconfidence: 0.9\nevidence: test", 64)
                st.success("LLM call worked")
                st.code(raw)
            except Exception as e:
                st.error(f"LLM test failed: {e}")

# -------------------------------------------------------------------
# 4. Main logic
# -------------------------------------------------------------------
if run_btn:
    try:
        if use_llm and llm_provider == "openai" and not openai_api_key:
            st.error("Please enter an OpenAI API key or configure OPENAI_API_KEY in Streamlit secrets/environment.")
            st.stop()

        df: Optional[pd.DataFrame] = None
        ext: Optional[str] = None
        trino_metadata: Dict[str, Any] = {}

        manual_metadata_prompt_source = "not_used"

        combined_manual_meta_text = "\n\n".join(
            part for part in [manual_meta_file_text, manual_meta_text] if (part or "").strip()
        )

        manual_metadata_raw = dict(manual_meta_file_raw)
        manual_metadata_raw.update(parse_kv_metadata(manual_meta_text))
        manual_metadata = normalize_metadata_to_symbols(manual_metadata_raw)

        manual_metadata_llm_raw = ""
        manual_metadata_llm = {}

        if use_llm and manual_meta_text.strip():
            try:
                manual_llm_runner = get_llm_runner(
                    provider=llm_provider,
                    model_name=llm_model_name,
                    api_key=openai_api_key,
                )
                with open(PROMPTS_YAML, "r", encoding="utf-8") as f:
                    prompts_cfg = yaml.safe_load(f) or {}
                
                manual_metadata_llm, manual_metadata_llm_raw, manual_metadata_prompt_source = infer_manual_metadata_symbols(
                    combined_manual_meta_text,
                    manual_llm_runner,
                    prompts_cfg=prompts_cfg,
                    prompt_regime=prompt_regime,
                )

                # explicit key:value wins, AI fills gaps
                merged_manual_metadata = dict(manual_metadata_llm)
                merged_manual_metadata.update(manual_metadata)
                manual_metadata = merged_manual_metadata

            except Exception:
                pass

        # prioriteet:
        # AI-first:
        # explicit key:value wins, AI fills the rest
        merged_manual_metadata = dict(manual_metadata_llm)
        merged_manual_metadata.update(manual_metadata)
        manual_metadata = merged_manual_metadata

        trino_metadata_raw: Dict[str, Any] = {}

        conn = None

        # -----------------------
        # A) File mode
        # -----------------------
        if data_source == UPLOAD_MODE:
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

            if trino_meta_sql.strip():
                try:
                    meta_df = pd.read_sql(trino_meta_sql, conn)
                    if not meta_df.empty:
                        meta_row = meta_df.iloc[0].to_dict()
                        trino_metadata_raw = meta_row
                        trino_metadata = normalize_metadata_to_symbols(trino_metadata_raw)
                except Exception as e:
                    st.warning(f"Metadata query failed (continuing without it): {e}")

            if not trino_metadata_raw:
                trino_metadata_raw = manual_metadata_raw

        # -----------------------
        # Sanity check
        # -----------------------
        if df is None:
            st.error("No data could be loaded from the selected data source.")
            st.stop()

        df = make_arrow_safe(df)

        # Preview
        st.subheader("Preview of the dataset")
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
                prompt_regime=prompt_regime,
                use_llm=use_llm,
                llm_provider=llm_provider,
                llm_model_name=llm_model_name,
                openai_api_key=openai_api_key,
                file_ext=ext,
                manual_metadata=manual_metadata,
                manual_metadata_text=manual_meta_text,
                trino_metadata=trino_metadata if data_source == TRINO_MODE else {},
                trino_metadata_raw=trino_metadata_raw if data_source == TRINO_MODE else {},
            )

        metrics_df["value"] = metrics_df["value"].apply(
            lambda x: float(x) if isinstance(x, (int, float)) else None
        )

        st.subheader("Results overview")

        if metrics_df.empty or metrics_df["value"].dropna().empty:
            st.info("No metrics could be computed.")
        else:
            metrics_non_null = metrics_df.dropna(subset=["value"]).copy()
            metrics_non_null["value_clamped"] = metrics_non_null["value"].clip(0.0, 1.0)

            dimension_scores = (
                metrics_non_null.groupby("dimension", as_index=False)["value_clamped"]
                .mean()
                .sort_values("value_clamped", ascending=False)
            )

            overall_score = metrics_non_null["value_clamped"].mean()
            best_dimension = dimension_scores.iloc[0]
            weakest_dimension = dimension_scores.iloc[-1]
            weakest_metric = metrics_non_null.sort_values("value_clamped", ascending=True).iloc[0]

            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            kpi1.metric("Overall score", f"{overall_score:.2f}")
            kpi2.metric("Best dimension", str(best_dimension["dimension"]), f"{best_dimension['value_clamped']:.2f}")
            kpi3.metric("Lowest dimension", str(weakest_dimension["dimension"]), f"{weakest_dimension['value_clamped']:.2f}")
            kpi4.metric("Lowest metric", str(weakest_metric["metric_label"]), f"{weakest_metric['value_clamped']:.2f}")

            st.subheader("Average score by dimension")
            dim_fig = px.bar(
                dimension_scores,
                x="dimension",
                y="value_clamped",
                range_y=[0, 1],
                labels={
                    "dimension": "Dimension",
                    "value_clamped": "Average normalised value (0–1)",
                },
            )
            st.plotly_chart(dim_fig, width="stretch")

            st.subheader("Metric-level results")
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

            st.subheader("Suggested next improvements")

            ai_recommendations = ""
            if use_llm:
                try:
                    reco_runner = get_llm_runner(
                        provider=llm_provider,
                        model_name=llm_model_name,
                        api_key=openai_api_key,
                    )
                    ai_recommendations = reco_runner(
                        build_ai_recommendation_prompt(metrics_non_null, data_source),
                        350,
                    ).strip()
                except Exception:
                    ai_recommendations = ""

            if ai_recommendations:
                st.markdown(ai_recommendations)
            else:
                fallback_recommendations = build_quality_recommendations(metrics_non_null)
                if fallback_recommendations:
                    for item in fallback_recommendations:
                        st.markdown(f"- {item}")
                else:
                    st.caption("No specific recommendations were generated.")
        # Debug
        st.write("LLM calls:", len(details.get("llm_debug", {}).get("calls", [])))        
        st.markdown("**Manual metadata prompt source:**")
        st.write(manual_metadata_prompt_source)
        st.markdown("**Prompt sources:**")
        st.json(details.get("prompt_sources", {}))
        with st.expander("Debug: auto-derived inputs and AI/metadata inferences"):
            st.markdown("**Auto-derived inputs (from the table only):**")
            st.json(details.get("auto_inputs", {}))

            if data_source == TRINO_MODE:
                st.markdown("**Trino metadata (normalised to symbols):**")
                st.json(trino_metadata)

            st.markdown("**Manual metadata (normalised to symbols):**")
            st.json(manual_metadata)
            if manual_metadata_llm:
                st.markdown("**Manual metadata interpreted by AI:**")
                st.json(manual_metadata_llm)

            if manual_metadata_llm_raw:
                st.markdown("**Manual metadata AI raw output:**")
                st.code(manual_metadata_llm_raw, language="json")

            st.markdown("**LLM debug info:**")
            st.json(details.get("llm_debug", {}))

            st.markdown("**LLM raw outputs:**")
            st.json(details.get("llm_raw", {}))

            st.markdown("**LLM confidences:**")
            st.json(details.get("llm_confidence", {}))

            st.markdown("**LLM evidence:**")
            st.json(details.get("llm_evidence", {}))

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
                debug_df["value"] = debug_df["value"].astype(str)
                debug_df["llm_confidence"] = pd.to_numeric(debug_df["llm_confidence"], errors="coerce")
                debug_df["llm_evidence"] = debug_df["llm_evidence"].astype(str)

                st.dataframe(debug_df, width="stretch")

    except Exception as e:
        st.exception(e)