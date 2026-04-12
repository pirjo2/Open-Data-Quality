from __future__ import annotations

import io
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import plotly.express as px
import streamlit as st
import yaml

from core.llm import get_llm_runner, infer_manual_metadata_symbols
from core.pipeline import run_quality_assessment
from core.utils import make_arrow_safe


FORMULAS_YAML = "configs/formulas.yaml"
PROMPTS_YAML = "configs/prompts.yaml"

HF_MODEL_OPTIONS = [
    "google/flan-t5-small",
    "google/flan-t5-base",
    "google/flan-t5-large",
]

OPENAI_MODEL_OPTIONS = [
    "gpt-4.1-mini",
    "gpt-4.1",
    "gpt-5-mini",
    "gpt-5",
]

UPLOAD_MODE = "Upload file"
TRINO_MODE = "Trino SQL query (advanced)"

APP_AUTHOR = "Pirjo Vainjärv"
APP_SUPERVISOR = "Kristo Raun"
COMMON_FILE_TYPES = ["csv", "tsv", "txt", "xls", "xlsx", "json", "yaml", "yml"]
SUPPORTED_FILE_TYPES_HELP = "Supported formats: CSV, TSV, TXT, XLS, XLSX, JSON, YAML, YML."
METADATA_FILE_HELP = (
    "Supported formats: CSV, TSV, TXT, XLS, XLSX, JSON, YAML, YML. "
    "For tabular metadata, use a 2-column key/value file or a single-row table."
)

DIMENSION_ORDER = [
    "traceability",
    "currentness",
    "expiration",
    "completeness",
    "compliance",
    "understandability",
    "accuracy",
]

DEFAULT_RECOMMENDATION_TEMPLATE = """
You are reviewing open data quality results.
Dataset source mode: {data_source}
Weakest metrics:
{summary_yaml}

Write exactly 5 short markdown bullet points.
Each bullet must:
- explain what to improve
- be concrete and actionable
- stay under 24 words
Do not mention confidence.
Do not mention that you are an AI.
""".strip()


def inject_css() -> None:
    st.markdown(
        """
        <style>
            .block-container {
                padding-top: 1.5rem;
                padding-bottom: 2rem;
                max-width: 1350px;
            }
            .top-note {
                padding: 0.95rem 1.1rem;
                border: 1px solid rgba(120, 120, 120, 0.25);
                border-radius: 16px;
                background: rgba(250, 250, 250, 0.45);
                margin-bottom: 1rem;
            }
            .section-title {
                font-size: 1.02rem;
                font-weight: 700;
                margin-bottom: 0.2rem;
            }
            .section-subtitle {
                color: rgba(49, 51, 63, 0.75);
                font-size: 0.92rem;
                margin-bottom: 0.8rem;
            }
            .summary-card {
                border: 1px solid rgba(120, 120, 120, 0.22);
                border-radius: 18px;
                padding: 0.9rem 1rem;
                background: white;
                min-height: 110px;
            }
            .summary-label {
                font-size: 0.82rem;
                color: rgba(49, 51, 63, 0.7);
                margin-bottom: 0.2rem;
            }
            .summary-value {
                font-size: 1.55rem;
                font-weight: 700;
                line-height: 1.15;
            }
            .summary-small {
                font-size: 0.9rem;
                margin-top: 0.35rem;
                color: rgba(49, 51, 63, 0.76);
            }
            .dimension-pill {
                border: 1px solid rgba(120,120,120,0.22);
                border-radius: 14px;
                padding: 0.75rem 0.85rem;
                background: white;
                margin-bottom: 0.65rem;
            }
            .dimension-pill strong {
                display: block;
                font-size: 0.95rem;
                margin-bottom: 0.15rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def humanize_dimension(value: str) -> str:
    mapping = {
        "traceability": "Traceability",
        "currentness": "Currentness",
        "expiration": "Expiration",
        "completeness": "Completeness",
        "compliance": "Compliance",
        "understandability": "Understandability",
        "accuracy": "Accuracy",
    }
    return mapping.get(str(value).strip().lower(), str(value).replace("_", " ").title())


def parse_kv_metadata(text: str) -> Dict[str, Any]:
    meta: Dict[str, Any] = {}
    for line in (text or "").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        try:
            meta[key] = float(value)
        except Exception:
            meta[key] = value
    return meta


def normalize_metadata_to_symbols(meta: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    meta_l = {str(k).strip().lower(): v for k, v in (meta or {}).items()}
    direct_symbols = {
        "pb",
        "t",
        "d",
        "dc",
        "cv",
        "l",
        "id",
        "s",
        "c",
        "dp",
        "sd",
        "edp",
        "ed",
        "cd",
        "lu",
        "du",
    }
    for key, value in meta_l.items():
        if key in direct_symbols:
            out[key] = value

    def _is_explicit_missing(x: Any) -> bool:
        if x is None:
            return True
        if isinstance(x, float) and pd.isna(x):
            return True
        if isinstance(x, str):
            return x.strip().lower() in {
                "",
                "none",
                "null",
                "missing",
                "n/a",
                "na",
                "[]",
                "not provided",
                "not available",
            }
        return False

    def _presence_from_key(meta_dict: Dict[str, Any], *keys: str) -> Optional[float]:
        for key in keys:
            if key in meta_dict:
                return 0.0 if _is_explicit_missing(meta_dict[key]) else 1.0
        return None

    semantic_key_map = {
        "t": ("title",),
        "d": ("description", "notes"),
        "pb": ("publisher", "organization", "org_name"),
        "s": ("source",),
        "dc": ("metadata_created", "issued", "created", "date_of_creation"),
        "cv": ("coverage", "temporalcoverage"),
        "l": ("language", "lang"),
        "id": ("identifier", "dataset_id", "datasetidentifier"),
        "c": ("category", "theme"),
    }
    for target_key, source_keys in semantic_key_map.items():
        if target_key not in out:
            value = _presence_from_key(meta_l, *source_keys)
            if value is not None:
                out[target_key] = value

    if "dp" not in out:
        for key in ["date_of_publication", "metadata_modified", "modified", "releasedate", "modificationdate"]:
            if key in meta_l and str(meta_l[key]).strip():
                out["dp"] = str(meta_l[key])
                break

    for field in ["sd", "edp", "ed", "cd"]:
        if field not in out and field in meta_l:
            out[field] = str(meta_l[field])

    if "lu" not in out and "updates" in meta_l:
        out["lu"] = 0.0 if _is_explicit_missing(meta_l["updates"]) else 1.0
    if "du" not in out and "updates" in meta_l:
        out["du"] = 0.0 if _is_explicit_missing(meta_l["updates"]) else None

    return {k: v for k, v in out.items() if v is not None}


def dataframe_to_metadata_dict(df: pd.DataFrame) -> Dict[str, Any]:
    if df is None or df.empty:
        return {}

    clean_df = df.dropna(how="all").copy()
    if clean_df.empty:
        return {}

    clean_df.columns = [str(col).strip() for col in clean_df.columns]
    lower_cols = [col.lower() for col in clean_df.columns]

    key_aliases = {"key", "field", "name", "symbol", "parameter"}
    value_aliases = {"value", "content", "data", "answer"}

    key_col = None
    value_col = None
    for col in clean_df.columns:
        if col.lower() in key_aliases and key_col is None:
            key_col = col
        if col.lower() in value_aliases and value_col is None:
            value_col = col

    if key_col and value_col:
        out: Dict[str, Any] = {}
        for _, row in clean_df[[key_col, value_col]].dropna(how="all").iterrows():
            key = str(row.get(key_col, "")).strip()
            value = row.get(value_col)
            if key:
                out[key] = value
        return out

    if len(clean_df.columns) == 2:
        first_col, second_col = clean_df.columns[:2]
        out = {}
        for _, row in clean_df[[first_col, second_col]].dropna(how="all").iterrows():
            key = str(row.get(first_col, "")).strip()
            value = row.get(second_col)
            if key:
                out[key] = value
        if out:
            return out

    if len(clean_df) == 1:
        row = clean_df.iloc[0].dropna()
        return {str(k).strip(): v for k, v in row.to_dict().items() if str(k).strip()}

    return {}


def parse_text_metadata_content(text: str) -> Dict[str, Any]:
    clean_text = (text or "").strip()
    if not clean_text:
        return {}

    try:
        parsed = json.loads(clean_text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    try:
        parsed = yaml.safe_load(clean_text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    return parse_kv_metadata(clean_text)


def parse_uploaded_metadata_file(uploaded) -> Tuple[Dict[str, Any], str]:
    if uploaded is None:
        return {}, ""

    raw_bytes = uploaded.getvalue()
    suffix = Path(uploaded.name).suffix.lower()

    if suffix in {".csv", ".tsv", ".txt"}:
        text_content = raw_bytes.decode("utf-8", errors="ignore")
        if suffix in {".csv", ".tsv"}:
            try:
                df = pd.read_csv(io.BytesIO(raw_bytes), sep=None, engine="python")
                parsed = dataframe_to_metadata_dict(df)
                return parsed, text_content
            except Exception:
                return parse_text_metadata_content(text_content), text_content
        return parse_text_metadata_content(text_content), text_content

    if suffix in {".xls", ".xlsx"}:
        df = pd.read_excel(io.BytesIO(raw_bytes))
        text_content = df.to_csv(index=False)
        return dataframe_to_metadata_dict(df), text_content

    if suffix == ".json":
        text_content = raw_bytes.decode("utf-8", errors="ignore")
        try:
            parsed = json.loads(text_content)
            if isinstance(parsed, dict):
                return parsed, text_content
            if isinstance(parsed, list):
                df = pd.json_normalize(parsed)
                return dataframe_to_metadata_dict(df), text_content
        except Exception:
            pass
        return {}, text_content

    if suffix in {".yaml", ".yml"}:
        text_content = raw_bytes.decode("utf-8", errors="ignore")
        try:
            parsed = yaml.safe_load(text_content) or {}
            if isinstance(parsed, dict):
                return parsed, text_content
            if isinstance(parsed, list):
                df = pd.json_normalize(parsed)
                return dataframe_to_metadata_dict(df), text_content
        except Exception:
            pass
        return {}, text_content

    return {}, raw_bytes.decode("utf-8", errors="ignore")


def parse_manual_metadata_text(text: str) -> Dict[str, Any]:
    return parse_text_metadata_content(text)


def parse_uploaded_dataset_file(uploaded_file) -> Tuple[pd.DataFrame, str]:
    name = uploaded_file.name
    ext = Path(name).suffix.lower()

    if ext in {".csv", ".tsv", ".txt"}:
        return pd.read_csv(uploaded_file, sep=None, engine="python"), ext

    if ext in {".xls", ".xlsx"}:
        return pd.read_excel(uploaded_file), ext

    if ext == ".json":
        raw = uploaded_file.getvalue().decode("utf-8", errors="ignore")
        payload = json.loads(raw)
        if isinstance(payload, list):
            return pd.json_normalize(payload), ext
        if isinstance(payload, dict):
            for key in ["data", "items", "records", "results", "rows"]:
                if isinstance(payload.get(key), list):
                    return pd.json_normalize(payload[key]), ext
            return pd.json_normalize([payload]), ext
        raise ValueError("Unsupported JSON structure. Use a list of records or an object containing data/items/records/results/rows.")

    if ext in {".yaml", ".yml"}:
        raw = uploaded_file.getvalue().decode("utf-8", errors="ignore")
        payload = yaml.safe_load(raw)
        if isinstance(payload, list):
            return pd.json_normalize(payload), ext
        if isinstance(payload, dict):
            for key in ["data", "items", "records", "results", "rows"]:
                if isinstance(payload.get(key), list):
                    return pd.json_normalize(payload[key]), ext
            return pd.json_normalize([payload]), ext
        raise ValueError("Unsupported YAML structure. Use a list of records or an object containing data/items/records/results/rows.")

    raise ValueError(f"Unsupported file type: {ext}")


def load_prompts_cfg(path: str = PROMPTS_YAML) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_prompt_template(cfg: Dict[str, Any], regime: str, name: str, fallback: str = "") -> str:
    return (
        cfg.get("prompt_regimes", {})
        .get(regime, {})
        .get(name, fallback)
    ) or fallback



def build_ai_recommendation_prompt(
    metrics_df: pd.DataFrame,
    data_source: str,
    template: str = DEFAULT_RECOMMENDATION_TEMPLATE,
) -> str:
    weakest = (
        metrics_df.sort_values("value_clamped", ascending=True)
        .head(5)[["dimension", "metric_label", "metric_id", "value_clamped"]]
        .to_dict(orient="records")
    )
    summary_yaml = yaml.safe_dump(weakest, sort_keys=False, allow_unicode=True)
    return template.format(data_source=data_source, summary_yaml=summary_yaml)


def build_quality_recommendations(metrics_df: pd.DataFrame) -> List[str]:
    if metrics_df.empty:
        return []

    guidance = {
        "traceability.track_of_creation": "Add a clear data source and an explicit creation date in the metadata.",
        "traceability.track_of_updates": "Add update dates or a simple change log to document dataset updates.",
        "currentness.percentage_of_current_rows": "Review time-related rows and clearly separate current and outdated records.",
        "currentness.delay_in_publication": "Publish the dataset closer to the end of the reference period and include a publication date.",
        "currentness.delay_after_expiration": "Refresh, archive, or replace expired datasets faster.",
        "expiration.delay_after_expiration": "Refresh, archive, or replace expired datasets faster.",
        "completeness.percentage_of_complete_cells": "Reduce missing values in key fields before publication.",
        "completeness.percentage_of_complete_rows": "Improve row-level completeness for the most important records.",
        "compliance.percentage_of_standardized_columns": "Use standardised formats, identifiers, and code lists where possible.",
        "compliance.egms_compliance": "Add richer metadata such as title, description, publisher, identifier, language, category, source, and coverage.",
        "compliance.five_stars_open_data": "Prefer machine-readable non-proprietary formats and publish reusable linked identifiers where relevant.",
        "understandability.percentage_of_columns_with_metadata": "Add a column-level data dictionary or field descriptions.",
        "understandability.percentage_of_columns_in_comprehensible_format": "Rename unclear columns and explain abbreviations or coded values.",
        "accuracy.percentage_of_syntactically_accurate_cells": "Validate date, code, and numeric formats before publishing.",
        "accuracy.accuracy_in_aggregation": "Cross-check totals and aggregates against row-level values.",
    }

    low_metrics = (
        metrics_df.dropna(subset=["value_clamped"])
        .sort_values("value_clamped", ascending=True)
        .head(5)
    )

    recommendations: List[str] = []
    for _, row in low_metrics.iterrows():
        metric_id = str(row.get("metric_id", "")).strip()
        message = guidance.get(metric_id)
        if message:
            recommendations.append(f"{row['metric_label']} ({row['value_clamped']:.2f}): {message}")
    return recommendations


def apply_result_dimension_overrides(metrics_df: pd.DataFrame) -> pd.DataFrame:
    if metrics_df is None or metrics_df.empty:
        return metrics_df

    adjusted = metrics_df.copy()
    mask = adjusted["metric_id"].astype(str).eq("currentness.delay_after_expiration")
    adjusted.loc[mask, "dimension"] = "expiration"
    adjusted.loc[mask & adjusted["metric_label"].astype(str).eq("Freshness after expiration"), "metric_label"] = "Delay after expiration"
    return adjusted


def render_summary_cards(
    overall_score: float,
    best_dimension: pd.Series,
    weakest_dimension: pd.Series,
    weakest_metric: pd.Series,
) -> None:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            f"""
            <div class="summary-card">
                <div class="summary-label">Overall score</div>
                <div class="summary-value">{overall_score:.2f}</div>
                <div class="summary-small">Average normalised score</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
            <div class="summary-card">
                <div class="summary-label">Strongest dimension</div>
                <div class="summary-value">{humanize_dimension(best_dimension['dimension'])}</div>
                <div class="summary-small">Score: {best_dimension['value_clamped']:.2f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f"""
            <div class="summary-card">
                <div class="summary-label">Weakest dimension</div>
                <div class="summary-value">{humanize_dimension(weakest_dimension['dimension'])}</div>
                <div class="summary-small">Score: {weakest_dimension['value_clamped']:.2f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            f"""
            <div class="summary-card">
                <div class="summary-label">Lowest metric</div>
                <div class="summary-value" style="font-size:1.15rem;">{weakest_metric['metric_label']}</div>
                <div class="summary-small">Score: {weakest_metric['value_clamped']:.2f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_dimension_tiles(dimension_scores: pd.DataFrame) -> None:
    st.markdown("### Dimension overview")
    score_map = {row["dimension"]: row["value_clamped"] for _, row in dimension_scores.iterrows()}
    rows = [DIMENSION_ORDER[:3], DIMENSION_ORDER[3:6], DIMENSION_ORDER[6:]]
    for row_dimensions in rows:
        cols = st.columns(len(row_dimensions))
        for col, dim in zip(cols, row_dimensions):
            with col:
                value = score_map.get(dim)
                if value is None:
                    label = "Not available"
                else:
                    label = f"{value:.2f}"
                st.markdown(
                    f"""
                    <div class="dimension-pill">
                        <strong>{humanize_dimension(dim)}</strong>
                        <span>{label}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


def get_metadata_placeholder() -> str:
    return "publisher: Ministry\ntitle: Dataset name\nmetadata_created: 2024-01-15\nmetadata_modified: 2024-03-01"


st.set_page_config(page_title="Open Data Quality", layout="wide")
inject_css()

st.title("Open Data Quality")
with st.expander("More info", expanded=False):
    st.markdown(
        f"""
        **Author:** {APP_AUTHOR}  
        **Supervisor:** {APP_SUPERVISOR}

        This Streamlit prototype evaluates open datasets with the Vetrò et al. (2016) framework and an AI-assisted workflow.

        It combines table-based checks, optional metadata inputs, and prompt-based semantic inference for missing signals.
        """
    )

st.subheader("Data source")
data_source = st.radio(
    "Data source",
    options=[UPLOAD_MODE, TRINO_MODE],
    index=0,
    horizontal=True,
    label_visibility="collapsed",
)

base_prompts_cfg = load_prompts_cfg(PROMPTS_YAML)
prompt_regime_options = list((base_prompts_cfg.get("prompt_regimes") or {}).keys()) or ["few_shot"]
default_prompt_index = prompt_regime_options.index("few_shot") if "few_shot" in prompt_regime_options else 0
prompt_regime = prompt_regime_options[default_prompt_index]

st.subheader("Dataset input")
uploaded_file = None
trino_host = ""
trino_port = 443
trino_catalog = ""
trino_schema = ""
trino_user = ""
trino_password = ""
trino_sql = ""
trino_meta_sql = ""

if data_source == UPLOAD_MODE:
    uploaded_file = st.file_uploader(
        "Input file",
        type=COMMON_FILE_TYPES,
        help=SUPPORTED_FILE_TYPES_HELP,
    )
else:
    trino_left_col, trino_right_col = st.columns([1, 1.6], gap="large")

    with trino_left_col:
        trino_host = st.text_input("Host", value="trino.avaandmeait.ee")
        trino_port = st.number_input("Port", min_value=1, max_value=65535, value=443)
        trino_catalog = st.text_input("Catalog", value="")
        trino_schema = st.text_input("Schema", value="")
        trino_user = st.text_input("Username", value="")
        trino_password = st.text_input("Password", value="", type="password")

    with trino_right_col:
        trino_sql = st.text_area(
            "Data query",
            height=220,
            placeholder="SELECT * FROM some_table LIMIT 100000",
        )
        trino_meta_sql = st.text_area(
            "Metadata query",
            height=160,
            placeholder=(
                "SELECT\n"
                "  title,\n"
                "  notes AS description,\n"
                "  metadata_created,\n"
                "  metadata_modified,\n"
                "  organization.name AS publisher\n"
                "FROM landing.avaandmete_portaal.dataset_metadata\n"
                "LIMIT 1"
            ),
        )

st.subheader("AI settings")
use_llm = True
ai_row_1a, ai_row_1b, ai_row_1c = st.columns([1, 1, 1], gap="large")

with ai_row_1a:
    row_limit = st.number_input(
        "Rows",
        min_value=0,
        value=0 if data_source == TRINO_MODE else 500_000,
        step=10_000,
        help="0 means all rows for uploaded files. For Trino, the query itself controls the row count.",
        disabled=data_source == TRINO_MODE,
    )

with ai_row_1b:
    llm_provider = st.selectbox(
        "AI provider",
        options=["openai", "huggingface"],
        index=0,
    )

with ai_row_1c:
    openai_api_key: Optional[str] = None
    if llm_provider == "openai":
        llm_model_name = st.selectbox(
            "Model name",
            options=OPENAI_MODEL_OPTIONS,
            index=0,
        )
    else:
        llm_model_name = st.selectbox(
            "Model name",
            options=HF_MODEL_OPTIONS,
            index=0,
        )

ai_row_2a, ai_row_2b = st.columns([4, 1.5], gap="large")
with ai_row_2a:
    if llm_provider == "openai":
        openai_api_key = st.text_input("OpenAI API key", type="password", value="")
        if not openai_api_key:
            try:
                openai_api_key = st.secrets.get("OPENAI_API_KEY", "")
            except Exception:
                openai_api_key = ""
        if not openai_api_key:
            openai_api_key = os.getenv("OPENAI_API_KEY", "")
    else:
        st.text_input("API key", value="Not required for local Hugging Face setup.", disabled=True)

with ai_row_2b:
    st.markdown("<div style='height: 1.9rem;'></div>", unsafe_allow_html=True)
    test_clicked = st.button("Test connection", use_container_width=True)
    if test_clicked:
        try:
            runner = get_llm_runner(
                provider=llm_provider,
                model_name=llm_model_name,
                api_key=openai_api_key,
            )
            raw = runner("Return exactly 3 lines:\\nanswer: 1\\nconfidence: 0.9\\nevidence: test", 64)
            st.success("Connection worked.")
            st.code(raw)
        except Exception as exc:
            st.error(f"Connection test failed: {exc}")

st.subheader("Optional metadata")
meta_upload_col, meta_text_col = st.columns([1, 2], gap="large")

with meta_upload_col:
    manual_meta_file = st.file_uploader(
        "Metadata file",
        type=COMMON_FILE_TYPES,
        help=METADATA_FILE_HELP,
    )

with meta_text_col:
    manual_meta_text = st.text_area(
        "Metadata text",
        height=220,
        placeholder=get_metadata_placeholder(),
        help="Paste metadata as key: value pairs, JSON, or YAML.",
    )

with st.expander("Advanced settings", expanded=False):
    prompt_regime = st.selectbox(
        "Prompting strategy",
        options=prompt_regime_options,
        index=default_prompt_index,
        help="Select the prompt family loaded from prompts.yaml.",
    )

st.markdown("### Run assessment")
run_btn = st.button("Run assessment", type="primary")

if run_btn:
    prompts_yaml_path = PROMPTS_YAML
    conn = None
    try:
        if llm_provider == "openai" and not openai_api_key:
            st.error("Please enter an OpenAI API key or configure OPENAI_API_KEY in Streamlit secrets or environment variables.")
            st.stop()

        manual_meta_file_raw, manual_meta_file_text = parse_uploaded_metadata_file(manual_meta_file)
        manual_text_raw = parse_manual_metadata_text(manual_meta_text)

        combined_manual_meta_text = "\n\n".join(
            part for part in [manual_meta_file_text, manual_meta_text] if (part or "").strip()
        )

        manual_metadata_raw = dict(manual_meta_file_raw)
        manual_metadata_raw.update(manual_text_raw)
        manual_metadata = normalize_metadata_to_symbols(manual_metadata_raw)

        manual_metadata_prompt_source = "not_used"
        manual_metadata_llm_raw = ""
        manual_metadata_llm: Dict[str, Any] = {}

        if use_llm and combined_manual_meta_text.strip():
            try:
                llm_runner = get_llm_runner(
                    provider=llm_provider,
                    model_name=llm_model_name,
                    api_key=openai_api_key,
                )
                prompts_cfg_runtime = load_prompts_cfg(prompts_yaml_path)
                manual_metadata_llm, manual_metadata_llm_raw, manual_metadata_prompt_source = infer_manual_metadata_symbols(
                    combined_manual_meta_text,
                    llm_runner,
                    prompts_cfg=prompts_cfg_runtime,
                    prompt_regime=prompt_regime,
                )
                merged_manual_metadata = dict(manual_metadata_llm)
                merged_manual_metadata.update(manual_metadata)
                manual_metadata = merged_manual_metadata
            except Exception:
                pass

        df: Optional[pd.DataFrame] = None
        ext: Optional[str] = None
        trino_metadata: Dict[str, Any] = {}
        trino_metadata_raw: Dict[str, Any] = {}

        if data_source == UPLOAD_MODE:
            if uploaded_file is None:
                st.error("Please upload a dataset file first.")
                st.stop()
            df, ext = parse_uploaded_dataset_file(uploaded_file)
            if row_limit and row_limit > 0:
                df = df.head(int(row_limit))
        else:
            if not trino_sql.strip():
                st.error("Please enter a Trino data query.")
                st.stop()
            if not trino_host.strip() or not trino_user.strip():
                st.error("Please enter at least Trino host and username.")
                st.stop()

            try:
                from trino.auth import BasicAuthentication
                from trino.dbapi import connect as trino_connect
            except Exception as exc:
                st.error(f"The 'trino' package is required for Trino mode. Import error: {exc}")
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

            conn = trino_connect(**conn_kwargs)
            df = pd.read_sql(trino_sql, conn)
            ext = ".sql"

            if trino_meta_sql.strip():
                try:
                    meta_df = pd.read_sql(trino_meta_sql, conn)
                    if not meta_df.empty:
                        trino_metadata_raw = meta_df.iloc[0].to_dict()
                        trino_metadata = normalize_metadata_to_symbols(trino_metadata_raw)
                except Exception as exc:
                    st.warning(f"Metadata query failed, continuing without it: {exc}")

        if df is None:
            st.error("No data could be loaded from the selected data source.")
            st.stop()

        df = make_arrow_safe(df)
        if ext is None:
            ext = ".table"

        with st.spinner("Computing quality metrics..."):
            metrics_df, details = run_quality_assessment(
                df=df,
                formulas_yaml_path=FORMULAS_YAML,
                prompts_yaml_path=prompts_yaml_path,
                prompt_regime=prompt_regime,
                use_llm=use_llm,
                llm_provider=llm_provider,
                llm_model_name=llm_model_name,
                openai_api_key=openai_api_key,
                file_ext=ext,
                manual_metadata=manual_metadata,
                manual_metadata_text=combined_manual_meta_text,
                trino_metadata=trino_metadata if data_source == TRINO_MODE else {},
                trino_metadata_raw=trino_metadata_raw if data_source == TRINO_MODE else {},
            )

        metrics_df = apply_result_dimension_overrides(metrics_df)
        metrics_df["value"] = metrics_df["value"].apply(
            lambda x: float(x) if isinstance(x, (int, float)) else None
        )

        for col in df.columns:
            if df[col].dtype == "object":
                df[col] = df[col].apply(lambda x: str(x) if isinstance(x, (dict, list, tuple)) else x)

        st.markdown("---")
        st.markdown("## Results")

        preview_tab, overview_tab, detail_tab, debug_tab = st.tabs(
            ["Preview", "Overview", "Detailed metrics", "Debug"]
        )

        with preview_tab:
            st.subheader("Preview of the dataset")
            st.dataframe(df.head(20), use_container_width=True)
            st.caption(f"{len(df)} rows × {len(df.columns)} columns used for metrics.")

        with overview_tab:
            if metrics_df.empty or metrics_df["value"].dropna().empty:
                st.info("No metrics could be computed.")
            else:
                metrics_non_null = metrics_df.dropna(subset=["value"]).copy()
                metrics_non_null["value_clamped"] = metrics_non_null["value"].clip(0.0, 1.0)
                metrics_non_null["dimension"] = pd.Categorical(
                    metrics_non_null["dimension"],
                    categories=DIMENSION_ORDER,
                    ordered=True,
                )

                dimension_scores = (
                    metrics_non_null.groupby("dimension", as_index=False)["value_clamped"]
                    .mean()
                    .sort_values("dimension")
                )
                dimension_scores = dimension_scores.dropna(subset=["dimension"])

                overall_score = metrics_non_null["value_clamped"].mean()
                best_dimension = dimension_scores.sort_values("value_clamped", ascending=False).iloc[0]
                weakest_dimension = dimension_scores.sort_values("value_clamped", ascending=True).iloc[0]
                weakest_metric = metrics_non_null.sort_values("value_clamped", ascending=True).iloc[0]

                render_summary_cards(overall_score, best_dimension, weakest_dimension, weakest_metric)
                st.markdown("")
                render_dimension_tiles(dimension_scores)

                chart_col, reco_col = st.columns([1.2, 0.9], gap="large")
                with chart_col:
                    st.markdown("### Average score by category")
                    dim_fig = px.bar(
                        dimension_scores,
                        x="value_clamped",
                        y="dimension",
                        orientation="h",
                        range_x=[0, 1],
                        labels={
                            "dimension": "Category",
                            "value_clamped": "Average normalised value (0–1)",
                        },
                    )
                    dim_fig.update_yaxes(
                        tickvals=dimension_scores["dimension"].tolist(),
                        ticktext=[humanize_dimension(x) for x in dimension_scores["dimension"].tolist()],
                    )
                    dim_fig.update_layout(height=420)
                    st.plotly_chart(dim_fig, use_container_width=True)

                with reco_col:
                    st.markdown("### Suggested next improvements")
                    ai_recommendations = ""
                    if use_llm:
                        try:
                            reco_runner = get_llm_runner(
                                provider=llm_provider,
                                model_name=llm_model_name,
                                api_key=openai_api_key,
                            )
                            ai_recommendations = reco_runner(
                                build_ai_recommendation_prompt(
                                    metrics_non_null,
                                    data_source,
                                ),
                                350,
                            ).strip()
                        except Exception:
                            ai_recommendations = ""

                    with st.container(border=True):
                        if ai_recommendations:
                            st.markdown(ai_recommendations)
                        else:
                            fallback_recommendations = build_quality_recommendations(metrics_non_null)
                            if fallback_recommendations:
                                for item in fallback_recommendations:
                                    st.markdown(f"- {item}")
                            else:
                                st.caption("No specific recommendations were generated.")

        with detail_tab:
            metrics_non_null = metrics_df.dropna(subset=["value"]).copy()
            if metrics_non_null.empty:
                st.info("No detailed metric table is available.")
            else:
                metrics_non_null["value_clamped"] = metrics_non_null["value"].clip(0.0, 1.0)
                metrics_non_null["dimension"] = metrics_non_null["dimension"].astype(str)
                metrics_non_null["dimension_display"] = metrics_non_null["dimension"].apply(humanize_dimension)

                metric_fig = px.bar(
                    metrics_non_null.sort_values(["dimension", "metric_label"]),
                    x="value_clamped",
                    y="metric_label",
                    color="dimension_display",
                    orientation="h",
                    range_x=[0, 1],
                    labels={
                        "metric_label": "Metric",
                        "value_clamped": "Normalised value (0–1)",
                        "dimension_display": "Category",
                    },
                )
                metric_fig.update_layout(height=680)
                st.plotly_chart(metric_fig, use_container_width=True)

                st.dataframe(
                    metrics_non_null[["dimension_display", "metric_label", "value", "metric_id"]]
                    .rename(columns={"dimension_display": "dimension"})
                    .sort_values(["dimension", "metric_id"]),
                    use_container_width=True,
                )

        with debug_tab:
            st.markdown("**Auto-derived inputs / inferred symbols**")
            st.json(details.get("raw_inputs", {}))
            st.markdown("**Metric evaluation details**")
            st.dataframe(metrics_df, use_container_width=True)
            st.markdown("**Prompt regime used**")
            st.write(prompt_regime)
            st.markdown("**LLM calls**")
            st.write(len(details.get("llm_debug", {}).get("calls", [])))
            st.markdown("**Manual metadata prompt source**")
            st.write(manual_metadata_prompt_source)
            st.markdown("**Manual metadata (raw)**")
            st.json(manual_metadata_raw)
            st.markdown("**Manual metadata (normalised to symbols)**")
            st.json(manual_metadata)
            if data_source == TRINO_MODE:
                st.markdown("**Trino metadata (normalised to symbols)**")
                st.json(trino_metadata)
            if manual_metadata_llm_raw:
                st.markdown("**Manual metadata LLM raw output**")
                st.code(manual_metadata_llm_raw)
            llm_calls = details.get("llm_debug", {}).get("calls", [])
            if llm_calls:
                st.markdown("**LLM call details**")
                st.json(llm_calls)

    except Exception as exc:
        st.error(f"Assessment failed: {exc}")
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
