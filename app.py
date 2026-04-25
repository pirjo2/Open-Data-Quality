from __future__ import annotations

import io
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import streamlit.components.v1 as components

import pandas as pd
import plotly.express as px
import streamlit as st
import yaml

from core.llm import get_llm_runner, infer_manual_metadata_symbols
from core.metadata_utils import (
    dataframe_to_metadata_dict,
    extract_symbols_from_realistic_text,
    normalize_metadata_to_symbols,
    parse_kv_metadata,
    parse_text_metadata_content,
)
from core.pipeline import run_quality_assessment
from core.utils import make_arrow_safe

from datetime import date, datetime
from decimal import Decimal

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
You are reviewing the results of an open data quality assessment based on Vetrò-style dimensions.

Dataset context:
{dataset_context_yaml}

Dimension scores:
{dimension_scores_yaml}

Lowest-scoring metrics:
{weakest_metrics_yaml}

Selected raw inputs and inferred signals:
{raw_inputs_yaml}

Write exactly 5 short markdown bullet points.

Rules:
- Base every recommendation only on the results above.
- Prioritise the lowest scores first.
- Mention the related metric or category in each bullet.
- Explain what should be improved in the dataset, metadata, or publication workflow.
- Give concrete actions, not generic advice.
- Keep each bullet under 28 words.
- Do not mention confidence.
- Do not mention that you are an AI.
- Do not invent missing facts.
""".strip()

def make_json_safe(value):
    if value is None:
        return None

    if isinstance(value, (date, datetime)):
        return value.isoformat()

    if isinstance(value, Decimal):
        return float(value)

    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")

    if isinstance(value, dict):
        return {str(k): make_json_safe(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [make_json_safe(v) for v in value]

    return value


def trino_query_to_df(conn, sql: str) -> pd.DataFrame:
    cur = conn.cursor()
    cur.execute(sql)
    rows = cur.fetchall()
    cols = [desc[0] for desc in cur.description] if cur.description else []

    safe_rows = [
        [make_json_safe(v) for v in row]
        for row in rows
    ]

    return pd.DataFrame(safe_rows, columns=cols)


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


def capitalise_dimension(value: str) -> str:
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

    if suffix == ".xls":
        df = pd.read_excel(io.BytesIO(raw_bytes), engine="xlrd")
        text_content = df.to_csv(index=False)
        return dataframe_to_metadata_dict(df), text_content

    if suffix == ".xlsx":
        df = pd.read_excel(io.BytesIO(raw_bytes), engine="openpyxl")
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

def _make_prompt_safe(value: Any, max_list_items: int = 12, max_dict_items: int = 40) -> Any:
    if isinstance(value, dict):
        out: Dict[str, Any] = {}
        for idx, (k, v) in enumerate(value.items()):
            if idx >= max_dict_items:
                out["truncated"] = True
                break
            out[str(k)] = _make_prompt_safe(v, max_list_items=max_list_items, max_dict_items=10)
        return out

    if isinstance(value, (list, tuple)):
        items = list(value)[:max_list_items]
        out = [_make_prompt_safe(x, max_list_items=max_list_items, max_dict_items=10) for x in items]
        if len(value) > max_list_items:
            out.append("truncated")
        return out

    if value is None:
        return None

    if isinstance(value, float) and pd.isna(value):
        return None

    if isinstance(value, (str, int, float, bool)):
        return value

    return str(value)


def build_ai_recommendation_prompt(
    metrics_df: pd.DataFrame,
    dimension_scores: pd.DataFrame,
    details: Dict[str, Any],
    data_source: str,
    df: Optional[pd.DataFrame] = None,
    template: str = DEFAULT_RECOMMENDATION_TEMPLATE,
) -> str:
    weakest_metrics_df = (
        metrics_df.sort_values("value_clamped", ascending=True)
        .head(7)[["dimension", "metric_label", "metric_id", "value", "value_clamped"]]
        .copy()
    )
    weakest_metrics_df["dimension"] = weakest_metrics_df["dimension"].astype(str).apply(capitalise_dimension)

    dimension_scores_df = dimension_scores.copy()
    dimension_scores_df["dimension"] = dimension_scores_df["dimension"].astype(str).apply(capitalise_dimension)

    dataset_context = {
        "data_source": data_source,
        "rows_used": int(len(df)) if df is not None else None,
        "column_count": int(len(df.columns)) if df is not None else None,
        "columns_sample": [str(col) for col in list(df.columns[:20])] if df is not None else [],
    }

    raw_inputs = _make_prompt_safe(details.get("raw_inputs", {}) or {})

    dataset_context_yaml = yaml.safe_dump(dataset_context, sort_keys=False, allow_unicode=True)
    dimension_scores_yaml = yaml.safe_dump(
        dimension_scores_df[["dimension", "value_clamped"]].to_dict(orient="records"),
        sort_keys=False,
        allow_unicode=True,
    )
    weakest_metrics_yaml = yaml.safe_dump(
        weakest_metrics_df.to_dict(orient="records"),
        sort_keys=False,
        allow_unicode=True,
    )
    raw_inputs_yaml = yaml.safe_dump(raw_inputs, sort_keys=False, allow_unicode=True)

    return template.format(
        dataset_context_yaml=dataset_context_yaml,
        dimension_scores_yaml=dimension_scores_yaml,
        weakest_metrics_yaml=weakest_metrics_yaml,
        raw_inputs_yaml=raw_inputs_yaml,
    )


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
                <div class="summary-value">{capitalise_dimension(best_dimension['dimension'])}</div>
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
                <div class="summary-value">{capitalise_dimension(weakest_dimension['dimension'])}</div>
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
                        <strong>{capitalise_dimension(dim)}</strong>
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

        The app combines:
        - structure-based checks from the uploaded table or SQL result
        - optional metadata from text or file upload
        - AI-based inference for missing semantic signals
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
        #help=SUPPORTED_FILE_TYPES_HELP,
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
            height=250,
            placeholder="SELECT * FROM some_table LIMIT 100000",
        )
        trino_meta_sql = st.text_area(
            "Metadata query",
            height=250,
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
    test_clicked = st.button("Test connection", width="stretch")
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
    st.session_state["scroll_to_results"] = True

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
        manual_metadata_llm_debug: Dict[str, Any] = {
            "evidence": {},
            "confidence": {},
            "parsed": {},
        }

        if use_llm and combined_manual_meta_text.strip():
            try:
                llm_runner = get_llm_runner(
                    provider=llm_provider,
                    model_name=llm_model_name,
                    api_key=openai_api_key,
                )
                prompts_cfg_runtime = load_prompts_cfg(prompts_yaml_path)
                (
                    manual_metadata_llm,
                    manual_metadata_llm_raw,
                    manual_metadata_prompt_source,
                    manual_metadata_llm_debug,
                ) = infer_manual_metadata_symbols(
                    combined_manual_meta_text,
                    llm_runner,
                    prompts_cfg=prompts_cfg_runtime,
                    prompt_regime=prompt_regime,
                )
                def _is_empty_or_weak_manual_value(x):
                    return x is None or x == "" or x in {0, 0.0}

                merged_manual_metadata = dict(manual_metadata)

                for k, v in manual_metadata_llm.items():
                    if _is_empty_or_weak_manual_value(merged_manual_metadata.get(k)) and v is not None:
                        merged_manual_metadata[k] = v

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
            df = trino_query_to_df(conn, trino_sql)
            ext = ".sql"

            if trino_meta_sql.strip():
                try:
                    meta_df = trino_query_to_df(conn, trino_meta_sql)
                    if not meta_df.empty:
                        trino_metadata_raw = {
                            str(k): make_json_safe(v)
                            for k, v in meta_df.iloc[0].to_dict().items()
                        }
                        trino_metadata = normalize_metadata_to_symbols(trino_metadata_raw)
                except Exception as exc:
                    st.warning(f"Metadata query failed, continuing without it: {exc}")

        if df is None:
            st.error("No data could be loaded from the selected data source.")
            st.stop()

        df_preview = make_arrow_safe(df.head(20).copy())

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

        details["manual_metadata_llm"] = manual_metadata_llm
        details["manual_metadata_llm_raw"] = manual_metadata_llm_raw
        details["manual_metadata_prompt_source"] = manual_metadata_prompt_source
        details["manual_metadata_llm_debug"] = manual_metadata_llm_debug
                
        metrics_df = apply_result_dimension_overrides(metrics_df)
        metrics_df["value"] = metrics_df["value"].apply(
            lambda x: float(x) if isinstance(x, (int, float)) else None
        )

        for col in df.columns:
            if df[col].dtype == "object":
                df[col] = df[col].apply(lambda x: str(x) if isinstance(x, (dict, list, tuple)) else x)

        st.markdown('<div id="results-anchor"></div>', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("## Results")

        recommendation_prompt = ""
        recommendation_raw = ""
        recommendation_error = ""

        preview_tab, overview_tab, detail_tab, debug_tab = st.tabs(
            ["Preview", "Overview", "Detailed metrics", "Debug"]
        )

        with preview_tab:
            st.subheader("Preview of the dataset")
            #st.dataframe(df.head(20), use_container_width=True)
            st.dataframe(df_preview, width="stretch")
            st.caption(f"{len(df)} rows × {len(df.columns)} columns used for metrics.")

        with overview_tab:
            if metrics_df.empty:
                st.info("No metrics could be computed.")
            else:
                NULL_BAR_EPS = 0.000001

                metrics_all = metrics_df.copy()
                metrics_all["dimension"] = pd.Categorical(
                    metrics_all["dimension"],
                    categories=DIMENSION_ORDER,
                    ordered=True,
                )

                metrics_non_null = metrics_all.dropna(subset=["value"]).copy()
                if not metrics_non_null.empty:
                    metrics_non_null["value_clamped"] = metrics_non_null["value"].clip(0.0, 1.0)

                # Keep ALL dimensions in overview, even if their score is NULL
                dimension_scores = pd.DataFrame({"dimension": DIMENSION_ORDER})
                if not metrics_non_null.empty:
                    dimension_means = (
                        metrics_non_null.groupby("dimension", as_index=False)["value_clamped"]
                        .mean()
                    )
                    dimension_scores = dimension_scores.merge(
                        dimension_means,
                        on="dimension",
                        how="left",
                    )
                else:
                    dimension_scores["value_clamped"] = pd.NA

                available_dimension_scores = dimension_scores.dropna(subset=["value_clamped"]).copy()

                # Summary cards only for dimensions that actually have numeric values
                if not metrics_non_null.empty and not available_dimension_scores.empty:
                    overall_score = metrics_non_null["value_clamped"].mean()
                    best_dimension = available_dimension_scores.sort_values("value_clamped", ascending=False).iloc[0]
                    weakest_dimension = available_dimension_scores.sort_values("value_clamped", ascending=True).iloc[0]
                    weakest_metric = metrics_non_null.sort_values("value_clamped", ascending=True).iloc[0]

                    render_summary_cards(overall_score, best_dimension, weakest_dimension, weakest_metric)
                    st.markdown("")
                else:
                    st.info("No dimensions with numeric values are available for summary cards.")

                # Tiles already support missing values -> "Not available"
                render_dimension_tiles(dimension_scores)

                chart_col, reco_col = st.columns([1.2, 0.9], gap="large")

                with chart_col:
                    st.markdown("### Average score by category")

                    dimension_chart = dimension_scores.copy()
                    dimension_chart["dimension_display"] = dimension_chart["dimension"].astype(str).apply(capitalise_dimension)
                    dimension_chart["value_for_chart"] = dimension_chart["value_clamped"].apply(
                        lambda x: NULL_BAR_EPS if pd.isna(x) else float(max(0.0, min(x, 1.0)))
                    )
                    dimension_chart["value_label"] = dimension_chart["value_clamped"].apply(
                        lambda x: "NULL" if pd.isna(x) else f"{float(x):.2f}"
                    )
                    dimension_chart["status"] = dimension_chart["value_clamped"].apply(
                        lambda x: "NULL / not enough data" if pd.isna(x) else "Computed"
                    )

                    dim_fig = px.bar(
                        dimension_chart,
                        x="value_for_chart",
                        y="dimension_display",
                        orientation="h",
                        text="value_label",
                        range_x=[0, 1],
                        labels={
                            "dimension_display": "Category",
                            "value_for_chart": "Average normalised value (0–1)",
                        },
                        hover_data={
                            "value_clamped": True,
                            "status": True,
                            "value_for_chart": False,
                            "value_label": False,
                        },
                    )
                    dim_fig.update_traces(textposition="outside", cliponaxis=False)
                    dim_fig.update_layout(height=420)
                    st.plotly_chart(dim_fig, width="stretch")

                with reco_col:
                    st.markdown("### Suggested next improvements")
                    ai_recommendations = ""

                    if use_llm and not metrics_non_null.empty and not available_dimension_scores.empty:
                        try:
                            recommendation_prompt = build_ai_recommendation_prompt(
                                metrics_df=metrics_non_null,
                                dimension_scores=available_dimension_scores,
                                details=details,
                                data_source=data_source,
                                df=df,
                            )
                            llm_runner = get_llm_runner(
                                provider=llm_provider,
                                model_name=llm_model_name,
                                api_key=openai_api_key,
                            )
                            recommendation_raw = llm_runner(recommendation_prompt, 220)
                            ai_recommendations = recommendation_raw.strip()
                        except Exception as exc:
                            recommendation_error = str(exc)

                    if ai_recommendations:
                        st.markdown(ai_recommendations)
                    elif recommendation_error:
                        st.warning(f"Recommendations could not be generated: {recommendation_error}")
                    else:
                        st.info("No recommendations available.")

        with detail_tab:
            metrics_table = metrics_df.copy()

            if metrics_table.empty:
                st.info("No detailed metric table is available.")
            else:
                metrics_table["dimension"] = metrics_table["dimension"].astype(str)
                metrics_table["dimension_display"] = metrics_table["dimension"].apply(capitalise_dimension)

                formula_trace = details.get("formula_trace", {})

                def _missing_inputs(metric_id: str) -> str:
                    missing = formula_trace.get(metric_id, {}).get("missing_required_inputs", [])
                    return ", ".join(missing) if missing else ""

                def _value_status(row) -> str:
                    metric_id = row["metric_id"]
                    value = row["value"]
                    missing = formula_trace.get(metric_id, {}).get("missing_required_inputs", [])

                    if pd.isna(value):
                        if missing:
                            return "NULL (missing inputs)"
                        return "NULL"
                    if value == 0:
                        return "0 (computed)"
                    return "Computed"

                def _value_display(x):
                    if pd.isna(x):
                        return "NULL"
                    return x

                metrics_table["missing_inputs"] = metrics_table["metric_id"].apply(_missing_inputs)
                metrics_table["status"] = metrics_table.apply(_value_status, axis=1)
                metrics_table["value_display"] = metrics_table["value"].apply(_value_display)

                NULL_BAR_EPS = 0.0000001

                metrics_chart = metrics_table.copy()
                metrics_chart["value_for_chart"] = metrics_chart["value"].apply(
                    lambda x: NULL_BAR_EPS if pd.isna(x) else float(max(0.0, min(x, 1.0)))
                )
                metrics_chart["value_label"] = metrics_chart["value"].apply(
                    lambda x: "NULL" if pd.isna(x) else f"{float(x):.2f}"
                )
                if not metrics_chart.empty:
                    metrics_chart["value_clamped"] = metrics_chart["value"].clip(0.0, 1.0)

                    metric_fig = px.bar(
                        metrics_chart.sort_values(["dimension", "metric_label"]),
                        x="value_for_chart",
                        y="metric_label",
                        color="dimension_display",
                        text="value_label",
                        orientation="h",
                        range_x=[0, 1],
                        labels={
                            "metric_label": "Metric",
                            "value_for_chart": "Normalised value (0–1)",
                            "dimension_display": "Category",
                        },
                        hover_data={
                            "value": True,
                            "status": True,
                            "missing_inputs": True,
                            "value_for_chart": False,
                            "value_label": False,
                        },
                    )

                    metric_fig.update_traces(textposition="outside", cliponaxis=False)
                    metric_fig.update_layout(height=680)
                    st.plotly_chart(metric_fig, width="stretch")
                else:
                    st.info("No metrics with numeric values are available for the chart.")

                st.dataframe(
                    metrics_table[
                        [
                            "dimension_display",
                            "metric_label",
                            "value_display",
                            "status",
                            "missing_inputs",
                            "metric_id",
                        ]
                    ]
                    .rename(columns={
                        "dimension_display": "dimension",
                        "value_display": "value",
                        "missing_inputs": "missing_required_inputs",
                    })
                    .sort_values(["dimension", "metric_id"]),
                    width="stretch",
                )

        with debug_tab:
            st.subheader("Debug information")

            st.markdown("### Manual metadata LLM extraction")
            st.write(f"Prompt source: {details.get('manual_metadata_prompt_source', 'not_used')}")
            st.json(details.get("manual_metadata_llm", {}))
            st.json(details.get("manual_metadata_llm_debug", {}))
            if details.get("manual_metadata_llm_raw"):
                st.code(details["manual_metadata_llm_raw"], language="json")

            st.markdown("### Symbol status table")

            symbol_trace = details.get("symbol_trace", {})
            symbol_values = details.get("symbol_values", {})
            symbol_source = details.get("symbol_source", {})

            symbol_rows = []
            for sym in sorted(set(symbol_values.keys()) | set(symbol_trace.keys())):
                val = symbol_values.get(sym)
                src = symbol_source.get(sym, "")
                evidence = symbol_trace.get(sym, {}).get("evidence", "")
                confidence = symbol_trace.get(sym, {}).get("confidence", None)

                if pd.isna(val):
                    status = "NULL / unresolved"
                elif val == 0:
                    status = "0 (explicit)"
                else:
                    status = "Resolved"

                symbol_rows.append({
                    "symbol": sym,
                    "value": "NULL" if pd.isna(val) else val,
                    "status": status,
                    "source": src,
                    "evidence": evidence,
                    "confidence": confidence,
                })

            symbol_df = pd.DataFrame(symbol_rows)
            st.dataframe(symbol_df, width="stretch")

            st.markdown("### Symbol trace")
            st.json(details.get("symbol_trace", {}))

            st.markdown("### Formula trace")
            formula_trace = details.get("formula_trace", {})
            if formula_trace:
                selected_metric = st.selectbox(
                    "Choose metric trace",
                    options=list(formula_trace.keys()),
                )
                st.json(formula_trace[selected_metric])
            else:
                st.info("No formula trace available.")

            st.markdown("### LLM semantic calls")
            llm_calls = details.get("llm_debug", {}).get("calls", [])
            if llm_calls:
                for idx, call in enumerate(llm_calls, start=1):
                    with st.expander(f"LLM call {idx}: {', '.join(call.get('symbols', []))}"):
                        if call.get("prompt"):
                            st.code(call["prompt"], language="text")
                        if call.get("parsed") is not None:
                            st.json(call["parsed"])
                        if call.get("raw"):
                            st.code(call["raw"], language="json")
            else:
                st.info("No semantic LLM calls recorded.")

            st.markdown("### Existing details JSON")
            st.json(details)

            st.markdown("**Auto-derived inputs / inferred symbols**")
            st.json(details.get("raw_inputs", {}))
            st.markdown("**Metric evaluation details**")
            st.dataframe(metrics_df, width="stretch")
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
            if recommendation_prompt:
                st.markdown("**Recommendation prompt**")
                st.code(recommendation_prompt)

            if recommendation_raw:
                st.markdown("**Recommendation raw output**")
                st.code(recommendation_raw)

            if recommendation_error:
                st.markdown("**Recommendation error**")
                st.write(recommendation_error)   

    except Exception as exc:
        st.error(f"Assessment failed: {exc}")
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
    if st.session_state.get("scroll_to_results"):
        components.html(
            """
            <script>
            const anchor = window.parent.document.getElementById("results-anchor");
            if (anchor) {
                anchor.scrollIntoView({behavior: "smooth", block: "start"});
            }
            </script>
            """,
            height=0,
        )
        st.session_state["scroll_to_results"] = False 
