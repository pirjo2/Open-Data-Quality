from __future__ import annotations

import os
from typing import Optional, Dict, Any

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

# -------------------------------------------------------------------
# Common settings
# -------------------------------------------------------------------
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
# 2C. Manual metadata textbox
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

if st.button("Test OpenAI connection"):
    from core.llm import get_llm_runner

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

        manual_metadata_raw = parse_kv_metadata(manual_meta_text)
        manual_metadata = normalize_metadata_to_symbols(manual_metadata_raw)

        # uus reeglipõhine vabateksti parser
        #manual_metadata_rule = extract_symbols_from_realistic_text(manual_meta_text)

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
                manual_metadata_prompt_source = "not_used"
                manual_metadata_llm, manual_metadata_llm_raw, manual_metadata_prompt_source = infer_manual_metadata_symbols(
                    manual_meta_text,
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
                trino_metadata=trino_metadata if data_source == "Trino SQL query (beta)" else {},
                trino_metadata_raw=trino_metadata_raw if data_source == "Trino SQL query (beta)" else {},
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
        st.write("LLM calls:", len(details["llm_debug"].get("calls", [])))
        st.markdown("**Manual metadata prompt source:**")
        st.write(manual_metadata_prompt_source)
        st.markdown("**Prompt sources:**")
        {
        "currentness_anchor": "yaml:zero_shot",
        "semantic_metric_inference": "yaml:zero_shot"
        }
        st.json(details.get("prompt_sources", {}))
        with st.expander("Debug: auto-derived inputs and AI/metadata inferences"):
            st.markdown("**Auto-derived inputs (from the table only):**")
            st.json(details.get("auto_inputs", {}))

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