from __future__ import annotations

import io
import os
import hashlib
import pandas as pd
import streamlit as st

from core.pipeline import run_quality_assessment

# ---- Paths (repo root) ----
FORMULAS = "formulas.yaml"
PROMPTS = "prompts.yaml"


def _sha16(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:16]


def _stable_cache_key(file_bytes: bytes, params: dict) -> str:
    h = hashlib.sha256()
    h.update(file_bytes)
    # stable parameter encoding
    for k in sorted(params.keys()):
        h.update(str(k).encode("utf-8"))
        h.update(str(params[k]).encode("utf-8"))
    return h.hexdigest()[:24]


def _button(label: str, **kwargs):
    """
    Streamlit deprecates use_container_width in favor of width.
    This wrapper keeps compatibility across versions.
    """
    try:
        return st.button(label, **kwargs)
    except TypeError:
        if "width" in kwargs:
            w = kwargs.pop("width")
            kwargs["use_container_width"] = (w == "stretch")
        return st.button(label, **kwargs)


def _dataframe(df: pd.DataFrame, **kwargs):
    try:
        return st.dataframe(df, **kwargs)
    except TypeError:
        if "width" in kwargs:
            w = kwargs.pop("width")
            kwargs["use_container_width"] = (w == "stretch")
        return st.dataframe(df, **kwargs)


def _download_button(label: str, data: bytes, file_name: str, mime: str, **kwargs):
    try:
        return st.download_button(label, data=data, file_name=file_name, mime=mime, **kwargs)
    except TypeError:
        if "width" in kwargs:
            w = kwargs.pop("width")
            kwargs["use_container_width"] = (w == "stretch")
        return st.download_button(label, data=data, file_name=file_name, mime=mime, **kwargs)


def load_dataset(uploaded_file, max_rows: int) -> tuple[pd.DataFrame, str, str]:
    """
    Returns (df, file_name, file_ext). max_rows=0 => load all rows.
    """
    file_name = uploaded_file.name
    ext = os.path.splitext(file_name)[1].lower().lstrip(".")
    data = uploaded_file.getvalue()

    if ext in ("csv", "txt"):
        df = None
        for enc in ("utf-8", "utf-8-sig", "cp1252", "latin1"):
            try:
                df = pd.read_csv(io.BytesIO(data), encoding=enc, low_memory=False)
                break
            except Exception:
                df = None
        if df is None:
            raise ValueError("CSV read failed (encoding). Try saving as UTF-8.")
    elif ext in ("xlsx", "xls"):
        df = pd.read_excel(io.BytesIO(data))
    elif ext in ("json",):
        df = pd.read_json(io.BytesIO(data))
    else:
        raise ValueError(f"Unsupported file type: .{ext}. Use CSV/XLSX/JSON.")

    if max_rows and max_rows > 0:
        df = df.head(max_rows).copy()

    return df, file_name, ext


def nice_bar_chart_df(metrics_df: pd.DataFrame) -> pd.DataFrame:
    out = metrics_df.copy()
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out = out.dropna(subset=["value"])
    out["label"] = out["metric_label"].fillna(out["metric_id"])
    out = out.sort_values(["dimension", "metric"])
    return out[["label", "value", "dimension", "metric_id"]]


st.set_page_config(page_title="Open Data Quality Assessment", layout="wide")
st.title("Open Data Quality Assessment (Vetrò + optional AI)")

st.markdown(
    """
Upload a dataset (CSV/XLSX/JSON), compute Vetrò (2016) quality metrics, and optionally use a local Hugging Face model
to estimate metadata-related inputs.  
**Note:** some Vetrò metrics require portal metadata (license/publisher/etc). If it's not in the file, AI can only guess
based on context you provide.
"""
)

uploaded = st.file_uploader("Dataset file", type=["csv", "txt", "xlsx", "xls", "json"], key="uploader")

col1, col2, col3 = st.columns([1.2, 1.2, 2.0])

with col1:
    max_rows = st.number_input("Max rows (0 = all)", min_value=0, value=0, step=1000)

with col2:
    use_llm = st.toggle("Use local Hugging Face model", value=True)

with col3:
    model_choices = [
        "google/flan-t5-small",
        "google/flan-t5-base",
        "google/flan-t5-large",
        "google/mt5-small",
    ]
    hf_model = st.selectbox("Model", options=model_choices, index=1)
    custom_model = st.text_input("Or type a custom Hugging Face model id (optional)", value="")
    if custom_model.strip():
        hf_model = custom_model.strip()

dataset_description = st.text_area(
    "Optional context (improves AI results): 1–5 sentences about the dataset source, meaning of columns, time period, license, publisher, etc.",
    value="",
    height=110,
)

run_btn = _button("Analyze", type="primary", width="stretch")

if uploaded is None:
    st.info("Upload a file, then click **Analyze**.")
    st.stop()

file_bytes = uploaded.getvalue()

params = {
    "max_rows": int(max_rows),
    "use_llm": bool(use_llm),
    "hf_model": hf_model,
    "dataset_description": dataset_description.strip(),
}
cache_key = _stable_cache_key(file_bytes, params)

if "results" not in st.session_state:
    st.session_state["results"] = {}

if run_btn:
    with st.spinner("Loading data and computing metrics..."):
        df, file_name, ext = load_dataset(uploaded, int(max_rows))

        # Tip: First AI run can be slow because the model downloads once.
        _, metrics_df, details = run_quality_assessment(
            df=df,
            formulas_yaml_path=FORMULAS,
            prompts_yaml_path=PROMPTS,
            use_llm=use_llm,
            hf_model_name=hf_model,
            dataset_description=dataset_description,
            file_name=file_name,
            file_ext=ext,
        )

        st.session_state["results"][cache_key] = (df, metrics_df, details)

if cache_key not in st.session_state["results"]:
    st.warning("Click **Analyze** to see results.")
    st.stop()

df, metrics_df, details = st.session_state["results"][cache_key]

st.subheader("Preview")
_dataframe(df.head(20), width="stretch")

st.subheader("Metrics")
_dataframe(metrics_df[["dimension", "metric", "metric_id", "value", "metric_label"]], width="stretch")

chart_df = nice_bar_chart_df(metrics_df)
if not chart_df.empty:
    st.subheader("Bar chart")
    st.bar_chart(chart_df.set_index("label")["value"], height=380)

# Downloads
st.subheader("Download")
metrics_csv = metrics_df.to_csv(index=False).encode("utf-8")
_download_button("Download metrics CSV", data=metrics_csv, file_name="metrics.csv", mime="text/csv", width="stretch")

# Debug / transparency
with st.expander("Debug: auto inputs, symbols, raw AI outputs"):
    st.write("Auto inputs (derived from data):")
    st.json(details.get("auto_inputs", {}))

    sym_vals = details.get("symbol_values", {}) or {}
    sym_src = details.get("symbol_source", {}) or {}
    sym_conf = details.get("llm_confidence", {}) or {}
    sym_ev = details.get("llm_evidence", {}) or {}
    sym_raw = details.get("llm_raw", {}) or {}

    rows = []
    all_keys = sorted(set(sym_vals.keys()) | set(sym_src.keys()) | set(sym_conf.keys()) | set(sym_raw.keys()))
    for k in all_keys:
        rows.append(
            {
                "symbol": k,
                "value (None=did not work)": sym_vals.get(k, None),
                "source": sym_src.get(k, ""),
                "confidence": sym_conf.get(k, None),
                "evidence": sym_ev.get(k, ""),
                "raw": sym_raw.get(k, ""),
            }
        )
    _dataframe(pd.DataFrame(rows), width="stretch")
