from __future__ import annotations

import io
import pandas as pd
import plotly.express as px
import streamlit as st

from core.pipeline import run_quality_assessment

FORMULAS = "configs/formulas.yaml"
PROMPTS = "configs/prompts.yaml"


def _read_csv(file_bytes: bytes) -> pd.DataFrame:
    try:
        return pd.read_csv(io.BytesIO(file_bytes))
    except Exception:
        return pd.read_csv(io.BytesIO(file_bytes), encoding="latin-1")


def load_dataset(uploaded, ext: str, max_rows: int = 0) -> pd.DataFrame:
    file_bytes = uploaded.getvalue()

    if ext == "csv":
        df = _read_csv(file_bytes)
    elif ext in ("xlsx", "xls"):
        df = pd.read_excel(io.BytesIO(file_bytes))
    elif ext == "json":
        df = pd.read_json(io.BytesIO(file_bytes))
    else:
        raise ValueError(f"Unsupported file type: .{ext}")

    if max_rows and max_rows > 0:
        df = df.head(max_rows)

    return df


def _symbol_table(details: dict) -> pd.DataFrame:
    symbols = sorted(set(details.get("symbol_values", {}).keys()))
    rows = []
    for sym in symbols:
        rows.append(
            {
                "symbol": sym,
                "value": details.get("symbol_values", {}).get(sym, None),
                "source": details.get("symbol_source", {}).get(sym, ""),
                "confidence": details.get("symbol_confidence", {}).get(sym, None),
                "evidence": details.get("llm_evidence", {}).get(sym, ""),
                "raw": details.get("llm_raw", {}).get(sym, ""),
            }
        )
    df = pd.DataFrame(rows)

    # Arrow compatibility (fixes your pyarrow error)
    df["value"] = df["value"].astype(str)
    df["source"] = df["source"].astype(str)
    df["evidence"] = df["evidence"].astype(str)
    df["raw"] = df["raw"].astype(str)
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
    return df


def _download_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


st.set_page_config(page_title="Open Data Quality Assessment", layout="wide")
st.title("Open Data Quality Assessment (Vetrò-style)")

with st.expander("How to get better results", expanded=False):
    st.markdown(
        """
- Paste the dataset/portal description into **Dataset description** (this helps LLM infer metadata fields).
- If you see many **fail / None**, lower the confidence threshold slightly or try a different model.
- Cold starts can be slow because models are downloaded and loaded.
"""
    )

col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    uploaded_file = st.file_uploader("Upload a dataset (CSV, XLSX, JSON)", type=["csv", "xlsx", "xls", "json"])
    dataset_description = st.text_area(
        "Dataset description (optional, paste from portal)",
        placeholder="Paste the dataset description/metadata snippet here...",
        height=150,
    )

with col_right:
    max_rows = st.number_input("Max rows to load (0 = all rows)", min_value=0, value=0, step=1000)
    use_llm = st.checkbox("Use LLM for missing symbols", value=True)

    model_options = [
        "google/flan-t5-small",
        "google/flan-t5-base",
        "LaMini-Flan-T5-248M",
    ]
    hf_model_name = st.selectbox("Hugging Face model", model_options, index=1)
    custom_model = st.text_input("Or type a custom HF model name (optional)", value="")
    if custom_model.strip():
        hf_model_name = custom_model.strip()

    min_symbol_conf = st.slider("Minimum confidence to accept LLM output", 0.0, 1.0, 0.35, 0.05)
    apply_conf_weight = st.checkbox("Apply confidence weighting (value *= confidence)", value=True)

run_btn = st.button("Run assessment", type="primary", use_container_width=True)
clear_btn = st.button("Clear results", use_container_width=True)

if clear_btn:
    for k in ["metrics_df", "details", "file_name", "file_ext"]:
        st.session_state.pop(k, None)
    st.success("Cleared.")

if run_btn:
    if uploaded_file is None:
        st.error("Please upload a file first.")
    else:
        file_name = uploaded_file.name
        ext = file_name.split(".")[-1].lower()

        try:
            df = load_dataset(uploaded_file, ext=ext, max_rows=int(max_rows))
        except Exception as e:
            st.error(f"Failed to read dataset: {e}")
            st.stop()

        st.session_state["file_name"] = file_name
        st.session_state["file_ext"] = ext

        with st.spinner("Running quality assessment..."):
            try:
                _, metrics_df, details = run_quality_assessment(
                    df=df,
                    formulas_yaml_path=FORMULAS,
                    prompts_yaml_path=PROMPTS,
                    use_llm=use_llm,
                    hf_model_name=hf_model_name,
                    dataset_description=dataset_description,
                    file_name=file_name,
                    file_ext=ext,
                    min_symbol_confidence=float(min_symbol_conf),
                    apply_confidence_weighting=bool(apply_conf_weight),
                )
            except FileNotFoundError as e:
                st.error(
                    "Config file not found. Make sure your repo contains:\n"
                    "- configs/formulas.yaml\n"
                    "- configs/prompts.yaml\n\n"
                    f"Details: {e}"
                )
                st.stop()
            except Exception as e:
                st.error(f"Failed to run assessment: {e}")
                st.stop()

        st.session_state["metrics_df"] = metrics_df
        st.session_state["details"] = details
        st.success("Done!")

if "metrics_df" in st.session_state and "details" in st.session_state:
    metrics_df = st.session_state["metrics_df"].copy()
    metrics_df["value"] = pd.to_numeric(metrics_df["value"], errors="coerce")

    details = st.session_state["details"]

    st.subheader("Metrics (normalized 0..1)")
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    chart_df = metrics_df.dropna(subset=["value"]).copy()
    if len(chart_df):
        fig = px.bar(chart_df, x="metric_id", y="value", hover_data=["description"])
        st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "Download metrics CSV",
            data=_download_csv(metrics_df),
            file_name="metrics.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with c2:
        st.download_button(
            "Download debug symbols CSV",
            data=_download_csv(_symbol_table(details)),
            file_name="symbols_debug.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.subheader("Symbols (debug)")
    st.dataframe(_symbol_table(details), use_container_width=True, hide_index=True)

    with st.expander("Auto inputs (derived from data)", expanded=False):
        st.json(details.get("auto_inputs", {}))
