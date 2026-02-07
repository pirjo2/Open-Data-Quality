from __future__ import annotations

import hashlib
import io
import json
import os
from typing import Optional, Tuple

import pandas as pd
import plotly.express as px
import streamlit as st

from core.pipeline import run_quality_assessment


# ---- Paths (repo root) ----
FORMULAS = "configs/formulas.yaml"
PROMPTS = "configs/prompts.yaml"

# ---- Small, CPU-friendly default models (can be overridden) ----
DEFAULT_MODELS = [
    "google/flan-t5-small",
    "google/flan-t5-base",
    "sshleifer/tiny-t5",
]


def _stable_hash_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:12]


def _read_uploaded_file(upload, max_rows: int) -> Tuple[pd.DataFrame, str]:
    """Read CSV/XLSX/JSON into a DataFrame. Returns (df, file_ext)."""
    name = upload.name
    ext = os.path.splitext(name)[1].lower().lstrip(".")
    raw = upload.getvalue()

    # 0 means "all"
    nrows = None if (max_rows in (None, 0)) else int(max_rows)

    if ext in ("xlsx", "xls"):
        df = pd.read_excel(raw, engine="openpyxl", nrows=nrows)
        return df, ext

    if ext in ("json",):
        try:
            df = pd.read_json(raw, lines=True)
        except Exception:
            df = pd.read_json(raw)
        if nrows is not None:
            df = df.head(nrows)
        return df, ext

    # CSV/TXT fallback
    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin1"]
    last_err: Optional[Exception] = None
    for enc in encodings:
        try:
            df = pd.read_csv(
                io.BytesIO(raw),
                encoding=enc,
                encoding_errors="replace",
                sep=None,           # auto-detect delimiter
                engine="python",
                nrows=nrows,
            )
            return df, ext or "csv"
        except Exception as e:
            last_err = e
            continue

    raise last_err or RuntimeError("Could not read the uploaded file.")


def _compute_cache_key(
    file_bytes: bytes,
    max_rows: int,
    dataset_description: str,
    metadata_text: str,
    use_llm: bool,
    model_name: str,
    weight_by_confidence: bool,
) -> str:
    return "|".join(
        [
            _stable_hash_bytes(file_bytes),
            str(max_rows),
            hashlib.sha256((dataset_description or "").encode("utf-8")).hexdigest()[:10],
            hashlib.sha256((metadata_text or "").encode("utf-8")).hexdigest()[:10],
            "llm" if use_llm else "nollm",
            model_name,
            "w" if weight_by_confidence else "nw",
        ]
    )


def main() -> None:
    st.set_page_config(page_title="Open Data Quality Assessment", layout="wide")

    st.title("Open Data Quality Assessment (Vetrò-style)")
    st.write(
        "Upload an open data file (CSV/XLSX/JSON). The app computes Vetrò-style quality metrics. "
        "If you enable the LLM option, it can also estimate metadata-related symbols."
    )

    upload = st.file_uploader("Upload data file", type=["csv", "txt", "xlsx", "xls", "json"])
    dataset_description = st.text_input("Dataset description (optional)", value="")
    metadata_text = st.text_area(
        "Paste portal metadata / documentation text (optional)",
        value="",
        height=140,
        help="If you paste the dataset page text (title, description, license, publisher, update frequency, etc.), "
             "LLM-based metrics become much more accurate.",
    )

    cols = st.columns(3)
    with cols[0]:
        max_rows = st.number_input(
            "Max rows to load (0 = all)",
            min_value=0,
            value=0,
            step=10000,
        )
    with cols[1]:
        use_llm = st.checkbox("Use LLM for metadata symbols", value=True)
        weight_by_confidence = st.checkbox("Weight LLM answers by confidence", value=False)
    with cols[2]:
        model_name = st.selectbox("Hugging Face model", DEFAULT_MODELS, index=1)
        custom_model = st.text_input("Custom model name (optional)", value="")
        if custom_model.strip():
            model_name = custom_model.strip()

    run_btn = st.button("Run assessment", type="primary", disabled=(upload is None))

    if upload is None:
        st.info("Upload a file to start.")
        return

    # ---- Persist results between reruns (e.g., after download button) ----
    if "results" not in st.session_state:
        st.session_state["results"] = {}

    file_bytes = upload.getvalue()
    cache_key = _compute_cache_key(
        file_bytes=file_bytes,
        max_rows=int(max_rows),
        dataset_description=dataset_description,
        metadata_text=metadata_text,
        use_llm=use_llm,
        model_name=model_name,
        weight_by_confidence=weight_by_confidence,
    )

    if run_btn:
        with st.spinner("Reading file..."):
            df, ext = _read_uploaded_file(upload, int(max_rows))

        with st.spinner("Computing metrics..."):
            _, metrics_df, details = run_quality_assessment(
                df=df,
                formulas_yaml_path=FORMULAS,
                prompts_yaml_path=PROMPTS,
                use_llm=use_llm,
                hf_model_name=model_name,
                dataset_description=dataset_description,
                metadata_text=metadata_text,
                file_name=upload.name,
                file_ext=ext,
                weight_by_confidence=weight_by_confidence,
            )

        st.session_state["results"][cache_key] = {
            "df": df,
            "ext": ext,
            "metrics_df": metrics_df,
            "details": details,
            "file_name": upload.name,
            "model_name": model_name,
            "use_llm": use_llm,
            "dataset_description": dataset_description,
            "metadata_text": metadata_text,
            "weight_by_confidence": weight_by_confidence,
        }

    res = st.session_state["results"].get(cache_key)
    if not res:
        st.info("Click **Run assessment** to compute results.")
        return

    df = res["df"]
    metrics_df = res["metrics_df"]
    details = res["details"]

    st.subheader("Data preview")
    st.dataframe(df.head(50), width="stretch")

    st.subheader("Quality metrics")
    show_df = metrics_df.copy()
    show_df = show_df[["dimension", "metric_label", "value", "metric_id"]].sort_values(["dimension", "metric_label"])
    st.dataframe(show_df, width="stretch")

    # Bar chart
    chart_df = show_df.dropna(subset=["value"]).copy()
    if len(chart_df) > 0:
        fig = px.bar(chart_df, x="metric_label", y="value", hover_data=["dimension", "metric_id"])
        fig.update_layout(xaxis_title="", yaxis_title="Score (0–1)", xaxis_tickangle=-30)
        st.plotly_chart(fig, width="stretch")
    else:
        st.warning("No metric values were computed (all are NaN).")

    # Downloads (metrics + debug)
    st.subheader("Downloads")
    metrics_csv = show_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download metrics (CSV)", data=metrics_csv, file_name="metrics.csv", mime="text/csv")

    sym_rows = []
    symvals = details.get("symbol_values", {}) or {}
    symsrc = details.get("symbol_source", {}) or {}
    symconf = details.get("llm_confidence", {}) or {}
    symraw = details.get("llm_raw", {}) or {}
    symev = details.get("llm_evidence", {}) or {}
    for sym in sorted(symvals.keys()):
        sym_rows.append(
            {
                "symbol": sym,
                "value": symvals.get(sym, None),
                "source": symsrc.get(sym, ""),
                "confidence": symconf.get(sym, None),
                "evidence": symev.get(sym, ""),
                "raw": symraw.get(sym, ""),
            }
        )
    symbols_df = pd.DataFrame(sym_rows)
    st.dataframe(symbols_df, width="stretch")

    st.download_button(
        "Download symbol debug (CSV)",
        data=symbols_df.to_csv(index=False).encode("utf-8"),
        file_name="symbols_debug.csv",
        mime="text/csv",
    )

    st.download_button(
        "Download full debug (JSON)",
        data=json.dumps(details, ensure_ascii=False, indent=2).encode("utf-8"),
        file_name="details.json",
        mime="application/json",
    )


if __name__ == "__main__":
    main()
