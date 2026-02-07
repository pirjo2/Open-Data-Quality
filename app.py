from __future__ import annotations

import io
import hashlib
import os
import pandas as pd
import streamlit as st

from core.pipeline import run_quality_assessment

# ---- Paths (Streamlit Cloud repo paths) ----
FORMULAS = "test2.yaml"
PROMPTS  = "vetro_prompts.yaml"

st.set_page_config(page_title="Avaandmete kvaliteedi analüüs", layout="wide")

def _hash_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:16]

def load_dataset(uploaded_file, max_rows: int) -> tuple[pd.DataFrame, str, str]:
    """
    Returns df, file_name, file_ext
    max_rows=0 => load all
    """
    file_name = uploaded_file.name
    ext = os.path.splitext(file_name)[1].lower().lstrip(".")
    data = uploaded_file.getvalue()

    if ext in ("csv", "txt"):
        # try a few encodings
        for enc in ("utf-8", "utf-8-sig", "cp1252", "latin1"):
            try:
                df = pd.read_csv(
                    io.BytesIO(data),
                    encoding=enc,
                    low_memory=False,
                )
                break
            except Exception:
                df = None
        if df is None:
            raise ValueError("CSV lugemine ebaõnnestus (encoding). Proovi salvestada UTF-8 kujul.")
    elif ext in ("xlsx", "xls"):
        df = pd.read_excel(io.BytesIO(data))
    elif ext in ("json",):
        df = pd.read_json(io.BytesIO(data))
    else:
        raise ValueError(f"Failitüüp .{ext} pole praegu toetatud. Proovi CSV/XLSX/JSON.")

    if max_rows and max_rows > 0:
        df = df.head(max_rows).copy()

    return df, file_name, ext

def nice_bar_chart_df(metrics_df: pd.DataFrame) -> pd.DataFrame:
    out = metrics_df.copy()
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out = out.dropna(subset=["value"])
    out = out.sort_values(["dimension", "metric"])
    out["label"] = out["metric_label"].fillna(out["metric_id"])
    return out[["label", "value", "dimension", "metric_id"]]

st.title("Avaandmete kvaliteedi analüüs (Vetrò + AI)")

with st.sidebar:
    st.header("Sisend")

    uploaded = st.file_uploader("Laadi üles andmestik (CSV / XLSX / JSON)", type=["csv","txt","xlsx","xls","json"])

    max_rows = st.number_input("Max read (0 = kõik read)", min_value=0, value=0, step=1000)

    st.divider()
    st.header("AI (valikuline)")

    use_llm = st.toggle("Kasuta Hugging Face mudelit", value=True)

    model_choices = [
        "google/flan-t5-small",
        "google/flan-t5-base",
    ]
    hf_model = st.selectbox("Mudeli valik", options=model_choices, index=1)
    custom_model = st.text_input("…või kirjuta oma HF mudel (optional)", value="")
    if custom_model.strip():
        hf_model = custom_model.strip()

    st.divider()
    st.header("Lisainfo (parandab AI vastuseid)")

    dataset_description = st.text_area(
        "Lühikirjeldus (1–5 lauset). Nt kust andmed pärinevad, mida veerud tähendavad, mis ajaperiood jne.",
        value="",
        height=120,
    )

run_btn = st.button("Analüüsi", type="primary", use_container_width=True)

if uploaded is None:
    st.info("Lae üles fail ja vajuta **Analüüsi**.")
    st.stop()

# Session cache key
file_bytes = uploaded.getvalue()
cache_key = _hash_bytes(file_bytes) + f":{max_rows}:{use_llm}:{hf_model}:{hash(dataset_description)}"

if "results" not in st.session_state:
    st.session_state["results"] = {}

if run_btn:
    with st.spinner("Loen faili ja arvutan metrikad..."):
        df, file_name, ext = load_dataset(uploaded, int(max_rows))
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
    st.warning("Vajuta **Analüüsi**, et näha tulemusi.")
    st.stop()

df, metrics_df, details = st.session_state["results"][cache_key]

st.subheader("Eelvaade")
st.dataframe(df.head(20), width="stretch")

st.subheader("Metrikad")
st.dataframe(metrics_df[["dimension","metric","metric_id","value","metric_label"]], width="stretch")

chart_df = nice_bar_chart_df(metrics_df)
if not chart_df.empty:
    st.subheader("Tulpdiagramm")
    st.bar_chart(chart_df.set_index("label")["value"], height=380)

with st.expander("Debug (auto_inputs, symbolid, LLM raw)"):
    st.write("Auto inputs (derived from data):")
    st.json(details.get("auto_inputs", {}))

    # nicer symbol table
    sym_vals = details.get("symbol_values", {}) or {}
    sym_src = details.get("symbol_source", {}) or {}
    sym_conf = details.get("llm_confidence", {}) or {}
    sym_ev = details.get("llm_evidence", {}) or {}
    sym_raw = details.get("llm_raw", {}) or {}

    rows = []
    for k in sorted(set(sym_vals.keys()) | set(sym_src.keys())):
        rows.append({
            "symbol": k,
            "value": sym_vals.get(k, None),
            "source": sym_src.get(k, ""),
            "confidence": sym_conf.get(k, None),
            "evidence": sym_ev.get(k, ""),
            "raw": sym_raw.get(k, ""),
        })
    st.dataframe(pd.DataFrame(rows), width="stretch")
