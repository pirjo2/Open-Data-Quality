import io
import yaml
import pandas as pd
import streamlit as st
import plotly.express as px

from core.pipeline import run_quality_assessment

st.set_page_config(page_title="Open data quality", layout="wide")

st.title("Open data quality assessment (Vetrò + AI)")

# Paths inside the repo
FORMULAS = "configs/formulas.yaml"
PROMPTS = "configs/prompts.yaml"

# Load formulas to get labels (pretty names for chart)
with open(FORMULAS, "r", encoding="utf-8") as f:
    formulas_cfg = yaml.safe_load(f) or {}
labels_map = formulas_cfg.get("labels", {}) or {}

st.sidebar.header("Settings")
use_llm = st.sidebar.checkbox("Use Hugging Face AI", value=False)

MODEL_OPTIONS = [
    "google/flan-t5-base",
    "google/flan-t5-large",
    "google/mt5-small",
]
hf_model = st.sidebar.selectbox("Hugging Face model", MODEL_OPTIONS, index=0)

# 0 = load ALL rows
max_rows = st.sidebar.number_input(
    "Max rows (0 = all rows)",
    min_value=0,
    max_value=2_000_000,
    value=0,
    step=10_000,
)

uploaded = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])
run_btn = st.button("Run quality assessment", type="primary", disabled=uploaded is None)

def load_df(file) -> pd.DataFrame:
    if file.name.lower().endswith(".csv"):
        data = file.getvalue()
        for enc in ("utf-8", "utf-8-sig", "latin1"):
            try:
                return pd.read_csv(io.BytesIO(data), encoding=enc)
            except Exception:
                pass
        return pd.read_csv(io.BytesIO(data), engine="python")
    else:
        return pd.read_excel(file)

if run_btn and uploaded is not None:
    with st.spinner("Loading dataset..."):
        df = load_df(uploaded)

        if int(max_rows) > 0 and len(df) > int(max_rows):
            df = df.head(int(max_rows)).copy()

    st.success(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns.")
    st.dataframe(df.head(20), width="stretch")

    with st.spinner("Computing metrics..."):
        _, metrics_df, details = run_quality_assessment(
            df=df,
            formulas_yaml_path=FORMULAS,
            prompts_yaml_path=PROMPTS,
            use_llm=use_llm,
            hf_model_name=hf_model,
        )

    st.subheader("Results (table)")
    st.dataframe(metrics_df, width="stretch")

    # Chart with pretty labels
    chart_df = metrics_df.dropna(subset=["value"]).copy()
    chart_df["label"] = chart_df["metric_id"].map(labels_map).fillna(chart_df["metric_id"])
    chart_df = chart_df.sort_values("value", ascending=False)

    st.subheader("Quality scores (bar chart)")
    fig = px.bar(chart_df, x="label", y="value", color="dimension", text="value")
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig.update_layout(
        xaxis_title="Metric",
        yaxis_title="Score",
        xaxis_tickangle=-30,
        margin=dict(t=40, b=140),
        legend_title_text="Dimension",
    )
    st.plotly_chart(fig, width="stretch")

    with st.expander("Debug"):
        st.write("auto_inputs", details.get("auto_inputs", {}))
        st.write("llm_confidence", details.get("llm_confidence", {}))
        st.write("llm_raw", details.get("llm_raw", {}))
        st.write("llm_evidence", details.get("llm_evidence", {}))

