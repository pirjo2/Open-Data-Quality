# Open Data Quality (Vetrò 2016) — YAML-driven + optional Hugging Face LLM

## Files
- `configs/formulas.yaml` - Vetrò methodology (dimensions, metrics, formulas)
- `configs/prompts.yaml` - prompt templates for symbols that can be inferred by an LLM
- `core/` - Python implementation (YAML loader, expression evaluator, metrics, LLM helpers)
- `app.py` - Streamlit UI

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy options
- https://avaandmete-kvaliteet.streamlit.app/ 
