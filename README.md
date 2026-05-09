# Open Data Quality

A Streamlit-based prototype for assessing the quality of open datasets using a YAML-driven implementation of the Vetrò et al. open data quality framework. The project was developed as part of the master's thesis **“Assessing the quality of open data using artificial intelligence based methods”**.

Live application: https://avaandmete-kvaliteet.streamlit.app/

## Overview

The application supports semi-automated open data quality assessment. It combines rule-based checks, metadata-based interpretation and optional AI-assisted inference for quality indicators that cannot be derived directly from the dataset structure.

The assessment logic is based on configurable YAML files instead of hard-coded formulas. This makes it easier to inspect, adapt and extend the implemented quality metrics.

## Main features

- Upload-based dataset assessment through a Streamlit interface.
- Advanced Trino SQL query option for assessing datasets from the Estonian open data infrastructure.
- YAML-based metric formulas and prompt templates.
- Quality dimensions based on the adapted Vetrò et al. framework:
  - traceability
  - currentness
  - completeness
  - compliance
  - understandability
  - accuracy
- Support for structural checks, metadata extraction and AI-assisted fallback inference.
- Metric-level results, dimension-level overview, debug information and improvement suggestions.
- Synthetic PLUS/MINUS test cases for evaluating the implemented metrics and prompting regimes.

## Project structure

```text
Open-Data-Quality/
├── app.py
├── configs/
│   ├── formulas.yaml
│   └── prompts.yaml
├── core/
│   ├── pipeline.py
│   ├── metrics_eval.py
│   ├── llm.py
│   ├── metadata_utils.py
│   └── utils.py
├── scripts/
│   ├── run_experiments.py
│   └── run_trino_batch_assessment_v2.py
├── testkomplekt/
│   └── vetro_tests/
├── requirements.txt
├── runtime.txt
└── README.md
```

## Requirements

The project is intended to run with Python 3.11.

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Running locally

Clone the repository:

```bash
git clone https://github.com/pirjo2/Open-Data-Quality.git
cd Open-Data-Quality
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit app:

```bash
streamlit run app.py
```

## Using AI-assisted inference

The application can use AI models to infer missing semantic quality indicators from metadata and dataset context.

For OpenAI-based inference, provide an API key either through the Streamlit interface or as an environment variable:

```bash
export OPENAI_API_KEY="your-api-key"
```

On Windows PowerShell:

```powershell
$env:OPENAI_API_KEY="your-api-key"
```

If deployed on Streamlit Cloud, the key can be configured in the app secrets.

## Trino option

The application also includes an advanced Trino SQL query mode. This is intended for assessing datasets that are accessible through the Estonian open data infrastructure.

The Trino option is optional and mainly intended for advanced users or batch assessment workflows.

## Running synthetic experiments

Synthetic metric test cases are located in:

```text
testkomplekt/vetro_tests/
```

To run the experiment script:

```bash
python scripts/run_experiments.py
```

The script is mainly intended for thesis-related testing and comparison of prompting regimes or model configurations.

## Notes

- The metric definitions are stored in `configs/formulas.yaml`.
- Prompt templates are stored in `configs/prompts.yaml`.
- Scores are normalized to the range 0–1 where possible.
- The Streamlit app is the main user-facing interface.
- The scripts folder contains supporting evaluation and batch-processing tools used during development and thesis experiments.

## Author

Pirjo Vainjärv  
University of Tartu  
Master's thesis project
