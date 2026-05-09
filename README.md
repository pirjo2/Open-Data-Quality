# Open Data Quality

This repository contains a Streamlit application for assessing the quality of open datasets. The project was created as part of the master's thesis **“Assessing the quality of open data using artificial intelligence based methods”**.

Live application: https://avaandmete-kvaliteet.streamlit.app/

## This project

The application helps to assess open data quality in a more structured way. It is based on the open data quality framework by Vetrò et al., but the framework has been adapted into a practical YAML-based implementation.

The goal of the project is not to fully automate all open data quality assessment. Instead, the application combines:

- rule-based checks from the dataset itself,
- metadata-based checks,
- AI-assisted inference for indicators that are harder to calculate automatically.

The app can be used with uploaded files, and it also has an advanced Trino option for working with datasets from the Estonian open data infrastructure.

## Main features

- Upload-based dataset assessment in Streamlit.
- Advanced Trino SQL query option.
- YAML-based metric definitions and formulas.
- AI-assisted inference for missing semantic quality inputs.
- Quality scores by metric and dimension.
- Debug information for checking how the result was calculated.
- Synthetic PLUS/MINUS test cases for testing the implemented metrics.

## Requirements

The project is intended to run with Python 3.11.

Install the required packages with:

```bash
pip install -r requirements.txt
```

## Running the Streamlit app locally

Clone the repository:

```bash
git clone https://github.com/pirjo2/Open-Data-Quality.git
cd Open-Data-Quality
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

## Using AI-assisted inference

The application can use AI to infer missing quality inputs from metadata and dataset context.

For OpenAI-based inference, provide an API key either in the Streamlit interface or as an environment variable.

On macOS/Linux:

```bash
export OPENAI_API_KEY="your-api-key"
```

On Windows PowerShell:

```powershell
$env:OPENAI_API_KEY="your-api-key"
```

## Using the Trino option

The Trino option is meant for accessing datasets from the Estonian open data infrastructure. This access is not public by default.

To use Trino, a username and password are needed. These credentials can be requested from **Kristjan Lõhmus**.

The Trino connection uses these environment variables:

- `TRINO_USER`
- `TRINO_PASSWORD`

On macOS/Linux:

```bash
export TRINO_USER="your-trino-username"
export TRINO_PASSWORD="your-trino-password"
```

On Windows PowerShell:

```powershell
$env:TRINO_USER="your-trino-username"
$env:TRINO_PASSWORD="your-trino-password"
```

## Running the Trino batch assessment script

The batch script is located here:

```text
scripts/run_trino_batch_assessment_v2.py
```

Example command for a small test run:

```bash
python scripts/run_trino_batch_assessment_v2.py --sample-rows 100 --max-datasets 5 --use-llm --llm-provider openai --llm-model gpt-4.1-mini
```

The script saves results under:

```text
outputs/trino_batch_assessment/
```

## Running synthetic experiments

Synthetic test cases are located in:

```text
testkomplekt/vetro_tests/
```

To run the experiment script:

```bash
python scripts/run_experiments.py
```

This script was mainly used for thesis-related testing and for comparing different prompting regimes or model configurations.

## Notes

- Metric definitions are stored in `configs/formulas.yaml`.
- Prompt templates are stored in `configs/prompts.yaml`.
- Scores are normalized to the range 0–1 where possible.
- The Streamlit app is the main user-facing part of the project.
- The scripts folder contains additional tools used during development and thesis experiments.
- AI tools were used to support code explanation, troubleshooting and the exploration of possible implementation solutions. Final decisions and code changes were made by the author.

## Author

Pirjo Vainjärv  
University of Tartu  
Master's thesis
Superviser: Kristo Raun