from __future__ import annotations

from typing import Any, Dict, Tuple
import yaml
import pandas as pd

from core.llm import get_hf_runner
from core.metrics_eval import compute_metrics


def run_quality_assessment(
    df: pd.DataFrame,
    formulas_yaml_path: str,
    prompts_yaml_path: str,
    use_llm: bool,
    hf_model_name: str,
    dataset_description: str = "",
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:

    with open(formulas_yaml_path, "r", encoding="utf-8") as f:
        formulas_cfg = yaml.safe_load(f) or {}

    with open(prompts_yaml_path, "r", encoding="utf-8") as f:
        prompt_cfg = yaml.safe_load(f) or {}

    hf_runner = None
    if use_llm:
        hf_runner = get_hf_runner(hf_model_name)

    metrics_df, details = compute_metrics(
        df=df,
        formulas_cfg=formulas_cfg,
        prompt_cfg=prompt_cfg,
        use_llm=use_llm,
        hf_runner=hf_runner,
        dataset_description=dataset_description or "",
    )

    return df, metrics_df, details
