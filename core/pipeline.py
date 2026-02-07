from __future__ import annotations

from typing import Any, Dict, Tuple

import pandas as pd

from core.metrics_eval import compute_metrics
from core.yaml_loader import load_yaml
from core.llm import get_hf_runner


def run_quality_assessment(
    df: pd.DataFrame,
    formulas_yaml_path: str,
    prompts_yaml_path: str,
    use_llm: bool,
    hf_model_name: str,
    dataset_description: str = "",
    metadata_text: str = "",
    file_name: str = "",
    file_ext: str = "",
    weight_by_confidence: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Main pipeline entry point.

    Returns:
      (df, metrics_df, details)
    """
    formulas_cfg = load_yaml(formulas_yaml_path) or {}
    prompt_cfg = load_yaml(prompts_yaml_path) or {}

    hf_runner = None
    if use_llm:
        hf_runner = get_hf_runner(hf_model_name)

    metrics_df, details = compute_metrics(
        df=df,
        formulas_cfg=formulas_cfg,
        prompt_cfg=prompt_cfg,
        use_llm=use_llm,
        hf_runner=hf_runner,
        dataset_description=dataset_description,
        metadata_text=metadata_text,
        file_name=file_name,
        file_ext=file_ext,
        weight_by_confidence=weight_by_confidence,
    )

    return df, metrics_df, details
