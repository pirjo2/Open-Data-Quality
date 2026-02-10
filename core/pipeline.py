from __future__ import annotations

from typing import Any, Dict, Tuple

import pandas as pd

from core.metrics_eval import compute_metrics
from core.yaml_loader import load_vetro_yaml, load_prompts_yaml
from core.llm import get_hf_runner


def run_quality_assessment(
    df: pd.DataFrame,
    formulas_yaml_path: str,
    prompts_yaml_path: str,
    use_llm: bool,
    hf_model_name: str,
    dataset_description: str = "",
    file_name: str | None = None,
    file_ext: str | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    High-level orchestration:
      1) load Vetrò formulas + symbol prompts
      2) optionally initialise a local HF model
      3) compute metrics & diagnostics

    Returns:
      (df_input, metrics_df, details_dict)
    """
    vetro_cfg, _ = load_vetro_yaml(formulas_yaml_path)
    prompts_cfg, _ = load_prompts_yaml(prompts_yaml_path)

    hf_runner = None
    if use_llm:
        hf_model_name = (hf_model_name or "").strip() or "google/flan-t5-base"
        hf_runner = get_hf_runner(hf_model_name)

    metrics_df, details = compute_metrics(
        df=df,
        vetro_cfg=vetro_cfg,
        prompts_cfg=prompts_cfg,
        use_llm=use_llm,
        hf_runner=hf_runner,
        dataset_description=dataset_description,
        file_name=file_name,
        file_ext=file_ext,
    )

    return df, metrics_df, details
