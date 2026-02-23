from __future__ import annotations

from typing import Any, Dict, Tuple, Optional

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
    file_ext: Optional[str] = None,
    manual_metadata: Optional[Dict[str, Any]] = None,
    trino_metadata: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:

    with open(formulas_yaml_path, "r", encoding="utf-8") as f:
        formulas_cfg = yaml.safe_load(f)

    with open(prompts_yaml_path, "r", encoding="utf-8") as f:
        prompts_cfg = yaml.safe_load(f)

    prompt_defs: Dict[str, Any] = prompts_cfg.get("symbols", {})

    hf_runner = None
    if use_llm:
        hf_runner = get_hf_runner(hf_model_name)

    metrics_df, details = compute_metrics(
        df=df,
        formulas_cfg=formulas_cfg,
        prompt_defs=prompt_defs,
        use_llm=use_llm,
        hf_runner=hf_runner,
        file_ext=file_ext,
        manual_metadata=manual_metadata or {},
        trino_metadata=trino_metadata or {},
    )

    return metrics_df, details