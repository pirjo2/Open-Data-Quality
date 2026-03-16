'''from __future__ import annotations

from typing import Any, Dict, Tuple, Optional

import yaml
import pandas as pd

from core.llm import get_llm_runner
from core.metrics_eval import compute_metrics


def run_quality_assessment(
    df: pd.DataFrame,
    formulas_yaml_path: str,
    prompts_yaml_path: str,
    use_llm: bool,
    llm_provider: str,
    llm_model_name: str,
    openai_api_key: Optional[str] = None,
    file_ext: Optional[str] = None,
    manual_metadata: Optional[Dict[str, Any]] = None,
    manual_metadata_text: Optional[str] = None,
    trino_metadata: Optional[Dict[str, Any]] = None,
    trino_metadata_raw: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    with open(formulas_yaml_path, "r", encoding="utf-8") as f:
        formulas_cfg = yaml.safe_load(f)

    with open(prompts_yaml_path, "r", encoding="utf-8") as f:
        prompts_cfg = yaml.safe_load(f)

    prompt_defs: Dict[str, Any] = prompts_cfg.get("symbols", {})

    llm_runner = None
    if use_llm:
        llm_runner = get_llm_runner(
            provider=llm_provider,
            model_name=llm_model_name,
            api_key=openai_api_key,
        )

    metrics_df, details = compute_metrics(
        df=df,
        formulas_cfg=formulas_cfg,
        prompt_defs=prompt_defs,
        use_llm=use_llm,
        llm_runner=llm_runner,
        file_ext=file_ext,
        manual_metadata=manual_metadata or {},
        manual_metadata_text=manual_metadata_text or "",
        trino_metadata=trino_metadata or {},
        trino_metadata_raw=trino_metadata_raw or {},
    )

    return metrics_df, details'''
from __future__ import annotations

from typing import Any, Dict, Tuple, Optional

import yaml
import pandas as pd

from core.llm import get_llm_runner
from core.metrics_eval import compute_metrics


def run_quality_assessment(
    df: pd.DataFrame,
    formulas_yaml_path: str,
    prompts_yaml_path: str,
    use_llm: bool,
    llm_provider: str,
    llm_model_name: str,
    openai_api_key: Optional[str] = None,
    file_ext: Optional[str] = None,
    manual_metadata: Optional[Dict[str, Any]] = None,
    manual_metadata_text: Optional[str] = None,
    trino_metadata: Optional[Dict[str, Any]] = None,
    trino_metadata_raw: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    with open(formulas_yaml_path, "r", encoding="utf-8") as f:
        formulas_cfg = yaml.safe_load(f)

    with open(prompts_yaml_path, "r", encoding="utf-8") as f:
        prompts_cfg = yaml.safe_load(f)

    prompt_defs: Dict[str, Any] = prompts_cfg.get("symbols", {})

    llm_runner = None
    if use_llm:
        llm_runner = get_llm_runner(
            provider=llm_provider,
            model_name=llm_model_name,
            api_key=openai_api_key,
        )

    metrics_df, details = compute_metrics(
        df=df,
        formulas_cfg=formulas_cfg,
        prompt_defs=prompt_defs,
        use_llm=use_llm,
        llm_runner=llm_runner,
        file_ext=file_ext,
        manual_metadata=manual_metadata or {},
        manual_metadata_text=manual_metadata_text or "",
        trino_metadata=trino_metadata or {},
        trino_metadata_raw=trino_metadata_raw or {},
    )

    return metrics_df, details