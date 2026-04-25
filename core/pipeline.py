from __future__ import annotations

from typing import Any, Dict, Tuple, Optional

import yaml
import pandas as pd

from core.llm import get_llm_runner
from core.metrics_eval import compute_metrics


REQUIRED_REGIME_PROMPTS = (
    "manual_metadata_extraction",
    "currentness_anchor",
    "semantic_metric_inference",
)


def _load_yaml_dict(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping at the top level.")

    return data


def _validate_prompts_cfg(prompts_cfg: Dict[str, Any], prompt_regime: str) -> None:
    symbols = prompts_cfg.get("symbols")
    if not isinstance(symbols, dict) or not symbols:
        raise ValueError("prompts.yaml is missing top-level 'symbols' mapping.")

    prompt_regimes = prompts_cfg.get("prompt_regimes")
    if not isinstance(prompt_regimes, dict) or not prompt_regimes:
        raise ValueError("prompts.yaml is missing top-level 'prompt_regimes' mapping.")

    regime_cfg = prompt_regimes.get(prompt_regime)
    if not isinstance(regime_cfg, dict):
        available = ", ".join(sorted(str(k) for k in prompt_regimes.keys()))
        raise ValueError(
            f"Prompt regime '{prompt_regime}' not found in prompts.yaml. "
            f"Available regimes: {available}"
        )

    missing = [
        name
        for name in REQUIRED_REGIME_PROMPTS
        if not isinstance(regime_cfg.get(name), str) or not regime_cfg.get(name).strip()
    ]
    if missing:
        raise ValueError(
            f"prompts.yaml regime '{prompt_regime}' is missing required prompt(s): "
            + ", ".join(missing)
        )


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
    prompt_regime: str = "zero_shot",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    formulas_cfg = _load_yaml_dict(formulas_yaml_path)
    prompts_cfg = _load_yaml_dict(prompts_yaml_path)

    if use_llm:
        _validate_prompts_cfg(prompts_cfg, prompt_regime)

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
        prompts_cfg=prompts_cfg,
        prompt_regime=prompt_regime,
        use_llm=use_llm,
        llm_runner=llm_runner,
        file_ext=file_ext,
        manual_metadata=manual_metadata or {},
        manual_metadata_text=manual_metadata_text or "",
        trino_metadata=trino_metadata or {},
        trino_metadata_raw=trino_metadata_raw or {},
    )

    return metrics_df, details