from __future__ import annotations

import json
import os
import traceback
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
TEST_ROOT = REPO_ROOT / "testkomplekt" / "vetro_tests"
FORMULAS_YAML = REPO_ROOT / "configs" / "formulas.yaml"
PROMPTS_YAML = REPO_ROOT / "configs" / "prompts.yaml"
OUTPUT_ROOT = REPO_ROOT / "experiment_outputs"
API_KEY_FILE = REPO_ROOT / "api-key.txt"
OPENAI_API_KEY = API_KEY_FILE.read_text(encoding="utf-8").strip()


import pandas as pd

from core.pipeline import run_quality_assessment
from core.metadata_utils import (
    parse_kv_metadata,
    extract_symbols_from_realistic_text,
    normalize_metadata_to_symbols,
)
from core.llm import get_llm_runner, infer_manual_metadata_symbols
import yaml

# Muuda siin oma jooksuvalikud
MODEL_SPECS = [
    {"provider": "openai", "model": "gpt-4.1"},
    {"provider": "openai", "model": "gpt-4.1-mini"},
    {"provider": "openai", "model": "gpt-5-mini"},
    {"provider": "openai", "model": "gpt-5"},
    {"provider": "huggingface", "model": "google/flan-t5-small"},
    {"provider": "huggingface", "model": "google/flan-t5-base"},
    {"provider": "huggingface", "model": "google/flan-t5-large"},
]

PROMPT_REGIMES = [
    "zero_shot",
    "few_shot",
    "reasoning",
]

USE_LLM = True
DATA_EXTS = {".csv", ".tsv", ".txt", ".xls", ".xlsx"}


@dataclass
class TestCase:
    case_id: str
    dataset_path: Path
    metadata_path: Optional[Path]
    label: str


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_table(path: Path) -> pd.DataFrame:
    ext = path.suffix.lower()
    if ext in {".csv", ".tsv", ".txt"}:
        return pd.read_csv(path, sep=None, engine="python")
    if ext in {".xls", ".xlsx"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported dataset type: {ext}")


def detect_label(name: str) -> str:
    up = name.upper()
    if "PLUS" in up:
        return "PLUS"
    if "MINUS" in up:
        return "MINUS"
    return "UNKNOWN"


def build_case_id(path: Path) -> str:
    stem = path.stem
    stem = stem.replace("_PLUS", "").replace("_MINUS", "")
    stem = stem.replace("PLUS", "").replace("MINUS", "")
    return stem.strip("_-")


def find_metadata_for_dataset(dataset_path: Path) -> Optional[Path]:
    same_stem_txt = dataset_path.with_suffix(".txt")
    if same_stem_txt.exists():
        return same_stem_txt

    same_stem_md = dataset_path.with_suffix(".md")
    if same_stem_md.exists():
        return same_stem_md

    # fallback: otsi sama prefiksiga txt
    #candidates = list(dataset_path.parent.glob(f"{dataset_path.stem}*.txt"))
    #if candidates:
    #    return sorted(candidates)[0]

    return None


def discover_test_cases(test_root: Path) -> List[TestCase]:
    cases: List[TestCase] = []
    for path in sorted(test_root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in DATA_EXTS:
            continue
        if path.suffix.lower() == ".txt":
            continue

        metadata_path = find_metadata_for_dataset(path)
        cases.append(
            TestCase(
                case_id=build_case_id(path),
                dataset_path=path,
                metadata_path=metadata_path,
                label=detect_label(path.name),
            )
        )
    return cases


def prepare_manual_metadata(
    metadata_text: str,
    llm_provider: str,
    llm_model_name: str,
    prompt_regime: str,
) -> Dict[str, Any]:
    manual_metadata_raw = parse_kv_metadata(metadata_text)
    manual_metadata = normalize_metadata_to_symbols(manual_metadata_raw)
    manual_metadata_rule = extract_symbols_from_realistic_text(metadata_text)

    manual_metadata_llm = {}
    if USE_LLM and metadata_text.strip():
        with open(PROMPTS_YAML, "r", encoding="utf-8") as f:
            prompts_cfg = yaml.safe_load(f) or {}

        runner = get_llm_runner(
            provider=llm_provider,
            model_name=llm_model_name,
            api_key=OPENAI_API_KEY,
        )
        try:
            manual_metadata_llm, _, _ = infer_manual_metadata_symbols(
                metadata_text,
                runner,
                prompts_cfg=prompts_cfg,
                prompt_regime=prompt_regime,
            )
        except Exception:
            manual_metadata_llm = {}

    merged = dict(manual_metadata_llm)
    merged.update(manual_metadata_rule)
    merged.update(manual_metadata)
    return merged


def run_one_case(
    case: TestCase,
    provider: str,
    model: str,
    prompt_regime: str,
) -> Dict[str, Any]:
    started_at = utc_now_iso()
    start_dt = datetime.now(timezone.utc)

    try:
        df = load_table(case.dataset_path)
        metadata_text = ""
        if case.metadata_path and case.metadata_path.exists():
            metadata_text = case.metadata_path.read_text(encoding="utf-8", errors="ignore")

        manual_metadata = prepare_manual_metadata(
            metadata_text=metadata_text,
            llm_provider=provider,
            llm_model_name=model,
            prompt_regime=prompt_regime,
        )

        metrics_df, details = run_quality_assessment(
            df=df,
            formulas_yaml_path=str(FORMULAS_YAML),
            prompts_yaml_path=str(PROMPTS_YAML),
            use_llm=USE_LLM,
            llm_provider=provider,
            llm_model_name=model,
            openai_api_key=OPENAI_API_KEY,
            file_ext=case.dataset_path.suffix.lower(),
            manual_metadata=manual_metadata,
            manual_metadata_text=metadata_text,
            trino_metadata={},
            trino_metadata_raw={},
            prompt_regime=prompt_regime,
        )

        end_dt = datetime.now(timezone.utc)
        ended_at = utc_now_iso()
        duration_s = (end_dt - start_dt).total_seconds()

        return {
            "status": "ok",
            "started_at": started_at,
            "ended_at": ended_at,
            "duration_s": duration_s,
            "metrics_df": metrics_df,
            "details": details,
            "error": "",
        }

    except Exception as e:
        end_dt = datetime.now(timezone.utc)
        ended_at = utc_now_iso()
        duration_s = (end_dt - start_dt).total_seconds()

        return {
            "status": "error",
            "started_at": started_at,
            "ended_at": ended_at,
            "duration_s": duration_s,
            "metrics_df": pd.DataFrame(),
            "details": {},
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(),
        }


def main() -> None:
    cases = discover_test_cases(TEST_ROOT)

    if not cases:
        raise RuntimeError(f"No test cases found under {TEST_ROOT}")

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = OUTPUT_ROOT / f"synthetic_run_{run_stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: List[Dict[str, Any]] = []
    metric_rows: List[Dict[str, Any]] = []
    details_path = out_dir / "details.jsonl"

    total_runs = len(cases) * len(MODEL_SPECS) * len(PROMPT_REGIMES)
    counter = 0

    for case in cases:
        for spec in MODEL_SPECS:
            for regime in PROMPT_REGIMES:
                counter += 1
                provider = spec["provider"]
                model = spec["model"]

                print(f"[{counter}/{total_runs}] {case.dataset_path.name} | {provider}:{model} | {regime}")

                result = run_one_case(
                    case=case,
                    provider=provider,
                    model=model,
                    prompt_regime=regime,
                )

                summary_row = {
                    "case_id": case.case_id,
                    "label": case.label,
                    "dataset_file": str(case.dataset_path.relative_to(REPO_ROOT)),
                    "metadata_file": str(case.metadata_path.relative_to(REPO_ROOT)) if case.metadata_path else "",
                    "provider": provider,
                    "model": model,
                    "prompt_regime": regime,
                    "status": result["status"],
                    "started_at": result["started_at"],
                    "ended_at": result["ended_at"],
                    "duration_s": result["duration_s"],
                    "error": result.get("error", ""),
                }

                details = result.get("details", {})
                prompt_sources = details.get("prompt_sources", {}) if isinstance(details, dict) else {}
                llm_debug = details.get("llm_debug", {}) if isinstance(details, dict) else {}

                summary_row["prompt_sources_json"] = json.dumps(prompt_sources, ensure_ascii=False)
                summary_row["llm_calls"] = len(llm_debug.get("calls", [])) if isinstance(llm_debug, dict) else 0

                summary_rows.append(summary_row)

                metrics_df = result["metrics_df"]
                if not metrics_df.empty:
                    for _, row in metrics_df.iterrows():
                        metric_rows.append(
                            {
                                "case_id": case.case_id,
                                "label": case.label,
                                "dataset_file": str(case.dataset_path.relative_to(REPO_ROOT)),
                                "provider": provider,
                                "model": model,
                                "prompt_regime": regime,
                                "status": result["status"],
                                "started_at": result["started_at"],
                                "ended_at": result["ended_at"],
                                "duration_s": result["duration_s"],
                                "metric_id": row.get("metric_id"),
                                "metric_label": row.get("metric_label"),
                                "dimension": row.get("dimension"),
                                "value": row.get("value"),
                            }
                        )

                with details_path.open("a", encoding="utf-8") as f:
                    f.write(
                        json.dumps(
                            {
                                **summary_row,
                                "details": details,
                                "traceback": result.get("traceback", ""),
                            },
                            ensure_ascii=False,
                            default=str,
                        )
                        + "\n"
                    )

    summary_df = pd.DataFrame(summary_rows)
    metrics_long_df = pd.DataFrame(metric_rows)

    if metrics_long_df.empty:
        raise RuntimeError(
            "metrics_long_df is empty. No metric rows were collected."
        )

    target_metric_map = {
        "01": "traceability.track_of_creation",
        "02": "traceability.track_of_updates",
        "03": "currentness.percentage_of_current_rows",
        "04": "currentness.delay_in_publication",
        "05": "currentness.delay_after_expiration",
        "06": "completeness.percentage_of_complete_cells",
        "07": "completeness.percentage_of_complete_rows",
        "08": "compliance.percentage_of_standardized_columns",
        "09": "compliance.egms_compliance",
        "10": "compliance.five_stars_open_data",
        "11": "understandability.percentage_of_columns_with_metadata",
        "12": "understandability.percentage_of_columns_in_comprehensible_format",
        "13": "accuracy.percentage_of_syntactically_accurate_cells",
        "14": "accuracy.accuracy_in_aggregation",
    }

    def get_target_metric(case_id: str) -> str:
        prefix = str(case_id)[:2]
        return target_metric_map.get(prefix, "")

    metrics_long_df["target_metric_id"] = metrics_long_df["case_id"].apply(get_target_metric)

    target_only_df = metrics_long_df[
        metrics_long_df["metric_id"] == metrics_long_df["target_metric_id"]
    ].copy()

    target_overview_df = (
        target_only_df.pivot_table(
            index=["case_id", "provider", "model", "prompt_regime", "metric_id", "metric_label"],
            columns="label",
            values="value",
            aggfunc="first",
        )
        .reset_index()
    )

    target_overview_df["gap_plus_minus"] = target_overview_df["PLUS"] - target_overview_df["MINUS"]
    target_overview_df["plus_gt_minus"] = target_overview_df["gap_plus_minus"] > 0

    target_summary_df = (
        target_overview_df.groupby("case_id", dropna=False)
        .agg(
            runs=("plus_gt_minus", "size"),
            success_count=("plus_gt_minus", "sum"),
            success_rate=("plus_gt_minus", "mean"),
            avg_gap=("gap_plus_minus", "mean"),
            min_minus=("MINUS", "min"),
            max_minus=("MINUS", "max"),
            min_plus=("PLUS", "min"),
            max_plus=("PLUS", "max"),
        )
        .reset_index()
    )

    summary_csv = out_dir / "run_summary.csv"
    metrics_csv = out_dir / "metric_results_long.csv"

    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    metrics_long_df.to_csv(metrics_csv, index=False, encoding="utf-8-sig")

    # Soovi korral ka Excel
    excel_path = out_dir / "experiment_results.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="run_summary", index=False)
        metrics_long_df.to_excel(writer, sheet_name="metric_results", index=False)
        target_only_df.to_excel(writer, sheet_name="target_metric_rows", index=False)
        target_overview_df.to_excel(writer, sheet_name="target_metric_results", index=False)
        target_summary_df.to_excel(writer, sheet_name="target_metric_summary", index=False)
        
    print()
    print(f"Done. Output folder: {out_dir}")
    print(f"- {summary_csv.name}")
    print(f"- {metrics_csv.name}")
    print(f"- {excel_path.name}")
    print(f"- {details_path.name}")


if __name__ == "__main__":
    main()