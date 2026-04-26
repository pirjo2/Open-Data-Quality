from __future__ import annotations

from typing import Any, Dict, Tuple, Optional
from itertools import combinations
import math
import re
import json

import pandas as pd
from core.llm import get_prompt_template, parse_llm_json_loose

DEBUG_PRINT_PROMPTS = False

# --- Turvaline tingimusavaldiste eval --- #
COND_ALLOWED_RE = re.compile(r"^[0-9a-zA-Z_ .<>=!()+\-*/]+$")

SYMBOL_HINTS: Dict[str, str] = {
    "pb": "publisher is present",
    "t": "title is present",
    "d": "description is present",
    "dc": "date of creation is present",
    "cv": "coverage is present",
    "l": "language is present",
    "id": "identifier is present",
    "s": "source is present",
    "c": "category/theme is present",

    "dp": "date of publication",
    "sd": "start date of the covered period",
    "edp": "end date of the covered period",
    "ed": "expiration date or end of validity",
    "cd": "current date or date when dataset became available again",

    "lu": "list or history of updates is present, including natural-language update frequency",
    "du": "dates of updates are explicitly present",

    "ncr": "number of rows that are not current",
    "ns": "number of columns for which a standard should apply",
    "nsc": "number of standardized columns",
    "ncm": "number of columns that have metadata/description",
    "ncuf": "number of columns in comprehensible format",
    "nce": "number of cells that are inaccurate or syntactically invalid",

    "s1": "open data is available on the web",
    "s2": "open data is machine-readable",
    "s3": "open data is in a non-proprietary format",
    "s4": "open data uses URIs / stable identifiers",
    "s5": "open data is linked open data",

    "sc": "scale constant used in aggregation comparison",
    "oav": "observed aggregate value calculated from detailed data",
    "dav": "declared aggregate value reported in the data",
    "e": "error or difference used in aggregation accuracy",
}


def _safe_eval_condition(expr: str, env: Dict[str, Any]) -> bool:
    """
    Safely evaluate a simple boolean expression used in normalization conditionals.
    Only a restricted character set is allowed and builtins are disabled.
    """
    if not isinstance(expr, str) or not COND_ALLOWED_RE.match(expr):
        return False
    try:
        return bool(eval(expr, {"__builtins__": {}}, env))
    except Exception:
        return False


# --- Väljendipuu hindamine (Variant B) --- #
def _eval_expr(node: Any, env: Dict[str, Any]) -> float:
    """
    Evaluate a metric expression tree against the environment `env`.
    """
    if node is None:
        return float("nan")

    # Numeric literal
    if isinstance(node, (int, float)):
        return float(node)

    # Symbol / string literal
    if isinstance(node, str):
        if node in env:
            v = env[node]
            if v is None:
                return 0.0
            try:
                return float(v)
            except Exception:
                pass
        try:
            return float(node)
        except Exception:
            return 0.0

    # Operator node
    if isinstance(node, dict):
        op = node.get("operator")

        if op == "identity":
            return _eval_expr(node.get("operand", node.get("of")), env)

        if op == "add":
            if "terms" in node:
                total = 0.0
                for term in (node.get("terms") or []):
                    v = _eval_expr(term, env)
                    if isinstance(v, float) and math.isnan(v):
                        v = 0.0
                    total += v
                return total
            return _eval_expr(node.get("left"), env) + _eval_expr(node.get("right"), env)

        if op == "subtract":
            return _eval_expr(node.get("left"), env) - _eval_expr(node.get("right"), env)

        if op == "multiply":
            return _eval_expr(node.get("left"), env) * _eval_expr(node.get("right"), env)

        if op == "divide":
            denom = _eval_expr(node.get("right"), env)
            if denom == 0 or math.isnan(denom):
                return float("nan")
            return _eval_expr(node.get("left"), env) / denom

        if op == "abs_diff":
            return abs(_eval_expr(node.get("left"), env) - _eval_expr(node.get("right"), env))

        if op == "sum":
            items = node.get("of")
            if not isinstance(items, list):
                items = [items]
            total = 0.0
            for term in items:
                v = _eval_expr(term, env)
                if isinstance(v, float) and math.isnan(v):
                    v = 0.0
                total += v
            return total

        if op == "conditional":
            for rule in node.get("conditions", []) or []:
                if not isinstance(rule, dict):
                    continue
                if "if" in rule and _safe_eval_condition(rule["if"], env):
                    return _eval_expr(rule.get("then"), env)
                elif "elif" in rule and _safe_eval_condition(rule["elif"], env):
                    return _eval_expr(rule.get("then"), env)
                elif "else" in rule:
                    return _eval_expr(rule.get("else"), env)
            return float("nan")

    return float("nan")


def _date_to_num(value: Any) -> Optional[float]:
    """
    Convert a date-like value into a numeric day count using .toordinal().
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        ts = pd.to_datetime(value, errors="coerce", utc=True)
        if pd.isna(ts):
            return None
        return float(ts.date().toordinal())
    except Exception:
        return None
    
def _extract_currentness_anchor_from_text(
    text: str,
    df: pd.DataFrame,
) -> Tuple[Optional[str], Optional[str]]:
    if not text or not text.strip():
        return None, None

    columns = [str(c) for c in df.columns]

    found_col = None
    for col in sorted(columns, key=len, reverse=True):
        if re.search(rf"\bcolumn\s+[`\"']?{re.escape(col)}[`\"']?\b", text, re.I):
            found_col = col
            break
        if re.search(rf"\b[`\"']?{re.escape(col)}[`\"']?\s+column\b", text, re.I):
            found_col = col
            break

    if not found_col:
        return None, None

    value_patterns = [
        r"(?:current|latest|most recent)\s+(?:reporting|reference)?\s*period\s*(?:is|=|:)\s*([0-9]{4}-[0-9]{2}-[0-9]{2})",
        r"(?:current|latest|most recent)\s+(?:reporting|reference)?\s*period\s*(?:is|=|:)\s*([0-9]{4}-[0-9]{2})",
        r"(?:current|latest|most recent)\s+(?:reporting|reference)?\s*period\s*(?:is|=|:)\s*([0-9]{4}Q[1-4])",
        r"(?:current|latest|most recent)\s+(?:reporting|reference)?\s*period\s*(?:is|=|:)\s*([0-9]{4})",
    ]

    candidates = []
    for pat in value_patterns:
        for m in re.finditer(pat, text, re.I):
            candidates.append(m.group(1).strip())

    if not candidates:
        return None, None

    series_values = set(df[found_col].astype(str).str.strip().tolist())

    for candidate in candidates:
        if candidate in series_values:
            return found_col, candidate

    return found_col, candidates[0]

def _infer_currentness_anchor(
    df: pd.DataFrame,
    manual_metadata_text: str,
    llm_runner,
    prompts_cfg: Optional[Dict[str, Any]] = None,
    prompt_regime: str = "zero_shot",
) -> Tuple[Optional[str], Optional[str], str]:
    if not llm_runner or not manual_metadata_text or not manual_metadata_text.strip():
        return None, None, "not_used"

    rule_col, rule_val = _extract_currentness_anchor_from_text(
        manual_metadata_text,
        df,
    )
    if rule_col and rule_val:
        return rule_col, rule_val, "rule_text"

    if not llm_runner:
        return None, None, "not_used"

    sample_rows = df.head(5).to_dict(orient="records")

    fallback_prompt = """
Return ONLY valid JSON with these keys:
- current_column
- current_value

Task:
Infer which dataset column is used to evaluate whether a row is current,
and what value represents the current reference period.

Rules:
- Use only a column that actually exists in the dataset
- current_value must match the dataset format if possible
- If uncertain, return {{}}

Metadata text:
{manual_metadata_text}

Columns:
{columns}

Sample rows:
{sample_rows}
"""

    prompt_template, prompt_source = get_prompt_template(
        prompts_cfg,
        prompt_regime,
        "currentness_anchor",
    )

    prompt = prompt_template.format(
        manual_metadata_text=manual_metadata_text,
        columns=list(df.columns),
        sample_rows=json.dumps(sample_rows, default=str),
    )
    if DEBUG_PRINT_PROMPTS:
        print("\n--- CURRENTNESS PROMPT START ---\n")
        print(prompt)
        print("\n--- CURENTNESS PROMPT END ---\n")

    raw = llm_runner(prompt, 384)

    try:
        data = json.loads(raw)
        if not isinstance(data, dict):
            return None, None, prompt_source
    except Exception:
        return None, None, prompt_source

    col = data.get("current_column")
    val = data.get("current_value")

    if col not in df.columns:
        return None, None, prompt_source
    if val is None:
        return None, None, prompt_source

    return str(col), str(val), prompt_source

def _chunk_list(items, chunk_size):
    for i in range(0, len(items), chunk_size):
        yield items[i:i + chunk_size]

COUNT_SYMBOLS = {
    "ncr", "ns", "nsc", "ncm", "ncuf", "nce"
}

COUNT_SYMBOL_MAX = {
    "ncr": "nr",
    "ns": "nc",
    "nsc": "ns",
    "ncm": "nc",
    "ncuf": "nc",
    "nce": "ncl",
}



def _get_numeric_series_map(
    df: pd.DataFrame,
    min_valid_ratio: float = 0.95,
) -> Dict[str, pd.Series]:
    """
    Return columns that are numeric or almost numeric after coercion.
    """
    numeric_map: Dict[str, pd.Series] = {}

    for col in df.columns:
        s = pd.to_numeric(df[col], errors="coerce")
        if float(s.notna().mean()) >= min_valid_ratio:
            numeric_map[str(col)] = s.astype(float)

    return numeric_map


def _infer_aggregation_symbols_from_table(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Language-independent aggregation inference.

    Idea:
    - find numeric columns
    - try each numeric column as declared aggregate (dav)
    - try subsets of the other numeric columns as components
    - choose the combination where row-wise component sums best match the candidate column

    Returns:
      {
        "dav": ...,
        "oav": ...,
        "e": ...,
        "sc": ...
      }
    or {} if no plausible aggregation structure is found.
    """
    numeric_map = _get_numeric_series_map(df)
    cols = list(numeric_map.keys())

    # Require at least 4 numeric columns:
    # one aggregate candidate + at least 2 components + usually one id/extra column
    if len(cols) < 4:
        return {}

    best: Optional[Dict[str, Any]] = None
    max_subset_size = min(4, len(cols) - 1)
    tol = 1e-9

    for target_col in cols:
        other_cols = [c for c in cols if c != target_col]

        for r in range(2, max_subset_size + 1):
            for comp_cols in combinations(other_cols, r):
                temp = pd.concat(
                    [numeric_map[target_col]] + [numeric_map[c] for c in comp_cols],
                    axis=1,
                )

                valid = temp.notna().all(axis=1)
                if int(valid.sum()) == 0:
                    continue

                dav_series = numeric_map[target_col][valid]
                component_frame = pd.concat(
                    [numeric_map[c][valid] for c in comp_cols],
                    axis=1,
                )
                oav_series = component_frame.sum(axis=1)

                diff_series = (oav_series - dav_series).abs()

                exact_ratio = float((diff_series <= tol).mean())
                total_abs_error = float(diff_series.sum())
                mean_abs_error = float(diff_series.mean())
                scale = max(float(dav_series.abs().sum()), 1.0)

                score = (
                    exact_ratio,
                    -(total_abs_error / scale),
                    -mean_abs_error,
                    len(comp_cols),
                )

                candidate = {
                    "target_col": target_col,
                    "component_cols": list(comp_cols),
                    "exact_ratio": exact_ratio,
                    "total_abs_error": total_abs_error,
                    "mean_abs_error": mean_abs_error,
                    "scale": scale,
                    "dav_total": float(dav_series.sum()),
                    "oav_total": float(oav_series.sum()),
                    "score": score,
                }

                if best is None or candidate["score"] > best["score"]:
                    best = candidate

    if best is None:
        return {}

    # Loose plausibility check:
    # reject only if the best candidate is still clearly nonsense
    if best["exact_ratio"] == 0.0 and best["total_abs_error"] > best["scale"]:
        return {}

    return {
        "dav": best["dav_total"],
        "oav": best["oav_total"],
        "e": best["total_abs_error"],
        "sc": best["scale"],
    }

def _to_json_safe(value: Any) -> Any:
    """
    Convert numpy scalar values into normal Python values
    so prompt examples look cleaner for the LLM.
    """
    if isinstance(value, dict):
        return {k: _to_json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_to_json_safe(v) for v in value]

    try:
        import numpy as np
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
    except Exception:
        pass

    return value

def _filter_prompt_metadata(meta: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Keep only meaningful metadata values for prompt context.
    Avoid crashing on list/dict values coming from Trino.
    """
    out: Dict[str, Any] = {}

    for k, v in (meta or {}).items():
        if v is None:
            continue

        if isinstance(v, str):
            if not v.strip():
                continue
            if v.strip() in {"0", "0.0"}:
                continue
            out[k] = v
            continue

        if isinstance(v, (list, tuple)):
            if len(v) == 0:
                continue
            out[k] = _to_json_safe(v)
            continue

        if isinstance(v, dict):
            if len(v) == 0:
                continue
            out[k] = _to_json_safe(v)
            continue

        if isinstance(v, (int, float)):
            if v == 0:
                continue
            out[k] = v
            continue

        out[k] = _to_json_safe(v)

    return out

ALLOW_PARTIAL_RESULT_METRICS = {
    "traceability.track_of_creation",
    "traceability.track_of_updates",
    "compliance.egms_compliance",
    "compliance.five_stars_open_data",
}

def _metric_allows_partial_result(metric_id: str) -> bool:
    return metric_id in ALLOW_PARTIAL_RESULT_METRICS

def _finalise_metric_result(
    metric_id: str,
    normalised_value: Optional[float],
    missing_required_inputs: list[str],
) -> Optional[float]:
    """
    Strict metrics:
        any missing required input -> None

    Partial metrics:
        if the normalised value is computable, keep it even when some
        required inputs are missing.
    """
    if normalised_value is None:
        return None

    if _metric_allows_partial_result(metric_id):
        return normalised_value

    if missing_required_inputs:
        return None

    return normalised_value

# --- Automaatsete sisendite arvutamine --- #
def _auto_inputs(df: pd.DataFrame, file_ext: Optional[str] = None) -> Dict[str, Any]:
    auto: Dict[str, Any] = {}

    N = int(df.shape[1])
    R = int(df.shape[0])
    auto["nc"] = float(N)
    auto["nr"] = float(R)

    # Completeness
    auto["ncl"] = float(N * R)
    na_mask = df.isna()
    auto["ic"] = float(na_mask.sum().sum())
    auto["nir"] = float(na_mask.any(axis=1).sum())

    # Accuracy: simple baseline

    # Currentness: detect a primary date column
        # ------------------------------------------------------------------
    # AI-first approach:
    # Do NOT hard-code semantic interpretation from column names here.
    # Keep only deterministic table statistics in _auto_inputs().
    # Semantic symbols will be inferred later from:
    # - raw manual metadata text
    # - Trino metadata
    # - column profiles
    # - sample rows
    # ------------------------------------------------------------------

    # Currentness / publication / expiration
    auto["ncr"] = None
    auto["sd"] = None
    auto["edp"] = None
    auto["ed"] = None
    auto["cd"] = None
    auto["dp"] = None
    auto["du"] = None

    # Standardization / understandability
    auto["ns"] = None
    auto["nsc"] = None
    auto["ncm"] = None
    auto["ncuf"] = None

    # 5-star open data
    auto["s1"] = None
    auto["s2"] = None
    auto["s3"] = None
    auto["s4"] = None
    auto["s5"] = None

    # Traceability / updates
    auto["s"] = None
    auto["dc"] = None
    auto["lu"] = None

    # Accuracy / aggregation
    auto["nce"] = None
    auto["sc"] = None
    auto["oav"] = None
    auto["dav"] = None
    auto["e"] = None

    return auto

def compute_metrics(
    df: pd.DataFrame,
    formulas_cfg: Dict[str, Any],
    prompt_defs: Optional[Dict[str, Any]],
    use_llm: bool,
    llm_runner,
    file_ext: Optional[str] = None,
    manual_metadata: Optional[Dict[str, Any]] = None,
    manual_metadata_text: Optional[str] = None,
    trino_metadata: Optional[Dict[str, Any]] = None,
    trino_metadata_raw: Optional[Dict[str, Any]] = None,
    prompts_cfg: Optional[Dict[str, Any]] = None,
    prompt_regime: str = "zero_shot",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    # Accept either the full config or just vetro_methodology
    vetro = formulas_cfg.get("vetro_methodology", formulas_cfg)

    # Labels are stored as flattened keys like "completeness.percentage_of_complete_cells"
    label_map: Dict[str, str] = {
        k: v for k, v in vetro.items() if isinstance(v, str) and "." in k
    }

    N = int(df.shape[1])
    R = int(df.shape[0])

    auto_inputs = _auto_inputs(df, file_ext)

    env: Dict[str, Any] = {"N": float(N), "R": float(R)}

    # Install numeric auto inputs into env (dates converted later)
    for k, v in auto_inputs.items():
        if k in ("sd", "edp", "ed", "cd", "dp"):
            continue
        if isinstance(v, (int, float)):
            env[k] = float(v)

    details: Dict[str, Any] = {
        "auto_inputs": auto_inputs,
        "symbol_values": {},
        "symbol_source": {},
        "llm_confidence": {},
        "llm_raw": {},
        "llm_evidence": {},
        "prompt_sources": {},
        "symbol_trace": {},
        "formula_trace": {},
    }

    def _record_symbol_trace(sym: str, value: Any, source: str, evidence: str = "", confidence: Any = None) -> None:
        details["symbol_trace"][sym] = {
            "value": value,
            "source": source,
            "evidence": evidence or "",
            "confidence": confidence,
        }

    # Collect symbols used by metrics
    required_symbols = set()
    for dim, dim_obj in vetro.items():
        if not isinstance(dim_obj, dict):
            continue
        for metric_key, metric_obj in dim_obj.items():
            if not isinstance(metric_obj, dict):
                continue
            for inp in metric_obj.get("inputs", []) or []:
                if isinstance(inp, dict):
                    for sym in inp.values():
                        if isinstance(sym, str):
                            required_symbols.add(sym)

    manual_metadata = manual_metadata or {}
    manual_metadata_text = manual_metadata_text or ""
    trino_metadata = trino_metadata or {}
    trino_metadata_raw = trino_metadata_raw or {}
    prompts_cfg = prompts_cfg or {}
    aggregation_auto = _infer_aggregation_symbols_from_table(df)
    if aggregation_auto:
        details["aggregation_auto"] = aggregation_auto
        for sym, val in aggregation_auto.items():
            details["symbol_values"][sym] = val
            details["symbol_source"][sym] = "auto_aggregation"
            _record_symbol_trace(sym, val, "auto_aggregation", evidence="Derived from table aggregation heuristics")
            env[sym] = val
        # AI infers currentness semantics, code computes ncr deterministically
    if auto_inputs.get("ncr") is None and use_llm and llm_runner is not None:
        current_col, current_val, currentness_prompt_source = _infer_currentness_anchor(
            df=df,
            manual_metadata_text=manual_metadata_text,
            llm_runner=llm_runner,
            prompts_cfg=prompts_cfg,
            prompt_regime=prompt_regime,
        )
        details["currentness_anchor_debug"] = {
            "column": current_col,
            "value": current_val,
            "prompt_source": currentness_prompt_source,
        }

        details["prompt_sources"]["currentness_anchor"] = currentness_prompt_source

        if current_col and current_val:
            series = df[current_col].astype(str).str.strip()
            ncr = float((series != current_val).sum())

            details["symbol_values"]["ncr"] = ncr
            details["symbol_source"]["ncr"] = "ai+auto"
            details["llm_evidence"]["ncr"] = f"current_column={current_col}, current_value={current_val}"
            _record_symbol_trace(
                "ncr",
                ncr,
                "ai+auto",
                evidence=f"Computed from currentness anchor: column={current_col}, current_value={current_val}",
            )
            env["ncr"] = ncr

    cov = trino_metadata_raw.get("temporalcoverage")

    mod_date = trino_metadata_raw.get("modificationdate")
    if mod_date:
        details["symbol_values"]["du"] = 1.0
        details["symbol_source"]["du"] = "parser"
        _record_symbol_trace("du", 1.0, "parser", evidence="modificationdate present in trino metadata raw")

    if isinstance(cov, str):
        years = re.findall(r"\d{4}", cov)
        if len(years) >= 2:
            sd_val = f"{years[0]}-01-01"
            edp_val = f"{years[1]}-12-31"

            details["symbol_values"]["sd"] = sd_val
            details["symbol_source"]["sd"] = "parser"
            _record_symbol_trace("sd", sd_val, "parser", evidence=f"Parsed from temporalcoverage: {cov}")

            details["symbol_values"]["edp"] = edp_val
            details["symbol_source"]["edp"] = "parser"
            _record_symbol_trace("edp", edp_val, "parser", evidence=f"Parsed from temporalcoverage: {cov}")

    # --------- PRIORITY RESOLUTION: auto -> trino -> manual -> missing ----------
    details["llm_debug"] = {
        "use_llm": use_llm,
        "llm_runner_available": llm_runner is not None,
        "prompt_defs_available": bool(prompt_defs),
        "prompt_regimes_available": bool(prompts_cfg.get("prompt_regimes", {})),
        "required_symbols": sorted(required_symbols),
        "missing_syms": [],
        "calls": [],
    }

    for sym in sorted(required_symbols):
        if details["symbol_source"].get(sym) in {"parser", "ai+auto", "auto_aggregation"}:
            continue

        # 1) explicit/manual metadata is strongest
        if sym in manual_metadata and manual_metadata[sym] is not None:
            val = manual_metadata[sym]
            details["symbol_values"][sym] = val
            details["symbol_source"][sym] = "manual"
            _record_symbol_trace(sym, val, "manual", evidence="Provided by user/manual metadata")
            env[sym] = val
            continue

        # 2) trino metadata
        if sym in trino_metadata and trino_metadata[sym] is not None:
            val = trino_metadata[sym]
            details["symbol_values"][sym] = val
            details["symbol_source"][sym] = "trino"
            _record_symbol_trace(sym, val, "trino", evidence="Resolved from normalized Trino metadata")
            env[sym] = val
            continue

        # 3) auto inputs last
        if sym in auto_inputs and auto_inputs[sym] is not None:
            val = auto_inputs[sym]
            details["symbol_values"][sym] = val
            details["symbol_source"][sym] = "auto"
            _record_symbol_trace(sym, val, "auto", evidence="Derived automatically from dataframe")
            env[sym] = val
            continue

        details["symbol_values"][sym] = None
        details["symbol_source"][sym] = "missing"

    # --------- LLM fallback (missing + auto=None refinement) ----------
    if use_llm and llm_runner is not None:

        '''refinable_symbols = {
            "pb", "t", "d", "dc", "cv", "l", "id", "s", "c",
            "dp", "sd", "edp", "ed", "cd",
            "lu", "du"
        }'''
        refinable_symbols = set(required_symbols)

        missing_syms = []

        # Collect symbols first
        for sym in sorted(required_symbols):
            #if sym not in prompt_defs:
                #continue

            source = details["symbol_source"].get(sym)
            val = details["symbol_values"].get(sym)

            # Case 1: missing
            if source == "missing":
                missing_syms.append(sym)
                continue

            # Case 2: auto=None and refinable
            if (
                sym in refinable_symbols
                and source == "auto"
                and val is None
            ):
                missing_syms.append(sym)

        details["llm_debug"]["missing_syms"] = list(missing_syms)

        # Only now build context
        if missing_syms:
            chunk_size = 4
            context_lines = []
            context_lines.append("Columns:")
            context_lines.append(", ".join(str(c) for c in df.columns))
            context_lines.append("")

            context_lines.append("Basic column profiles:")
            for col in df.columns:
                s = df[col]
                dtype = str(s.dtype)
                missing_ratio = float(s.isna().mean())
                sample_vals = _to_json_safe(list(s.dropna().unique()[:3]))                
                context_lines.append(
                    f"- {col}: dtype={dtype}, missing={missing_ratio:.3f}, samples={sample_vals}"
                )

            context_lines.append("")
            context_lines.append("Raw metadata record from portal (JSON):")
            context_lines.append(
                json.dumps(trino_metadata_raw, indent=2, default=str)
            )

            filtered_manual_metadata = _filter_prompt_metadata(manual_metadata)

            if filtered_manual_metadata:
                context_lines.append("")
                context_lines.append("Manual metadata overrides:")
                context_lines.append(
                    json.dumps(filtered_manual_metadata, indent=2, default=str)
                )

            if manual_metadata_text.strip():
                context_lines.append("")
                context_lines.append("Raw manual metadata text:")
                context_lines.append(manual_metadata_text)

            context_lines.append("")
            context_lines.append("Sample rows (first 5):")
            sample_rows = _to_json_safe(df.head(5).to_dict(orient="records"))
            context_lines.append(json.dumps(sample_rows, indent=2, default=str))

            all_data: Dict[str, Any] = {}
            chunk_raw_map: Dict[str, str] = {}
            chunk_evidence_map = {}
            chunk_confidence_map = {}

            fallback_prompt = """
You are evaluating metadata and table semantics for an open dataset.

Infer the requested Vetrò symbols from:
- raw metadata text
- portal metadata JSON
- column names
- data types
- sample values
- sample rows

Rules:
- Return ONLY valid JSON.
- Use only the requested symbols as keys.
- Binary/presence symbols must be 0 or 1 only.
- Count / numeric symbols may be integers or floats if clearly inferable.
- Date symbols (dp, sd, edp, ed, cd) must be YYYY-MM-DD only if clearly supported.
- If evidence is insufficient, omit the symbol.
- Do not invent facts.

Requested symbols:
{requested_symbols}

Dataset context:
{context}
"""

            prompt_template, prompt_source = get_prompt_template(
                prompts_cfg,
                prompt_regime,
                "semantic_metric_inference",
            )

            details["prompt_sources"]["semantic_metric_inference"] = prompt_source

            base_context = "\n".join(context_lines)

            for chunk in _chunk_list(missing_syms, chunk_size):
                chunk_context_lines = [base_context]

                chunk_hint_lines = []
                for sym in chunk:
                    if sym in SYMBOL_HINTS:
                        chunk_hint_lines.append(f"- {sym}: {SYMBOL_HINTS[sym]}")

                if chunk_hint_lines:
                    chunk_context_lines.append("")
                    chunk_context_lines.append("Requested symbol meanings:")
                    chunk_context_lines.extend(chunk_hint_lines)

                chunk_context = "\n".join(chunk_context_lines)

                prompt = prompt_template.format(
                    requested_symbols=", ".join(chunk),
                    context=chunk_context,
                )
                if DEBUG_PRINT_PROMPTS:
                    print("\n--- SEMANTIC PROMPT START ---\n")
                    print(prompt)
                    print("\n--- SEMANTIC PROMPT END ---\n")

                raw = llm_runner(prompt, 384)

                details["llm_debug"]["calls"].append(
                    {
                        "symbols": list(chunk),
                        "raw": raw,
                    }
                )

                data = parse_llm_json_loose(raw)
                if not isinstance(data, dict):
                    data = {}

                top_level_values = _extract_top_level_llm_values(raw, list(chunk))
                data.update(top_level_values)

                evidence_data = data.get("__evidence__", {}) if isinstance(data.get("__evidence__"), dict) else {}
                confidence_data = data.get("__confidence__", {}) if isinstance(data.get("__confidence__"), dict) else {}

                for k, v in data.items():
                    if str(k).startswith("__"):
                        continue
                    all_data[k] = v
                    chunk_raw_map[k] = raw
                    chunk_evidence_map[k] = str(evidence_data.get(k, "") or "")
                    chunk_confidence_map[k] = confidence_data.get(k)

            date_symbols = {"dp", "sd", "edp", "ed", "cd"}

            binary_symbols = {
                "pb", "t", "d", "dc", "cv", "l", "id", "s", "c",
                "lu", "du",
                "s1", "s2", "s3", "s4", "s5",
            }

            numeric_symbols = {
                "ncr", "ns", "nsc", "ncm", "ncuf", "nce",
                "sc", "oav", "dav", "e",
            }

            count_symbols = {
                "ncr", "ns", "nsc", "ncm", "ncuf", "nce"
            }


            def _parse_llm_numeric_symbol(sym: str, raw_val: Any) -> Optional[float]:

                if raw_val is None:
                    return None

                if isinstance(raw_val, bool):
                    val = float(int(raw_val))

                elif isinstance(raw_val, (int, float)):
                    val = float(raw_val)

                elif isinstance(raw_val, str):
                    text = raw_val.strip()

                    if not re.fullmatch(r"[-+]?\d+(\.\d+)?", text):
                        return None

                    val = float(text)

                else:
                    return None

                count_symbols = {
                    "ncr", "ns", "nsc", "ncm", "ncuf", "nce"
                }

                if sym in count_symbols and val < 0:
                    return None

                return val

            def _extract_top_level_llm_values(raw_text: str, requested_symbols: list[str]) -> Dict[str, Any]:

                if not raw_text:
                    return {}

                head = re.split(
                    r'["\']?__(?:evidence|confidence)__["\']?\s*:',
                    raw_text,
                    maxsplit=1,
                    flags=re.IGNORECASE,
                )[0]

                out: Dict[str, Any] = {}

                for sym in requested_symbols:
                    pattern = rf'(?m)^\s*["\']?{re.escape(sym)}["\']?\s*:\s*([-+]?\d+(?:\.\d+)?)\s*,?\s*$'
                    m = re.search(pattern, head)

                    if m:
                        out[sym] = float(m.group(1))

                return out


            for sym in missing_syms:
                val = all_data.get(sym)

                if sym in binary_symbols:
                    if val in [0, 1]:
                        val = float(val)
                    else:
                        val = None

                elif sym in numeric_symbols:
                    val = _parse_llm_numeric_symbol(sym, val)

                elif sym in date_symbols:
                    if isinstance(val, str):
                        m = re.search(r"\d{4}-\d{2}-\d{2}", val)
                        val = m.group(0) if m else None
                    else:
                        val = None

                else:
                    if isinstance(val, (int, float)):
                        val = float(val)
                    else:
                        val = None

                details["llm_raw"][sym] = chunk_raw_map.get(sym, "")
                details["llm_confidence"][sym] = chunk_confidence_map.get(sym)
                details["llm_evidence"][sym] = chunk_evidence_map.get(sym, "")

                if details["symbol_source"].get(sym) in {"trino", "manual", "ai+auto"}:
                    continue

                if val is None:
                    details["symbol_source"][sym] = "llm_fail"
                    env.setdefault(sym, 0.0)
                else:
                    details["symbol_source"][sym] = "llm"
                    details["symbol_values"][sym] = val
                    _record_symbol_trace(
                        sym,
                        val,
                        "llm",
                        evidence=chunk_evidence_map.get(sym, ""),
                        confidence=chunk_confidence_map.get(sym),
                    )
                    env[sym] = val
    # Convert date-like symbols into numeric
    for sym in ("sd", "edp", "ed", "cd", "dp"):
        raw_val = details["symbol_values"].get(sym, auto_inputs.get(sym))
        num = _date_to_num(raw_val)
        if num is not None:
            env[sym] = num

    # Ensure all symbols exist
    for sym in required_symbols:
        env.setdefault(sym, 0.0)

    rows = []
    for dim, dim_obj in vetro.items():
        if not isinstance(dim_obj, dict):
            continue

        for metric_key, metric_obj in dim_obj.items():
            if not isinstance(metric_obj, dict):
                continue

            desc = metric_obj.get("description", "")
            inputs = metric_obj.get("inputs", [])
            if not inputs:
                continue

            interm = metric_obj.get("intermediate_calculation")
            if interm:
                if isinstance(interm, dict) and "assign" in interm:
                    interms = [interm]
                elif isinstance(interm, list):
                    interms = [x for x in interm if isinstance(x, dict)]
                else:
                    interms = []

                for ic in interms:
                    name = ic.get("assign")
                    expr = ic.get("expression")
                    if name and expr:
                        env[name] = _eval_expr(expr, env)

            formula = metric_obj.get("formula", {})
            norm = metric_obj.get("normalization", {})
            if not formula or not norm:
                continue

            f_assign = formula.get("assign")
            f_expr = formula.get("expression")
            n_assign = norm.get("assign")
            n_expr = norm.get("expression")
            if not (f_assign and f_expr and n_assign and n_expr):
                continue

            env[f_assign] = _eval_expr(f_expr, env)
            val = _eval_expr(n_expr, env)

            metric_id = f"{dim}.{metric_key}"
            label = label_map.get(metric_id, metric_id)

            normalised_val: Optional[float]
            if isinstance(val, (int, float)) and not math.isnan(val):
                normalised_val = float(val)
            else:
                normalised_val = None

            input_map: Dict[str, str] = {}
            for inp in inputs:
                if isinstance(inp, dict):
                    input_map.update(inp)

            required_symbols_for_metric = list(input_map.values())
            missing_required_inputs = [
                sym for sym in required_symbols_for_metric
                if details["symbol_values"].get(sym) is None
            ]

            out_val = _finalise_metric_result(
                metric_id=metric_id,
                normalised_value=normalised_val,
                missing_required_inputs=missing_required_inputs,
            )

            intermediate_values = {}
            if interm:
                if isinstance(interm, dict) and "assign" in interm:
                    interm_list = [interm]
                elif isinstance(interm, list):
                    interm_list = [x for x in interm if isinstance(x, dict)]
                else:
                    interm_list = []
                for ic in interm_list:
                    assign_name = ic.get("assign")
                    if assign_name:
                        intermediate_values[assign_name] = env.get(assign_name)
            else:
                intermediate_values = {}

            details["formula_trace"][metric_id] = {
                "metric_label": label,
                "dimension": dim,
                "inputs": {
                    input_name: {
                        "symbol": symbol_name,
                        "raw_value": details["symbol_values"].get(symbol_name),
                        "numeric_env_value": env.get(symbol_name),
                        "source": details["symbol_source"].get(symbol_name),
                        "evidence": details["symbol_trace"].get(symbol_name, {}).get("evidence", ""),
                    }
                    for input_name, symbol_name in input_map.items()
                },
                "missing_required_inputs": missing_required_inputs,
                "intermediate_values": intermediate_values,
                "formula_assign": f_assign,
                "formula_value": env.get(f_assign),
                "normalised_assign": n_assign,
                "result": out_val,
                "null_policy": "allow_partial" if _metric_allows_partial_result(metric_id) else "strict",
            }

            rows.append(
                {
                    "dimension": dim,
                    "metric": metric_key,
                    "metric_id": metric_id,
                    "value": out_val,
                    "description": desc,
                    "metric_label": label,
                }
            )

    metrics_df = pd.DataFrame(rows)
    return metrics_df, details