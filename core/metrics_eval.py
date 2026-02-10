from __future__ import annotations

from typing import Any, Dict, Tuple, Optional
import math
import re
from datetime import date, datetime

import pandas as pd

from core.llm import infer_symbol

COND_ALLOWED_RE = re.compile(r"^[0-9a-zA-Z_ .<>=!()+\-*/]+$")


def _safe_eval_condition(expr: str, env: Dict[str, Any]) -> bool:
    if not isinstance(expr, str) or not COND_ALLOWED_RE.match(expr):
        return False
    try:
        return bool(eval(expr, {"__builtins__": {}}, env))
    except Exception:
        return False


def _eval_expr(node: Any, env: Dict[str, Any]) -> float:
    if node is None:
        return float("nan")

    if isinstance(node, str):
        v = env.get(node, None)
        if v is None:
            return float("nan")
        try:
            return float(v)
        except Exception:
            return float("nan")

    if isinstance(node, (int, float)):
        return float(node)

    if isinstance(node, dict) and "operator" in node:
        op = node["operator"]

        if op == "identity":
            return _eval_expr(node.get("of"), env)
        if op == "add":
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
        if op == "conditional":
            for rule in node.get("conditions", []) or []:
                if not isinstance(rule, dict):
                    continue
                if "if" in rule and _safe_eval_condition(rule["if"], env):
                    return _eval_expr(rule.get("then"), env)
                if "elif" in rule and _safe_eval_condition(rule["elif"], env):
                    return _eval_expr(rule.get("then"), env)
                if "else" in rule:
                    return _eval_expr(rule.get("else"), env)
            return float("nan")

    return float("nan")


def _profile_df(df: pd.DataFrame, sample_n: int = 3, max_cols: int = 40) -> Dict[str, Any]:
    profile: Dict[str, Any] = {}
    cols = list(df.columns)[:max_cols]
    for col in cols:
        s = df[col]
        missing = float(s.isna().mean()) if len(s) else 0.0
        dtype = str(s.dtype)
        samples = [x for x in s.dropna().head(sample_n).astype(str).tolist()]
        profile[str(col)] = {"dtype": dtype, "missing": round(missing, 6), "samples": samples}
    return profile


def _build_llm_context(df: pd.DataFrame, dataset_description: str = "", file_name: str = "", file_ext: str = "") -> str:
    cols = list(df.columns)
    profile = _profile_df(df, sample_n=3, max_cols=40)

    parts = []
    if file_name:
        parts.append(f"File name: {file_name}")
    if file_ext:
        parts.append(f"File type/extension: {file_ext}")
    if dataset_description:
        parts.append("Dataset description (from portal/user):")
        parts.append(dataset_description.strip())

    parts.append(f"The dataset has {len(cols)} columns (N={len(cols)}) and {len(df)} rows (R={len(df)}).")
    parts.append("Column names: " + ", ".join([str(c) for c in cols[:40]]))
    parts.append("Column profile (dtype, missing ratio, samples):")
    for col, info in profile.items():
        parts.append(f"- {col}: dtype={info['dtype']}, missing={info['missing']}, samples={info['samples']}")
    return "\n".join(parts)


def _date_to_num(d: str | date | datetime) -> Optional[int]:
    try:
        if isinstance(d, datetime):
            return d.date().toordinal()
        if isinstance(d, date):
            return d.toordinal()
        s = str(d).strip()
        dt = pd.to_datetime(s, errors="coerce")
        if pd.isna(dt):
            return None
        return dt.date().toordinal()
    except Exception:
        return None


def _find_date_column(df: pd.DataFrame) -> Optional[str]:
    for c in df.columns:
        if "date" in str(c).lower():
            return str(c)
    return None


def _count_empty_like(df: pd.DataFrame) -> Tuple[int, int]:
    empty_mask = df.isna()
    for c in df.columns:
        if df[c].dtype == object:
            empty_mask = empty_mask | df[c].astype(str).str.strip().eq("")
    empty_cells = int(empty_mask.sum().sum())
    incomplete_rows = int(empty_mask.any(axis=1).sum())
    return empty_cells, incomplete_rows


def _looks_like_identifier(col_name: str) -> bool:
    cl = col_name.lower()
    return cl in {"id", "uuid", "identifier"} or cl.endswith("_id") or "uuid" in cl


def _is_readable_colname(col_name: str) -> bool:
    s = str(col_name)
    if len(s) < 3:
        return False
    if re.fullmatch(r"[A-Z0-9_]{2,}", s):
        return False
    if re.fullmatch(r"[A-Za-z]\d+", s):
        return False
    if re.search(r"\s", s):
        return True
    return bool(re.fullmatch(r"[A-Za-z0-9_\-]+", s))


def _auto_symbol(
    sym: str,
    df: pd.DataFrame,
    *,
    dataset_description: str,
    file_name: str,
    file_ext: str,
    auto_inputs: Dict[str, Any],
) -> Tuple[Optional[Any], str, float, str]:
    if sym in auto_inputs and auto_inputs[sym] is not None:
        return auto_inputs[sym], "auto", 1.0, "derived from data"

    cols_lower = [str(c).lower() for c in df.columns]
    N = int(df.shape[1])

    # human-like metadata inference (reduces "None")
    if sym in {"t", "d"}:
        if dataset_description.strip() or file_name.strip():
            return 1.0, "auto", 0.6, "inferred: dataset has a name/description"
        if N > 0:
            return 1.0, "auto", 0.35, "inferred: column names imply dataset topic"

    if sym == "cv":
        has_geo = any(k in " ".join(cols_lower) for k in ["ehak", "county", "municipality", "country", "region", "maakond", "vald", "linn"])
        has_time = _find_date_column(df) is not None
        if has_geo or has_time:
            return 1.0, "auto", 0.55, "inferred: geo/date columns imply coverage"

    if sym == "id":
        if any(_looks_like_identifier(str(c)) for c in df.columns):
            return 1.0, "auto", 0.8, "inferred: Id/UUID column present"

    if sym == "s2":
        return 1.0, "auto", 0.9, "inferred: tabular structured dataset"

    if sym == "s3":
        if file_ext.lower() in {"csv", "json", "xml"}:
            return 1.0, "auto", 0.9, f"inferred: file format {file_ext}"
        return None, "", 0.0, ""

    if sym == "s4":
        if any(_looks_like_identifier(str(c)) for c in df.columns) or any("uri" in str(c).lower() for c in df.columns):
            return 1.0, "auto", 0.55, "inferred: identifier-like columns"
        if any("ehak" in str(c).lower() for c in df.columns):
            return 1.0, "auto", 0.45, "inferred: official code columns (EHAK)"

    if sym == "ncuf":
        readable = sum(1 for c in df.columns if _is_readable_colname(str(c)))
        return float(readable), "auto", 0.6, "inferred: readable column names"

    if sym == "ncm":
        readable = sum(1 for c in df.columns if _is_readable_colname(str(c)))
        if readable / max(1, N) >= 0.7:
            return float(N), "auto", 0.45, "inferred: column names are self-explanatory"
        return float(readable), "auto", 0.35, "inferred: partial from readable names"

    return None, "", 0.0, ""


def compute_metrics(
    df: pd.DataFrame,
    formulas_cfg: Dict[str, Any],
    prompt_cfg: Dict[str, Any],
    use_llm: bool,
    hf_runner,
    *,
    dataset_description: str = "",
    file_name: str = "",
    file_ext: str = "",
    min_symbol_confidence: float = 0.35,
    apply_confidence_weighting: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:

    vetro = (formulas_cfg or {}).get("vetro_methodology") or {}
    if not isinstance(vetro, dict):
        vetro = {}

    prompts = (prompt_cfg or {}).get("symbols", {}) or {}
    if not isinstance(prompts, dict):
        prompts = {}

    symbol_types: Dict[str, str] = {k: str(v.get("type", "binary")) for k, v in prompts.items() if isinstance(v, dict)}

    env: Dict[str, Any] = {"N": int(df.shape[1]), "R": int(df.shape[0])}

    auto_inputs: Dict[str, Any] = {}
    auto_inputs["nc"] = float(df.shape[1])
    auto_inputs["nr"] = float(df.shape[0])
    auto_inputs["ncl"] = float(df.shape[0] * df.shape[1])

    empty_cells, incomplete_rows = _count_empty_like(df)
    auto_inputs["ic"] = float(empty_cells)
    auto_inputs["nir"] = float(incomplete_rows)
    auto_inputs["nce"] = 0.0

    date_col = _find_date_column(df)
    if date_col is not None:
        dt = pd.to_datetime(df[date_col], errors="coerce")
        if dt.notna().any():
            sd_iso = dt.min().date().isoformat()
            edp_iso = dt.max().date().isoformat()
            auto_inputs["sd_col"] = str(date_col)
            auto_inputs["sd"] = sd_iso
            auto_inputs["edp"] = edp_iso
            auto_inputs["max_date"] = edp_iso
            auto_inputs["ncr"] = float(int((dt.dt.date != dt.max().date()).sum()))

    auto_inputs["cd"] = date.today().isoformat()

    for c in df.columns:
        cl = str(c).lower()
        if cl in {"modifiedat", "modified_at", "updatedat", "updated_at", "lastmodified", "last_modified"}:
            mod = pd.to_datetime(df[c], errors="coerce")
            if mod.notna().any():
                auto_inputs["dp"] = mod.max().date().isoformat()
                break
    if "dp" not in auto_inputs and auto_inputs.get("max_date"):
        auto_inputs["dp"] = auto_inputs["max_date"]

    if auto_inputs.get("edp"):
        auto_inputs["ed"] = auto_inputs["edp"]

    for k, v in auto_inputs.items():
        if v is None:
            env[k] = None
        elif symbol_types.get(k) == "date" or k in {"cd", "sd", "edp", "dp", "ed", "max_date"}:
            env[k] = _date_to_num(v)
        else:
            env[k] = v

    context = _build_llm_context(df, dataset_description=dataset_description, file_name=file_name, file_ext=file_ext)

    llm_raw: Dict[str, str] = {}
    llm_evidence: Dict[str, str] = {}
    llm_conf: Dict[str, float] = {}

    symbol_values: Dict[str, Any] = {}
    symbol_source: Dict[str, str] = {}
    symbol_confidence: Dict[str, float] = {}

    required_symbols = set()
    for dim, dim_obj in vetro.items():
        if not isinstance(dim_obj, dict):
            continue
        for _, metric_obj in dim_obj.items():
            if not isinstance(metric_obj, dict):
                continue
            for inp in metric_obj.get("inputs", []) or []:
                if isinstance(inp, dict):
                    for _, sym in inp.items():
                        required_symbols.add(sym)

    extra_values = {
        "dataset_description": dataset_description,
        "file_name": file_name,
        "file_ext": file_ext,
        "columns": ", ".join([str(c) for c in df.columns[:40]]),
        "profile": "",
        "N": int(df.shape[1]),
    }

    for sym in sorted(required_symbols):
        auto_val, auto_src, auto_conf, _ = _auto_symbol(
            sym,
            df,
            dataset_description=dataset_description,
            file_name=file_name,
            file_ext=file_ext,
            auto_inputs=auto_inputs,
        )
        if auto_src == "auto" and auto_val is not None:
            symbol_source[sym] = "auto"
            symbol_confidence[sym] = float(auto_conf)
            symbol_values[sym] = auto_val

            if symbol_types.get(sym) == "date":
                env[sym] = _date_to_num(auto_val)
            else:
                env[sym] = float(auto_val) * float(auto_conf) if apply_confidence_weighting and isinstance(auto_val, (int, float)) else auto_val
            continue

        if use_llm and hf_runner is not None and sym in prompts:
            val, raw, conf, evid = infer_symbol(
                symbol=sym,
                context=context,
                N=int(df.shape[1]),
                prompt_defs=prompts,
                hf_runner=hf_runner,
                extra_values=extra_values,
            )

            llm_raw[sym] = str(raw or "").strip()
            llm_conf[sym] = float(conf)
            llm_evidence[sym] = str(evid or "").strip()

            if val is None or float(conf) < float(min_symbol_confidence):
                env[sym] = None
                symbol_values[sym] = None
                symbol_source[sym] = "fail"
                symbol_confidence[sym] = float(conf)
                continue

            symbol_source[sym] = "llm"
            symbol_confidence[sym] = float(conf)
            symbol_values[sym] = val

            if symbol_types.get(sym) == "date":
                env[sym] = _date_to_num(val)
            else:
                env[sym] = float(val) * float(conf) if apply_confidence_weighting and isinstance(val, (int, float)) else val
            continue

        env[sym] = None
        symbol_values[sym] = None
        symbol_source[sym] = "fail"
        symbol_confidence[sym] = 0.0

    rows = []
    for dim, dim_obj in vetro.items():
        if not isinstance(dim_obj, dict):
            continue
        for metric_key, metric_obj in dim_obj.items():
            if not isinstance(metric_obj, dict):
                continue

            f_assign = (metric_obj.get("formula") or {}).get("assign")
            f_expr = (metric_obj.get("formula") or {}).get("expression")
            if f_assign and f_expr:
                env[f_assign] = _eval_expr(f_expr, env)

            n_assign = (metric_obj.get("normalization") or {}).get("assign")
            n_expr = (metric_obj.get("normalization") or {}).get("expression")
            value = float("nan")
            if n_assign and n_expr:
                value = _eval_expr(n_expr, env)
                env[n_assign] = value

            rows.append(
                {
                    "dimension": dim,
                    "metric": metric_key,
                    "metric_id": f"{dim}.{metric_key}",
                    "value": value,
                    "description": metric_obj.get("description", ""),
                }
            )

    metrics_df = pd.DataFrame(rows)

    details = {
        "auto_inputs": auto_inputs,
        "symbol_values": symbol_values,
        "symbol_source": symbol_source,
        "symbol_confidence": symbol_confidence,
        "llm_confidence": llm_conf,
        "llm_raw": llm_raw,
        "llm_evidence": llm_evidence,
    }
    return metrics_df, details
