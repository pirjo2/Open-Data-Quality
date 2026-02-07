from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import math
import re

import pandas as pd

from core.llm import infer_symbol


COND_ALLOWED_RE = re.compile(r"^[0-9a-zA-Z_ .<>=!()+\-*/]+$")
LICENSE_RE = re.compile(r"\b(cc[- ]?by|cc0|open\s+license|licen[cs]e)\b", re.IGNORECASE)
PUBLISHER_RE = re.compile(r"\b(publisher|published\s+by|ria|information\s+system\s+authority)\b", re.IGNORECASE)


def _safe_eval_condition(expr: str, env: Dict[str, float]) -> bool:
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


def _profile_df(df: pd.DataFrame, sample_n: int = 2, max_cols: int = 30) -> Dict[str, Any]:
    profile: Dict[str, Any] = {}
    cols = list(df.columns)[:max_cols]
    for col in cols:
        s = df[col]
        missing = float(s.isna().mean()) if len(s) else 0.0
        dtype = str(s.dtype)
        samples = [x for x in s.dropna().head(sample_n).astype(str).tolist()]
        samples = [x[:60] for x in samples]
        profile[str(col)] = {"dtype": dtype, "missing": round(missing, 6), "samples": samples}
    return profile


def _build_data_context(df: pd.DataFrame, max_cols: int = 30) -> str:
    cols = list(df.columns)
    profile = _profile_df(df, sample_n=2, max_cols=max_cols)

    parts = []
    parts.append(f"The dataset has {len(cols)} columns (N={len(cols)}).")
    parts.append("Column names: " + ", ".join([str(c) for c in cols[:max_cols]]))
    parts.append("Column profile (dtype, missing ratio, sample values):")
    for col, info in profile.items():
        parts.append(f"- {col}: dtype={info['dtype']}, missing={info['missing']}, samples={info['samples']}")
    return "\n".join(parts)


def _build_meta_context(dataset_description: str, metadata_text: str, file_name: str, file_ext: str) -> str:
    parts = []
    if file_name:
        parts.append(f"File name: {file_name}")
    if file_ext:
        parts.append(f"File extension: {file_ext}")
    if dataset_description:
        parts.append("User description:")
        parts.append(dataset_description.strip())
    if metadata_text:
        t = metadata_text.strip()
        if len(t) > 3500:
            t = t[:3500] + " …"
        parts.append("Additional metadata/documentation text:")
        parts.append(t)
    return "\n".join(parts).strip()


def _find_first_date_col(df: pd.DataFrame) -> Optional[str]:
    for c in df.columns:
        cn = str(c).lower()
        if cn in ("date", "datetime", "time", "timestamp"):
            return str(c)
    for c in df.columns:
        cn = str(c).lower()
        if "date" in cn or "time" in cn or "timestamp" in cn:
            return str(c)
    return None


def _infer_standardizable_columns(df: pd.DataFrame) -> int:
    count = 0
    for c in df.columns:
        cn = str(c).lower()
        if any(k in cn for k in ["date", "time", "timestamp", "country", "county", "commune", "region", "ehak", "iso", "code", "id", "uuid", "uri", "url"]):
            count += 1
    return count


def _infer_standardized_columns(df: pd.DataFrame) -> int:
    nsc = 0
    for c in df.columns:
        cn = str(c).lower()
        s = df[c].dropna()
        if len(s) == 0:
            continue

        if "date" in cn:
            dt = pd.to_datetime(s.astype(str), errors="coerce", utc=True)
            if dt.notna().mean() > 0.95:
                nsc += 1
            continue

        if any(k in cn for k in ["ehak", "code", "iso"]):
            vals = s.astype(str).head(500)
            ok = vals.str.fullmatch(r"[A-Za-z0-9\-_/]{2,15}").mean()
            if ok > 0.90:
                nsc += 1
            continue

    return nsc


def _infer_comprehensible_columns(df: pd.DataFrame) -> int:
    good = 0
    for c in df.columns:
        cn = str(c)
        if len(cn) < 2:
            continue
        if cn.isnumeric():
            continue
        if re.fullmatch(r"[A-Z]{1,4}\d{0,4}", cn):
            continue
        good += 1
    return good


def _auto_symbol(sym: str, df: pd.DataFrame, auto_inputs: Dict[str, Any], meta_context: str, file_ext: str, dataset_description: str) -> Tuple[Optional[Any], str]:
    if sym in auto_inputs:
        return auto_inputs[sym], "auto"

    cols_lower = [str(c).lower() for c in df.columns]
    ext = (file_ext or "").lower().lstrip(".")

    if sym == "t":
        if dataset_description.strip() or auto_inputs.get("file_name"):
            return 1.0, "auto"

    if sym == "d":
        if dataset_description.strip() or meta_context.strip():
            return 1.0, "auto"

    if sym == "id":
        if any(x in cols_lower for x in ["id", "uuid", "identifier", "uri", "url"]):
            return 1.0, "auto"

    if sym == "dp":
        for c in df.columns:
            if str(c).lower() in ("modifiedat", "modified_at", "updatedat", "updated_at", "lastmodified", "last_modified"):
                dt = pd.to_datetime(df[c], errors="coerce", utc=True)
                if dt.notna().any():
                    return dt.max().date().isoformat(), "auto"
        if "max_date" in auto_inputs:
            return auto_inputs["max_date"], "auto"

    if sym == "du":
        if any(x in cols_lower for x in ["modifiedat", "modified_at", "updatedat", "updated_at", "lastmodified", "last_modified"]):
            return 1.0, "auto"

    if sym == "s2":
        if ext in ("csv", "json", "xml", "parquet"):
            return 1.0, "auto"
        if ext in ("xlsx", "xls"):
            return 1.0, "auto"
        return 0.0, "auto"

    if sym == "s3":
        return (1.0 if ext in ("csv", "json", "xml") else 0.0), "auto"

    if sym == "s4":
        if any(k in cols_lower for k in ["uri", "url", "id", "uuid"]):
            return 1.0, "auto"
        return 0.0, "auto"

    if sym == "s5":
        for c in df.columns:
            if "url" in str(c).lower() or "uri" in str(c).lower():
                s = df[c].dropna().astype(str).head(50)
                if (s.str.contains(r"https?://", regex=True).mean() or 0.0) > 0.2:
                    return 1.0, "auto"
        return 0.0, "auto"

    if sym == "s1":
        if LICENSE_RE.search(meta_context):
            return 1.0, "auto"
        return None, ""

    if sym == "pb":
        if PUBLISHER_RE.search(meta_context):
            return 1.0, "auto"
        return None, ""

    if sym == "ncuf":
        return float(_infer_comprehensible_columns(df)), "auto"

    return None, ""


def compute_metrics(
    df: pd.DataFrame,
    formulas_cfg: Dict[str, Any],
    prompt_cfg: Dict[str, Any],
    use_llm: bool,
    hf_runner,
    dataset_description: str = "",
    metadata_text: str = "",
    file_name: str = "",
    file_ext: str = "",
    weight_by_confidence: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:

    vetro = (formulas_cfg or {}).get("vetro_methodology") or {}
    if not isinstance(vetro, dict):
        vetro = {}

    prompts = (prompt_cfg or {}).get("symbols", {}) or {}
    if not isinstance(prompts, dict):
        prompts = {}

    env: Dict[str, Any] = {}
    env["N"] = int(df.shape[1])
    env["R"] = int(df.shape[0])

    auto_inputs: Dict[str, Any] = {}
    auto_inputs["file_name"] = file_name
    auto_inputs["file_ext"] = (file_ext or "").lower().lstrip(".")

    auto_inputs["nc"] = int(df.shape[1])
    auto_inputs["nr"] = int(df.shape[0])
    auto_inputs["ncl"] = float(df.shape[0] * df.shape[1])

    empty_mask = df.isna()
    try:
        obj = df.select_dtypes(include=["object", "string"])
        if not obj.empty:
            empty_str = obj.fillna("").astype(str).apply(lambda s: s.str.strip().eq("")).to_numpy().sum()
            empty_mask = empty_mask | (df.select_dtypes(include=["object", "string"]).fillna("").astype(str).apply(lambda s: s.str.strip().eq("")))
            auto_inputs["nce"] = float(empty_mask.sum().sum())
        else:
            auto_inputs["nce"] = float(empty_mask.sum().sum())
    except Exception:
        auto_inputs["nce"] = float(empty_mask.sum().sum())

    auto_inputs["nir"] = float(empty_mask.any(axis=1).sum())
    auto_inputs["ic"] = float(df.shape[0] * df.shape[1] - auto_inputs["nce"])

    auto_inputs["ns"] = float(_infer_standardizable_columns(df))
    auto_inputs["nsc"] = float(_infer_standardized_columns(df))

    date_col = _find_first_date_col(df)
    if date_col is not None:
        dt = pd.to_datetime(df[date_col], errors="coerce", utc=True)
        if dt.notna().any():
            auto_inputs["sd_col"] = str(date_col)
            auto_inputs["sd"] = dt.min().date().isoformat()
            auto_inputs["edp"] = dt.max().date().isoformat()
            auto_inputs["max_date"] = dt.max().date().isoformat()

    auto_inputs["cd"] = pd.Timestamp.utcnow().date().isoformat()

    if "max_date" in auto_inputs:
        auto_inputs["ed"] = auto_inputs["max_date"]

    for k, v in auto_inputs.items():
        env[k] = v

    if "sd_col" in auto_inputs and "max_date" in auto_inputs:
        col = auto_inputs["sd_col"]
        dt = pd.to_datetime(df[col], errors="coerce", utc=True)
        max_dt = pd.to_datetime(auto_inputs["max_date"], errors="coerce", utc=True)
        if dt.notna().any() and max_dt is not pd.NaT:
            auto_inputs["ncr_definition"] = "rows_not_at_max_date"
            auto_inputs["ncr"] = float((dt.dt.date != max_dt.date()).sum())
            env["ncr"] = auto_inputs["ncr"]

    data_context = _build_data_context(df)
    meta_context = _build_meta_context(dataset_description, metadata_text, file_name, file_ext)

    metadata_symbols = {
        "c", "cv", "d", "dc", "du", "lu", "s", "pb", "l",
        "s1", "s2", "s3", "s4", "s5", "t",
        "dp", "sd", "edp", "ed",
    }

    llm_raw: Dict[str, str] = {}
    llm_evidence: Dict[str, str] = {}
    llm_conf: Dict[str, float] = {}

    symbol_values: Dict[str, Any] = {}
    symbol_source: Dict[str, str] = {}

    required_symbols = set()
    for _, dim_obj in vetro.items():
        if not isinstance(dim_obj, dict):
            continue
        for _, metric_obj in dim_obj.items():
            if not isinstance(metric_obj, dict):
                continue
            for inp in metric_obj.get("inputs", []) or []:
                if isinstance(inp, dict):
                    for _, sym in inp.items():
                        required_symbols.add(sym)

    CONF_THRESHOLD = 0.25

    for sym in sorted(required_symbols):
        auto_val, auto_src = _auto_symbol(sym, df, auto_inputs, meta_context, file_ext, dataset_description)
        if auto_src == "auto" and auto_val is not None:
            env[sym] = auto_val
            symbol_values[sym] = auto_val
            symbol_source[sym] = "auto"
            continue

        llm_context = meta_context if sym in metadata_symbols else data_context
        if sym in metadata_symbols and not meta_context:
            llm_context = data_context

        if use_llm and hf_runner is not None and sym in prompts:
            val, evidence, conf, raw = infer_symbol(
                symbol=sym,
                context=llm_context,
                N=int(df.shape[1]),
                prompt_defs=prompts,
                hf_runner=hf_runner,
            )

            llm_raw[sym] = str(raw or "").strip()
            llm_conf[sym] = float(conf or 0.0)
            llm_evidence[sym] = str(evidence or "").strip()

            if val is None or float(conf or 0.0) < CONF_THRESHOLD:
                symbol_values[sym] = None
                symbol_source[sym] = "fail"
                typ = str(prompts.get(sym, {}).get("type", "binary"))
                env[sym] = 0.0 if typ in ("binary", "count_0_to_N", "count", "number") else None
            else:
                v = val
                if weight_by_confidence and isinstance(v, (int, float)):
                    v = float(v) * float(conf)
                env[sym] = v
                symbol_values[sym] = v
                symbol_source[sym] = "llm"
            continue

        symbol_values[sym] = None
        symbol_source[sym] = "fail"
        typ = str(prompts.get(sym, {}).get("type", "binary"))
        env[sym] = 0.0 if typ in ("binary", "count_0_to_N", "count", "number") else None

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
                    "metric_label": metric_obj.get("metric_label", f"{dim}.{metric_key}"),
                }
            )

    metrics_df = pd.DataFrame(rows)

    details = {
        "auto_inputs": auto_inputs,
        "symbol_values": symbol_values,
        "symbol_source": symbol_source,
        "llm_confidence": llm_conf,
        "llm_raw": llm_raw,
        "llm_evidence": llm_evidence,
        "contexts": {
            "meta_context_used": bool(meta_context),
            "data_context_preview": data_context[:800] + (" …" if len(data_context) > 800 else ""),
            "meta_context_preview": meta_context[:800] + (" …" if len(meta_context) > 800 else ""),
        },
    }
    return metrics_df, details
