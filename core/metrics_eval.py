from __future__ import annotations

from typing import Any, Dict, Tuple, Optional
import math
import re
import datetime as dt

import pandas as pd

from core.llm import infer_symbol

COND_ALLOWED_RE = re.compile(r"^[0-9a-zA-Z_ .<>=!()+\-*/]+$")
ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
URL_RE = re.compile(r"^https?://", re.IGNORECASE)

def _safe_eval_condition(expr: str, env: Dict[str, Any]) -> bool:
    if not isinstance(expr, str) or not COND_ALLOWED_RE.match(expr):
        return False
    try:
        return bool(eval(expr, {"__builtins__": {}}, env))
    except Exception:
        return False

def _parse_date(x: Any) -> Optional[dt.date]:
    if x is None:
        return None
    if isinstance(x, dt.date) and not isinstance(x, dt.datetime):
        return x
    try:
        s = str(x).strip()
        m = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", s)
        if not m:
            return None
        return dt.date.fromisoformat(m.group(1))
    except Exception:
        return None

def _days_between(a: Any, b: Any) -> float:
    da = _parse_date(a)
    db = _parse_date(b)
    if not da or not db:
        return float("nan")
    return float(abs((da - db).days))

def _eval_expr(node: Any, env: Dict[str, Any]) -> float:
    if node is None:
        return float("nan")

    if isinstance(node, str):
        v = env.get(node, None)
        if v is None:
            return float("nan")
        if isinstance(v, (int, float)):
            return float(v)
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

        if op == "days_between":
            a = env.get(node.get("left"), None) if isinstance(node.get("left"), str) else node.get("left")
            b = env.get(node.get("right"), None) if isinstance(node.get("right"), str) else node.get("right")
            return _days_between(a, b)

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

def _profile_df(df: pd.DataFrame, sample_n: int = 3, max_cols: int = 25) -> Dict[str, Any]:
    profile: Dict[str, Any] = {}
    cols = list(df.columns)[:max_cols]
    for col in cols:
        s = df[col]
        missing = float(s.isna().mean()) if len(s) else 0.0
        dtype = str(s.dtype)
        samples = [x for x in s.dropna().head(sample_n).astype(str).tolist()]
        profile[str(col)] = {"dtype": dtype, "missing": round(missing, 6), "samples": samples}
    return profile

def _build_llm_context(df: pd.DataFrame, dataset_description: str, file_name: str, file_ext: str) -> str:
    cols = list(df.columns)
    profile = _profile_df(df, sample_n=3, max_cols=25)

    parts = []
    if file_name:
        parts.append(f"File name: {file_name}")
    if file_ext:
        parts.append(f"File type/extension: {file_ext}")
    if dataset_description.strip():
        parts.append("User-provided dataset description:")
        parts.append(dataset_description.strip())

    parts.append(f"Rows={len(df)}, Columns={len(cols)} (N={len(cols)}).")
    parts.append("Column names (first 25): " + ", ".join([str(c) for c in cols[:25]]))
    parts.append("Column profile (dtype, missing, samples):")
    for col, info in profile.items():
        parts.append(f"- {col}: dtype={info['dtype']}, missing={info['missing']}, samples={info['samples']}")
    return "\n".join(parts)

def _count_incomplete_cells(df: pd.DataFrame) -> int:
    # Treat empty strings as missing too
    tmp = df.copy()
    for c in tmp.columns:
        if tmp[c].dtype == object:
            tmp[c] = tmp[c].astype(str).replace({"": None, "nan": None, "None": None})
            tmp[c] = tmp[c].apply(lambda x: None if isinstance(x, str) and x.strip() == "" else x)
    return int(tmp.isna().sum().sum())

def _count_incomplete_rows(df: pd.DataFrame) -> int:
    tmp = df.copy()
    for c in tmp.columns:
        if tmp[c].dtype == object:
            tmp[c] = tmp[c].astype(str).replace({"": None, "nan": None, "None": None})
            tmp[c] = tmp[c].apply(lambda x: None if isinstance(x, str) and x.strip() == "" else x)
    return int(tmp.isna().any(axis=1).sum())

def _count_syntactic_errors(df: pd.DataFrame) -> int:
    """
    Simple syntactic error heuristic:
    - For numeric-looking columns: non-empty values that fail to parse as numeric
    - For date-looking columns: non-empty values that fail ISO YYYY-MM-DD
    - Otherwise: 0 errors
    """
    errors = 0
    for col in df.columns:
        s = df[col]
        non_null = s.dropna()
        if non_null.empty:
            continue

        name = str(col).lower()
        sample = non_null.astype(str).head(200)

        looks_date = ("date" in name) or (sample.map(lambda x: bool(ISO_DATE_RE.match(x.strip()))).mean() > 0.8)
        if looks_date:
            bad = sample.map(lambda x: 0 if ISO_DATE_RE.match(x.strip()) else 1).sum()
            errors += int(bad)
            continue

        # numeric-ish
        parsed = pd.to_numeric(sample.astype(str), errors="coerce")
        numeric_ratio = float(parsed.notna().mean())
        if numeric_ratio > 0.85:
            errors += int(parsed.isna().sum())

    return int(errors)

def _infer_standards(df: pd.DataFrame) -> Tuple[int, int]:
    """
    ns  = number of columns where a standard could apply
    nsc = number of columns that actually follow the standard
    """
    ns = 0
    nsc = 0

    for col in df.columns:
        name = str(col).lower()
        s = df[col].dropna().astype(str).head(500)

        if s.empty:
            continue

        # ISO date
        if "date" in name:
            ns += 1
            ok = s.map(lambda x: bool(ISO_DATE_RE.match(x.strip()))).mean()
            if ok >= 0.95:
                nsc += 1
            continue

        # URL
        if "url" in name or "link" in name:
            ns += 1
            ok = s.map(lambda x: bool(URL_RE.match(x.strip()))).mean()
            if ok >= 0.95:
                nsc += 1
            continue

        # codes (simple heuristic)
        if any(k in name for k in ["ehak", "code", "id"]):
            ns += 1
            ok = s.map(lambda x: x.strip().isdigit()).mean()
            if ok >= 0.95:
                nsc += 1
            continue

    return int(ns), int(nsc)

def _auto_symbol(sym: str, df: pd.DataFrame, auto_inputs: Dict[str, Any], file_ext: str) -> Tuple[Optional[Any], str]:
    """
    Returns (value_or_None, source) where source in {"auto",""}.
    """
    # direct auto_inputs
    if sym in auto_inputs:
        return auto_inputs[sym], "auto"

    cols_lower = [str(c).lower() for c in df.columns]

    # Basic dataset derived counts
    if sym == "nc":   # number of columns
        return float(df.shape[1]), "auto"
    if sym == "nr":   # number of rows
        return float(df.shape[0]), "auto"
    if sym == "ncl":  # number of cells
        return float(df.shape[0] * df.shape[1]), "auto"

    if sym == "ic":   # incomplete cells
        return float(_count_incomplete_cells(df)), "auto"
    if sym == "nir":  # incomplete rows
        return float(_count_incomplete_rows(df)), "auto"
    if sym == "nce":  # cells with errors
        return float(_count_syntactic_errors(df)), "auto"

    if sym == "ns" or sym == "nsc":
        ns, nsc = _infer_standards(df)
        if sym == "ns":
            return float(ns), "auto"
        return float(nsc), "auto"

    # dates: cd = current date
    if sym == "cd":
        return dt.date.today().isoformat(), "auto"

    # dp: publication/last updated date – from ModifiedAt / UpdatedAt if exists
    if sym == "dp":
        for c in df.columns:
            if str(c).lower() in ("modifiedat", "modified_at", "updatedat", "updated_at", "lastmodified", "last_modified"):
                dtv = pd.to_datetime(df[c], errors="coerce", utc=True)
                if dtv.notna().any():
                    return dtv.max().date().isoformat(), "auto"
        if "max_date" in auto_inputs:
            return auto_inputs["max_date"], "auto"

    # du (update dates mentioned) -> if timestamp column exists
    if sym == "du":
        if any(x in cols_lower for x in ["modifiedat", "modified_at", "updatedat", "updated_at", "lastmodified", "last_modified"]):
            return 1.0, "auto"

    # id (identifier present)
    if sym == "id":
        if any(x in cols_lower for x in ["id", "uuid", "identifier"]):
            return 1.0, "auto"

    # 5-star heuristics from file type
    if sym in ("s2", "s3"):
        ext = (file_ext or "").lower().lstrip(".")
        # structured + machine-readable (CSV, JSON, XML, XLSX are structured)
        if sym == "s2":
            if ext in ("csv", "json", "xml", "xlsx", "xls", "parquet"):
                return 1.0, "auto"
            return 0.0, "auto"
        # non-proprietary open format
        if sym == "s3":
            if ext in ("csv", "json", "xml", "parquet"):
                return 1.0, "auto"
            return 0.0, "auto"

    # s4: URIs/identifiers used (if URL columns exist)
    if sym == "s4":
        if any("url" in c or "link" in c for c in cols_lower):
            return 1.0, "auto"
        return 0.0, "auto"

    # current rows count ncr (rows at max_date) if time column exists
    if sym == "ncr":
        date_col = auto_inputs.get("sd_col")
        max_date = auto_inputs.get("max_date")
        if date_col and max_date and date_col in df.columns:
            dtv = pd.to_datetime(df[date_col], errors="coerce")
            if dtv.notna().any():
                return float((dtv.dt.date.astype(str) == str(max_date)).sum()), "auto"

    return None, ""

def compute_metrics(
    df: pd.DataFrame,
    formulas_cfg: Dict[str, Any],
    prompt_cfg: Dict[str, Any],
    use_llm: bool,
    hf_runner,
    dataset_description: str = "",
    file_name: str = "",
    file_ext: str = "",
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

    # auto date inputs
    auto_inputs: Dict[str, Any] = {}
    date_col = None
    for c in df.columns:
        if "date" in str(c).lower():
            date_col = c
            break

    if date_col is not None:
        dtv = pd.to_datetime(df[date_col], errors="coerce", utc=True)
        if dtv.notna().any():
            auto_inputs["sd_col"] = str(date_col)
            auto_inputs["sd"] = dtv.min().date().isoformat()
            auto_inputs["edp"] = dtv.max().date().isoformat()
            auto_inputs["max_date"] = dtv.max().date().isoformat()

    # inject
    for k, v in auto_inputs.items():
        env[k] = v

    context = _build_llm_context(df, dataset_description, file_name, file_ext)

    # debug collections
    llm_raw: Dict[str, str] = {}
    llm_evidence: Dict[str, str] = {}
    llm_conf: Dict[str, float] = {}

    symbol_values: Dict[str, Any] = {}      # None = did not work; numeric 0 = real zero
    symbol_source: Dict[str, str] = {}      # auto / llm / fail

    # required symbols from YAML
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

    # fill symbols
    CONF_THRESHOLD = 0.45

    for sym in sorted(required_symbols):
        # 1) auto first
        auto_val, src = _auto_symbol(sym, df, auto_inputs, file_ext)
        if src == "auto" and auto_val is not None:
            env[sym] = auto_val
            symbol_values[sym] = auto_val
            symbol_source[sym] = "auto"
            continue

        # 2) LLM
        if use_llm and hf_runner is not None and sym in prompts:
            val, evidence, conf, raw = infer_symbol(
                symbol=sym,
                context=context,
                N=int(df.shape[1]),
                prompt_defs=prompts,
                hf_runner=hf_runner,
            )
            llm_raw[sym] = (raw or "").strip()
            llm_evidence[sym] = (evidence or "").strip()
            llm_conf[sym] = float(conf)

            if val is None or conf < CONF_THRESHOLD:
                # did not work
                symbol_values[sym] = None
                symbol_source[sym] = "fail"
                # but for formulas: binary/count -> safe default 0
                typ = prompts.get(sym, {}).get("type", "binary")
                if typ in ("binary", "count_0_to_N", "count"):
                    env[sym] = 0.0
                else:
                    env[sym] = None
            else:
                env[sym] = val
                symbol_values[sym] = val
                symbol_source[sym] = "llm"
            continue

        # 3) no LLM, fail
        symbol_values[sym] = None
        symbol_source[sym] = "fail"
        # safe for formulas if numeric expected
        env[sym] = 0.0

    # compute metrics
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
                    "metric_label": metric_obj.get("label", f"{dim}: {metric_key}"),
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
    }
    return metrics_df, details
