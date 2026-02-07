from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import datetime as dt
import math
import re
import pandas as pd

from core.llm import infer_symbol

ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
URL_RE = re.compile(r"^https?://", re.IGNORECASE)

COND_ALLOWED_RE = re.compile(r"^[0-9a-zA-Z_ .<>=!()+\-*/]+$")


def _safe_eval_condition(expr: str, env: Dict[str, Any]) -> bool:
    if not isinstance(expr, str) or not COND_ALLOWED_RE.match(expr):
        return False
    try:
        return bool(eval(expr, {"__builtins__": {}}, env))
    except Exception:
        return False


def _date_ordinal(v: Any) -> Optional[float]:
    """
    Converts 'YYYY-MM-DD' -> ordinal float (days).
    """
    if isinstance(v, str):
        s = v.strip()
        if ISO_DATE_RE.match(s):
            try:
                return float(dt.date.fromisoformat(s).toordinal())
            except Exception:
                return None
    return None


def _leaf_value_to_number(v: Any) -> float:
    """
    Numeric leaf conversion:
    - numbers -> float
    - date strings -> ordinal float
    - else -> NaN
    """
    if v is None:
        return float("nan")
    if isinstance(v, (int, float)):
        return float(v)
    ordv = _date_ordinal(v)
    if ordv is not None:
        return float(ordv)
    return float("nan")


def _eval_expr(node: Any, env: Dict[str, Any]) -> float:
    """
    Evaluates formula expressions from formulas.yaml.
    Supports:
      - identity (operand/of)
      - add (left/right OR terms list)
      - subtract/multiply/divide
      - abs_diff
      - days_between
      - conditional
    Also supports date arithmetic via ordinal conversion.
    """
    if node is None:
        return float("nan")

    # variable reference
    if isinstance(node, str):
        return _leaf_value_to_number(env.get(node, None))

    if isinstance(node, (int, float)):
        return float(node)

    if isinstance(node, dict) and "operator" in node:
        op = node["operator"]

        if op == "identity":
            return _eval_expr(node.get("operand", node.get("of")), env)

        if op == "add":
            # support "terms"
            if "terms" in node and isinstance(node["terms"], list):
                vals = [_eval_expr(t, env) for t in node["terms"]]
                if any(math.isnan(x) for x in vals):
                    # allow partial sums: ignore NaN terms
                    vals = [x for x in vals if not math.isnan(x)]
                return float(sum(vals)) if vals else float("nan")
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
            # evaluate children, which may be date ordinals already
            a = _eval_expr(node.get("left"), env)
            b = _eval_expr(node.get("right"), env)
            if math.isnan(a) or math.isnan(b):
                return float("nan")
            return float(abs(b - a))

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

        parsed = pd.to_numeric(sample.astype(str), errors="coerce")
        numeric_ratio = float(parsed.notna().mean())
        if numeric_ratio > 0.85:
            errors += int(parsed.isna().sum())

    return int(errors)


def _infer_standards(df: pd.DataFrame) -> Tuple[int, int]:
    ns = 0
    nsc = 0

    for col in df.columns:
        name = str(col).lower()
        s = df[col].dropna().astype(str).head(500)
        if s.empty:
            continue

        if "date" in name:
            ns += 1
            ok = s.map(lambda x: bool(ISO_DATE_RE.match(x.strip()))).mean()
            if ok >= 0.95:
                nsc += 1
            continue

        if "url" in name or "link" in name:
            ns += 1
            ok = s.map(lambda x: bool(URL_RE.match(x.strip()))).mean()
            if ok >= 0.95:
                nsc += 1
            continue

        if any(k in name for k in ["ehak", "code", "id"]):
            ns += 1
            ok = s.map(lambda x: x.strip().isdigit()).mean()
            if ok >= 0.95:
                nsc += 1
            continue

    return int(ns), int(nsc)


def _auto_symbol(
    sym: str,
    df: pd.DataFrame,
    auto_inputs: Dict[str, Any],
    file_ext: str,
    file_name: str,
    dataset_description: str,
) -> Tuple[Optional[Any], str]:
    """
    Returns (value_or_None, source) where source in {"auto",""}.
    """
    # direct auto inputs
    if sym in auto_inputs:
        return auto_inputs[sym], "auto"

    cols_lower = [str(c).lower() for c in df.columns]

    # dataset derived counts
    if sym == "nc":
        return float(df.shape[1]), "auto"
    if sym == "nr":
        return float(df.shape[0]), "auto"
    if sym == "ncl":
        return float(df.shape[0] * df.shape[1]), "auto"

    if sym == "ic":
        return float(_count_incomplete_cells(df)), "auto"
    if sym == "nir":
        return float(_count_incomplete_rows(df)), "auto"
    if sym == "nce":
        return float(_count_syntactic_errors(df)), "auto"

    if sym in ("ns", "nsc"):
        ns, nsc = _infer_standards(df)
        return (float(ns), "auto") if sym == "ns" else (float(nsc), "auto")

    # dates
    if sym == "cd":
        return dt.date.today().isoformat(), "auto"

    # dp: try timestamp columns
    if sym == "dp":
        for c in df.columns:
            if str(c).lower() in ("modifiedat", "modified_at", "updatedat", "updated_at", "lastmodified", "last_modified"):
                dtv = pd.to_datetime(df[c], errors="coerce", utc=True)
                if dtv.notna().any():
                    return dtv.max().date().isoformat(), "auto"
        if "max_date" in auto_inputs:
            return auto_inputs["max_date"], "auto"

    # ed: previous version expiration date – heuristic fallback (better than NaN)
    if sym == "ed":
        if "max_date" in auto_inputs:
            return auto_inputs["max_date"], "auto"

    # du: update dates mentioned -> timestamp column exists
    if sym == "du":
        if any(x in cols_lower for x in ["modifiedat", "modified_at", "updatedat", "updated_at", "lastmodified", "last_modified"]):
            return 1.0, "auto"

    # identifier
    if sym == "id":
        if any(x in cols_lower for x in ["id", "uuid", "identifier"]):
            return 1.0, "auto"

    # light "human-ish" heuristics:
    if sym == "t":
        # treat file name as a title proxy
        if file_name:
            return 1.0, "auto"
    if sym == "d":
        # treat user description as "description present"
        if dataset_description and dataset_description.strip():
            return 1.0, "auto"

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

    labels_map = vetro.get("labels", {}) if isinstance(vetro.get("labels", {}), dict) else {}

    prompts = (prompt_cfg or {}).get("symbols", {}) or {}
    if not isinstance(prompts, dict):
        prompts = {}

    env: Dict[str, Any] = {"N": int(df.shape[1]), "R": int(df.shape[0])}

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

    for k, v in auto_inputs.items():
        env[k] = v

    context = _build_llm_context(df, dataset_description or "", file_name or "", file_ext or "")

    llm_raw: Dict[str, str] = {}
    llm_evidence: Dict[str, str] = {}
    llm_conf: Dict[str, float] = {}

    symbol_values: Dict[str, Any] = {}   # None = did not work; numeric 0 = real zero
    symbol_source: Dict[str, str] = {}  # auto / llm / fail

    # required symbols
    required_symbols = set()
    for dim, dim_obj in vetro.items():
        if dim == "labels":
            continue
        if not isinstance(dim_obj, dict):
            continue
        for _, metric_obj in dim_obj.items():
            if not isinstance(metric_obj, dict):
                continue
            for inp in metric_obj.get("inputs", []) or []:
                if isinstance(inp, dict):
                    for _, sym in inp.items():
                        required_symbols.add(sym)

    CONF_THRESHOLD = 0.45

    for sym in sorted(required_symbols):
        # 1) auto first
        auto_val, src = _auto_symbol(sym, df, auto_inputs, file_ext, file_name, dataset_description)
        if src == "auto" and auto_val is not None:
            env[sym] = auto_val
            symbol_values[sym] = auto_val
            symbol_source[sym] = "auto"
            continue

        # 2) LLM second
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
                symbol_values[sym] = None
                symbol_source[sym] = "fail"

                typ = prompts.get(sym, {}).get("type", "binary")
                if typ in ("binary", "count_0_to_N", "count"):
                    env[sym] = 0.0  # safe default for formulas
                else:
                    env[sym] = None
            else:
                env[sym] = val
                symbol_values[sym] = val
                symbol_source[sym] = "llm"
            continue

        # 3) fail
        symbol_values[sym] = None
        symbol_source[sym] = "fail"
        env[sym] = 0.0  # safe numeric default

    # compute metrics
    rows = []
    for dim, dim_obj in vetro.items():
        if dim == "labels":
            continue
        if not isinstance(dim_obj, dict):
            continue

        for metric_key, metric_obj in dim_obj.items():
            if not isinstance(metric_obj, dict):
                continue

            # 0) intermediate calculations (important!)
            inter = metric_obj.get("intermediate_calculation", None)
            if inter:
                inter_list = inter if isinstance(inter, list) else [inter]
                for step in inter_list:
                    if not isinstance(step, dict):
                        continue
                    a = step.get("assign")
                    e = step.get("expression")
                    if a and e is not None:
                        env[a] = _eval_expr(e, env)

            # 1) formula assign
            f_assign = (metric_obj.get("formula") or {}).get("assign")
            f_expr = (metric_obj.get("formula") or {}).get("expression")
            if f_assign and f_expr is not None:
                env[f_assign] = _eval_expr(f_expr, env)

            # 2) normalization
            n_assign = (metric_obj.get("normalization") or {}).get("assign")
            n_expr = (metric_obj.get("normalization") or {}).get("expression")
            value = float("nan")
            if n_assign and n_expr is not None:
                value = _eval_expr(n_expr, env)
                env[n_assign] = value

            metric_id = f"{dim}.{metric_key}"
            metric_label = labels_map.get(metric_id, metric_obj.get("label", metric_id))

            rows.append(
                {
                    "dimension": dim,
                    "metric": metric_key,
                    "metric_id": metric_id,
                    "value": value,
                    "description": metric_obj.get("description", ""),
                    "metric_label": metric_label,
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
