from __future__ import annotations

from typing import Any, Dict, Tuple, Optional
import math
import re
import datetime as _dt

import pandas as pd

from core.llm import infer_symbol

COND_ALLOWED_RE = re.compile(r"^[0-9a-zA-Z_ .<>=!()+\-*/]+$")
DATE_ONLY_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})$")
DATE_INNER_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")


def _safe_eval_condition(expr: str, env: Dict[str, float]) -> bool:
    """
    Evaluate a simple numeric condition like "x >= 0.5" safely.
    """
    if not isinstance(expr, str) or not COND_ALLOWED_RE.match(expr):
        return False
    try:
        return bool(eval(expr, {"__builtins__": {}}, env))
    except Exception:
        return False


def _to_float_value(v: Any) -> float:
    """
    Convert an environment value into a float for expression evaluation.
    Handles ints/floats, booleans, date strings, and pandas timestamps.
    """
    if v is None:
        return float("nan")

    if isinstance(v, bool):
        return 1.0 if v else 0.0

    if isinstance(v, (int, float)):
        return float(v)

    # Pandas / datetime objects
    try:
        import pandas as _pd  # local import to avoid hard dependency at import time

        if isinstance(v, _pd.Timestamp):
            return float(v.to_pydatetime().date().toordinal())
    except Exception:
        pass

    if isinstance(v, (_dt.date, _dt.datetime)):
        d = v.date() if isinstance(v, _dt.datetime) else v
        return float(d.toordinal())

    if isinstance(v, str):
        s = v.strip()
        # Try ISO date
        m = DATE_ONLY_RE.fullmatch(s) or DATE_INNER_RE.search(s)
        if m:
            try:
                d = _dt.date.fromisoformat(m.group(1))
                return float(d.toordinal())
            except Exception:
                pass
        # Try numeric
        try:
            return float(s)
        except Exception:
            return float("nan")

    return float("nan")


def _eval_expr(node: Any, env: Dict[str, Any]) -> float:
    """
    Evaluate a formula expression tree coming from formulas.yaml.
    Supports:
      - literals (int/float)
      - variable references (strings -> env[name])
      - {operator: "add"/"subtract"/"multiply"/"divide"/"abs_diff"/"identity"/"conditional"}
    """
    if node is None:
        return float("nan")

    if isinstance(node, str):
        v = env.get(node, None)
        return _to_float_value(v)

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
            # conditions: list of {if/elif/else, then}
            for rule in node.get("conditions", []) or []:
                if not isinstance(rule, dict):
                    continue
                # Copy numeric env only to avoid weirdness
                num_env = {k: _to_float_value(v) for k, v in env.items()}
                if "if" in rule and _safe_eval_condition(str(rule["if"]), num_env):
                    return _eval_expr(rule.get("then"), env)
                if "elif" in rule and _safe_eval_condition(str(rule["elif"]), num_env):
                    return _eval_expr(rule.get("then"), env)
                if "else" in rule:
                    return _eval_expr(rule.get("else"), env)
            return float("nan")

    return float("nan")


def _profile_df(df: pd.DataFrame, sample_n: int = 2, max_cols: int = 25) -> Dict[str, Any]:
    """
    Compact profiling: dtype, missing ratio, and a couple of sample values per column.
    """
    profile: Dict[str, Any] = {}
    cols = list(df.columns)[:max_cols]
    for col in cols:
        s = df[col]
        missing = float(s.isna().mean()) if len(s) else 0.0
        dtype = str(s.dtype)
        samples = [x for x in s.dropna().astype(str).head(sample_n).tolist()]
        profile[str(col)] = {
            "dtype": dtype,
            "missing": round(missing, 6),
            "samples": samples,
        }
    return profile


def _build_llm_context(
    df: pd.DataFrame,
    dataset_description: str = "",
    file_name: str | None = None,
    file_ext: str | None = None,
) -> Tuple[str, Dict[str, Any]]:
    """
    Build a textual context for the LLM and a small dict of extra values
    that can be used in the prompt templates.
    """
    parts: list[str] = []

    if dataset_description:
        parts.append("Dataset description:")
        parts.append(dataset_description.strip())
        parts.append("")

    if file_name:
        parts.append(f"File: {file_name} (type: {file_ext or 'unknown'})")

    R, N = df.shape
    parts.append(f"Tabular data: {R} rows, {N} columns.")
    cols = [str(c) for c in df.columns]
    parts.append("Column names: " + ", ".join(cols[:40]))

    profile = _profile_df(df, sample_n=2, max_cols=25)
    parts.append("Column profile (dtype, missing ratio, sample values):")
    for col, info in profile.items():
        parts.append(
            f"- {col}: dtype={info['dtype']}, missing={info['missing']}, samples={info['samples']}"
        )

    context = "\n".join(parts)

    extra_values = {
        "dataset_description": dataset_description or "",
        "columns": ", ".join(cols),
        "profile": profile,
        "N": N,
        "R": R,
        "file_name": file_name or "",
        "file_ext": file_ext or "",
    }
    return context, extra_values


def _auto_base_inputs(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Heuristic automatic inputs that can be derived directly from the dataframe.
    These do NOT use the LLM.
    """
    R, N = df.shape
    auto: Dict[str, Any] = {}

    # Basic counts
    auto["N"] = float(N)
    auto["R"] = float(R)
    auto["nc"] = float(N)  # number of columns
    auto["nr"] = float(R)  # number of rows

    # Cells / completeness
    if R > 0 and N > 0:
        na_mask = df.isna()
        ic = int(na_mask.to_numpy().sum())
        nir = int(na_mask.any(axis=1).sum())
        ncl = float(R * N)
    else:
        ic = 0
        nir = 0
        ncl = 0.0

    auto["ic"] = float(ic)   # number of incomplete cells
    auto["nir"] = float(nir) # number of incomplete rows
    auto["ncl"] = float(ncl) # total number of cells
    auto["nce"] = 0.0        # syntactically incorrect cells (we don't detect -> assume 0)

    # Current date for time-related metrics
    auto["cd"] = _dt.date.today().isoformat()

    # Detect a date column (e.g. StatisticsDate)
    date_col: Optional[str] = None
    for c in df.columns:
        name = str(c).lower()
        if any(tok in name for tok in ("date", "kuupäev", "kuupaev", "kp")):
            date_col = str(c)
            break
    if date_col is None:
        # Fallback: first datetime64 column if present
        for c in df.columns:
            try:
                if pd.api.types.is_datetime64_any_dtype(df[c]):
                    date_col = str(c)
                    break
            except Exception:
                continue

    if date_col is not None and R > 0:
        dt = pd.to_datetime(df[date_col], errors="coerce", utc=True)
        dt_valid = dt[dt.notna()]
        if not dt_valid.empty:
            sd = dt_valid.min().date().isoformat()
            edp = dt_valid.max().date().isoformat()
            auto["sd_col"] = date_col
            auto["sd"] = sd
            auto["edp"] = edp
            auto["max_date"] = edp

            # "ed" (expiration of previous version) is often equal to end_date or previous period.
            # We don't know previous version, so we approximate with current end date.
            auto.setdefault("ed", edp)

            # rows not at max date -> not current rows
            ncr = int((dt_valid.dt.date.astype("string") != edp).sum())
            auto["ncr"] = float(ncr)

    # Publication date from typical metadata columns
    for c in df.columns:
        name = str(c).lower()
        if name in (
            "modifiedat",
            "modified_at",
            "updatedat",
            "updated_at",
            "lastmodified",
            "last_modified",
        ):
            ts = pd.to_datetime(df[c], errors="coerce", utc=True)
            ts_valid = ts[ts.notna()]
            if not ts_valid.empty:
                auto["dp"] = ts_valid.max().date().isoformat()
                break

    # If we still don't have dp but we have max_date, reuse that cautiously
    if "dp" not in auto and "max_date" in auto:
        auto["dp"] = auto["max_date"]

    # du (update dates mentioned) -> if any modified/updated column exists
    if any(
        str(c).lower()
        in (
            "modifiedat",
            "modified_at",
            "updatedat",
            "updated_at",
            "lastmodified",
            "last_modified",
        )
        for c in df.columns
    ):
        auto["du"] = 1.0

    # id (identifier present) -> common patterns
    if any(
        str(c).lower() in ("id", "uuid", "identifier") or str(c).lower().endswith("_id")
        for c in df.columns
    ):
        auto["id"] = 1.0

    return auto


def compute_metrics(
    df: pd.DataFrame,
    vetro_cfg: Dict[str, Any],
    prompts_cfg: Dict[str, Any],
    use_llm: bool,
    hf_runner,
    dataset_description: str = "",
    file_name: str | None = None,
    file_ext: str | None = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Core quality metric computation.

    vetro_cfg: content of formulas.yaml["vetro_methodology"]
    prompts_cfg: content of prompts.yaml["symbols"]
    """
    vetro = vetro_cfg or {}
    if not isinstance(vetro, dict):
        vetro = {}

    prompts = (prompts_cfg or {})
    if not isinstance(prompts, dict):
        prompts = {}

    R, N = df.shape

    # 1) Build LLM context
    context, extra_values = _build_llm_context(
        df=df,
        dataset_description=dataset_description,
        file_name=file_name,
        file_ext=file_ext,
    )

    # 2) Seed environment with basic values and auto-derived inputs
    env: Dict[str, Any] = {}
    env["N"] = float(N)
    env["R"] = float(R)

    auto_inputs = _auto_base_inputs(df)
    env.update(auto_inputs)

    # Debug collections
    llm_raw: Dict[str, str] = {}
    llm_evidence: Dict[str, str] = {}
    llm_conf: Dict[str, float] = {}

    symbol_values: Dict[str, Any] = {}
    symbol_source: Dict[str, str] = {}  # "auto" / "llm" / "none"

    # 3) Collect required symbols from YAML inputs
    required_symbols = set()
    for dim_name, dim_obj in vetro.items():
        if not isinstance(dim_obj, dict):
            continue
        for metric_key, metric_obj in dim_obj.items():
            if not isinstance(metric_obj, dict):
                continue
            for inp in metric_obj.get("inputs", []) or []:
                if isinstance(inp, dict):
                    for _, sym in inp.items():
                        required_symbols.add(str(sym))

    # 4) Fill each symbol (auto first, then LLM if enabled, else None)
    CONF_THRESHOLD_COUNTS = 0.25  # slightly stricter only for count symbols

    for sym in sorted(required_symbols):
        # auto base
        if sym in auto_inputs:
            val = auto_inputs[sym]
            env[sym] = val
            symbol_values[sym] = val
            symbol_source[sym] = "auto"
            continue

        # LLM
        if use_llm and hf_runner is not None and sym in prompts:
            val, raw, conf, evidence = infer_symbol(
                symbol=sym,
                context=context,
                N=N,
                prompt_defs=prompts,
                hf_runner=hf_runner,
                extra_values=extra_values,
            )

            raw_str = str(raw or "").strip()
            llm_raw[sym] = raw_str
            llm_conf[sym] = float(conf or 0.0)
            llm_evidence[sym] = evidence or ""

            typ = str(prompts[sym].get("type", "binary"))

            # For binary & dates: if we parsed something, trust it (no hard threshold)
            if val is None:
                env[sym] = None
                symbol_values[sym] = None
                symbol_source[sym] = "none"
            else:
                if typ in ("count", "count_0_to_N") and (conf or 0.0) < CONF_THRESHOLD_COUNTS:
                    # discard very low-confidence counts (they heavily affect percentages)
                    env[sym] = None
                    symbol_values[sym] = None
                    symbol_source[sym] = "none"
                else:
                    env[sym] = val
                    symbol_values[sym] = val
                    symbol_source[sym] = "llm"
        else:
            # No auto and no LLM
            env[sym] = None
            symbol_values[sym] = None
            symbol_source[sym] = "none"

    # 5) Compute metrics from formulas
    rows: list[Dict[str, Any]] = []
    for dim_name, dim_obj in vetro.items():
        if not isinstance(dim_obj, dict):
            continue

        for metric_key, metric_obj in dim_obj.items():
            if not isinstance(metric_obj, dict):
                continue

            formula_cfg = metric_obj.get("formula") or {}
            norm_cfg = metric_obj.get("normalization") or {}

            # Optional intermediate formula
            f_assign = formula_cfg.get("assign")
            f_expr = formula_cfg.get("expression")
            if f_assign and f_expr:
                env[str(f_assign)] = _eval_expr(f_expr, env)

            # Normalized / final value
            n_assign = norm_cfg.get("assign")
            n_expr = norm_cfg.get("expression")
            value = float("nan")
            if n_assign and n_expr:
                value = _eval_expr(n_expr, env)
                env[str(n_assign)] = value

            rows.append(
                {
                    "dimension": dim_name,
                    "metric": metric_key,
                    "metric_id": f"{dim_name}.{metric_key}",
                    "value": value,
                    "description": metric_obj.get("description", ""),
                }
            )

    metrics_df = pd.DataFrame(rows)

    details = {
        "auto_inputs": auto_inputs,
        "symbol_values": symbol_values,
        "symbol_source": symbol_source,
        "required_symbols": sorted(required_symbols),
        "llm_confidence": llm_conf,
        "llm_raw": llm_raw,
        "llm_evidence": llm_evidence,
    }

    return metrics_df, details
