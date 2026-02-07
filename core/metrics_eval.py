from __future__ import annotations

from typing import Any, Dict, Tuple, Optional
import math
import re
import pandas as pd

from core.llm import infer_symbol

COND_ALLOWED_RE = re.compile(r"^[0-9a-zA-Z_ .<>=!()+\-*/]+$")
EVIDENCE_LINE_RE = re.compile(r"(?im)^\s*evidence\s*[:=]\s*(.*)\s*$")


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


def _profile_df(df: pd.DataFrame, sample_n: int = 2, max_cols: int = 40) -> Dict[str, Any]:
    profile: Dict[str, Any] = {}
    cols = list(df.columns)[:max_cols]
    for col in cols:
        s = df[col]
        missing = float(s.isna().mean()) if len(s) else 0.0
        dtype = str(s.dtype)
        samples = [x for x in s.dropna().head(sample_n).astype(str).tolist()]
        profile[str(col)] = {"dtype": dtype, "missing": round(missing, 6), "samples": samples}
    return profile


def _build_llm_context(df: pd.DataFrame, dataset_description: str = "") -> str:
    cols = list(df.columns)
    profile = _profile_df(df, sample_n=2, max_cols=40)

    parts = []

    # ✅ NEW: include pasted portal metadata/description first (keep it short)
    if isinstance(dataset_description, str) and dataset_description.strip():
        dd = dataset_description.strip()
        if len(dd) > 2500:
            dd = dd[:2500] + "..."
        parts.append("User-provided portal metadata / description:")
        parts.append(dd)
        parts.append("")  # spacer

    parts.append(f"The dataset has {len(cols)} columns (N={len(cols)}).")
    parts.append("Column names: " + ", ".join([str(c) for c in cols[:40]]))
    parts.append("Column profile (dtype, missing ratio, samples):")
    for col, info in profile.items():
        parts.append(f"- {col}: dtype={info['dtype']}, missing={info['missing']}, samples={info['samples']}")
    return "\n".join(parts)


# ---------- Fix #4: Auto-first symbol derivation ----------
def _auto_symbol(sym: str, df: pd.DataFrame, auto_inputs: Dict[str, Any]) -> Tuple[Optional[Any], str]:
    """
    Returns (value or None, source_string)
    source_string: "auto" if computed, "" if not available.
    """
    # direct auto_inputs
    if sym in auto_inputs:
        return auto_inputs[sym], "auto"

    cols_lower = [str(c).lower() for c in df.columns]

    # sd/edp/max_date already in auto_inputs; dp can be inferred from ModifiedAt if present
    if sym == "dp":
        # try ModifiedAt column as "publication/last updated"
        for c in df.columns:
            if str(c).lower() in ("modifiedat", "modified_at", "updatedat", "updated_at", "lastmodified", "last_modified"):
                dt = pd.to_datetime(df[c], errors="coerce", utc=True)
                if dt.notna().any():
                    return dt.max().date().isoformat(), "auto"
        # fallback to max_date if present
        if "max_date" in auto_inputs:
            return auto_inputs["max_date"], "auto"

    # du (update dates mentioned) -> if there is a modified/updated timestamp column
    if sym == "du":
        if any(x in cols_lower for x in ["modifiedat", "modified_at", "updatedat", "updated_at", "lastmodified", "last_modified"]):
            return 1.0, "auto"

    # id (identifier present) -> common if there's Id/uuid columns
    if sym == "id":
        if any(x in cols_lower for x in ["id", "uuid", "identifier"]):
            return 1.0, "auto"

    return None, ""


def compute_metrics(
    df: pd.DataFrame,
    formulas_cfg: Dict[str, Any],
    prompt_cfg: Dict[str, Any],
    use_llm: bool,
    hf_runner,
    dataset_description: str = "",   # ✅ NEW
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

    # ---------- auto inputs (sd/edp/max_date etc) ----------
    auto_inputs: Dict[str, Any] = {}

    date_col = None
    for c in df.columns:
        if "date" in str(c).lower():
            date_col = c
            break

    if date_col is not None:
        dt = pd.to_datetime(df[date_col], errors="coerce", utc=True)
        if dt.notna().any():
            auto_inputs["sd_col"] = str(date_col)
            auto_inputs["sd"] = dt.min().date().isoformat()
            auto_inputs["edp"] = dt.max().date().isoformat()
            auto_inputs["max_date"] = dt.max().date().isoformat()

    # Put auto_inputs into env
    for k, v in auto_inputs.items():
        env[k] = v

    # ✅ NEW: build context with dataset_description
    context = _build_llm_context(df, dataset_description=dataset_description or "")

    # Debug collections
    llm_raw: Dict[str, str] = {}
    llm_evidence: Dict[str, str] = {}
    llm_conf: Dict[str, float] = {}

    symbol_values: Dict[str, Any] = {}
    symbol_source: Dict[str, str] = {}  # auto / llm / fail

    # ---------- required symbols from YAML ----------
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

    # ---------- fill symbols ----------
    CONF_THRESHOLD = 0.4

    for sym in sorted(required_symbols):
        # auto-first
        auto_val, auto_src = _auto_symbol(sym, df, auto_inputs)
        if auto_src == "auto" and auto_val is not None:
            env[sym] = auto_val
            symbol_values[sym] = auto_val
            symbol_source[sym] = "auto"
            continue

        # If no auto worked, try LLM
        if use_llm and hf_runner is not None and sym in prompts:
            val, raw, conf = infer_symbol(
                symbol=sym,
                context=context,
                N=int(df.shape[1]),
                prompt_defs=prompts,
                hf_runner=hf_runner,
            )

            raw_str = str(raw or "").strip()
            llm_raw[sym] = raw_str
            llm_conf[sym] = float(conf)

            em = EVIDENCE_LINE_RE.search(raw_str)
            llm_evidence[sym] = (em.group(1).strip() if em else "")

            # None means "did not work", 0.0 means "worked and answer was zero"
            if val is None or conf < CONF_THRESHOLD:
                env[sym] = None
                symbol_values[sym] = None
                symbol_source[sym] = "fail"
            else:
                env[sym] = val
                symbol_values[sym] = val
                symbol_source[sym] = "llm"

        else:
            env[sym] = None
            symbol_values[sym] = None
            symbol_source[sym] = "fail"

    # ---------- compute metrics ----------
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
        "llm_confidence": llm_conf,
        "llm_raw": llm_raw,
        "llm_evidence": llm_evidence,
    }
    return metrics_df, details
