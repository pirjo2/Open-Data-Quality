from __future__ import annotations

from typing import Any, Dict, Tuple, Optional

import math
import re
from datetime import date as _date_type
import json

import pandas as pd

# --- Turvaline tingimusavaldiste eval --- #
COND_ALLOWED_RE = re.compile(r"^[0-9a-zA-Z_ .<>=!()+\-*/]+$")


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
                if "elif" in rule and _safe_eval_condition(rule["elif"], env):
                    return _eval_expr(rule.get("then"), env)
                if "else" in rule:
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
    auto["nce"] = None

    # Currentness: detect a primary date column
    date_col = None
    for c in df.columns:
        if "date" in str(c).lower() or "kuup" in str(c).lower():
            date_col = c
            break

    if date_col is not None:
        dt_series = pd.to_datetime(df[date_col], errors="coerce", utc=True)
        if dt_series.notna().any():
            sd = dt_series.min().date()
            edp = dt_series.max().date()
            auto["sd_col"] = str(date_col)
            auto["sd"] = sd.isoformat()
            auto["edp"] = edp.isoformat()
            auto["max_date"] = edp.isoformat()
            auto["ed"] = edp.isoformat()  # expiration ≈ previous end of period
            auto["ncr"] = float((dt_series != dt_series.max()).sum())

    auto["cd"] = _date_type.today().isoformat()

    # Publication / update dates from typical columns like ModifiedAt
    mod_col = None
    for c in df.columns:
        cl = str(c).lower()
        if cl in ("modifiedat", "modified_at", "updatedat", "updated_at", "lastmodified", "last_modified"):
            mod_col = c
            break

    if mod_col is not None:
        dtm = pd.to_datetime(df[mod_col], errors="coerce", utc=True)
        if dtm.notna().any():
            dp = dtm.max().date()
            auto["dp"] = dp.isoformat()
            auto["du"] = 1.0

    if "dp" not in auto and "max_date" in auto:
        auto["dp"] = auto["max_date"]

    # Standardised columns
    def _infer_ns_nsc(df2: pd.DataFrame) -> Tuple[Optional[float], Optional[float]]:
        ns = 0.0
        nsc = 0.0
        for col in df2.columns:
            lname = str(col).lower()
            s = df2[col].dropna()
            if s.empty:
                continue

            is_candidate = False
            is_standardised = False

            if "date" in lname or "kuup" in lname:
                is_candidate = True
                dt2 = pd.to_datetime(s, errors="coerce", utc=True)
                if dt2.notna().mean() > 0.9:
                    is_standardised = True
            elif "year" in lname or "aasta" in lname:
                is_candidate = True
                vals = pd.to_numeric(s, errors="coerce")
                if vals.notna().mean() > 0.9 and vals.between(1900, 2100).mean() > 0.9:
                    is_standardised = True
            elif any(tok in lname for tok in ("ehak", "iso", "code", "kood")):
                is_candidate = True
                is_standardised = True
            elif any(tok in lname for tok in ("country", "county", "commune", "region", "maakond", "vald", "linn")):
                is_candidate = True

            if is_candidate:
                ns += 1.0
                if is_standardised:
                    nsc += 1.0

        if ns == 0:
            return None, None
        return ns, nsc

    ns, nsc = _infer_ns_nsc(df)
    if ns is not None:
        auto["ns"] = ns
    if nsc is not None:
        auto["nsc"] = nsc

    # Understandability heuristics
    auto["ncm"] = float(N)
    auto["ncuf"] = float(N)

    # 5-star open data heuristics
    ext = (file_ext or "").lower()
    auto["s1"] = 1.0
    auto["s2"] = 1.0
    auto["s3"] = 1.0 if ext in (".csv", ".tsv", ".json", ".xml") else 0.5
    cols_lower = [str(c).lower() for c in df.columns]
    auto["s4"] = 1.0 if any(("id" in c or "uuid" in c or "uri" in c) for c in cols_lower) else 0.0
    auto["s5"] = None

    # Traceability proxies
    auto["s"] = 1.0 if ("dp" in auto or "sd" in auto) else None
    auto["dc"] = 1.0 if "dp" in auto else None

    auto["lu"] = None

    # Aggregation accuracy defaults
    auto["sc"] = 1.0
    auto["oav"] = None
    auto["dav"] = None
    auto["e"] = None

    return auto

def compute_metrics(
    df: pd.DataFrame,
    formulas_cfg: Dict[str, Any],
    prompt_defs: Optional[Dict[str, Any]],
    use_llm: bool,
    hf_runner,
    file_ext: Optional[str] = None,
    manual_metadata: Optional[Dict[str, Any]] = None,
    trino_metadata: Optional[Dict[str, Any]] = None,
    trino_metadata_raw: Optional[Dict[str, Any]] = None,
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
    trino_metadata = trino_metadata or {}
    trino_metadata_raw = trino_metadata_raw or {}

    cov = trino_metadata_raw.get("temporalcoverage")

    mod_date = trino_metadata_raw.get("modificationdate")
    if mod_date:
        details["symbol_values"]["du"] = 1.0
        details["symbol_source"]["du"] = "parser"

    if isinstance(cov, str):
        years = re.findall(r"\d{4}", cov)
        if len(years) >= 2:
            sd_val = f"{years[0]}-01-01"
            edp_val = f"{years[1]}-12-31"

            details["symbol_values"]["sd"] = sd_val
            details["symbol_source"]["sd"] = "parser"

            details["symbol_values"]["edp"] = edp_val
            details["symbol_source"]["edp"] = "parser"

    # --------- PRIORITY RESOLUTION: auto -> trino -> manual -> missing ----------
    for sym in sorted(required_symbols):
        if details["symbol_source"].get(sym) == "parser":
            continue
        # Auto only if NOT traceability core symbols
        if sym in auto_inputs and sym not in {"s", "dc"}:
            details["symbol_values"][sym] = auto_inputs[sym]
            details["symbol_source"][sym] = "auto"
            env[sym] = auto_inputs[sym]
            continue

        if sym in trino_metadata:
            val = trino_metadata[sym]
            details["symbol_values"][sym] = val
            details["symbol_source"][sym] = "trino"
            env[sym] = val
            continue

        if sym in manual_metadata:
            val = manual_metadata[sym]
            details["symbol_values"][sym] = val
            details["symbol_source"][sym] = "manual"
            env[sym] = val
            continue

        details["symbol_values"][sym] = None
        details["symbol_source"][sym] = "missing"

    # --------- LLM fallback (missing + auto=0 refinement) ----------
    if use_llm and hf_runner is not None and prompt_defs:

        refinable_symbols = {
            "s", "dc", "dp", "du", "sd", "edp", "ed",
            "cv", "l", "id", "c"
        }

        missing_syms = []

        # Collect symbols first
        for sym in sorted(required_symbols):
            if sym not in prompt_defs:
                continue

            source = details["symbol_source"].get(sym)
            val = details["symbol_values"].get(sym)

            # Case 1: missing
            if source == "missing":
                missing_syms.append(sym)
                continue

            # Case 2: auto=0 and refinable
            if (
                sym in refinable_symbols
                and source == "auto"
                and val is None
            ):
                missing_syms.append(sym)

        # Only now build context
        if missing_syms:

            context_lines = []
            context_lines.append("Columns:")
            context_lines.append(", ".join(str(c) for c in df.columns))
            context_lines.append("")

            context_lines.append("Basic column profiles:")
            for col in df.columns:
                s = df[col]
                dtype = str(s.dtype)
                missing_ratio = float(s.isna().mean())
                sample_vals = list(s.dropna().unique()[:3])
                context_lines.append(
                    f"- {col}: dtype={dtype}, missing={missing_ratio:.3f}, samples={sample_vals}"
                )
            context_lines.append("")
            context_lines.append("Raw metadata record from portal (JSON):")
            context_lines.append(
                json.dumps(trino_metadata_raw, indent=2, default=str)
            )

            if manual_metadata:
                context_lines.append("")
                context_lines.append("Manual metadata overrides:")
                context_lines.append(
                    json.dumps(manual_metadata, indent=2, default=str)
                )

            context = "\n".join(context_lines)

            from core.llm import infer_symbol as _infer_symbol

            # Then call LLM
            for sym in missing_syms:
                val, raw, conf, evid = _infer_symbol(
                    symbol=sym,
                    context=context,
                    N=N,
                    prompt_defs=prompt_defs,
                    hf_runner=hf_runner,
                    extra_values={"N": N},
                )

                details["llm_raw"][sym] = raw
                details["llm_confidence"][sym] = conf
                details["llm_evidence"][sym] = evid

                # Do NOT override trino or manual
                if details["symbol_source"].get(sym) in {"trino", "manual"}:
                    continue

                if val is None or (conf is not None and conf < 0.4):
                    details["symbol_source"][sym] = "llm_fail"
                    env.setdefault(sym, 0.0)
                else:
                    details["symbol_source"][sym] = "llm"
                    details["symbol_values"][sym] = val
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

            out_val: Optional[float]
            if isinstance(val, (int, float)) and not math.isnan(val):
                out_val = float(val)
            else:
                out_val = None

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