from __future__ import annotations

from typing import Any, Dict, Tuple, Optional, List
import datetime as dt
import math
import re

import pandas as pd

from core.llm import infer_symbol

COND_ALLOWED_RE = re.compile(r"^[0-9a-zA-Z_ .<>=!()+\-*/]+$")
URL_RE = re.compile(r"https?://", re.IGNORECASE)


def _safe_eval_condition(expr: str, env: Dict[str, Any]) -> bool:
    if not isinstance(expr, str) or not COND_ALLOWED_RE.match(expr):
        return False
    try:
        return bool(eval(expr, {"__builtins__": {}}, env))
    except Exception:
        return False


def _to_numeric(x: Any) -> float:
    if x is None:
        return float("nan")
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        s = x.strip()
        try:
            d = dt.date.fromisoformat(s[:10])
            return float(d.toordinal())
        except Exception:
            pass
        try:
            return float(s)
        except Exception:
            return float("nan")
    return float("nan")


def _eval_expr(node: Any, env: Dict[str, Any]) -> float:
    if node is None:
        return float("nan")

    if isinstance(node, str):
        return _to_numeric(env.get(node, None))

    if isinstance(node, (int, float)):
        return float(node)

    if isinstance(node, dict) and "operator" in node:
        op = node["operator"]

        if op == "identity":
            return _eval_expr(node.get("operand") or node.get("of"), env)

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


def _build_llm_context(
    df: pd.DataFrame,
    dataset_description: str,
    file_name: str,
    file_ext: str,
) -> str:
    cols = list(df.columns)
    profile = _profile_df(df, sample_n=3, max_cols=40)

    parts = []
    if dataset_description.strip():
        parts.append("Portal metadata / description:")
        parts.append(dataset_description.strip())
        parts.append("")
    parts.append(f"File name: {file_name}")
    parts.append(f"File format/extension: {file_ext}")
    parts.append(f"Dataset shape: rows={len(df)}, columns={len(cols)} (N={len(cols)}).")
    parts.append("Column names: " + ", ".join([str(c) for c in cols[:40]]))
    parts.append("Column profile (dtype, missing ratio, sample values):")
    for col, info in profile.items():
        parts.append(f"- {col}: dtype={info['dtype']}, missing={info['missing']}, samples={info['samples']}")
    return "\n".join(parts)


def _pick_date_column(df: pd.DataFrame) -> Optional[str]:
    for c in df.columns:
        if "date" in str(c).lower():
            return str(c)
    return None


def _auto_inputs_from_df(df: pd.DataFrame) -> Dict[str, Any]:
    auto_inputs: Dict[str, Any] = {}

    auto_inputs["nr"] = float(len(df))
    auto_inputs["nc"] = float(df.shape[1])
    auto_inputs["ic"] = float(df.shape[0] * df.shape[1])
    auto_inputs["nce"] = float(df.isna().sum().sum())
    auto_inputs["ncl"] = float(df.shape[0] * df.shape[1])
    auto_inputs["cd"] = dt.date.today().isoformat()

    date_col = _pick_date_column(df)
    if date_col and date_col in df.columns:
        dt_series = pd.to_datetime(df[date_col], errors="coerce", utc=True)
        if dt_series.notna().any():
            auto_inputs["sd_col"] = date_col
            auto_inputs["sd"] = dt_series.min().date().isoformat()
            auto_inputs["edp"] = dt_series.max().date().isoformat()
            auto_inputs["max_date"] = dt_series.max().date().isoformat()

            max_dt = dt_series.max()
            nir = float((dt_series == max_dt).sum())
            auto_inputs["nir"] = nir
            auto_inputs["ncr_definition"] = "rows_not_at_max_date"
            auto_inputs["ncr"] = float(len(df) - int(nir))

    for c in df.columns:
        cl = str(c).lower()
        if cl in ("modifiedat", "modified_at", "updatedat", "updated_at", "lastmodified", "last_modified"):
            ts = pd.to_datetime(df[c], errors="coerce", utc=True)
            if ts.notna().any():
                auto_inputs["dp"] = ts.max().date().isoformat()
                auto_inputs["du"] = 1.0
                break

    cols_lower = [str(c).lower() for c in df.columns]
    if any(x in cols_lower for x in ["id", "uuid", "identifier"]):
        auto_inputs["id"] = 1.0

    return auto_inputs


def _looks_like_url_series(s: pd.Series, sample_n: int = 200) -> bool:
    vals = s.dropna().astype(str).head(sample_n).tolist()
    return any(URL_RE.search(v) for v in vals)


def _comprehensible_column_name(name: str) -> bool:
    n = str(name).strip()
    if not n:
        return False
    if len(n) <= 2:
        return False
    if re.fullmatch(r"\d+", n):
        return False
    if re.fullmatch(r"[A-Za-z]{1,2}\d{1,3}", n):
        return False
    return True


def _heuristic_ncuf(df: pd.DataFrame) -> float:
    return float(sum(1 for c in df.columns if _comprehensible_column_name(str(c))))


def _heuristic_ns_nsc(df: pd.DataFrame, sample_n: int = 5000) -> Tuple[float, float]:
    ns = 0
    nsc = 0
    sample = df.head(sample_n)

    for c in df.columns:
        name = str(c).lower()
        s = sample[c]
        if s.dropna().empty:
            continue

        if "date" in name:
            ns += 1
            parsed = pd.to_datetime(s, errors="coerce", utc=True)
            rate = float(parsed.notna().mean())
            if rate >= 0.9:
                nsc += 1
            continue

        if "ehak" in name or "code" in name or name.endswith("id") or "uuid" in name:
            ns += 1
            vals = s.dropna().astype(str).head(200).tolist()
            ok = 0
            for v in vals:
                if re.fullmatch(r"\d+", v.strip()):
                    ok += 1
            if vals and (ok / len(vals)) >= 0.9:
                nsc += 1
            continue

    return float(ns), float(nsc)


def _heuristic_five_star(symbol: str, df: pd.DataFrame, file_ext: str, dataset_description: str) -> Optional[float]:
    ext = (file_ext or "").lower()

    if symbol == "s2":
        if ext in ("csv", "tsv", "xlsx", "xls", "json", "xml", "parquet"):
            return 1.0
        return 0.0

    if symbol == "s3":
        if ext in ("csv", "tsv", "json", "xml"):
            return 1.0
        if ext in ("xlsx", "xls"):
            return 0.0
        return None

    if symbol == "s4":
        cols_lower = [str(c).lower() for c in df.columns]
        if any(("id" == c) or c.endswith("id") or "uuid" in c or "uri" in c or "url" in c or "ehak" in c for c in cols_lower):
            return 1.0
        for c in df.columns:
            if _looks_like_url_series(df[c]):
                return 1.0
        return 0.0

    if symbol == "s5":
        for c in df.columns:
            if _looks_like_url_series(df[c]):
                return 1.0
        return 0.0

    if symbol == "s1":
        desc = dataset_description.lower()
        if "license" in desc or "licence" in desc or "cc by" in desc or "creative commons" in desc:
            return 1.0
        if dataset_description.strip():
            return 0.0
        return None

    return None


def _auto_symbol(
    sym: str,
    df: pd.DataFrame,
    auto_inputs: Dict[str, Any],
    dataset_description: str,
    file_name: str,
    file_ext: str,
) -> Tuple[Optional[Any], str]:
    if sym in auto_inputs:
        return auto_inputs[sym], "auto"

    if sym in ("s1", "s2", "s3", "s4", "s5"):
        v = _heuristic_five_star(sym, df, file_ext, dataset_description)
        if v is not None:
            return v, "auto"

    if sym == "ncuf":
        return _heuristic_ncuf(df), "auto"

    if sym in ("ns", "nsc"):
        ns, nsc = _heuristic_ns_nsc(df)
        return (ns, "auto") if sym == "ns" else (nsc, "auto")

    desc = (dataset_description or "").strip()

    if sym == "t":
        if desc or file_name:
            return 1.0, "auto"
        return 0.0, "auto"

    if sym == "d":
        if len(desc) >= 20:
            return 1.0, "auto"
        if desc:
            return 0.0, "auto"
        return None, ""

    if sym == "dc":
        if re.search(r"\b(created|issued|publication|published)\b", desc, re.IGNORECASE) and re.search(r"\d{4}-\d{2}-\d{2}", desc):
            return 1.0, "auto"
        if desc:
            return 0.0, "auto"
        return None, ""

    if sym in ("pb", "s"):
        if re.search(r"\b(publisher|publisher organization|publishing|source|andmeallikas|asutus|amet|ministeerium)\b", desc, re.IGNORECASE):
            return 1.0, "auto"
        if desc:
            return 0.0, "auto"
        return None, ""

    if sym == "c":
        if re.search(r"\b(category|theme|topic|subject|valdkond|teema)\b", desc, re.IGNORECASE):
            return 1.0, "auto"
        if desc:
            return 0.0, "auto"
        return None, ""

    if sym == "l":
        if re.search(r"\b(language|keel)\b", desc, re.IGNORECASE):
            return 1.0, "auto"
        if desc:
            return 0.0, "auto"
        return None, ""

    if sym == "lu":
        if re.search(r"\b(changelog|version|history|muudat)\b", desc, re.IGNORECASE):
            return 1.0, "auto"
        if desc:
            return 0.0, "auto"
        return None, ""

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
    weight_by_confidence: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:

    vetro = (formulas_cfg or {}).get("vetro_methodology") or {}
    if not isinstance(vetro, dict):
        vetro = {}

    labels_map = {}
    if isinstance(vetro.get("labels"), dict):
        labels_map = vetro.get("labels") or {}

    prompts = (prompt_cfg or {}).get("symbols", {}) or {}
    if not isinstance(prompts, dict):
        prompts = {}

    auto_inputs = _auto_inputs_from_df(df)

    required_symbols = set()
    for dim, dim_obj in vetro.items():
        if not isinstance(dim_obj, dict):
            continue
        if dim == "labels":
            continue
        for _, metric_obj in dim_obj.items():
            if not isinstance(metric_obj, dict):
                continue
            for inp in metric_obj.get("inputs", []) or []:
                if isinstance(inp, dict):
                    for _, sym in inp.items():
                        required_symbols.add(sym)

    context = _build_llm_context(df, dataset_description, file_name, file_ext)

    env: Dict[str, Any] = {}
    for k, v in auto_inputs.items():
        env[k] = v

    llm_raw: Dict[str, str] = {}
    symbol_rows: List[Dict[str, Any]] = []

    def sym_type(sym: str) -> str:
        if sym in ("sd", "edp", "dp", "ed", "cd", "max_date"):
            return "date"
        cfg = prompts.get(sym) if isinstance(prompts.get(sym), dict) else {}
        t = str(cfg.get("type", "")).strip()
        return t if t else "number"

    for sym in sorted(required_symbols):
        stype = sym_type(sym)

        auto_val, auto_src = _auto_symbol(sym, df, auto_inputs, dataset_description, file_name, file_ext)
        if auto_src == "auto":
            raw_val = auto_val
            conf = 1.0
            evidence = ""
            raw_text = ""
            source = "auto"
        else:
            raw_val = None
            conf = 0.0
            evidence = ""
            raw_text = ""
            source = "fail"

            if use_llm and hf_runner is not None and sym in prompts:
                val, raw_text, conf, evidence = infer_symbol(
                    symbol=sym,
                    context=context,
                    N=int(df.shape[1]),
                    prompt_defs=prompts,
                    hf_runner=hf_runner,
                    extra_values={
                        "dataset_description": dataset_description,
                        "columns": ", ".join([str(c) for c in df.columns]),
                        "profile": "",
                        "file_name": file_name,
                        "file_ext": file_ext,
                    },
                )
                llm_raw[sym] = raw_text
                raw_val = val
                source = "llm" if val is not None else "fail"

        effective_val = raw_val
        if raw_val is None and stype != "date":
            effective_val = 0.0

        if weight_by_confidence and source == "llm" and effective_val is not None and stype != "date":
            try:
                effective_val = float(effective_val) * float(conf)
            except Exception:
                pass

        env[sym] = effective_val

        symbol_rows.append(
            {
                "symbol": sym,
                "value (None=did not work)": raw_val,
                "effective_value": effective_val,
                "source": source,
                "confidence": round(float(conf), 3) if conf is not None else None,
                "evidence": evidence,
                "raw": raw_text,
            }
        )

    rows = []
    for dim, dim_obj in vetro.items():
        if not isinstance(dim_obj, dict) or dim == "labels":
            continue

        for metric_key, metric_obj in dim_obj.items():
            if not isinstance(metric_obj, dict):
                continue

            intermediate = metric_obj.get("intermediate_calculation")
            if intermediate:
                steps = intermediate if isinstance(intermediate, list) else [intermediate]
                for step in steps:
                    if not isinstance(step, dict):
                        continue
                    assign = step.get("assign")
                    expr = step.get("expression")
                    if assign and expr:
                        env[assign] = _eval_expr(expr, env)

            f = metric_obj.get("formula") or {}
            if isinstance(f, dict) and f.get("assign") and f.get("expression") is not None:
                env[str(f["assign"])] = _eval_expr(f.get("expression"), env)

            n = metric_obj.get("normalization") or {}
            value = float("nan")
            if isinstance(n, dict) and n.get("assign") and n.get("expression") is not None:
                value = _eval_expr(n.get("expression"), env)
                env[str(n["assign"])] = value

            metric_id = f"{dim}.{metric_key}"
            label = labels_map.get(metric_id, metric_id)

            rows.append(
                {
                    "dimension": dim,
                    "metric": metric_key,
                    "metric_id": metric_id,
                    "value": value,
                    "description": metric_obj.get("description", ""),
                    "metric_label": label,
                }
            )

    metrics_df = pd.DataFrame(rows)

    details = {
        "auto_inputs": auto_inputs,
        "symbols": symbol_rows,
        "llm_raw": llm_raw,
    }
    return metrics_df, details
