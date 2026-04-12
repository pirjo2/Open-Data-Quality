from __future__ import annotations

from typing import Any, Dict
import re
import pandas as pd


def parse_kv_metadata(text: str) -> Dict[str, Any]:
    meta: Dict[str, Any] = {}
    for line in (text or "").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        k = k.strip()
        v = v.strip()
        if not k:
            continue
        try:
            meta[k] = float(v)
        except Exception:
            meta[k] = v
    return meta


def extract_symbols_from_realistic_text(text: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    low = (text or "").lower()

    if re.search(r"updates\s*:\s*\[\s*\]", low):
        out["lu"] = 0.0
        out["du"] = 0.0

    if "update history" in low or "updated regularly" in low:
        out["lu"] = 1.0

    if re.search(r"updated on\s+\d{4}-\d{2}-\d{2}", low):
        out["du"] = 1.0

    m = re.search(
        r"covers the period from\s+(20\d{2}-\d{2}-\d{2})\s+to\s+(20\d{2}-\d{2}-\d{2})",
        low
    )
    if m:
        out["sd"] = m.group(1)
        out["edp"] = m.group(2)

    m = re.search(r"published on\s+(20\d{2}-\d{2}-\d{2})", low)
    if m:
        out["dp"] = m.group(1)

    m = re.search(r"expired on\s+(20\d{2}-\d{2}-\d{2})", low)
    if m:
        out["ed"] = m.group(1)

    m = re.search(r"became available on\s+(20\d{2}-\d{2}-\d{2})", low)
    if m:
        out["cd"] = m.group(1)

    if re.search(r"^title\s*:", text, re.I | re.M):
        out["t"] = 1.0
    elif "title is missing" in low:
        out["t"] = 0.0

    if re.search(r"^description\s*:", text, re.I | re.M):
        out["d"] = 1.0
    elif "description is missing" in low:
        out["d"] = 0.0

    if re.search(r"^identifier\s*:", text, re.I | re.M):
        out["id"] = 1.0
    elif "no identifier" in low or "identifier is missing" in low:
        out["id"] = 0.0

    if re.search(r"^publisher\s*:", text, re.I | re.M):
        out["pb"] = 1.0
    elif "publisher is missing" in low:
        out["pb"] = 0.0

    if re.search(r"^coverage\s*:", text, re.I | re.M):
        out["cv"] = 1.0
    elif "no coverage information" in low:
        out["cv"] = 0.0

    if re.search(r"^language\s*:", text, re.I | re.M):
        out["l"] = 1.0
    elif "language is missing" in low:
        out["l"] = 0.0

    if re.search(r"^source\s*:", text, re.I | re.M):
        out["s"] = 1.0
    elif "no source information" in low:
        out["s"] = 0.0

    if re.search(r"^date of creation\s*:\s*(20\d{2}-\d{2}-\d{2})", text, re.I | re.M):
        out["dc"] = 1.0
    elif "no creation date" in low or "creation date is missing" in low:
        out["dc"] = 0.0

    if re.search(r"^category\s*:", text, re.I | re.M):
        out["c"] = 1.0
    elif "no category" in low or "category is missing" in low:
        out["c"] = 0.0

    return out


def normalize_metadata_to_symbols(meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Accept either:
      - direct symbols
      - or common field names

    Important:
    - do NOT fabricate 0-values for fields that are not explicitly present
    - only normalize fields that actually exist in `meta`
    """
    out: Dict[str, Any] = {}

    direct_symbols = {
        "pb", "t", "d", "dc", "cv", "l", "id", "s",
        "dp", "sd", "edp", "ed", "cd",
        "lu", "du", "c",
    }

    for k, v in meta.items():
        kk = str(k).strip().lower()
        if kk in direct_symbols:
            out[kk] = v

    def _is_missing(x: Any) -> bool:
        if x is None:
            return True
        if isinstance(x, float) and pd.isna(x):
            return True
        if isinstance(x, str):
            s = x.strip().lower()
            return s in {"", "[]", "none", "null", "missing", "n/a", "na"}
        return False

    def _set_presence(symbol: str, *keys: str) -> None:
        for key in keys:
            if key in meta:
                out[symbol] = 0.0 if _is_missing(meta.get(key)) else 1.0
                return

    def _set_date(symbol: str, *keys: str) -> None:
        for key in keys:
            if key in meta and not _is_missing(meta.get(key)):
                out[symbol] = str(meta.get(key))
                return

    _set_presence("t", "title")
    _set_presence("d", "description", "notes")
    _set_presence("pb", "publisher", "organization", "org_name")
    _set_presence("s", "source")
    _set_presence("dc", "metadata_created", "issued", "created", "date_of_creation")
    _set_presence("cv", "coverage", "temporalcoverage")
    _set_presence("l", "language", "lang")
    _set_presence("id", "identifier", "dataset_id", "id")
    _set_presence("c", "category", "theme")
    _set_presence("lu", "updates", "update_history")
    _set_presence("du", "update_dates", "updates")

    _set_date("dp", "date_of_publication", "metadata_modified", "modified")

    return {k: v for k, v in out.items() if v is not None}