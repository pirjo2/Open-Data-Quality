from __future__ import annotations

import json
import re
from typing import Any, Dict

import pandas as pd
import yaml


def parse_kv_metadata(text: str) -> Dict[str, Any]:
    meta: Dict[str, Any] = {}

    for line in (text or "").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue

        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()

        if not key:
            continue

        try:
            meta[key] = float(value)
        except Exception:
            meta[key] = value

    return meta


def dataframe_to_metadata_dict(df: pd.DataFrame) -> Dict[str, Any]:
    if df is None or df.empty:
        return {}

    clean_df = df.copy()
    clean_df.columns = [str(col).strip() for col in clean_df.columns]

    key_aliases = {"key", "field", "name", "symbol", "parameter"}
    value_aliases = {"value", "content", "data", "answer"}

    key_col = None
    value_col = None

    for col in clean_df.columns:
        col_l = col.lower()
        if col_l in key_aliases and key_col is None:
            key_col = col
        if col_l in value_aliases and value_col is None:
            value_col = col

    if key_col and value_col:
        out: Dict[str, Any] = {}
        for _, row in clean_df[[key_col, value_col]].dropna(subset=[key_col]).iterrows():
            k = str(row[key_col]).strip()
            if not k:
                continue
            out[k] = row[value_col]
        return out

    if len(clean_df) == 1:
        row = clean_df.iloc[0].dropna()
        return {str(k).strip(): v for k, v in row.to_dict().items() if str(k).strip()}

    return {}


def parse_text_metadata_content(text: str) -> Dict[str, Any]:
    clean_text = (text or "").strip()
    if not clean_text:
        return {}

    try:
        parsed = json.loads(clean_text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    try:
        parsed = yaml.safe_load(clean_text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    return parse_kv_metadata(clean_text)


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
        low,
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

    if re.search(r"^publisher\s*:", text, re.I | re.M):
        out["pb"] = 1.0
    elif "publisher is missing" in low:
        out["pb"] = 0.0

    if re.search(r"^language\s*:", text, re.I | re.M):
        out["l"] = 1.0
    elif "language is missing" in low:
        out["l"] = 0.0

    if re.search(r"^source\s*:", text, re.I | re.M):
        out["s"] = 1.0
    elif "source is missing" in low:
        out["s"] = 0.0

    if re.search(r"^coverage\s*:", text, re.I | re.M):
        out["cv"] = 1.0
    elif "coverage is missing" in low:
        out["cv"] = 0.0

    if re.search(r"^category\s*:", text, re.I | re.M):
        out["c"] = 1.0
    elif "no category" in low or "category is missing" in low:
        out["c"] = 0.0

    return out


def normalize_metadata_to_symbols(meta: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    meta_l = {str(k).strip().lower(): v for k, v in (meta or {}).items()}

    key_aliases = {
        "publisher_name": "publisher",
        "organisation": "organization",
        "organisation_name": "organization",
        "dataset_title": "title",
        "name": "title",
        "metadata_created": "metadata_created",
        "created_at": "created",
        "creation_date": "date_of_creation",
        "metadata_modified": "metadata_modified",
        "modified_at": "modified",
        "update_date": "modified",
    }

    meta_l = {key_aliases.get(k, k): v for k, v in meta_l.items()}

    direct_symbols = {
        "pb",
        "t",
        "d",
        "dc",
        "cv",
        "l",
        "id",
        "s",
        "c",
        "dp",
        "sd",
        "edp",
        "ed",
        "cd",
        "lu",
        "du",
    }

    for key, value in meta_l.items():
        if key in direct_symbols:
            out[key] = value

    def _is_explicit_missing(value: Any) -> bool:
        if value is None:
            return True
        if isinstance(value, float) and pd.isna(value):
            return True
        if isinstance(value, str):
            return value.strip().lower() in {"", "none", "null", "missing", "n/a", "na", "[]"}
        if isinstance(value, (list, tuple, set, dict)):
            return len(value) == 0
        return False

    def _presence_from_key(meta_dict: Dict[str, Any], *keys: str) -> Any:
        for key in keys:
            if key in meta_dict:
                return 0.0 if _is_explicit_missing(meta_dict[key]) else 1.0
        return None

    semantic_key_map = {
        "t": ("title",),
        "d": ("description", "notes"),
        "pb": ("publisher", "organization", "org_name"),
        "s": ("source",),
        "dc": ("metadata_created", "issued", "created", "date_of_creation"),
        "cv": ("coverage", "temporalcoverage"),
        "l": ("language", "lang"),
        "id": ("identifier", "dataset_id", "datasetidentifier"),
        "c": ("category", "theme"),
    }

    for target_key, source_keys in semantic_key_map.items():
        if target_key not in out:
            value = _presence_from_key(meta_l, *source_keys)
            if value is not None:
                out[target_key] = value

    if "dp" not in out:
        for key in [
            "date_of_publication",
            "metadata_modified",
            "modified",
            "releasedate",
            "modificationdate",
        ]:
            if key in meta_l and str(meta_l[key]).strip():
                out["dp"] = str(meta_l[key])
                break

    for field in ["sd", "edp", "ed", "cd"]:
        if field not in out and field in meta_l:
            value = meta_l[field]
            if value is not None and str(value).strip():
                out[field] = str(value)

    if "lu" not in out:
        value = _presence_from_key(meta_l, "update_history", "updates", "modifications")
        if value is not None:
            out["lu"] = value

    if "du" not in out:
        value = _presence_from_key(
            meta_l,
            "update_dates",
            "modified",
            "metadata_modified",
            "modificationdate",
        )
        if value is not None:
            out["du"] = value

    return out