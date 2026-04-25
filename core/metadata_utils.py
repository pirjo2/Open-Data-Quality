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

    meta_l = {
        key_aliases.get(k, k): v
        for k, v in meta_l.items()
    }

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

    def _is_explicit_missing(x: Any) -> bool:
        if x is None:
            return True
        if isinstance(x, float) and pd.isna(x):
            return True
        if isinstance(x, str):
            return x.strip().lower() in {
                "",
                "none",
                "null",
                "missing",
                "n/a",
                "na",
                "[]",
                "not provided",
                "not available",
            }
        return False

    def _presence_from_key(meta_dict: Dict[str, Any], *keys: str):
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
            out[field] = str(meta_l[field])

    if "lu" not in out and "updates" in meta_l:
        out["lu"] = 0.0 if _is_explicit_missing(meta_l["updates"]) else 1.0

    if "du" not in out and "updates" in meta_l:
        out["du"] = 0.0 if _is_explicit_missing(meta_l["updates"]) else None

    return {k: v for k, v in out.items() if v is not None}