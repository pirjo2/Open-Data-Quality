"""
Batch Trino assessment script for the Open-Data-Quality thesis prototype.

Run from the repository root, for example:

    python scripts/run_trino_batch_assessment.py \
        --tables-csv tabelite_nimed2026-05-02T18-59_export.csv \
        --metadata-csv metadata_tabelid2026-05-02T18-46_export.csv \
        --sample-rows 100 \
        --use-llm

Environment variables:
    TRINO_HOST       default: trino.avaandmeait.ee
    TRINO_PORT       default: 443
    TRINO_USER       required
    TRINO_PASSWORD   optional, but usually required
    OPENAI_API_KEY   required if --use-llm --llm-provider openai

What this does:
1. Reads physical Trino table names from a CSV export or from landing.information_schema.tables.
2. Reads portal metadata from a CSV export or from landing.avaandmete_portaal.dataset_metadata.
3. Builds a best-effort mapping between physical tables and metadata records.
4. For each matched dataset, samples rows from Trino.
5. Runs the existing run_quality_assessment() function.
6. Saves an Excel workbook and CSV checkpoints under the output folder.

Important:
- The mapping is systematic but not perfect. Always review the Mapping sheet before using final thesis results.
- Running all datasets with LLM enabled can be slow and can cost money. Use --max-datasets for testing.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import re
import sys
import time
import unicodedata
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from difflib import SequenceMatcher

import pandas as pd

# Make imports work when this file is placed in scripts/ and run from repo root.
REPO_ROOT = Path(__file__).resolve().parents[1] if Path(__file__).resolve().parent.name == "scripts" else Path.cwd()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.pipeline import run_quality_assessment
from core.metadata_utils import normalize_metadata_to_symbols


FORMULAS_YAML = "configs/formulas.yaml"
PROMPTS_YAML = "configs/prompts.yaml"
DEFAULT_OUTPUT_DIR = "outputs/trino_batch_assessment"


ESTONIAN_TRANSLATION = str.maketrans(
    {
        "õ": "o", "ä": "a", "ö": "o", "ü": "u", "š": "s", "ž": "z",
        "Õ": "o", "Ä": "a", "Ö": "o", "Ü": "u", "Š": "s", "Ž": "z",
    }
)


# -----------------------------
# Generic helpers
# -----------------------------


def make_json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, dict):
        return {str(k): make_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [make_json_safe(v) for v in value]
    try:
        if isinstance(value, float) and math.isnan(value):
            return None
    except Exception:
        pass
    return value


def to_excel_safe(value: Any, max_len: int = 32000) -> Any:
    value = make_json_safe(value)
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False, default=str)
    if value is None:
        return ""
    if isinstance(value, str) and len(value) > max_len:
        return value[:max_len] + "…[truncated]"
    return value


def read_csv_auto(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    encodings = ["utf-8", "utf-8-sig", "cp1257", "iso-8859-13", "latin1", "cp1252"]
    last_exc: Optional[Exception] = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError as exc:
            last_exc = exc
    raise RuntimeError(f"Could not read CSV file {path}: {last_exc}")


def normalise_text(value: Any) -> str:
    """Lowercase, remove accents/punctuation and collapse to comparable ASCII-ish text."""
    if value is None:
        return ""
    text = str(value)
    if text.lower() in {"nan", "none", "null"}:
        return ""
    text = text.translate(ESTONIAN_TRANSLATION)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def compact(value: Any) -> str:
    return normalise_text(value).replace(" ", "")


def strip_file_suffixes(name: str) -> str:
    out = compact(name)
    for suffix in ["xlsx", "xls", "csv", "json", "xml", "txt", "zip"]:
        if out.endswith(suffix) and len(out) > len(suffix) + 3:
            out = out[: -len(suffix)]
    return out


def tokens(value: Any) -> set[str]:
    return {t for t in normalise_text(value).split() if len(t) >= 3}


def parse_jsonish(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, (list, dict)):
        return value
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    try:
        return ast.literal_eval(text)
    except Exception:
        return text


def parse_organization(row: pd.Series) -> Tuple[str, str]:
    """Return (organization_name, organization_slug) from either flat or CSV-exported fields."""
    for name_col in ["organization_name", "publisher"]:
        if name_col in row and pd.notna(row.get(name_col)) and str(row.get(name_col)).strip():
            name = str(row.get(name_col)).strip()
            slug = str(row.get("organization_slug", "") or "").strip()
            return name, slug

    org = parse_jsonish(row.get("organization", None))
    if isinstance(org, dict):
        return str(org.get("name") or ""), str(org.get("slug") or "")
    if isinstance(org, list):
        # Avaandmete portal export usually has: [id, regCode, name, slug, email, contact, ...]
        name = str(org[2]) if len(org) > 2 and org[2] is not None else ""
        slug = str(org[3]) if len(org) > 3 and org[3] is not None else ""
        return name, slug
    return "", ""


def quote_identifier(identifier: str) -> str:
    """Always quote table/schema/column identifiers for safety in Trino."""
    return '"' + str(identifier).replace('"', '""') + '"'


def table_full_name(schema: str, table: str, catalog: str = "landing") -> str:
    return f"{quote_identifier(catalog)}.{quote_identifier(schema)}.{quote_identifier(table)}"


# -----------------------------
# Trino helpers
# -----------------------------


def connect_trino():
    try:
        from trino.auth import BasicAuthentication
        from trino.dbapi import connect as trino_connect
    except Exception as exc:
        raise RuntimeError("The 'trino' package is required. Install with: pip install trino") from exc

    host = os.getenv("TRINO_HOST", "trino.avaandmeait.ee")
    port = int(os.getenv("TRINO_PORT", "443"))
    user = os.getenv("TRINO_USER", "").strip()
    password = os.getenv("TRINO_PASSWORD", "")

    if not user:
        raise RuntimeError("TRINO_USER environment variable is missing.")

    kwargs: Dict[str, Any] = {
        "host": host,
        "port": port,
        "user": user,
        "catalog": "landing",
        "http_scheme": "https",
    }
    if password:
        kwargs["auth"] = BasicAuthentication(user, password)
    return trino_connect(**kwargs)


def trino_query_to_df(conn, sql: str) -> pd.DataFrame:
    cur = conn.cursor()
    cur.execute(sql)
    rows = cur.fetchall()
    cols = [desc[0] for desc in cur.description] if cur.description else []
    safe_rows = [[make_json_safe(v) for v in row] for row in rows]
    return pd.DataFrame(safe_rows, columns=cols)


def load_tables(conn=None, csv_path: Optional[str] = None) -> pd.DataFrame:
    if csv_path:
        df = read_csv_auto(csv_path)
    else:
        if conn is None:
            raise ValueError("conn is required when tables CSV is not provided")
        df = trino_query_to_df(
            conn,
            """
            SELECT table_schema, table_name, table_type
            FROM landing.information_schema.tables
            WHERE table_schema <> 'information_schema'
            ORDER BY table_schema, table_name
            """,
        )

    # Drop index columns from Trino UI exports.
    df = df[[c for c in df.columns if not str(c).startswith("Unnamed")]].copy()
    required = {"table_schema", "table_name"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Tables file is missing columns: {sorted(missing)}")
    if "table_type" not in df.columns:
        df["table_type"] = "BASE TABLE"
    df = df[df["table_schema"].astype(str).ne("information_schema")]
    df = df[df["table_schema"].astype(str).ne("avaandmete_portaal")]
    df = df[df["table_type"].astype(str).str.upper().eq("BASE TABLE")]
    return df.reset_index(drop=True)


def load_metadata(conn=None, csv_path: Optional[str] = None) -> pd.DataFrame:
    if csv_path:
        df = read_csv_auto(csv_path)
    else:
        if conn is None:
            raise ValueError("conn is required when metadata CSV is not provided")
        # Keep this live query simple. Nested array fields are intentionally not selected here.
        df = trino_query_to_df(
            conn,
            """
            SELECT
                id,
                datasetidentifier,
                title,
                description,
                shorttitle,
                landingpage,
                CAST(issued AS varchar) AS issued,
                CAST(temporalcoverage AS varchar) AS temporalcoverage,
                CAST(releasedate AS varchar) AS releasedate,
                CAST(modificationdate AS varchar) AS modificationdate,
                titleen,
                slug,
                descriptionen,
                organization.name AS organization_name,
                organization.slug AS organization_slug,
                access,
                accrualperiodicity,
                status,
                CAST(publishedat AS varchar) AS publishedat,
                CAST(createdat AS varchar) AS createdat,
                CAST(updatedat AS varchar) AS updatedat
            FROM landing.avaandmete_portaal.dataset_metadata
            WHERE COALESCE(tobedeletedat IS NULL, true)
            """,
        )

    df = df[[c for c in df.columns if not str(c).startswith("Unnamed")]].copy()
    for col in ["title", "description", "slug"]:
        if col not in df.columns:
            df[col] = ""
    return df.reset_index(drop=True)


def get_column_names(conn, schema: str, table: str) -> List[str]:
    sql = f"SHOW COLUMNS FROM {table_full_name(schema, table)}"
    df = trino_query_to_df(conn, sql)
    if df.empty:
        return []
    first_col = df.columns[0]
    return [str(v) for v in df[first_col].dropna().tolist()]


def build_data_query(conn, schema: str, table: str, sample_rows: int, max_columns: int) -> str:
    cols: List[str] = []
    try:
        cols = get_column_names(conn, schema, table)
    except Exception:
        cols = []

    if cols:
        selected_cols = cols[:max_columns] if max_columns and max_columns > 0 else cols
        select_expr = ",\n    ".join(quote_identifier(c) for c in selected_cols)
    else:
        select_expr = "*"

    base_query = f"SELECT\n    {select_expr}\nFROM {table_full_name(schema, table)}"
    if sample_rows and int(sample_rows) > 0:
        return base_query + f"\nLIMIT {int(sample_rows)}"
    return base_query


# -----------------------------
# Metadata matching
# -----------------------------


def metadata_search_text(row: pd.Series) -> str:
    org_name, org_slug = parse_organization(row)
    fields = [
        row.get("title", ""),
        row.get("slug", ""),
        row.get("description", ""),
        row.get("shorttitle", ""),
        row.get("landingpage", ""),
        row.get("titleen", ""),
        row.get("descriptionen", ""),
        row.get("distributions", ""),
        row.get("tables", ""),
        row.get("datasetfiles", ""),
        org_name,
        org_slug,
    ]
    return " ".join(str(x) for x in fields if pd.notna(x))


def score_metadata_match(schema: str, table: str, meta_row: pd.Series) -> Tuple[float, str]:
    table_c = compact(table)
    table_stem = strip_file_suffixes(table)
    schema_c = compact(schema)
    org_name, org_slug = parse_organization(meta_row)

    title_slug = compact(str(meta_row.get("title", "")) + " " + str(meta_row.get("slug", "")))
    dist_tables = compact(str(meta_row.get("distributions", "")) + " " + str(meta_row.get("tables", "")) + " " + str(meta_row.get("datasetfiles", "")))
    all_text = compact(metadata_search_text(meta_row))
    org_c = compact(org_name + " " + org_slug)

    score = 0.0
    reasons: List[str] = []

    if table_c and table_c in dist_tables:
        score += 100
        reasons.append("table name found in distributions/tables")
    elif table_stem and table_stem in dist_tables:
        score += 95
        reasons.append("table stem found in distributions/tables")

    if table_c and table_c in title_slug:
        score += 85
        reasons.append("table name found in title/slug")
    elif table_stem and table_stem in title_slug:
        score += 80
        reasons.append("table stem found in title/slug")

    if table_c and table_c in all_text:
        score += 45
        reasons.append("table name found in metadata text")
    elif table_stem and table_stem in all_text:
        score += 40
        reasons.append("table stem found in metadata text")

    if schema_c and (schema_c in org_c or schema_c in all_text):
        score += 30
        reasons.append("schema matched organization/text")

    table_tokens = tokens(table)
    meta_tokens = tokens(metadata_search_text(meta_row))
    if table_tokens:
        overlap = table_tokens & meta_tokens
        if overlap:
            token_score = min(25.0, 5.0 * len(overlap))
            score += token_score
            reasons.append(f"token overlap: {', '.join(sorted(list(overlap))[:5])}")

    # Similarity fallback helps long xlsx-style table names and long metadata slugs/titles.
    title_slug_text = compact(str(meta_row.get("title", "")) + " " + str(meta_row.get("slug", "")))
    if table_stem and title_slug_text:
        ratio = SequenceMatcher(None, table_stem, title_slug_text).ratio()
        if ratio >= 0.35:
            score += ratio * 30
            reasons.append(f"fuzzy title/slug ratio={ratio:.2f}")

    return score, "; ".join(reasons)


def build_mapping(tables_df: pd.DataFrame, metadata_df: pd.DataFrame, min_score: float) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for _, tbl in tables_df.iterrows():
        schema = str(tbl["table_schema"])
        table = str(tbl["table_name"])

        best_idx = None
        best_score = -1.0
        best_reason = ""

        for idx, meta_row in metadata_df.iterrows():
            score, reason = score_metadata_match(schema, table, meta_row)
            if score > best_score:
                best_score = score
                best_idx = idx
                best_reason = reason

        if best_idx is not None and best_score >= min_score:
            meta = metadata_df.loc[best_idx]
            org_name, org_slug = parse_organization(meta)
            rows.append(
                {
                    "table_schema": schema,
                    "table_name": table,
                    "table_type": tbl.get("table_type", ""),
                    "metadata_match_status": "matched",
                    "metadata_score": round(float(best_score), 2),
                    "metadata_reason": best_reason,
                    "metadata_index": int(best_idx),
                    "metadata_id": meta.get("id", ""),
                    "metadata_title": meta.get("title", ""),
                    "metadata_slug": meta.get("slug", ""),
                    "metadata_publisher": org_name,
                    "metadata_org_slug": org_slug,
                    "metadata_description": meta.get("description", ""),
                }
            )
        else:
            rows.append(
                {
                    "table_schema": schema,
                    "table_name": table,
                    "table_type": tbl.get("table_type", ""),
                    "metadata_match_status": "no_match",
                    "metadata_score": round(float(best_score), 2) if best_score >= 0 else None,
                    "metadata_reason": best_reason,
                    "metadata_index": None,
                    "metadata_id": "",
                    "metadata_title": "",
                    "metadata_slug": "",
                    "metadata_publisher": "",
                    "metadata_org_slug": "",
                    "metadata_description": "",
                }
            )

    return pd.DataFrame(rows)


def metadata_row_for_assessment(metadata_df: pd.DataFrame, metadata_index: Any) -> Dict[str, Any]:
    if metadata_index is None or metadata_index == "" or pd.isna(metadata_index):
        return {}
    row = metadata_df.loc[int(metadata_index)]
    org_name, org_slug = parse_organization(row)
    raw = {
        "id": row.get("id", ""),
        "datasetidentifier": row.get("datasetidentifier", ""),
        "title": row.get("title", ""),
        "description": row.get("description", ""),
        "shorttitle": row.get("shorttitle", ""),
        "landingpage": row.get("landingpage", ""),
        "issued": row.get("issued", ""),
        "temporalcoverage": row.get("temporalcoverage", ""),
        "releasedate": row.get("releasedate", ""),
        "modificationdate": row.get("modificationdate", ""),
        "createdat": row.get("createdat", ""),
        "updatedat": row.get("updatedat", ""),
        "publishedat": row.get("publishedat", ""),
        "slug": row.get("slug", ""),
        "publisher": org_name,
        "organization_name": org_name,
        "organization_slug": org_slug,
        "access": row.get("access", ""),
        "accrualperiodicity": row.get("accrualperiodicity", ""),
        "status": row.get("status", ""),
        "languages": row.get("languages", ""),
        "keywords": row.get("keywords", ""),
        # Keep these as raw text from CSV exports when available; app pipeline can use them as context.
        "distributions": row.get("distributions", ""),
        "tables": row.get("tables", ""),
        "datasetfiles": row.get("datasetfiles", ""),
    }
    return {k: make_json_safe(v) for k, v in raw.items() if v is not None and str(v).lower() != "nan"}


# -----------------------------
# Assessment and output
# -----------------------------


def safe_dataset_id(schema: str, table: str) -> str:
    base = f"{schema}__{table}"
    base = normalise_text(base).replace(" ", "_")
    return re.sub(r"_+", "_", base).strip("_")[:180]


def compute_dimension_scores(metrics_df: pd.DataFrame) -> pd.DataFrame:
    if metrics_df is None or metrics_df.empty:
        return pd.DataFrame(columns=["dimension", "value_clamped"])
    df = metrics_df.copy()
    value_col = "value_clamped" if "value_clamped" in df.columns else "value"
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    out = (
        df.dropna(subset=[value_col])
        .groupby("dimension", as_index=False)[value_col]
        .mean()
        .rename(columns={value_col: "value_clamped"})
    )
    return out


def run_one_dataset(
    conn,
    mapping_row: pd.Series,
    metadata_df: pd.DataFrame,
    args: argparse.Namespace,
) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    schema = str(mapping_row["table_schema"])
    table = str(mapping_row["table_name"])
    ds_id = safe_dataset_id(schema, table)
    started = datetime.now().isoformat(timespec="seconds")
    t0 = time.time()

    summary: Dict[str, Any] = {
        "dataset_id": ds_id,
        "table_schema": schema,
        "table_name": table,
        "metadata_match_status": mapping_row.get("metadata_match_status", ""),
        "metadata_score": mapping_row.get("metadata_score", None),
        "metadata_title": mapping_row.get("metadata_title", ""),
        "metadata_slug": mapping_row.get("metadata_slug", ""),
        "metadata_publisher": mapping_row.get("metadata_publisher", ""),
        "status": "started",
        "started_at": started,
        "ended_at": "",
        "duration_s": None,
        "rows_used": None,
        "columns_used": None,
        "overall_score": None,
        "error": "",
    }

    if mapping_row.get("metadata_match_status") != "matched" and not args.run_unmatched:
        summary.update(
            {
                "status": "skipped_no_metadata",
                "ended_at": datetime.now().isoformat(timespec="seconds"),
                "duration_s": round(time.time() - t0, 3),
            }
        )
        return summary, pd.DataFrame(), pd.DataFrame(), {}

    data_query = build_data_query(conn, schema, table, args.sample_rows, args.max_columns)
    df = trino_query_to_df(conn, data_query)

    # Avoid object columns containing lists/dicts causing downstream hash issues.
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].apply(lambda x: json.dumps(x, ensure_ascii=False, default=str) if isinstance(x, (dict, list, tuple)) else x)

    trino_metadata_raw = metadata_row_for_assessment(metadata_df, mapping_row.get("metadata_index"))
    trino_metadata = normalize_metadata_to_symbols(trino_metadata_raw) if trino_metadata_raw else {}

    metrics_df, details = run_quality_assessment(
        df=df,
        formulas_yaml_path=args.formulas_yaml,
        prompts_yaml_path=args.prompts_yaml,
        prompt_regime=args.prompt_regime,
        use_llm=bool(args.use_llm),
        llm_provider=args.llm_provider,
        llm_model_name=args.llm_model,
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        file_ext=".sql",
        manual_metadata={},
        manual_metadata_text="",
        trino_metadata=trino_metadata,
        trino_metadata_raw=trino_metadata_raw,
    )

    dimension_scores = compute_dimension_scores(metrics_df)
    overall_score = None
    if not dimension_scores.empty:
        overall_score = float(pd.to_numeric(dimension_scores["value_clamped"], errors="coerce").mean())

    metrics_out = metrics_df.copy()
    metrics_out.insert(0, "dataset_id", ds_id)
    metrics_out.insert(1, "table_schema", schema)
    metrics_out.insert(2, "table_name", table)
    metrics_out.insert(3, "metadata_title", mapping_row.get("metadata_title", ""))

    dimensions_out = dimension_scores.copy()
    if not dimensions_out.empty:
        dimensions_out.insert(0, "dataset_id", ds_id)
        dimensions_out.insert(1, "table_schema", schema)
        dimensions_out.insert(2, "table_name", table)
        dimensions_out.insert(3, "metadata_title", mapping_row.get("metadata_title", ""))

    summary.update(
        {
            "status": "ok",
            "ended_at": datetime.now().isoformat(timespec="seconds"),
            "duration_s": round(time.time() - t0, 3),
            "rows_used": int(len(df)),
            "columns_used": int(len(df.columns)),
            "overall_score": overall_score,
            "data_query": data_query,
        }
    )

    return summary, metrics_out, dimensions_out, details


def write_outputs(
    output_dir: Path,
    summary_rows: List[Dict[str, Any]],
    metrics_rows: List[pd.DataFrame],
    dimension_rows: List[pd.DataFrame],
    mapping_df: pd.DataFrame,
    excel_name: str = "trino_batch_assessment_results.xlsx",
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_df = pd.DataFrame(summary_rows)
    metrics_df = pd.concat(metrics_rows, ignore_index=True) if metrics_rows else pd.DataFrame()
    dimensions_df = pd.concat(dimension_rows, ignore_index=True) if dimension_rows else pd.DataFrame()

    # Excel cannot store nested objects; make everything string-safe where needed.
    def clean_df(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()
        out = df.copy()
        for col in out.columns:
            out[col] = out[col].apply(to_excel_safe)
        return out

    summary_clean = clean_df(summary_df)
    metrics_clean = clean_df(metrics_df)
    dimensions_clean = clean_df(dimensions_df)
    mapping_clean = clean_df(mapping_df)

    summary_clean.to_csv(output_dir / "summary_checkpoint.csv", index=False, encoding="utf-8-sig")
    metrics_clean.to_csv(output_dir / "metrics_checkpoint.csv", index=False, encoding="utf-8-sig")
    dimensions_clean.to_csv(output_dir / "dimension_scores_checkpoint.csv", index=False, encoding="utf-8-sig")
    mapping_clean.to_csv(output_dir / "mapping_checkpoint.csv", index=False, encoding="utf-8-sig")

    excel_path = output_dir / excel_name
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        summary_clean.to_excel(writer, sheet_name="Summary", index=False)
        dimensions_clean.to_excel(writer, sheet_name="Dimension scores", index=False)
        metrics_clean.to_excel(writer, sheet_name="Metrics", index=False)
        mapping_clean.to_excel(writer, sheet_name="Mapping", index=False)

        # Basic formatting for readability.
        wb = writer.book
        for ws in wb.worksheets:
            ws.freeze_panes = "A2"
            ws.auto_filter.ref = ws.dimensions
            for col_cells in ws.columns:
                header = str(col_cells[0].value or "")
                max_len = min(max([len(str(c.value or "")) for c in col_cells[: min(len(col_cells), 100)]] + [len(header)]), 45)
                ws.column_dimensions[col_cells[0].column_letter].width = max(10, max_len + 2)

    return excel_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run batch data quality assessment for Trino datasets.")
    parser.add_argument("--tables-csv", default="", help="CSV export of landing.information_schema.tables.")
    parser.add_argument("--metadata-csv", default="", help="CSV export of landing.avaandmete_portaal.dataset_metadata.")
    parser.add_argument("--mapping-csv", default="", help="Optional existing mapping CSV, e.g. outputs/trino_batch_assessment/mapping_preview.csv. Skips slow remapping.")
    parser.add_argument("--reuse-mapping", action="store_true", help="Reuse mapping_preview.csv from output-dir if it exists.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-rows", type=int, default=0, help="Rows read from each table. 0 means read the whole table; this can be very slow/heavy.")
    parser.add_argument("--max-columns", type=int, default=0, help="Maximum columns selected from each table. 0 means all columns.")
    parser.add_argument("--max-datasets", type=int, default=0, help="0 means all datasets; use small values for testing.")
    parser.add_argument("--min-match-score", type=float, default=45.0)
    parser.add_argument("--mapping-only", action="store_true", help="Only create/check mapping; do not run assessments.")
    parser.add_argument("--run-unmatched", action="store_true", help="Run assessment even if metadata mapping was not found.")
    parser.add_argument("--use-llm", action="store_true", help="Use LLM-backed inference. Can be slow/costly for all datasets.")
    parser.add_argument("--llm-provider", default="openai", choices=["openai", "huggingface"])
    parser.add_argument("--llm-model", default="gpt-4.1-mini")
    parser.add_argument("--prompt-regime", default="reasoning")
    parser.add_argument("--formulas-yaml", default=FORMULAS_YAML)
    parser.add_argument("--prompts-yaml", default=PROMPTS_YAML)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    details_dir = output_dir / "details"
    details_dir.mkdir(parents=True, exist_ok=True)

    conn = None
    if not args.mapping_only or not args.tables_csv or not args.metadata_csv:
        conn = connect_trino()

    tables_df = load_tables(conn=conn, csv_path=args.tables_csv or None)
    metadata_df = load_metadata(conn=conn, csv_path=args.metadata_csv or None)

    print(f"Loaded {len(tables_df)} physical tables.")
    print(f"Loaded {len(metadata_df)} metadata rows.")

    output_dir.mkdir(parents=True, exist_ok=True)
    mapping_path = output_dir / "mapping_preview.csv"

    existing_mapping_path = Path(args.mapping_csv) if args.mapping_csv else mapping_path
    if args.mapping_csv or (args.reuse_mapping and existing_mapping_path.exists()):
        if not existing_mapping_path.exists():
            raise FileNotFoundError(f"Mapping CSV not found: {existing_mapping_path}")
        mapping_df = read_csv_auto(existing_mapping_path)
        print(f"Loaded existing mapping: {existing_mapping_path}")
    else:
        print("Building table-to-metadata mapping. This may take a few minutes...")
        mapping_df = build_mapping(tables_df, metadata_df, min_score=args.min_match_score)
        mapping_df.to_csv(mapping_path, index=False, encoding="utf-8-sig")
        print(f"Saved mapping preview: {mapping_path}")

    print(mapping_df["metadata_match_status"].value_counts(dropna=False).to_string())

    if args.mapping_only:
        write_outputs(output_dir, [], [], [], mapping_df)
        print("Mapping-only mode finished.")
        return

    if conn is None:
        conn = connect_trino()

    run_df = mapping_df.copy()
    if args.max_datasets and args.max_datasets > 0:
        run_df = run_df.head(args.max_datasets)

    summary_rows: List[Dict[str, Any]] = []
    metrics_rows: List[pd.DataFrame] = []
    dimension_rows: List[pd.DataFrame] = []

    for i, row in run_df.iterrows():
        schema = row["table_schema"]
        table = row["table_name"]
        print(f"\n[{len(summary_rows) + 1}/{len(run_df)}] {schema}.{table}")

        try:
            summary, metrics, dimensions, details = run_one_dataset(conn, row, metadata_df, args)
            summary_rows.append(summary)
            if metrics is not None and not metrics.empty:
                metrics_rows.append(metrics)
            if dimensions is not None and not dimensions.empty:
                dimension_rows.append(dimensions)

            if details:
                details_path = details_dir / f"{summary['dataset_id']}.json"
                with open(details_path, "w", encoding="utf-8") as f:
                    json.dump(make_json_safe(details), f, ensure_ascii=False, indent=2, default=str)

            print(f"  status={summary['status']} rows={summary.get('rows_used')} cols={summary.get('columns_used')} overall={summary.get('overall_score')}")

        except Exception as exc:
            err_summary = {
                "dataset_id": safe_dataset_id(str(schema), str(table)),
                "table_schema": schema,
                "table_name": table,
                "metadata_match_status": row.get("metadata_match_status", ""),
                "metadata_score": row.get("metadata_score", None),
                "metadata_title": row.get("metadata_title", ""),
                "metadata_slug": row.get("metadata_slug", ""),
                "metadata_publisher": row.get("metadata_publisher", ""),
                "status": "error",
                "started_at": "",
                "ended_at": datetime.now().isoformat(timespec="seconds"),
                "duration_s": None,
                "rows_used": None,
                "columns_used": None,
                "overall_score": None,
                "error": repr(exc),
            }
            summary_rows.append(err_summary)
            print(f"  ERROR: {exc!r}")

        # Save CSV checkpoints after every dataset so progress is not lost.
        write_outputs(output_dir, summary_rows, metrics_rows, dimension_rows, mapping_df)

    excel_path = write_outputs(output_dir, summary_rows, metrics_rows, dimension_rows, mapping_df)
    print(f"\nDone. Excel results saved to: {excel_path}")


if __name__ == "__main__":
    main()
