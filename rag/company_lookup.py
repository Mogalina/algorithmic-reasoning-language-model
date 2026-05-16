import logging
import math
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PARQUET_PATH = PROJECT_ROOT / "datasets" / "companies.parquet"

SIZE_RANGE_ORDER = {
    "1-10": 0,
    "11-50": 1,
    "51-200": 2,
    "201-500": 3,
    "501-1000": 4,
    "1001-5000": 5,
    "5001-10000": 6,
    "10001+": 7,
}


def lookup_company(name: str) -> dict | None:
    """
    Search the Kaggle Parquet for a company by name.

    Returns a dict with keys: industry, size_range, employee_count,
    year_founded, locality.  Returns None if no match or Parquet missing.
    """
    if not PARQUET_PATH.exists():
        logger.warning("companies.parquet not found at %s", PARQUET_PATH)
        return None

    try:
        import polars as pl
    except ImportError:
        logger.warning("polars not installed — cannot look up unknown companies")
        return None

    query = name.strip().lower()
    if not query:
        return None

    # Lazy scan — only materializes matching rows
    lf = pl.scan_parquet(PARQUET_PATH)

    matches = (
        lf.filter(
            pl.col("name").str.to_lowercase().str.contains(query, literal=True)
        )
        .head(50)
        .collect()
    )

    if matches.is_empty():
        logger.info("No Kaggle match for '%s'", name)
        return None

    # Prefer exact match (case-insensitive), fall back to shortest name
    best = None
    for row in matches.iter_rows(named=True):
        row_name = (row.get("name") or "").strip().lower()
        if row_name == query:
            best = row
            break

    if best is None:
        sorted_rows = sorted(
            matches.iter_rows(named=True),
            key=lambda r: len(r.get("name") or ""),
        )
        best = sorted_rows[0]

    return {
        "industry": best.get("industry") or "",
        "size_range": best.get("size range") or "",
        "employee_count": best.get("current employee estimate") or 0,
        "year_founded": best.get("year founded") or 0,
        "locality": best.get("locality") or "",
    }


def encode_company_features(
    meta: dict,
    feature_columns: list[str],
    existing_features=None,
) -> list[float]:
    """
    Convert a raw company metadata dict into a numeric feature vector
    aligned with the columns used during clustering.

    Volume dimensions (TIME_BUCKETS) are set to 0 for unknown companies.
    Company feature dimensions are encoded the same way as during fit.
    """
    from rag.curve_fitting import COMPANY_FEATURE_COLS

    values = []
    for col in feature_columns:
        if col in COMPANY_FEATURE_COLS:
            values.append(_encode_single(col, meta, existing_features))
        else:
            # Volume dimension — unknown company has no LeetCode data
            values.append(0.0)

    return values


def _encode_single(col: str, meta: dict, existing_features) -> float:
    """Encode a single company feature dimension."""
    if col == "industry_enc":
        # If we have the existing label encoder results, find the closest
        # industry; otherwise use the median of existing values
        industry = (meta.get("industry") or "unknown").strip().lower()
        if existing_features is not None and "industry_enc" in existing_features.columns:
            known = existing_features.reset_index()
            match = known.loc[
                known.get("industry", pd.Series(dtype=str))
                .fillna("")
                .str.strip()
                .str.lower()
                == industry,
                "industry_enc",
            ] if "industry" in known.columns else pd.Series(dtype=float)
            if hasattr(match, "__len__") and len(match) > 0:
                return float(match.iloc[0])
            return float(existing_features["industry_enc"].median())
        return 0.0

    if col == "size_rank":
        size = meta.get("size_range", "")
        return float(SIZE_RANGE_ORDER.get(size, len(SIZE_RANGE_ORDER) // 2))

    if col == "log_employees":
        emp = meta.get("employee_count", 1)
        try:
            emp = max(1, float(emp))
        except (ValueError, TypeError):
            emp = 1
        return math.log1p(emp)

    if col == "year_founded":
        yr = meta.get("year_founded", 2000)
        try:
            return float(yr) if yr else 2000.0
        except (ValueError, TypeError):
            return 2000.0

    return 0.0
