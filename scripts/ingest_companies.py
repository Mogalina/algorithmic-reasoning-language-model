import re
import logging
from pathlib import Path

import pandas as pd
import polars as pl

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
KAGGLE_CSV = PROJECT_ROOT / "datasets" / "companies_sorted.csv"
PARQUET_OUT = PROJECT_ROOT / "datasets" / "companies.parquet"
META_OUT = PROJECT_ROOT / "datasets" / "company_meta.csv"
AGGREGATED_CSV = PROJECT_ROOT / "datasets" / "dataset_aggregated.csv"

KEEP_COLUMNS = [
    "name",
    "domain",
    "year founded",
    "industry",
    "size range",
    "locality",
    "current employee estimate",
]

STRIP_SUFFIXES = re.compile(
    r"\s*\b(inc\.?|llc\.?|ltd\.?|corp\.?|corporation|company|co\.?"
    r"|group|plc|gmbh|s\.?a\.?|n\.?v\.?|technologies|technology|software"
    r"|systems|solutions|services|consulting|labs?|,)\s*$",
    re.IGNORECASE,
)


def _normalize_name(name: str) -> str:
    """Lowercase, strip corporate suffixes, collapse whitespace."""
    name = name.lower().strip()
    for _ in range(3):
        name = STRIP_SUFFIXES.sub("", name).strip()
    name = re.sub(r"\s+", " ", name)
    return name


def convert_to_parquet():
    """Read the Kaggle CSV and write a trimmed Parquet file."""
    if not KAGGLE_CSV.exists():
        raise FileNotFoundError(
            f"Kaggle CSV not found at {KAGGLE_CSV}.\n"
            "Download it from: https://www.kaggle.com/datasets/"
            "peopledatalabssf/free-7-million-company-dataset\n"
            "and place it at datasets/companies_sorted.csv"
        )

    logger.info("Reading Kaggle CSV (this may take a moment)...")
    lf = pl.scan_csv(KAGGLE_CSV, infer_schema_length=10000)

    available_cols = lf.collect_schema().names()
    cols_to_keep = [c for c in KEEP_COLUMNS if c in available_cols]
    if not cols_to_keep:
        raise ValueError(
            f"None of the expected columns found. Available: {available_cols}"
        )

    missing = set(KEEP_COLUMNS) - set(cols_to_keep)
    if missing:
        logger.warning("Columns not found in CSV (will be skipped): %s", missing)

    lf = lf.select(cols_to_keep)
    lf = lf.filter(pl.col("name").is_not_null())

    logger.info("Writing Parquet to %s ...", PARQUET_OUT)
    lf.collect().write_parquet(PARQUET_OUT)

    size_mb = PARQUET_OUT.stat().st_size / (1024 * 1024)
    logger.info("Parquet written: %.1f MB", size_mb)


def _build_kaggle_lookup() -> dict[str, dict]:
    """Build a normalized-name -> row dict from the Parquet file."""
    df = pl.read_parquet(PARQUET_OUT)

    lookup: dict[str, dict] = {}
    for row in df.iter_rows(named=True):
        raw_name = row.get("name")
        if not raw_name or not isinstance(raw_name, str):
            continue
        norm = _normalize_name(raw_name)
        if norm and norm not in lookup:
            lookup[norm] = row

    logger.info("Kaggle lookup built: %d unique normalized names", len(lookup))
    return lookup


def _match_company(lc_name: str, kaggle_lookup: dict[str, dict]) -> dict | None:
    """Try to find a Kaggle row matching a LeetCode company name."""
    norm = _normalize_name(lc_name)

    # exact match
    if norm in kaggle_lookup:
        return kaggle_lookup[norm]

    # check if any Kaggle name starts with the LeetCode name
    for kaggle_norm, row in kaggle_lookup.items():
        if kaggle_norm.startswith(norm) and len(norm) >= 3:
            return row

    # check if the LeetCode name is a substring of a Kaggle name
    for kaggle_norm, row in kaggle_lookup.items():
        if norm in kaggle_norm and len(norm) >= 4:
            return row

    return None


def match_leetcode_companies(kaggle_lookup: dict[str, dict]):
    """Match LeetCode companies to Kaggle rows and write company_meta.csv. We use this to enrich the dataset, to help with clustering"""
    agg_df = pd.read_csv(AGGREGATED_CSV)
    lc_companies = sorted(agg_df["company"].unique())
    logger.info("LeetCode companies to match: %d", len(lc_companies))

    matched = []
    unmatched = []

    for lc_name in lc_companies:
        row = _match_company(lc_name, kaggle_lookup)
        if row is None:
            unmatched.append(lc_name)
            continue

        matched.append({
            "company": lc_name,
            "industry": row.get("industry", ""),
            "size_range": row.get("size range", ""),
            "employee_count": row.get("current employee estimate", 0),
            "year_founded": row.get("year founded", 0),
            "locality": row.get("locality", ""),
        })

    meta_df = pd.DataFrame(matched)
    meta_df.to_csv(META_OUT, index=False)
    logger.info(
        "Matched %d / %d companies. Output: %s",
        len(matched), len(lc_companies), META_OUT,
    )

    if unmatched:
        logger.warning(
            "Unmatched companies (%d): %s",
            len(unmatched), ", ".join(unmatched[:20]),
        )
        if len(unmatched) > 20:
            logger.warning("  ... and %d more", len(unmatched) - 20)


def main():
    convert_to_parquet()
    kaggle_lookup = _build_kaggle_lookup()
    match_leetcode_companies(kaggle_lookup)
    logger.info("Done. You can now delete datasets/companies_sorted.csv if desired.")


if __name__ == "__main__":
    main()
