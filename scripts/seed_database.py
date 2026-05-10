"""
Seed the database with companies and problems from the dataset.

Prefers datasets/dataset_enriched.jsonl (with problem descriptions).
Falls back to datasets/dataset.csv if the enriched file doesn't exist.

Usage:
    python scripts/seed_database.py
"""

import csv
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
APP_DIR = PROJECT_ROOT / "application"

sys.path.insert(0, str(APP_DIR))

from database import engine, get_db, Base
from models import Company, InterviewProblem

ENRICHED_PATH = PROJECT_ROOT / "datasets" / "dataset_enriched.jsonl"
CSV_PATH = PROJECT_ROOT / "datasets" / "dataset.csv"


def _load_rows():
    """Load rows from JSONL (preferred) or CSV (fallback)."""
    if ENRICHED_PATH.exists():
        print(f"Reading enriched dataset: {ENRICHED_PATH}")
        rows = []
        with open(ENRICHED_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    if CSV_PATH.exists():
        print(f"Enriched dataset not found. Falling back to: {CSV_PATH}")
        with open(CSV_PATH, "r", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    print(f"No dataset found. Expected one of:")
    print(f"  {ENRICHED_PATH}")
    print(f"  {CSV_PATH}")
    return None


def seed_db():
    Base.metadata.create_all(bind=engine)
    db = next(get_db())

    try:
        existing_count = db.query(InterviewProblem).count()
        if existing_count > 0:
            print(f"Database already seeded ({existing_count} problems). Skipping.")
            return

        rows = _load_rows()
        if rows is None:
            return

        company_cache: dict[str, Company] = {}
        problems_added = 0

        for row in rows:
            company_name = row["company"].strip()

            if company_name not in company_cache:
                company = db.query(Company).filter(Company.name == company_name).first()
                if not company:
                    company = Company(name=company_name)
                    db.add(company)
                    db.flush()
                company_cache[company_name] = company

            problem = InterviewProblem(
                title=row["title"].strip(),
                difficulty=row["difficulty"].strip(),
                description=row.get("body", "").strip() or None,
                url=row["link"].strip(),
                topics=row.get("topics", "").strip() or None,
                acceptance_rate=float(row["acceptance_rate"]) if row.get("acceptance_rate") else None,
                frequency=float(row["frequency"]) if row.get("frequency") else None,
                preparation_days=int(row["preparation_days"]) if row.get("preparation_days") else None,
                company_id=company_cache[company_name].id,
            )
            db.add(problem)
            problems_added += 1

        db.commit()
        companies_added = len(company_cache)
        print(f"Seeded {companies_added} companies and {problems_added} problems.")
    except Exception as e:
        db.rollback()
        print(f"Error seeding database: {e}")
        raise
    finally:
        db.close()


if __name__ == "__main__":
    seed_db()
