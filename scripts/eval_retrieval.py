"""
Evaluate the RAG retrieval pipeline.

Tests:
  1. Company-hit rate: what fraction of retrieved problems belong to the
     target company vs. peer companies vs. global fallback?
  2. Difficulty accuracy: do retrieved problems match the requested difficulty?
  3. Frequency ranking: are results sorted by descending frequency?
  4. Deduplication: no duplicate URLs in the final roadmap.
  5. End-to-end roadmap: recommend_problems returns the right count mix.

Usage:
    python scripts/eval_retrieval.py

Prerequisites:
    - data/app.db seeded (python scripts/seed_database.py)
    - data/chromadb built  (python scripts/build_vector_index.py)
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
APP_DIR = ROOT / "application"

sys.path.insert(0, str(APP_DIR))
sys.path.insert(0, str(ROOT))

from database import SessionLocal
from models import Company, InterviewProblem
from rag.curve_fitting import get_predictor
from rag.retriever import get_retriever
from rag.recommender import recommend_problems

SEPARATOR = "=" * 70


def _pick_test_companies(db, n=5):
    """Pick companies that actually have problems in the DB."""
    rows = (
        db.query(Company)
        .join(InterviewProblem, InterviewProblem.company_id == Company.id)
        .distinct()
        .limit(n * 3)
        .all()
    )
    return rows[:n]


def eval_company_hit_rate(db, retriever, predictor, companies):
    """What percentage of retrieved problems belong to the target or peer companies?"""
    print(f"\n{SEPARATOR}")
    print("TEST 1: Company Hit Rate")
    print(SEPARATOR)

    total_retrieved = 0
    target_hits = 0
    peer_hits = 0
    global_hits = 0

    for company in companies:
        peer_names = predictor.get_peer_company_names(company.name)
        peer_ids_in_db = {
            c.id for c in db.query(Company).filter(Company.name.in_(peer_names)).all()
        } if peer_names else set()

        for difficulty in ["EASY", "MEDIUM", "HARD"]:
            results = retriever.retrieve_by_company(
                company_id=company.id,
                company_name=company.name,
                difficulty=difficulty,
                n_results=5,
                peer_company_ids=list(peer_ids_in_db),
            )
            metas = results.get("metadatas", [[]])[0]

            for meta in metas:
                total_retrieved += 1
                cid = meta.get("company_id")
                if cid == company.id:
                    target_hits += 1
                elif cid in peer_ids_in_db:
                    peer_hits += 1
                else:
                    global_hits += 1

    if total_retrieved == 0:
        print("  SKIP: No problems retrieved (is the vector index built?)")
        return

    print(f"  Total retrieved:     {total_retrieved}")
    print(f"  Target company:      {target_hits} ({100 * target_hits / total_retrieved:.1f}%)")
    print(f"  Peer companies:      {peer_hits} ({100 * peer_hits / total_retrieved:.1f}%)")
    print(f"  Global fallback:     {global_hits} ({100 * global_hits / total_retrieved:.1f}%)")
    cluster_rate = 100 * (target_hits + peer_hits) / total_retrieved
    print(f"  Cluster relevance:   {cluster_rate:.1f}%  (target + peer)")
    print(f"  {'PASS' if cluster_rate > 50 else 'WARN'}: "
          f"{'Majority' if cluster_rate > 50 else 'Minority'} of results from same cluster")


def eval_difficulty_accuracy(db, retriever, companies):
    """Do retrieved problems match the requested difficulty?"""
    print(f"\n{SEPARATOR}")
    print("TEST 2: Difficulty Accuracy")
    print(SEPARATOR)

    correct = 0
    total = 0

    for company in companies:
        for difficulty in ["EASY", "MEDIUM", "HARD"]:
            results = retriever.retrieve_by_company(
                company_id=company.id,
                company_name=company.name,
                difficulty=difficulty,
                n_results=5,
            )
            metas = results.get("metadatas", [[]])[0]
            for meta in metas:
                total += 1
                if meta.get("difficulty", "").upper() == difficulty:
                    correct += 1

    if total == 0:
        print("  SKIP: No problems retrieved")
        return

    accuracy = 100 * correct / total
    print(f"  Total checked:       {total}")
    print(f"  Correct difficulty:  {correct} ({accuracy:.1f}%)")
    print(f"  {'PASS' if accuracy == 100 else 'WARN'}: Difficulty filter "
          f"{'perfectly enforced' if accuracy == 100 else 'has mismatches'}")


def eval_frequency_ranking(db, retriever, companies):
    """Are results sorted by descending frequency?"""
    print(f"\n{SEPARATOR}")
    print("TEST 3: Frequency Ranking")
    print(SEPARATOR)

    sorted_count = 0
    total_queries = 0

    for company in companies:
        for difficulty in ["EASY", "MEDIUM", "HARD"]:
            results = retriever.retrieve_by_company(
                company_id=company.id,
                company_name=company.name,
                difficulty=difficulty,
                n_results=5,
            )
            metas = results.get("metadatas", [[]])[0]
            freqs = [m.get("frequency", 0) for m in metas]
            total_queries += 1
            if freqs == sorted(freqs, reverse=True):
                sorted_count += 1
            else:
                print(f"  NOT SORTED: {company.name} / {difficulty} -> frequencies = {freqs}")

    if total_queries == 0:
        print("  SKIP: No queries executed")
        return

    pct = 100 * sorted_count / total_queries
    print(f"  Total queries:       {total_queries}")
    print(f"  Correctly sorted:    {sorted_count} ({pct:.1f}%)")
    print(f"  {'PASS' if pct == 100 else 'WARN'}: "
          f"{'All' if pct == 100 else 'Not all'} results sorted by frequency")


def eval_deduplication(db, companies):
    """recommend_problems should not return duplicate URLs."""
    print(f"\n{SEPARATOR}")
    print("TEST 4: Deduplication")
    print(SEPARATOR)

    all_pass = True
    for company in companies:
        problems = recommend_problems(db, company.id, days_until=90)
        urls = [p.url for p in problems if p.url]
        unique = set(u.lower() for u in urls)
        has_dupes = len(urls) != len(unique)
        status = "FAIL" if has_dupes else "PASS"
        if has_dupes:
            all_pass = False
        print(f"  {company.name:20s}  {len(problems)} problems, {len(urls)} URLs, "
              f"{len(unique)} unique  [{status}]")

    if all_pass:
        print("  PASS: No duplicate URLs found in any roadmap")


def eval_roadmap_counts(db, companies):
    """recommend_problems should respect the predicted difficulty mix."""
    print(f"\n{SEPARATOR}")
    print("TEST 5: End-to-End Roadmap Count Accuracy")
    print(SEPARATOR)

    predictor = get_predictor()

    for company in companies:
        for days in [30, 90, 180]:
            expected = predictor.predict_counts(company.name, days)
            problems = recommend_problems(db, company.id, days_until=days)

            actual_easy = sum(1 for p in problems if (p.difficulty or "").upper() == "EASY")
            actual_med = sum(1 for p in problems if (p.difficulty or "").upper() == "MEDIUM")
            actual_hard = sum(1 for p in problems if (p.difficulty or "").upper() == "HARD")
            actual_total = len(problems)

            match = (actual_total == expected["total"])
            status = "PASS" if match else "WARN"
            print(f"  {company.name:15s} @ {days:3d}d  "
                  f"expected T={expected['total']} (E={expected['easy']} M={expected['medium']} H={expected['hard']})  "
                  f"got T={actual_total} (E={actual_easy} M={actual_med} H={actual_hard})  [{status}]")


def main():
    print(SEPARATOR)
    print("EVALUATION: RAG Retrieval Pipeline")
    print(SEPARATOR)

    db = SessionLocal()
    try:
        retriever = get_retriever()
        predictor = get_predictor()
        companies = _pick_test_companies(db)

        if not companies:
            print("  ERROR: No companies with problems found. Run seed_database.py first.")
            return

        print(f"  Test companies: {[c.name for c in companies]}")

        eval_company_hit_rate(db, retriever, predictor, companies)
        eval_difficulty_accuracy(db, retriever, companies)
        eval_frequency_ranking(db, retriever, companies)
        eval_deduplication(db, companies)
        eval_roadmap_counts(db, companies)

        print(f"\n{SEPARATOR}")
        print("RETRIEVAL EVALUATION COMPLETE")
        print(SEPARATOR)
    finally:
        db.close()


if __name__ == "__main__":
    main()
