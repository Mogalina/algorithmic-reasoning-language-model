"""
Problem recommender -- determines which problems a user should solve.

Uses cluster-based PCHIP interpolation to predict the difficulty distribution,
then samples problems from the database accordingly.
"""

import random
from sqlalchemy.orm import Session

from rag.curve_fitting import get_predictor


def recommend_problems(db: Session, company_id: int, days_until: int) -> list:
    """
    Return a list of InterviewProblem rows for a given company and timeline.

    Uses the curve_fitting predictor to determine how many easy/medium/hard
    problems to recommend, then samples from the database.
    """
    from models import Company, InterviewProblem

    company = db.query(Company).filter(Company.id == company_id).first()
    if not company:
        return []

    predictor = get_predictor()
    counts = predictor.predict_counts(company.name, days_until)

    all_problems = (
        db.query(InterviewProblem)
        .filter(InterviewProblem.company_id == company_id)
        .all()
    )

    if not all_problems:
        return []

    by_difficulty = {"EASY": [], "MEDIUM": [], "HARD": []}
    for p in all_problems:
        key = p.difficulty.upper()
        if key in by_difficulty:
            by_difficulty[key].append(p)

    selected = []
    for difficulty, count in [("EASY", counts["easy"]), ("MEDIUM", counts["medium"]), ("HARD", counts["hard"])]:
        pool = by_difficulty[difficulty]
        n = min(count, len(pool))
        if n > 0:
            selected.extend(random.sample(pool, n))

    # If we got fewer than requested (pool too small), fill from remaining
    if len(selected) < counts["total"]:
        used_ids = {p.id for p in selected}
        remaining = [p for p in all_problems if p.id not in used_ids]
        fill = min(counts["total"] - len(selected), len(remaining))
        if fill > 0:
            selected.extend(random.sample(remaining, fill))

    return selected
