"""
Problem recommender — determines which problems a user should solve.

Currently a mock implementation that picks random problems from the database.
Will be replaced with PCHIP interpolation + RAG-based retrieval.
"""

import random
from sqlalchemy.orm import Session


def recommend_problems(db: Session, company_id: int, days_until: int, total: int = 10) -> list:
    """
    Return a list of InterviewProblem rows for a given company and timeline.

    Currently picks `total` random problems associated with the company.
    The real implementation will use interpolation on the aggregated dataset
    to determine the difficulty distribution, then use RAG to select the
    best-matching problems.
    """
    from models import InterviewProblem

    problems = (
        db.query(InterviewProblem)
        .filter(InterviewProblem.company_id == company_id)
        .all()
    )

    if not problems:
        return []

    count = min(total, len(problems))
    return random.sample(problems, count)
