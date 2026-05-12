from sqlalchemy.orm import Session

from rag.curve_fitting import get_predictor
from rag.retriever import get_retriever


def _resolve_peer_company_ids(db: Session, company_name: str) -> list[int]:
    """
    Get database IDs for companies in the same cluster as *company_name*.
    Returns an empty list when no peers are found.
    """
    from models import Company

    predictor = get_predictor()
    peer_names = predictor.get_peer_company_names(company_name)
    if not peer_names:
        return []

    peers = (
        db.query(Company.id)
        .filter(Company.name.in_(peer_names))
        .all()
    )
    return [p.id for p in peers]


def recommend_problems(db: Session, company_id: int, days_until: int) -> list:
    """
    Return a list of InterviewProblem rows for a given company and timeline.

    Uses the curve_fitting predictor to determine the difficulty mix,
    then uses ChromaDB vector retrieval to find the most relevant problems.
    Falls back to peer-company problems when the target company's pool is
    too small.
    """
    from models import Company, InterviewProblem

    company = db.query(Company).filter(Company.id == company_id).first()
    if not company:
        return []

    # predict how many of each difficulty we need
    predictor = get_predictor()
    counts = predictor.predict_counts(company.name, days_until)
    
    # resolve peer company IDs for cluster-aware fallback
    peer_ids = _resolve_peer_company_ids(db, company.name)

    # get the ChromaDB retriever
    retriever = get_retriever()
    selected_ids = []

    # for each difficulty, retrieve the most relevant problems via vector search
    for difficulty, count in [("EASY", counts["easy"]), ("MEDIUM", counts["medium"]), ("HARD", counts["hard"])]:
        if count <= 0:
            continue
            
        results = retriever.retrieve_by_company(
            company_id=company_id,
            company_name=company.name,
            difficulty=difficulty,
            n_results=count,
            peer_company_ids=peer_ids,
        )
        
        if results and results["ids"] and results["ids"][0]:
            selected_ids.extend(results["ids"][0])

    # fetch the full models from SQLite, preserving the frequency-ranked
    # order returned by the retriever, deduplicating by URL
    selected = []
    seen_urls: set[str] = set()

    if selected_ids:
        int_ids = [int(sid) for sid in selected_ids]
        rows = db.query(InterviewProblem).filter(InterviewProblem.id.in_(int_ids)).all()
        id_to_row = {p.id: p for p in rows}
        for pid in int_ids:
            problem = id_to_row.get(pid)
            if problem is None:
                continue
            url_key = (problem.url or "").lower()
            if url_key and url_key in seen_urls:
                continue
            selected.append(problem)
            if url_key:
                seen_urls.add(url_key)

    # fallback: fill with highest-frequency problems from the company or peers,
    # skipping any problems already selected (by URL)
    if len(selected) < counts["total"]:
        used_ids = {p.id for p in selected}

        # try the company's own problems first
        candidate_ids = [company_id]
        if peer_ids:
            candidate_ids.extend(peer_ids)

        fill_problems = (
            db.query(InterviewProblem)
            .filter(InterviewProblem.company_id.in_(candidate_ids))
            .filter(InterviewProblem.id.notin_(used_ids))
            .order_by(InterviewProblem.frequency.desc().nullslast())
            .all()
        )

        for problem in fill_problems:
            if len(selected) >= counts["total"]:
                break
            url_key = (problem.url or "").lower()
            if url_key and url_key in seen_urls:
                continue
            selected.append(problem)
            if url_key:
                seen_urls.add(url_key)

    return selected
