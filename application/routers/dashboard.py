import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy.orm import Session

from database import get_db
from deps import get_current_user, templates
from models import Company, Interview, Roadmap
from rag.recommender import recommend_problems

router = APIRouter()


@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request, db: Session = Depends(get_db)):
    user = get_current_user(request, db)
    if not user:
        return RedirectResponse(url="/login", status_code=302)
    companies = db.query(Company).all()
    interviews = db.query(Interview).filter(Interview.user_id == user.id).all()
    error = request.query_params.get("error")

    # Dynamic calculation and sorting
    today = datetime.now().date()
    for interview in interviews:
        if interview.interview_date:
            interview.days_remaining = (interview.interview_date - today).days
        else:
            interview.days_remaining = interview.days_until
            
        # Calculate completion for sorting
        total = len(interview.roadmaps)
        done = len([r for r in interview.roadmaps if r.is_completed])
        interview.is_finished = (done == total and total > 0)

    # Sort: Not finished first, then by days remaining
    interviews.sort(key=lambda x: (x.is_finished, x.days_remaining))

    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "user": user,
        "companies": companies,
        "interviews": interviews,
        "error": error
    })


@router.post("/dashboard/add", response_class=HTMLResponse)
async def add_interview(request: Request, company_id: int = Form(...), days_until: int = Form(...), db: Session = Depends(get_db)):
    user = get_current_user(request, db)
    if not user:
        return RedirectResponse(url="/login", status_code=302)

    # Check for duplicate track
    existing = db.query(Interview).filter(Interview.user_id == user.id, Interview.company_id == company_id).first()
    if existing:
        return RedirectResponse(url="/dashboard?error=You+already+have+a+track+for+this+company", status_code=302)

    interview_date = datetime.now().date() + timedelta(days=days_until)
    interview = Interview(
        user_id=user.id, 
        company_id=company_id, 
        days_until=days_until,
        interview_date=interview_date
    )
    db.add(interview)
    db.flush()

    problems = recommend_problems(db, company_id, days_until)
    for problem in problems:
        db.add(Roadmap(interview_id=interview.id, problem_id=problem.id))

    db.commit()
    return RedirectResponse(url="/dashboard", status_code=302)


@router.post("/dashboard/delete/{interview_id}")
async def delete_interview(request: Request, interview_id: int, db: Session = Depends(get_db)):
    user = get_current_user(request, db)
    if not user:
        return RedirectResponse(url="/login", status_code=302)

    interview = db.query(Interview).filter(Interview.id == interview_id, Interview.user_id == user.id).first()
    if interview:
        # Delete associated roadmaps first
        db.query(Roadmap).filter(Roadmap.interview_id == interview.id).delete()
        db.delete(interview)
        db.commit()

    return RedirectResponse(url="/dashboard", status_code=302)
