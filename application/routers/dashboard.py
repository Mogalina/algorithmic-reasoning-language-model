import sys
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
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "user": user,
        "companies": companies,
        "interviews": interviews,
    })


@router.post("/dashboard/add", response_class=HTMLResponse)
async def add_interview(request: Request, company_id: int = Form(...), days_until: int = Form(...), db: Session = Depends(get_db)):
    user = get_current_user(request, db)
    if not user:
        return RedirectResponse(url="/login", status_code=302)

    interview = Interview(user_id=user.id, company_id=company_id, days_until=days_until)
    db.add(interview)
    db.flush()

    problems = recommend_problems(db, company_id, days_until)
    for problem in problems:
        db.add(Roadmap(interview_id=interview.id, problem_id=problem.id))

    db.commit()
    return RedirectResponse(url="/dashboard", status_code=302)
