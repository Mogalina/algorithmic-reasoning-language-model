from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy.orm import Session

from database import get_db
from deps import get_current_user, templates
from models import Interview, Roadmap, InterviewProblem

router = APIRouter()


@router.get("/interview/{interview_id}/roadmap", response_class=HTMLResponse)
async def roadmap(request: Request, interview_id: int, db: Session = Depends(get_db)):
    user = get_current_user(request, db)
    if not user:
        return RedirectResponse(url="/login", status_code=302)

    interview = db.query(Interview).filter(
        Interview.id == interview_id,
        Interview.user_id == user.id,
    ).first()
    if not interview:
        return RedirectResponse(url="/dashboard", status_code=302)

    roadmap_items = (
        db.query(Roadmap)
        .filter(Roadmap.interview_id == interview_id)
        .all()
    )

    return templates.TemplateResponse("roadmap.html", {
        "request": request,
        "user": user,
        "interview": interview,
        "roadmap_items": roadmap_items,
    })


@router.post("/interview/{interview_id}/roadmap/{roadmap_id}/toggle", response_class=HTMLResponse)
async def toggle_completed(request: Request, interview_id: int, roadmap_id: int, db: Session = Depends(get_db)):
    user = get_current_user(request, db)
    if not user:
        return RedirectResponse(url="/login", status_code=302)

    item = (
        db.query(Roadmap)
        .join(Interview)
        .filter(Roadmap.id == roadmap_id, Interview.user_id == user.id)
        .first()
    )
    if item:
        item.is_completed = not item.is_completed
        db.commit()

    return RedirectResponse(url=f"/interview/{interview_id}/roadmap", status_code=302)


@router.get("/interview/{interview_id}/problem/{problem_id}", response_class=HTMLResponse)
async def problem_detail(request: Request, interview_id: int, problem_id: int, db: Session = Depends(get_db)):
    user = get_current_user(request, db)
    if not user:
        return RedirectResponse(url="/login", status_code=302)

    interview = db.query(Interview).filter(
        Interview.id == interview_id,
        Interview.user_id == user.id,
    ).first()
    if not interview:
        return RedirectResponse(url="/dashboard", status_code=302)

    problem = db.query(InterviewProblem).filter(InterviewProblem.id == problem_id).first()
    if not problem:
        return RedirectResponse(url=f"/interview/{interview_id}/roadmap", status_code=302)

    return templates.TemplateResponse("problem.html", {
        "request": request,
        "user": user,
        "interview": interview,
        "problem": problem,
    })
