import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from database import get_db
from deps import get_current_user
from models import InterviewProblem

router = APIRouter(prefix="/api/tutor")


class ChatRequest(BaseModel):
    problem_id: int
    messages: list[dict]


@router.post("/chat")
async def tutor_chat(payload: ChatRequest, request: Request, db: Session = Depends(get_db)):
    user = get_current_user(request, db)
    if not user:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    problem = db.query(InterviewProblem).filter(InterviewProblem.id == payload.problem_id).first()
    if not problem:
        return JSONResponse({"error": "Problem not found"}, status_code=404)

    description = problem.description or f"{problem.title} ({problem.difficulty})"

    try:
        from tutor.chat import get_tutor

        tutor = get_tutor()
        reply = tutor.reply(payload.messages, problem_description=description)
    except RuntimeError as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)
    except Exception as exc:
        return JSONResponse({"error": f"Tutor error: {exc}"}, status_code=502)

    return {"reply": reply}
