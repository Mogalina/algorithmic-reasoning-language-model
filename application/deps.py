import re

from fastapi import Request
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from models import User
from security import decode_token

templates = Jinja2Templates(directory="templates")
templates.env.filters["regex_replace"] = lambda s, pat, repl: re.sub(pat, repl, s)


def get_current_user(request: Request, db: Session):
    token = request.cookies.get("access_token")
    if not token:
        return None
    email = decode_token(token)
    if not email:
        return None
    return db.query(User).filter(User.email == email).first()
