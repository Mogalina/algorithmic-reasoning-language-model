from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from database import engine, Base
from deps import templates
from routers import auth, dashboard, interview, tutor

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

Base.metadata.create_all(bind=engine)

app.include_router(auth.router)
app.include_router(dashboard.router)
app.include_router(interview.router)
app.include_router(tutor.router)


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})
