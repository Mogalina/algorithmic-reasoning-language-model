# Algorithmic Reasoning Language Model

An AI-powered interview preparation platform. Users register upcoming coding interviews, receive a personalized problem roadmap, and practice with a Socratic tutor chatbot that guides them through each problem.

---

## Table of Contents

- [Project Structure](#project-structure)
- [Architecture Overview](#architecture-overview)
- [Quick Start](#quick-start)
- [Application — FastAPI Web App](#application--fastapi-web-app)
  - [Routes](#routes)
  - [Authentication](#authentication)
  - [Database Schema](#database-schema)
  - [Pages](#pages)
- [RAG — Retrieval & Recommendation](#rag--retrieval--recommendation)
- [Embeddings — Vector Pipeline](#embeddings--vector-pipeline)
- [Tutor — Socratic Chatbot Agent](#tutor--socratic-chatbot-agent)
- [Datasets](#datasets)
- [Scripts](#scripts)
- [Dependencies](#dependencies)

---

## Project Structure

```
algorithmic-reasoning-language-model/
├── application/                      # FastAPI web application
│   ├── main.py                       # App entry point, mounts routers
│   ├── database.py                   # SQLAlchemy engine & session
│   ├── models.py                     # ORM models
│   ├── security.py                   # Password hashing & JWT
│   ├── deps.py                       # Shared dependencies (templates, auth helper)
│   ├── routers/
│   │   ├── auth.py                   # /login, /register, /logout
│   │   ├── dashboard.py              # /dashboard, /dashboard/add
│   │   └── interview.py              # /interview/.../roadmap, .../problem
│   ├── static/
│   │   └── style.css                 # Global stylesheet
│   └── templates/
│       ├── base.html                 # Base layout with nav & footer
│       ├── home.html                 # Landing page
│       ├── login.html                # Login form
│       ├── register.html             # Registration form
│       ├── dashboard.html            # Interview list with progress bars
│       ├── roadmap.html              # Problem checklist for an interview
│       └── problem.html              # Split view: problem + tutor chat
│
├── rag/                              # Retrieval & recommendation
│   ├── __init__.py
│   └── recommender.py                # Selects problems for a roadmap
│
├── embeddings/                       # ML embedding & search pipeline
│   ├── config/
│   │   └── config.yaml               # Model, FAISS & logging config
│   ├── src/
│   │   ├── pipeline/
│   │   │   ├── embedder.py           # Text -> vector embeddings (HuggingFace)
│   │   │   └── searcher.py           # FAISS similarity search
│   │   └── utils/
│   │       ├── config.py             # YAML config loader
│   │       ├── download_model.py     # HuggingFace model downloader
│   │       └── logger.py             # Loguru structured logging
│   └── tests/                        # Pytest suite (fully mocked)
│
├── tutor/                            # Socratic chatbot agent (planned)
│   └── __init__.py
│
├── datasets/                         # Source data (tracked in git)
│   ├── dataset.csv                   # Raw LeetCode problems per company
│   ├── dataset_aggregated.csv        # Difficulty distributions per company/days
│   └── dataset_enriched.jsonl        # Problems with scraped descriptions
│
├── scripts/                          # Setup & maintenance scripts
│   ├── seed_database.py              # Populate DB from dataset
│   ├── enrich_dataset.py             # Scrape problem descriptions from LeetCode
│   ├── process_distributions.py      # Aggregate difficulty distributions
│   └── build_vector_index.py         # Build ChromaDB index (placeholder)
│
├── data/                             # Runtime artifacts (gitignored)
│   └── app.db                        # SQLite database
│
├── requirements.txt                  # Python dependencies
└── README.md
```

---

## Architecture Overview

```
User registers interview
        │
        ▼
┌─────────────────────┐
│    application/      │  FastAPI web app
│    routers/          │  Auth, dashboard, interview routes
└────────┬────────────┘
         │
    ┌────┴──────┐
    ▼           ▼
┌────────┐  ┌────────┐
│  rag/  │  │ tutor/ │
│        │  │        │
│ Picks  │  │Socratic│
│problems│  │chatbot │
│for the │  │with one│
│roadmap │  │tool    │
└───┬────┘  └────────┘
    │
    ▼
┌────────────┐
│ embeddings/│  Text → vectors → FAISS search
└────────────┘
```

**Dependency flow:** `application` → `rag` → `embeddings`. The `tutor` module is called directly by the application. Each module has a single clear responsibility.

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Seed the database

This reads `datasets/dataset_enriched.jsonl` (or falls back to `datasets/dataset.csv`) and populates the SQLite database at `data/app.db` with 464 companies and ~11K problems.

```bash
python scripts/seed_database.py
```

### 3. Run the app

```bash
cd application
uvicorn main:app --reload
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.

### 4. Use the app

1. **Register** an account
2. **Add an interview** — pick a company and enter how many days until the interview
3. A **roadmap** of 10 problems is generated automatically
4. Click **Roadmap →** to see your problem checklist
5. Click a problem to open the **problem detail page** with the tutor chat panel
6. Check off problems as you solve them — progress is tracked on the dashboard

---

## Application — FastAPI Web App

Server-rendered web application using Jinja2 templates and SQLite.

**Tech stack:** FastAPI, Jinja2, SQLAlchemy, Passlib (bcrypt), python-jose (JWT), Uvicorn.

### Routes

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Landing page |
| `GET/POST` | `/login` | Login |
| `GET/POST` | `/register` | Registration |
| `GET` | `/dashboard` | Interview list with progress |
| `POST` | `/dashboard/add` | Create interview + generate roadmap |
| `GET` | `/interview/{id}/roadmap` | Problem checklist |
| `POST` | `/interview/{id}/roadmap/{rid}/toggle` | Toggle problem completion |
| `GET` | `/interview/{id}/problem/{pid}` | Problem detail + tutor chat |
| `GET` | `/logout` | Clear auth cookie |

### Authentication

- Passwords are bcrypt-hashed before storage
- JWT in an HTTP-only cookie (`access_token`, 60 min expiry)
- `get_current_user()` in `deps.py` decodes the cookie on every protected route

### Database Schema

```
users                interviews              companies
┌──────────────┐     ┌──────────────────┐    ┌──────────────┐
│ id       PK  │──┐  │ id           PK  │ ┌──│ id       PK  │
│ email        │  └─>│ user_id      FK  │ │  │ name         │
│ hashed_pwd   │     │ company_id   FK  │<┘  └──────────────┘
└──────────────┘     │ days_until       │         │
                     └────────┬─────────┘         │
                              │                   │
                     roadmaps │         interview_problems
                     ┌────────┴─────┐   ┌─────────┴──────────┐
                     │ id       PK  │   │ id            PK   │
                     │ interview_id │   │ title              │
                     │ problem_id   │──>│ difficulty          │
                     │ is_completed │   │ description         │
                     └──────────────┘   │ url                 │
                                        │ topics              │
                                        │ acceptance_rate     │
                                        │ frequency           │
                                        │ preparation_days    │
                                        │ company_id     FK   │
                                        └─────────────────────┘
```

### Pages

- **Dashboard** — schedule interviews, see progress bars and status badges (Urgent/Soon/On Track), link to each roadmap
- **Roadmap** — checklist of problems with difficulty tags, topic labels, circular progress ring, and completion toggles
- **Problem detail** — split layout: problem description + topic chips on the left, Socratic tutor chat panel on the right, link to solve on LeetCode

---

## RAG — Retrieval & Recommendation

`rag/recommender.py` — selects which problems a user should solve for a given company and timeline.

**Current implementation:** mock that picks 10 random problems from the company.

**Planned implementation:**
1. **PCHIP interpolation** on `datasets/dataset_aggregated.csv` to determine how many easy/medium/hard problems to recommend for the given number of days
2. **ChromaDB vector search** to retrieve the most relevant problems per difficulty, using the `embeddings/` module to embed query text

---

## Embeddings — Vector Pipeline

Library-style module for embedding text into dense vectors and performing similarity search.

**Tech stack:** PyTorch, HuggingFace Transformers (`google/embeddinggemma-300m`), FAISS, Loguru.

- **`Embedder`** — loads a transformer model, performs mean pooling + L2 normalization, outputs `np.ndarray` vectors
- **`Searcher`** — lazy-loads a FAISS index + metadata pickle, returns top-k similar entries
- **Config** — `embeddings/config/config.yaml` (model ID, batch size, index paths, logging)
- **Tests** — fully mocked, no model weights needed to run

**Note:** This module requires additional dependencies not in the root `requirements.txt`:

```bash
pip install torch transformers faiss-cpu pyyaml loguru huggingface_hub rich
```

---

## Tutor — Socratic Chatbot Agent

**Status:** UI placeholder is implemented. The chat panel is visible on the problem detail page with a message input and placeholder responses.

**Planned architecture:**
- **`agent.py`** — single-tool agent that decides when to call the solution generator
- **`solver.py`** — fine-tuned SLM that produces ground-truth solutions
- **`prompts/`** — Socratic system prompt + few-shot examples
- The tutor knows the solution but never reveals it directly — it guides the student with questions

---

## Datasets

All source data lives in `datasets/` and is tracked in git.

| File | Description |
|---|---|
| `dataset.csv` | ~11K LeetCode problems with company, difficulty, topics, frequency, acceptance rate, and preparation days |
| `dataset_aggregated.csv` | Aggregated counts of easy/medium/hard problems per company per time bucket (30/90/180/360 days) |
| `dataset_enriched.jsonl` | Same as `dataset.csv` but enriched with problem descriptions scraped from LeetCode (JSONL format) |

---

## Scripts

Run all scripts from the **repository root**.

| Script | Command | Description |
|---|---|---|
| **Seed database** | `python scripts/seed_database.py` | Loads problems and companies into SQLite. Prefers `dataset_enriched.jsonl`, falls back to `dataset.csv`. Idempotent — skips if already seeded. |
| **Enrich dataset** | `python scripts/enrich_dataset.py` | Scrapes problem descriptions from LeetCode's GraphQL API. Resumable with checkpoints. Output: `datasets/dataset_enriched.jsonl`. |
| **Process distributions** | `python scripts/process_distributions.py` | Aggregates the raw dataset into difficulty distributions. Output: `datasets/dataset_aggregated.csv`. |
| **Build vector index** | `python scripts/build_vector_index.py` | Placeholder for building the ChromaDB vector index from the database. |

---

## Dependencies

### Root `requirements.txt` (web app + scripts)

FastAPI, SQLAlchemy, Jinja2, Passlib, python-jose, Uvicorn, pandas, numpy, beautifulsoup4, requests.

### Additional (embeddings module)

```
torch, transformers, faiss-cpu, pyyaml, loguru, huggingface_hub, rich
```

### Additional (planned — RAG + tutor)

```
chromadb, openai (or equivalent LLM client)
```
