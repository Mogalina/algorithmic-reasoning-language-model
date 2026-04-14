# Algorithmic Reasoning Language Model

An interview preparation platform that combines a **FastAPI web application**, a **transformer-based embedding and retrieval pipeline**, and a **data analytics module** for interview question analysis.

---

## Table of Contents

- [Project Structure](#project-structure)
- [Architecture Overview](#architecture-overview)
- [Application — FastAPI Web App](#application--fastapi-web-app)
  - [Routes](#routes)
  - [Authentication Flow](#authentication-flow)
  - [Database Schema](#database-schema)
- [Embeddings — ML Retrieval Pipeline](#embeddings--ml-retrieval-pipeline)
  - [Embedder](#embedder)
  - [Searcher](#searcher)
  - [How the Embedding Math Works](#how-the-embedding-math-works)
  - [Utilities](#utilities)
  - [Configuration](#configuration)
  - [Tests](#tests)
- [Layout Model — Data Processing & Analytics](#layout-model--data-processing--analytics)
- [Dependencies](#dependencies)
- [Getting Started](#getting-started)
- [Notes for Contributors](#notes-for-contributors)

---

## Project Structure

```
algorithmic-reasoning-language-model/
├── .gitignore
├── README.md
├── requirements.txt                      # Web app dependencies (pinned)
│
├── application/                          # FastAPI web application
│   ├── main.py                           # App entry point & routes
│   ├── auth.py                           # Password hashing & JWT tokens
│   ├── database.py                       # SQLAlchemy engine & session
│   ├── models.py                         # ORM models (User, Company, Interview)
│   ├── test_main.http                    # HTTP client scratch file
│   ├── static/
│   │   └── style.css                     # Global stylesheet
│   └── templates/
│       ├── base.html                     # Jinja2 base layout
│       ├── home.html                     # Landing page
│       ├── login.html                    # Login form
│       ├── register.html                 # Registration form
│       └── dashboard.html                # Authenticated dashboard
│
├── embeddings/                           # ML embedding & retrieval pipeline
│   ├── config/
│   │   └── config.yaml                   # Model, FAISS & logging settings
│   ├── src/
│   │   ├── __init__.py
│   │   ├── pipeline/
│   │   │   ├── __init__.py
│   │   │   ├── embedder.py               # Text → vector embeddings
│   │   │   └── searcher.py               # FAISS similarity search
│   │   └── utils/
│   │       ├── __init__.py
│   │       ├── config.py                 # YAML config loader
│   │       ├── download_model.py         # HuggingFace model downloader
│   │       └── logger.py                 # Loguru structured file logging
│   └── tests/
│       ├── conftest.py                   # Pytest path setup
│       ├── test_pipeline_embedder.py
│       ├── test_pipeline_searcher.py
│       └── test_utils_logger.py
│
└── layout-model/                         # Data analytics & future ML
    └── data_processing.py                # Pandas aggregation pipeline
```

---

## Architecture Overview

```
┌────────────────────┐      ┌────────────────────────┐      ┌────────────────────┐
│   application/     │      │     embeddings/         │      │   layout-model/    │
│                    │      │                         │      │                    │
│   FastAPI web app  │      │   ML retrieval pipeline │      │   Analytics script │
│   Auth + dashboard │      │   PyTorch + FAISS       │      │   Pandas + NumPy   │
│   SQLite via ORM   │      │   HuggingFace models    │      │   Future SVD       │
│                    │      │                         │      │                    │
└────────────────────┘      └────────────────────────┘      └────────────────────┘
         │                            │                              │
         │         (not yet wired)    │           (not yet wired)    │
         └────────────────────────────┴──────────────────────────────┘
```

The three modules are currently **independent**. The intended architecture is to wire the
embeddings search into the web app (so users can search for similar interview questions)
and integrate the layout-model analytics (difficulty distributions by company and timeframe).

---

## Application — FastAPI Web App

A server-rendered interview-tracking web application. Users register, log in, and manage
upcoming interviews on a personal dashboard.

**Tech stack:**

| Library | Role |
|---|---|
| **FastAPI** | Async web framework, routing, dependency injection |
| **Jinja2** | Server-side HTML templating |
| **SQLAlchemy** | ORM over local SQLite (`application/data/app.db`) |
| **Passlib + bcrypt** | Password hashing |
| **python-jose** | JWT creation and verification |
| **Uvicorn** | ASGI server |

### Routes

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Landing / home page |
| `GET` | `/login` | Login form |
| `POST` | `/login` | Validate credentials, set JWT cookie, redirect to dashboard |
| `GET` | `/register` | Registration form |
| `POST` | `/register` | Create user, set JWT cookie, redirect to dashboard |
| `GET` | `/dashboard` | *Requires auth* — lists companies and user's interviews |
| `POST` | `/dashboard/add` | *Requires auth* — creates a new interview entry |
| `GET` | `/logout` | Clears the auth cookie |

### Authentication Flow

1. User submits email + password via the registration or login form.
2. On **register**, the password is bcrypt-hashed and stored in the `users` table.
3. On **login**, the submitted password is verified against the stored hash.
4. A **JWT** is created with the user's email as the `sub` claim and an expiry of 60 minutes.
5. The JWT is set as an **HTTP-only cookie** named `access_token`.
6. Every protected route calls `get_current_user()`, which decodes the cookie and loads the `User` from the database.

### Database Schema

Three tables managed by SQLAlchemy:

```
┌──────────────┐       ┌──────────────────┐       ┌──────────────┐
│    users     │       │   interviews     │       │  companies   │
├──────────────┤       ├──────────────────┤       ├──────────────┤
│ id       PK  │──┐    │ id           PK  │    ┌──│ id       PK  │
│ email        │  └───>│ user_id      FK  │    │  │ name         │
│ hashed_pwd   │       │ company_id   FK  │<───┘  └──────────────┘
└──────────────┘       │ days_until       │
                       └──────────────────┘
```

- **`users`** — `id`, `email` (unique), `hashed_password`
- **`companies`** — `id`, `name` (unique)
- **`interviews`** — `id`, `days_until`, `user_id` → users, `company_id` → companies

On startup, `seed_db()` inserts a default company (*"ActuallyGoodAIAds"*) if none exists.

---

## Embeddings — ML Retrieval Pipeline

A library-style module for embedding text into dense vectors and performing similarity
search over a FAISS index. This is the AI core of the project.

**Tech stack:**

| Library | Role |
|---|---|
| **PyTorch** | Tensor operations, GPU/MPS/CPU device management |
| **HuggingFace Transformers** | Pre-trained model loading (`AutoModel`, `AutoTokenizer`) |
| **FAISS** | High-performance vector similarity search |
| **HuggingFace Hub** | Model weight downloads |
| **Loguru** | Structured JSON file logging with rotation |
| **PyYAML** | Configuration loading |
| **Rich** | Terminal spinner/status UI during downloads |

### Embedder

**Class:** `Embedder` in `embeddings/src/pipeline/embedder.py`

- Loads the model specified in `config.yaml` (default: **`google/embeddinggemma-300m`**)
  — either from a local cache or directly from HuggingFace.
- Automatically selects the best available device: **CUDA GPU → Apple MPS → CPU**.
- `embed(texts)` accepts a string or list of strings and returns an `np.ndarray` of
  shape `(N, dim)` with **L2-normalized float32** vectors.
- Processes inputs in configurable **batches** (default 32) to prevent out-of-memory errors.

### Searcher

**Class:** `Searcher` in `embeddings/src/pipeline/searcher.py`

- Reads a pre-built FAISS index (`index.faiss`) and companion metadata (`metadata.pkl`)
  from the configured directory.
- **Lazy-loads** — files are not read until the first `.search()` call.
- `search(query_vector, top_k=5)` returns the `top_k` most similar entries as a list of
  dicts, each containing metadata fields plus a `distance` score.

### How the Embedding Math Works

```
  Input text
      │
      ▼
┌─────────────────┐
│   Transformer    │    Tokenize → forward pass → one vector per token
│   (EmbedGemma)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Mean Pooling   │    Average token vectors (masked by attention) → 1 vector per text
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ L2 Normalization │    Scale to unit length so inner product = cosine similarity
└────────┬────────┘
         │
         ▼
   Dense vector (float32, unit norm)
         │
         ▼
┌─────────────────┐
│   FAISS Index    │    IndexFlatIP (inner product on unit vectors ≈ cosine similarity)
│   .search()      │    → top-k most similar entries
└─────────────────┘
```

### Utilities

| Module | Description |
|---|---|
| `config.py` | Resolves `PROJECT_ROOT` (the `embeddings/` folder) and provides `load_config()` with `lru_cache` for single-parse YAML loading |
| `download_model.py` | Wraps `huggingface_hub.snapshot_download` with error handling for gated/restricted models and a Rich progress spinner |
| `logger.py` | Configures Loguru with a file-only JSON sink, 100 MB rotation, 30-day retention, and zip compression |

### Configuration

All settings live in `embeddings/config/config.yaml`:

```yaml
embedding:
  model_id: "google/embeddinggemma-300m"
  local_model_dir: "./models/embeddinggemma-300m"
  batch_size: 32

database:
  faiss:
    path: "./data/faiss_index"       # index.faiss + metadata.pkl
    max_size_gb: 10
  sqlite:
    path: "./data/metadata.db"       # reserved for future use
    max_size_gb: 100

logging:
  log_file: "./logs/app.log"
  rotation: "100 MB"
  retention: "30 days"
  level: "INFO"
```

### Tests

Located in `embeddings/tests/`. All transformer and FAISS dependencies are **fully mocked**,
so tests run without model weights or index files on disk.

| Test file | Covers |
|---|---|
| `test_pipeline_embedder.py` | Embedder init, mean pooling logic, embed output shape |
| `test_pipeline_searcher.py` | Searcher paths, lazy loading, FAISS search results |
| `test_utils_logger.py` | Config caching, logger setup and binding |

---

## Layout Model — Data Processing & Analytics

A standalone script (`layout-model/data_processing.py`) that processes interview-related
data from a CSV into aggregated statistics.

**Pipeline:**

1. Reads `../embeddings/dataset/dataset.csv` (not checked into the repo).
2. Filters to four columns: `company`, `difficulty`, `preparation_days`, `title`.
3. Renames `preparation_days` → `days_until`.
4. Groups by `(company, days_until, difficulty)` and counts interview titles.
5. Pivots difficulty levels into separate columns: `easycount`, `mediumcount`, `hardcount`.
6. Writes the result to `dataset_aggregated.csv`.

**Planned feature:** `predict_missing_days_svd(df, value_col, k_components=3)` is a stub
for **SVD-based matrix completion** — predicting interview difficulty distributions for
time windows where data is missing.

---

## Dependencies

### Declared in root `requirements.txt` (web app)

FastAPI, Starlette, Uvicorn, Jinja2, SQLAlchemy, Pydantic, Passlib (bcrypt),
python-jose, python-multipart, numpy, pandas, and their transitive dependencies.

### Required by `embeddings/` (not yet in a requirements file)

| Package | Purpose |
|---|---|
| `torch` | Tensor operations, device management |
| `transformers` | HuggingFace model loading |
| `faiss-cpu` or `faiss-gpu` | Vector similarity search |
| `PyYAML` | Config parsing |
| `loguru` | Structured logging |
| `huggingface_hub` | Model downloads |
| `rich` | Terminal UI |
| `pytest` | Testing |

---

## Getting Started

### Web Application

```bash
cd application
pip install -r ../requirements.txt
uvicorn main:app --reload
```

The app creates `application/data/app.db` (SQLite) on first run and seeds it with a
default company. Visit `http://127.0.0.1:8000`.

### Embeddings Pipeline

```bash
cd embeddings
pip install torch transformers faiss-cpu pyyaml loguru huggingface_hub rich
```

Usage (programmatic):

```python
from src.pipeline import Embedder, Searcher

embedder = Embedder()
vectors = embedder.embed(["How do you design a distributed cache?"])

searcher = Searcher()
results = searcher.search(vectors, top_k=5)
```

> **Note:** The FAISS index (`index.faiss` + `metadata.pkl`) must be built separately.
> No index-building script exists in the repo yet.

### Data Processing

```bash
cd layout-model
python data_processing.py
```

> Requires the dataset CSV at `../embeddings/dataset/dataset.csv`.

---

## Notes for Contributors

- The three modules (`application/`, `embeddings/`, `layout-model/`) are currently
  **independent** — no imports exist between them yet.
- The `SECRET_KEY` in `auth.py` is hardcoded — move it to an environment variable
  before any deployment.
- `main.py` prints plaintext passwords during registration (`print(password)`) —
  this should be removed.
- The dataset CSV and model weights are **gitignored** and not in the repository.
- The `embeddings/` module needs its own `requirements.txt`.
- `test_main.http` references a route (`/hello/User`) that does not exist in `main.py`.
- The SQLite metadata path in `config.yaml` (`database.sqlite`) is reserved but not
  currently used by any Python code — metadata is stored as pickle alongside the FAISS index.
