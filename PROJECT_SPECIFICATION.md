# Algorithmic Reasoning Language Model — Project Specification

An AI-powered interview preparation platform that generates personalized problem roadmaps using company clustering and RAG, then guides students through each problem with a Socratic tutor agent backed by an expert solver, automated evaluation, and self-improving meta-prompting.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Datasets](#datasets)
  - [Raw Problem Dataset](#1-raw-problem-dataset)
  - [Enriched Dataset](#2-enriched-dataset)
  - [Aggregated Distributions](#3-aggregated-distributions)
  - [Company Metadata](#4-company-metadata)
  - [Kaggle 7M Company Dataset](#5-kaggle-7m-company-dataset)
  - [Fine-Tuning Data](#6-fine-tuning-data)
- [Company Clustering & Roadmap Prediction](#company-clustering--roadmap-prediction)
  - [Feature Engineering](#feature-engineering)
  - [Clustering Algorithm](#clustering-algorithm)
  - [Industry-Weighted Clustering](#industry-weighted-clustering)
  - [PCHIP Curve Fitting](#pchip-curve-fitting)
  - [Unknown Company Handling](#unknown-company-handling)
  - [Peer Company Retrieval](#peer-company-retrieval)
- [Embedding Pipeline](#embedding-pipeline)
  - [Model](#model)
  - [Architecture](#architecture)
  - [Vector Storage](#vector-storage)
- [Retrieval-Augmented Generation (RAG)](#retrieval-augmented-generation-rag)
  - [ChromaDB Collections](#chromadb-collections)
  - [Problem Retrieval Strategy](#problem-retrieval-strategy)
  - [Frequency-Based Ranking](#frequency-based-ranking)
  - [URL-Based Deduplication](#url-based-deduplication)
  - [Solution Retrieval (Few-Shot Injection)](#solution-retrieval-few-shot-injection)
- [Agentic Socratic Tutor](#agentic-socratic-tutor)
  - [Architecture](#tutor-architecture)
  - [End-to-End Flow](#end-to-end-flow)
  - [Expert Solver (Hybrid)](#expert-solver-hybrid)
  - [Solver Self-Evaluation Loop](#solver-self-evaluation-loop)
- [Reinforcement Learning from Human Feedback (RLHF)](#reinforcement-learning-from-human-feedback-rlhf)
  - [DPO Training](#dpo-training)
  - [QLoRA Configuration](#qlora-configuration)
  - [Preference Pair Generation](#preference-pair-generation)
  - [Reward Signal](#reward-signal)
- [Evaluation with DeepEval](#evaluation-with-deepeval)
  - [Socratic Quality (GEval)](#socratic-quality-geval)
  - [Code Logic Correctness (GEval)](#code-logic-correctness-geval)
  - [Two-Stage Solver Gating](#two-stage-solver-gating)
- [Meta-Prompting (Self-Improving Prompts)](#meta-prompting-self-improving-prompts)
  - [Closed-Loop Optimization](#closed-loop-optimization)
  - [Optimizer Model](#optimizer-model)
- [Prompting Strategies](#prompting-strategies)
  - [Tree of Thought (Socratic Tutor)](#tree-of-thought-socratic-tutor)
  - [Chain of Thought (Solver)](#chain-of-thought-solver)
  - [Few-Shot CoT (Evaluators)](#few-shot-cot-evaluators)
  - [Few-Shot CoT (Optimizer)](#few-shot-cot-optimizer)
  - [Prompt Extraction Architecture](#prompt-extraction-architecture)
- [Toxicity & Hallucination Handling](#toxicity--hallucination-handling)
  - [Anti-Hallucination Guards](#anti-hallucination-guards)
  - [Toxicity & Tone Controls](#toxicity--tone-controls)
  - [Evaluation-Level Enforcement](#evaluation-level-enforcement)
- [Models Used](#models-used)
- [Application Layer](#application-layer)
  - [Tech Stack](#tech-stack)
  - [Database Schema](#database-schema)
  - [API Routes](#api-routes)
- [Setup & Scripts](#setup--scripts)

---

## Architecture Overview

```
                                    ┌──────────────────────────────────────┐
                                    │         datasets/                    │
                                    │  dataset.csv  dataset_enriched.jsonl │
                                    │  dataset_aggregated.csv              │
                                    │  company_meta.csv  companies.parquet │
                                    └──────────┬───────────────────────────┘
                                               │
          ┌────────────────────────────────────┬┴──────────────────────────────┐
          ▼                                    ▼                               ▼
  ┌───────────────┐                  ┌──────────────────┐             ┌──────────────────┐
  │  scripts/      │                 │  rag/             │             │  tutor/           │
  │                │                 │                   │             │                   │
  │ seed_database  │                 │ curve_fitting.py  │◄──────┐    │ chat.py           │
  │ enrich_dataset │                 │  └ KMeans + PCHIP │       │    │  └ SocraticTutor  │
  │ ingest_company │                 │ retriever.py      │       │    │ solver.py         │
  │ build_*_index  │                 │  └ ChromaRetriever│       │    │  └ GemmaSolver    │
  │ process_dist.  │                 │ recommender.py    │       │    │ evaluator.py      │
  └───────┬────────┘                 │  └ orchestration  │       │    │  └ GEval judge    │
          │                          │ company_lookup.py │       │    │ optimizer.py      │
          ▼                          │  └ Parquet lookup │       │    │  └ meta-prompting │
  ┌───────────────┐                  └────────┬─────────┘       │    │ solver_evaluator  │
  │  data/app.db   │                          │                 │    │ solver_rewards.py │
  │  SQLite + ORM  │◄─────────────────────────┘                 │    │ train_rlhf.py     │
  └───────────────┘                                             │    │  └ DPO + QLoRA    │
          │                                                     │    │ prompts/           │
          ▼                                                     │    │  └ 11 .txt files  │
  ┌───────────────┐          ┌──────────────────┐               │    └──────────┬────────┘
  │ application/   │          │  embeddings/      │               │               │
  │ FastAPI + Jinja│─────────>│  embeddinggemma   │───────────────┘               │
  │ Auth, Dashboard│          │  300M transformer │                               │
  │ Roadmap, Chat  │──────────┤  mean pool + L2   │◄──────────────────────────────┘
  └────────────────┘          │  ChromaDB / FAISS  │
                              └───────────────────┘
```

---

## Datasets

### 1. Raw Problem Dataset

**File:** `datasets/dataset.csv` (~11K rows)

| Column | Description |
|--------|-------------|
| `company` | Company name (e.g., Google, Amazon) |
| `title` | Problem title |
| `link` | LeetCode URL |
| `difficulty` | Easy / Medium / Hard |
| `acceptance_rate` | Percentage of accepted submissions |
| `frequency` | How often this problem appears in interviews (0.0–1.0) |
| `topics` | Comma-separated topic tags (e.g., "Array, Hash Table, Two Pointers") |
| `preparation_days` | Time bucket: 30, 90, 180, or 360 days until interview |

**Source:** Scraped from LeetCode's company-tagged problem lists. Covers 464 companies.

### 2. Enriched Dataset

**File:** `datasets/dataset_enriched.jsonl` (~11K lines)

Same fields as the raw CSV plus:

| Column | Description |
|--------|-------------|
| `body` | Full problem description text, scraped from LeetCode's GraphQL API |

**Purpose:** Provides the problem statement text needed for embedding, RAG retrieval, and the tutor's problem context. Produced by `scripts/enrich_dataset.py` with rate-limited GraphQL calls and resume-safe JSONL checkpoints.

### 3. Aggregated Distributions

**File:** `datasets/dataset_aggregated.csv` (~935 rows)

| Column | Description |
|--------|-------------|
| `company` | Company name |
| `days_until` | Time bucket (30, 90, 180, 360) |
| `easycount` | Number of Easy problems in this bucket |
| `mediumcount` | Number of Medium problems |
| `hardcount` | Number of Hard problems |

**Purpose:** Drives the clustering and PCHIP curve fitting in `rag/curve_fitting.py`. Each row represents how many problems of each difficulty a company has at a given preparation horizon. Produced by `scripts/process_distributions.py`.

### 4. Company Metadata

**File:** `datasets/company_meta.csv` (~454 rows)

| Column | Description |
|--------|-------------|
| `company` | LeetCode company name (matched to Kaggle) |
| `industry` | Industry sector (e.g., "internet", "computer software") |
| `size_range` | Company size bucket (e.g., "10001+", "1001-5000") |
| `employee_count` | Current employee estimate |
| `year_founded` | Year the company was founded |
| `locality` | Geographic location |

**Purpose:** Enriches LeetCode companies with firmographic metadata from the Kaggle dataset. Used as the primary features for clustering (industry, size, employees, founding year). Produced by `scripts/ingest_companies.py` through fuzzy name matching.

### 5. Kaggle 7M Company Dataset

**File:** `datasets/companies.parquet` (153 MB, Git LFS) — trimmed from `companies_sorted.csv` (1 GB, gitignored)

**Source:** [Kaggle — Free 7+ Million Company Dataset](https://www.kaggle.com/datasets/peopledatalabssf/free-7-million-company-dataset)

| Column | Description |
|--------|-------------|
| `name` | Company name |
| `domain` | Website domain |
| `year founded` | Founding year |
| `industry` | Industry sector |
| `size range` | Employee count bucket |
| `locality` | Location |
| `current employee estimate` | Numeric employee count |

**Purpose:** Runtime lookup for unknown companies. When a user enters a company not in our LeetCode dataset, we scan this Parquet file to find the closest match, extract its features, predict which cluster it belongs to, and serve problems from structurally similar companies. Stored as Parquet for efficient columnar scans with Polars (lazy evaluation, no full file load).

### 6. Fine-Tuning Data

**File:** `dataset/fine-tune/train.jsonl`

| Field | Description |
|-------|-------------|
| `prompt` | Problem description |
| `completion` | Reference Python solution |

**Purpose:** Used by `scripts/build_solution_index.py` to build the ChromaDB `solutions` collection for few-shot injection into the solver prompt.

---

## Company Clustering & Roadmap Prediction

The roadmap prediction system determines how many Easy, Medium, and Hard problems to recommend based on who the user is interviewing with and how many days they have left.

### Feature Engineering

Companies are characterized by four metadata features extracted from the Kaggle dataset:

| Feature | Encoding | Description |
|---------|----------|-------------|
| `industry_enc` | `LabelEncoder` ordinal | Industry sector (e.g., "internet" → 5, "finance" → 3) |
| `size_rank` | Ordinal mapping (1–7) | Company size bucket ("1-10" → 1, "10001+" → 7) |
| `log_employees` | `log1p(employee_count)` | Log-scaled employee count for normalization |
| `year_founded` | Raw integer | Founding year as a proxy for company maturity |

### Clustering Algorithm

1. **StandardScaler** normalizes all four features to zero mean, unit variance
2. **KMeans** clustering with `k` chosen by **Silhouette Score** maximization over `k ∈ [2, 8]`
3. Companies with insufficient metadata are assigned to the nearest cluster by curve distance (sparse assignment)

### Industry-Weighted Clustering

Industry is the primary signal for determining "what kind of company is this." After scaling, the `industry_enc` column is multiplied by **`INDUSTRY_WEIGHT = 2.0`**, making it dominate cluster formation. This ensures that tech companies cluster together, finance companies cluster together, etc., rather than clustering being driven by volume data (how many LeetCode problems happen to be tagged to that company).

The volume data (`days_until` × difficulty counts from `dataset_aggregated.csv`) is **not** used for clustering — it only feeds the PCHIP curve fitting within each cluster.

### PCHIP Curve Fitting

Within each cluster, **Piecewise Cubic Hermite Interpolating Polynomial (PCHIP)** curves are fitted to model how problem counts change with preparation time:

- **Overall proportion curve:** `f(days) → proportion of total problems` (anchored at 0 for day 0, monotonically increasing)
- **Per-difficulty percentage curves:** `f(days) → % of easy/medium/hard` within that cluster

At prediction time: `predict_counts(company, days)` → `{easy: N, medium: M, hard: H}` by:
1. Finding the company's cluster
2. Interpolating the cluster's PCHIP curves at the given day count
3. Scaling by the company's maximum problem count
4. Defaulting to `{easy: 3, medium: 5, hard: 2}` if no cluster data exists

### Unknown Company Handling

When a user enters a company not in our LeetCode dataset:

1. **Parquet lookup:** `company_lookup.py` performs a Polars lazy scan of the 7M company Parquet, matching by name (exact → prefix → substring)
2. **Feature encoding:** The matched company's metadata is encoded into the same 4-feature vector format
3. **Cluster prediction:** `predict_cluster_for_unknown()` scales the features with the fitted `StandardScaler`, applies `INDUSTRY_WEIGHT`, and assigns to the nearest cluster center by Euclidean distance
4. **Persistence:** The new company's metadata is appended to `company_meta.csv` so it's remembered for future requests

### Peer Company Retrieval

`get_peer_company_names(company, top_n)` returns the most structurally similar companies by computing the Euclidean distance between the query company's scaled feature vector and all other companies' vectors. This is used by the retriever to source problems from similar companies when the target company has insufficient problems.

---

## Embedding Pipeline

### Model

**`google/embeddinggemma-300m`** — a 300M parameter transformer model from Google's Gemma family, specialized for text embedding.

### Architecture

The embedding pipeline in `embeddings/src/pipeline/embedder.py`:

1. **Tokenization:** `AutoTokenizer` with `max_length=2048`, padding and truncation enabled
2. **Forward pass:** `AutoModel` inference with **`torch.no_grad()`** (no gradient tracking for pure inference efficiency)
3. **Mean pooling:** Token embeddings are averaged using the attention mask to exclude padding tokens — this produces a single fixed-size vector per input, weighted by actual content
4. **L2 normalization:** Vectors are normalized to unit length so that **dot product equals cosine similarity**, enabling efficient retrieval without explicit cosine computation
5. **Batching:** Inputs are processed in configurable batches (`batch_size: 8` to avoid MPS out-of-memory on Apple Silicon)
6. **Device selection:** Automatic fallback: CUDA → MPS (Apple Silicon) → CPU

### Vector Storage

Two storage backends are used:

| Backend | Collection | Purpose |
|---------|-----------|---------|
| **ChromaDB** (persistent) | `problems` | Problem retrieval for roadmap recommendations |
| **ChromaDB** (persistent) | `solutions` | Reference solution retrieval for solver few-shot injection |
| **FAISS** (legacy) | — | Original similarity search backend (still available but ChromaDB is primary) |

**ChromaDB indexing format:** Each problem is embedded as `"{company} | {title} | {topics} | {difficulty}"` with metadata including `company_id`, `difficulty`, `frequency`, `title`, and `url`.

---

## Retrieval-Augmented Generation (RAG)

### ChromaDB Collections

| Collection | Contents | Indexed By | Used For |
|-----------|----------|-----------|----------|
| `problems` | ~11K LeetCode problems | Company + title + topics + difficulty | Roadmap problem selection |
| `solutions` | Reference Python solutions | Problem description embedding | Few-shot solver prompt injection |

### Problem Retrieval Strategy

`ChromaRetriever.retrieve_by_company(company_name, difficulty, n_results, peer_company_ids)`:

1. **Company-specific search:** Query `"{company} | {difficulty}"` with `where` filter on `company_id` AND `difficulty`
2. **Peer company expansion:** If results are insufficient, search across peer companies (from cluster analysis) using `company_id $in [peer_ids]`, capped at 50 results
3. **Global fallback:** If still short, broaden to all problems at the same difficulty level using semantic similarity on the problem text
4. **Merge & deduplicate:** Results from all three stages are merged, deduplicated by both ChromaDB `id` and problem `url`

### Frequency-Based Ranking

After retrieval, results are sorted by the `frequency` metadata field (descending). Problems that appear more frequently in real interviews are prioritized higher in the roadmap. This ensures the user practices the most commonly asked problems first.

### URL-Based Deduplication

Problems are deduplicated by `url` as the canonical identifier. This prevents the same logical problem from appearing multiple times in a roadmap even if it's associated with different companies (e.g., "Two Sum" tagged to both Google and Amazon).

### Solution Retrieval (Few-Shot Injection)

`ChromaRetriever.retrieve_solutions(problem_description, n_results=2)`:

- Embeds the problem description and queries the `solutions` collection
- Returns the top-2 most similar reference solutions
- These are injected into the solver's prompt as `--- Reference Solution 1 ---` / `--- Reference Solution 2 ---` blocks
- Provides the solver with concrete examples of working solutions to similar problems

---

## Agentic Socratic Tutor

### Tutor Architecture

The tutor follows an **agent architecture** with distinct components:

```
User message
     │
     ▼
┌─────────────┐     ┌──────────────┐     ┌────────────────┐
│ SocraticTutor│────►│ GemmaSolver  │────►│SolverEvaluator │
│ (OpenRouter) │     │ (HF API/Local)│     │ (GEval judge)  │
│              │     │ + RAG few-shot│     │ + ast.parse    │
└──────┬───────┘     └──────────────┘     └────────────────┘
       │                                          │
       ▼                                          ▼
┌──────────────┐     ┌──────────────┐     ┌────────────────┐
│TutorEvaluator│────►│TutorOptimizer│────►│ socratic_system │
│ (GEval judge)│     │ (Llama 70B)  │     │   .txt on disk  │
└──────────────┘     └──────────────┘     └────────────────┘
```

### End-to-End Flow

1. **User sends a message** via `POST /api/tutor/chat` with `problem_id` and conversation history
2. **Problem loading:** The `InterviewProblem` is fetched from SQLite, providing the problem description
3. **Expert solution generation:** `GemmaSolver.solve(problem_description)` produces a ground-truth solution (cached per problem). The solver:
   - Retrieves 2 similar solved problems from ChromaDB for **few-shot context**
   - Generates a solution via HF Inference API or local Gemma + PEFT adapter
   - **Self-evaluates** with `SolverEvaluator` (compilation check + GEval correctness)
   - Retries up to 3 times if score < 0.8, keeping the best attempt
4. **System prompt assembly:** Loads `socratic_system.txt` + appends the problem statement + appends the secret ground truth solution
5. **Tutor response:** OpenRouter chat completion (`temperature=0.7`, `max_tokens=512`) with the full conversation history
6. **Post-hoc evaluation:** `TutorEvaluator` scores the response on Socratic quality (0.0–1.0)
7. **Meta-prompting:** If score < 0.4, the `TutorOptimizer` rewrites `socratic_system.txt` on disk to prevent similar failures in future conversations

### Expert Solver (Hybrid)

The solver operates in two modes controlled by `TUTOR_MODE`:

| Mode | Backend | Model | Latency | Cost |
|------|---------|-------|---------|------|
| `api` | HuggingFace Inference API | `google/gemma-2-2b-it` | ~2-5s | Free tier |
| `local` | Local PEFT adapter | Gemma 2B + QLoRA adapter | ~5-15s | GPU/MPS only |

Both modes receive the same prompts (`solver_system.txt` / `solver_user.txt` / `solver_local.txt`) and benefit from RAG few-shot injection.

### Solver Self-Evaluation Loop

```
solve(problem) ──► generate solution ──► SolverEvaluator
                        ▲                      │
                        │                      ▼
                        │               score ≥ 0.8? ──► return solution
                        │                      │
                        │                 score < 0.8
                        │                      │
                        └── retry (up to 3x) ◄─┘
```

The evaluator uses a **two-stage gate:**
1. **`ast.parse`** — instant syntax/compilation check. If the code doesn't parse, score is 0.0 immediately (no LLM call)
2. **GEval** — LLM judge (Llama 70B) evaluates logic, edge cases, efficiency, and correctness

---

## Reinforcement Learning from Human Feedback (RLHF)

### DPO Training

The project uses **Direct Preference Optimization (DPO)** rather than traditional RLHF with a reward model. DPO directly optimizes the language model to prefer "chosen" responses over "rejected" ones without needing a separate reward model.

**Implementation:** `tutor/train_rlhf.py` using TRL's `DPOTrainer`.

### QLoRA Configuration

| Parameter | Value | Purpose |
|-----------|-------|---------|
| Base model | `google/gemma-2-2b-it` | 2B instruction-tuned Gemma |
| Quantization | 4-bit NF4 with double quantization | Reduces memory from ~8GB to ~2GB |
| LoRA rank | `r=8` | Low-rank adaptation dimension |
| LoRA alpha | `16` | Scaling factor (alpha/rank = 2) |
| Target modules | `q_proj`, `v_proj` | Attention query and value projections |
| Dropout | `0.05` | LoRA dropout for regularization |
| Learning rate | `5e-5` | DPO optimization learning rate |
| Gradient accumulation | `8` steps | Effective batch size of 8 with batch_size=1 |
| Max steps | `100` | Training steps |

### Preference Pair Generation

Preference pairs are generated **synthetically** from the model itself:

1. Fetch 10 real problem descriptions from the SQLite database
2. For each problem, generate **2 candidate solutions** with `temperature=0.9`, `do_sample=True`
3. Score both with `compute_solver_reward` (AST compilation check)
4. **Pair construction:**
   - One compiles, one doesn't → chosen = compiling, rejected = non-compiling
   - Both compile → chosen = shorter code (conciseness as quality proxy)
   - Both fail → skip (no useful signal)

### Reward Signal

The reward is **binary ±1** based on Python AST compilation:
- `+1` if `ast.parse(code)` succeeds (syntactically valid Python)
- `-1` if parsing fails (syntax error)

With a **tie-breaker** on code length: among two compiling solutions, the shorter one is preferred, encoding a bias toward concise, elegant solutions.

**Output:** Fine-tuned adapter saved to `./gemma_dsa_dpo_final/`.

---

## Evaluation with DeepEval

The project uses **DeepEval's GEval** (LLM-as-a-judge) framework for automated evaluation of both the tutor's pedagogical responses and the solver's generated code.

### Socratic Quality (GEval)

**Metric name:** `"Socratic Quality"`
**Judge model:** `meta-llama/Llama-3.1-8B-Instruct` (via OpenRouter)
**Threshold:** 0.7

**Evaluation criteria (6 pillars):**

| Pillar | What it checks |
|--------|---------------|
| Non-disclosure | Does the response avoid revealing the solution or working code? |
| Relevance | Is it specific to this problem and the student's last message? |
| Helpfulness | Does it provide a meaningful nudge that unblocks the student? |
| Pedagogical balance | Does it confirm correct reasoning, not just ask questions? |
| Hallucination | Does it contain factually incorrect claims (wrong complexity, fake APIs)? |
| Toxicity & tone | Is it respectful, patient, and non-condescending? |

**Scoring:** 0 (reveals code or hallucinates), 0.5 (vague/unhelpful), 1.0 (perfect Socratic nudge)

### Code Logic Correctness (GEval)

**Metric name:** `"Code Logic Correctness"`
**Judge model:** `meta-llama/llama-3.1-70b-instruct` (via OpenRouter)
**Threshold:** 0.8

**Evaluation criteria (5 pillars):**

| Pillar | What it checks |
|--------|---------------|
| Logic | Does the algorithm correctly address the problem? (mental trace with example) |
| Edge cases | Handles empty input, single element, negatives, duplicates? |
| Efficiency | Optimal complexity for interview context? |
| Correctness | Subtle bugs: off-by-one, missing returns, wrong operators? |
| Hallucinated APIs | Uses only real Python functions and standard library methods? |

**Scoring:** 1 (correct and optimal) or 0 (fundamentally wrong or uses non-existent APIs)

### Two-Stage Solver Gating

The solver evaluation uses a **cheap-then-expensive** two-stage approach:

1. **Stage 1 — `ast.parse`** (free, instant): If the generated code doesn't compile, return score 0.0 immediately. This avoids wasting an LLM judge call on obviously broken output.
2. **Stage 2 — GEval** (LLM call, ~2s): Only if the code compiles, the 70B Llama judge evaluates semantic correctness.

---

## Meta-Prompting (Self-Improving Prompts)

### Closed-Loop Optimization

The system implements **automated prompt repair** — a feedback loop where poor tutor responses trigger an automatic rewrite of the system prompt:

```
Tutor response
     │
     ▼
TutorEvaluator ──► score ≥ 0.4? ──► done (prompt is fine)
                        │
                   score < 0.4
                        │
                        ▼
              TutorOptimizer (Llama 70B)
                        │
                 analyze failure ──► rewrite prompt
                        │
                        ▼
              save_prompt("socratic_system", new_text)
              (overwrites socratic_system.txt on disk)
```

### Optimizer Model

The optimizer uses **`meta-llama/llama-3.1-70b-instruct`** (70B parameters) — deliberately more capable than the tutor's own model — to perform prompt engineering. It follows a structured 4-step Chain of Thought:

1. **Root Cause:** Identify which specific rule was violated
2. **Pattern:** Is this a one-off or structural weakness?
3. **Targeted Fix:** Draft a minimal change that prevents this failure class
4. **Verify:** Mentally simulate the failing scenario with the revised prompt

The optimizer is instructed to make **minimal, targeted changes** to avoid over-correction (fixing one rule while breaking another).

---

## Prompting Strategies

### Tree of Thought (Socratic Tutor)

The main tutor system prompt uses **Tree of Thought (ToT)** reasoning. Before each response, the model is instructed to silently consider three teaching strategies:

- **Branch A:** Ask a conceptual question highlighting the key insight
- **Branch B:** Propose a simpler sub-problem that builds toward the full solution
- **Branch C:** Use an analogy or real-world example to reframe the problem

The model evaluates which branch best matches the student's current understanding level and responds accordingly. Decision heuristics: stuck → Branch B, right idea but wrong direction → Branch A, lacks foundational intuition → Branch C.

### Chain of Thought (Solver)

The solver prompts use **Chain of Thought (CoT)** reasoning:

1. Identify the problem type (graph traversal, DP, two pointers, etc.)
2. Consider at least two algorithmic approaches with time/space complexity
3. Choose the most optimal correct approach
4. Implement clean Python 3 code

The solver is instructed to reason internally but **only output the final code** in triple backticks.

### Few-Shot CoT (Evaluators)

Both evaluator prompts combine **few-shot examples** with **Chain of Thought** reasoning:

- **Socratic Quality evaluator:** 5 worked examples (scores 1, 0, 0.5, 0 for hallucination, 0 for toxicity) with full reasoning traces showing how to evaluate each criterion
- **Code Correctness evaluator:** 3 worked examples (score 1, score 0 for logic bug, score 0 for hallucinated API) with step-by-step analysis

The few-shot examples calibrate the judge model's scoring and demonstrate the expected analysis depth.

### Few-Shot CoT (Optimizer)

The optimizer prompt includes a concrete worked example showing:
- A real failure scenario (tutor reveals code when student says "I give up")
- The 4-step analysis (root cause → pattern → fix → verification)
- The resulting prompt change

### Prompt Extraction Architecture

All prompts are extracted into `tutor/prompts/` as separate `.txt` files with a Python loader module:

```
tutor/prompts/
├── __init__.py                 # load_prompt(), render_prompt(), save_prompt()
├── socratic_system.txt         # Socratic tutor system prompt (ToT + rules)
├── problem_context.txt         # "--- PROBLEM ---" template
├── ground_truth.txt            # "--- GROUND TRUTH SOLUTION ---" template
├── optimizer_meta.txt          # Meta-prompt for rewriting system prompt
├── failure_analysis.txt        # Failure report template for optimizer
├── solver_system.txt           # Expert solver system message (CoT + example)
├── solver_user.txt             # Solver user message template
├── solver_local.txt            # Local Gemma solver prompt template
├── few_shot_preamble.txt       # RAG few-shot intro text
├── eval_socratic_quality.txt   # GEval criteria (6 pillars + 5 examples)
└── eval_code_correctness.txt   # GEval criteria (5 pillars + 3 examples)
```

**API:** `load_prompt(name)` for static prompts, `render_prompt(name, **kwargs)` for templates with `{placeholders}`, `save_prompt(name, content)` for the optimizer's prompt rewriting. All files are cached after first load.

---

## Toxicity & Hallucination Handling

### Anti-Hallucination Guards

Applied at three levels:

**1. Tutor system prompt (generation-time):**
- Only state facts about algorithms and complexity that are certain
- Do not invent problem constraints not in the problem statement
- Do not reference non-existent Python functions or standard library modules
- Stay in scope — redirect off-topic questions back to the problem
- Never fabricate example inputs/outputs — verify traces step by step

**2. Solver system prompt (generation-time):**
- Only use real Python standard library functions and built-in methods
- Do not assume constraints not stated in the problem
- Verify complexity claims match the actual algorithm

**3. Evaluator criteria (evaluation-time):**
- **Socratic evaluator Step 5:** Checks for wrong complexity claims, non-existent Python functions, invented constraints
- **Code evaluator Step 5:** Checks for hallucinated APIs (non-existent methods like `Counter.most_frequent()`)
- Hallucination is a **score-0 failure** — same severity as revealing the solution

### Toxicity & Tone Controls

**Tutor system prompt rules:**
- Maintain a patient, encouraging, professional tone at all times — even after repeated wrong answers
- Never mock or belittle a student's approach, even if far from optimal
- If the student sends inappropriate/offensive messages, calmly redirect without engaging
- Never use language implying the student should already know something ("this is obvious", "as you should know")
- Never express frustration, impatience, or judgment about skill level

**Evaluator enforcement:**
- **Socratic evaluator Step 6:** Checks for condescending language, sarcasm, dismissiveness
- Includes a few-shot example of a toxic response (score 0) where the tutor says "you should know this by now"
- Toxicity is a **score-0 failure** — triggers meta-prompting to fix the system prompt

### Evaluation-Level Enforcement

The combination of generation-time rules and evaluation-time checks creates a **defense-in-depth** approach:

1. **Prevention:** Prompt rules instruct the model what NOT to do
2. **Detection:** GEval judges score responses on hallucination and toxicity criteria
3. **Correction:** If score < 0.4, the optimizer rewrites the system prompt to be more robust

---

## Models Used

| Component | Model | Parameters | Provider | Purpose |
|-----------|-------|------------|----------|---------|
| Tutor chat | `google/gemma-3-4b-it:free` | 4B | OpenRouter | Socratic dialogue generation |
| Expert solver | `google/gemma-2-2b-it` | 2B | HF Inference API or local | Ground-truth solution generation |
| Fine-tuned adapter | Gemma 2B + QLoRA (DPO) | 2B + LoRA | Local PEFT | Improved code generation |
| Socratic judge | `meta-llama/Llama-3.1-8B-Instruct` | 8B | OpenRouter | GEval tutor response evaluation |
| Code judge | `meta-llama/llama-3.1-70b-instruct` | 70B | OpenRouter | GEval code correctness evaluation |
| Prompt optimizer | `meta-llama/llama-3.1-70b-instruct` | 70B | OpenRouter | Meta-prompting system prompt rewriting |
| Text embeddings | `google/embeddinggemma-300m` | 300M | Local (HF download) | Vector embeddings for ChromaDB |

**Grading tiers (per project requirements):**
- Using commercial models → max grade 8
- Using SLMs (not fine-tuned) → max grade 9
- **Using SLMs (fine-tuned) → max grade 10** ← this project uses DPO fine-tuning on Gemma 2B

---

## Application Layer

### Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | FastAPI (Python) |
| Templates | Jinja2 server-side rendering |
| Database | SQLite via SQLAlchemy ORM |
| Auth | bcrypt password hashing + JWT in HTTP-only cookies |
| Server | Uvicorn (ASGI) |

### Database Schema

```
users                interviews              companies
┌──────────────┐     ┌──────────────────┐    ┌──────────────┐
│ id       PK  │──┐  │ id           PK  │ ┌──│ id       PK  │
│ email        │  └─>│ user_id      FK  │ │  │ name         │
│ hashed_pwd   │     │ company_id   FK  │<┘  └──────────────┘
└──────────────┘     │ days_until       │         │
                     │ interview_date   │         │
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

### API Routes

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Landing page |
| `GET/POST` | `/login` | Authentication |
| `GET/POST` | `/register` | User registration |
| `GET` | `/dashboard` | Interview list with progress tracking |
| `POST` | `/dashboard/add` | Create interview + generate roadmap via RAG |
| `POST` | `/dashboard/delete/{id}` | Delete interview and its roadmap |
| `GET` | `/interview/{id}/roadmap` | Problem checklist with completion toggles |
| `POST` | `/interview/{id}/roadmap/{rid}/toggle` | Toggle problem completion |
| `GET` | `/interview/{id}/problem/{pid}` | Problem detail + tutor chat panel |
| `POST` | `/api/tutor/chat` | Send message to Socratic tutor (JSON API) |

---

## Setup & Scripts

### Prerequisites

```bash
pip install -r requirements.txt
cp .env.example .env  # Fill in API keys
```

### Data Pipeline

| Step | Command | Description |
|------|---------|-------------|
| 1 | `python scripts/enrich_dataset.py` | Scrape problem descriptions from LeetCode GraphQL API (resumable) |
| 2 | `python scripts/process_distributions.py` | Aggregate difficulty distributions per company/time bucket |
| 3 | `python scripts/ingest_companies.py` | Convert Kaggle CSV → Parquet, match LeetCode companies to Kaggle metadata |
| 4 | `python scripts/seed_database.py` | Populate SQLite with companies and problems |
| 5 | `python scripts/build_vector_index.py` | Build ChromaDB `problems` collection with embeddings |
| 6 | `python scripts/build_solution_index.py` | Build ChromaDB `solutions` collection from fine-tuning data |

### Run the Application

```bash
cd application
uvicorn main:app --reload
```

### Train the DPO Adapter

```bash
python -m tutor.train_rlhf
```

### Clustering Visualization

```bash
jupyter notebook notebooks/clustering_analysis.ipynb
```

Produces elbow/silhouette plots, cluster composition analysis, PCHIP difficulty curves, PCA projections, and feature importance charts.
