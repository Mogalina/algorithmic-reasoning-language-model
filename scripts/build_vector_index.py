"""
Build the vector index from problems in the database.

Embeds all InterviewProblem rows using the embeddings module and stores
them in a vector store (ChromaDB) for RAG retrieval.

Usage:
    cd application
    python ../scripts/build_vector_index.py

Prerequisites:
    - Run seed_database.py first to populate problems
    - Install: pip install chromadb torch transformers
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
APP_DIR = PROJECT_ROOT / "application"
EMBEDDINGS_SRC = PROJECT_ROOT / "embeddings" / "src"

sys.path.insert(0, str(APP_DIR))
sys.path.insert(0, str(EMBEDDINGS_SRC))

import os
from decouple import Config, RepositoryEnv

# Load environment variables from .env
env_path = PROJECT_ROOT / ".env"
if env_path.exists():
    env = Config(RepositoryEnv(str(env_path)))
    hf_token = env.get("HF_TOKEN", default=None)
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        print("HF_TOKEN loaded from .env")

from database import SessionLocal
from models import InterviewProblem
from pipeline import Embedder
import chromadb
import numpy as np

def build_index():
    # 1. Load all InterviewProblem rows from the database
    db = SessionLocal()
    problems = db.query(InterviewProblem).all()
    
    if not problems:
        print("No problems found in the database. Please run seed_database.py first.")
        db.close()
        return

    # 2. Initialize Embedder
    # By default it looks for config/config.yaml relative to embeddings/src/
    # which we've added to sys.path.
    print("Initializing Embedder...")
    embedder = Embedder()

    # 3. Initialize ChromaDB client
    chroma_path = str(PROJECT_ROOT / "data" / "chromadb")
    print(f"Connecting to ChromaDB at {chroma_path}...")
    client = chromadb.PersistentClient(path=chroma_path)
    
    # Create or get collection
    collection_name = "problems"
    collection = client.get_or_create_collection(name=collection_name)

    # 4. Embed and store in batches
    batch_size = 64
    total = len(problems)
    print(f"Starting to index {total} problems into ChromaDB...")

    for i in range(0, total, batch_size):
        batch = problems[i : i + batch_size]
        
        # Build text representations: "{company} | {title} | {topics} | {difficulty}"
        # Fetching company names for the batch
        texts = []
        for p in batch:
            company_name = p.company.name if p.company else "Unknown"
            texts.append(f"{company_name} | {p.title} | {p.topics or ''} | {p.difficulty}")
        
        # Embed them
        vectors = embedder.embed(texts)
        
        # Store vectors + metadata
        # ChromaDB expects embeddings as a list of lists
        collection.add(
            ids=[str(p.id) for p in batch],
            embeddings=vectors.tolist(),
            metadatas=[{
                "company_id": p.company_id,
                "difficulty": p.difficulty,
                "frequency": p.frequency or 0.0,
                "title": p.title,
                "url": p.url
            } for p in batch],
            documents=[p.description[:1000] if p.description else "" for p in batch] # Store a snippet/full description
        )
        print(f"Indexed {min(i + batch_size, total)}/{total} problems...")

    print(f"Successfully indexed {total} problems into ChromaDB collection '{collection_name}'.")
    db.close()

if __name__ == "__main__":
    try:
        build_index()
    except Exception as e:
        print(f"Error building vector index: {e}")
        sys.exit(1)
