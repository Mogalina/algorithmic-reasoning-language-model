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

# TODO: Implement once ChromaDB is added as a dependency
#
# Intended flow:
#   1. Load all InterviewProblem rows from the database
#   2. Build text representations: "{title} | {topics} | {difficulty}"
#   3. Embed them in batches using embeddings.src.pipeline.Embedder
#   4. Store vectors + metadata (problem ID, company, difficulty) in ChromaDB
#   5. Persist the ChromaDB collection to disk at data/chromadb/
#
# Example:
#   from database import SessionLocal
#   from models import InterviewProblem
#   from pipeline import Embedder
#   import chromadb
#
#   db = SessionLocal()
#   problems = db.query(InterviewProblem).all()
#   embedder = Embedder()
#
#   client = chromadb.PersistentClient(path=str(PROJECT_ROOT / "data" / "chromadb"))
#   collection = client.get_or_create_collection("problems")
#
#   batch_size = 64
#   for i in range(0, len(problems), batch_size):
#       batch = problems[i : i + batch_size]
#       texts = [f"{p.title} | {p.topics or ''} | {p.difficulty}" for p in batch]
#       vectors = embedder.embed(texts)
#       collection.add(
#           ids=[str(p.id) for p in batch],
#           embeddings=vectors.tolist(),
#           metadatas=[{
#               "company_id": p.company_id,
#               "difficulty": p.difficulty,
#               "title": p.title,
#           } for p in batch],
#       )
#
#   print(f"Indexed {len(problems)} problems into ChromaDB.")

if __name__ == "__main__":
    print("build_vector_index.py is not yet implemented.")
    print("See the TODO comments in this file for the intended flow.")
