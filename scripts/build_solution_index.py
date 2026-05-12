import sys
import os
import json
from pathlib import Path
import chromadb
from decouple import Config, RepositoryEnv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
APP_DIR = PROJECT_ROOT / "application"
EMBEDDINGS_SRC = PROJECT_ROOT / "embeddings" / "src"

sys.path.insert(0, str(APP_DIR))
sys.path.insert(0, str(EMBEDDINGS_SRC))

# Load environment variables
env_path = PROJECT_ROOT / ".env"
if env_path.exists():
    env = Config(RepositoryEnv(str(env_path)))
    hf_token = env.get("HF_TOKEN", default=None)
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token

from pipeline import Embedder

def build_solution_index():
    # 1. Load the training data with solutions
    train_path = PROJECT_ROOT / "dataset" / "fine-tune" / "train.jsonl"
    if not train_path.exists():
        print(f"Error: {train_path} not found.")
        return

    print(f"Loading data from {train_path}...")
    data = []
    with open(train_path, "r") as f:
        for line in f:
            data.append(json.loads(line))

    # 2. Initialize Embedder
    print("Initializing Embedder...")
    embedder = Embedder()

    # 3. Initialize ChromaDB
    chroma_path = str(PROJECT_ROOT / "data" / "chromadb")
    client = chromadb.PersistentClient(path=chroma_path)
    
    # Create a separate collection for solutions
    collection_name = "solutions"
    try:
        client.delete_collection(collection_name)
    except:
        pass
    collection = client.create_collection(name=collection_name)

    # 4. Embed and store
    batch_size = 64
    total = len(data)
    print(f"Indexing {total} solution pairs into ChromaDB...")

    for i in range(0, total, batch_size):
        batch = data[i : i + batch_size]
        
        # We embed the PROMPT (problem description) so we can find matches based on the query
        prompts = [item["prompt"] for item in batch]
        vectors = embedder.embed(prompts)
        
        collection.add(
            ids=[f"sol_{i+j}" for j in range(len(batch))],
            embeddings=vectors.tolist(),
            # We store the SOLUTION in the documents so it can be retrieved
            documents=[item["completion"] for item in batch],
            metadatas=[{"type": "reference_solution"} for _ in batch]
        )
        print(f"Indexed {min(i + batch_size, total)}/{total}...")

    print(f"Successfully built solution index in collection '{collection_name}'.")

if __name__ == "__main__":
    build_solution_index()
