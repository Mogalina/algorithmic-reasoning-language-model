import sys
import os
from pathlib import Path
import chromadb
from decouple import Config, RepositoryEnv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EMBEDDINGS_SRC = PROJECT_ROOT / "embeddings" / "src"

# Ensure embeddings/src is in path for imports
if str(EMBEDDINGS_SRC) not in sys.path:
    sys.path.insert(0, str(EMBEDDINGS_SRC))

from pipeline import Embedder

class ChromaRetriever:
    """
    Handles retrieval from the ChromaDB vector store.
    """
    def __init__(self, collection_name: str = "problems"):
        # Load environment variables for gated models
        env_path = PROJECT_ROOT / ".env"
        if env_path.exists():
            env = Config(RepositoryEnv(str(env_path)))
            hf_token = env.get("HF_TOKEN", default=None)
            if hf_token:
                os.environ["HF_TOKEN"] = hf_token

        # Initialize Embedder (expects config relative to embeddings/src/)
        self.embedder = Embedder()
        
        # Initialize ChromaDB client
        chroma_path = str(PROJECT_ROOT / "data" / "chromadb")
        self.client = chromadb.PersistentClient(path=chroma_path)
        self.collection = self.client.get_or_create_collection(name=collection_name)
        # Solutions collection for few-shot RAG
        self.solutions_collection = self.client.get_or_create_collection(name="solutions")

    def retrieve_solutions(self, query_text: str, n_results: int = 2):
        """
        Search for similar problem-solution pairs in the training dataset.
        """
        query_vector = self.embedder.embed(query_text).tolist()[0]
        results = self.solutions_collection.query(
            query_embeddings=[query_vector],
            n_results=n_results
        )
        return results

    def retrieve_by_company(
        self,
        company_id: int,
        company_name: str,
        difficulty: str,
        n_results: int = 5,
        peer_company_ids: list[int] | None = None,
    ):
        """
        Search for problems for a specific company and difficulty.

        When the company's own pool is too small, falls back to problems
        from peer companies (same cluster) rather than a generic global
        search.  If no peer_company_ids are provided, falls back to the
        original global semantic search.
        """
        query = f"{company_name} | {difficulty}"
        
        # primary search: strict filter by company and difficulty
        query_vector = self.embedder.embed(query).tolist()[0]
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=n_results,
            where={
                "$and": [
                    {"company_id": company_id},
                    {"difficulty": difficulty}
                ]
            }
        )
        
        # fallback if not enough results
        found_ids = results.get("ids", [[]])[0]
        if len(found_ids) >= n_results:
            self._sort_by_frequency(results)
            return results

        remaining = n_results - len(found_ids)

        # cluster-aware fallback: search across peer companies
        if peer_company_ids:
            peer_results = self._search_peer_companies(
                query_vector, difficulty, peer_company_ids,
                remaining + len(found_ids), found_ids,
            )
            if peer_results:
                self._merge_results(results, peer_results, found_ids, n_results)

        # if still not enough, fall back to global difficulty search
        found_ids = results.get("ids", [[]])[0]
        if len(found_ids) < n_results:
            remaining = n_results - len(found_ids)

            seed_text = query
            if results.get("documents") and results["documents"][0]:
                seed_text += " | " + " | ".join(results["documents"][0])
            
            fallback_vector = self.embedder.embed(seed_text).tolist()[0]
            fallback_results = self.collection.query(
                query_embeddings=[fallback_vector],
                n_results=remaining + len(found_ids), 
                where={"difficulty": difficulty}
            )
            self._merge_results(results, fallback_results, found_ids, n_results)

        self._sort_by_frequency(results)
        return results

    def _search_peer_companies(
        self,
        query_vector: list[float],
        difficulty: str,
        peer_company_ids: list[int],
        n_results: int,
        exclude_ids: list[str],
    ) -> dict | None:
        """Query ChromaDB for problems from peer companies."""
        if not peer_company_ids:
            return None

        # ChromaDB $in filter has a practical limit; cap the peer list
        capped_peers = peer_company_ids[:50]

        try:
            return self.collection.query(
                query_embeddings=[query_vector],
                n_results=n_results,
                where={
                    "$and": [
                        {"company_id": {"$in": capped_peers}},
                        {"difficulty": difficulty},
                    ]
                },
            )
        except Exception as e:
            print(f"[ChromaRetriever] Peer search failed (non-fatal): {e}")
            return None

    @staticmethod
    def _sort_by_frequency(results: dict):
        """Re-order results so highest-frequency problems come first."""
        ids = results.get("ids", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        docs = results.get("documents", [[]])[0]

        if not ids:
            return

        triples = list(zip(ids, metas, docs))
        triples.sort(key=lambda t: t[1].get("frequency", 0), reverse=True)

        results["ids"][0] = [t[0] for t in triples]
        results["metadatas"][0] = [t[1] for t in triples]
        results["documents"][0] = [t[2] for t in triples]

    @staticmethod
    def _merge_results(
        target: dict, source: dict, existing_ids: list[str], limit: int,
    ):
        """Merge source results into target, avoiding duplicates by ID and title."""
        src_ids = source.get("ids", [[]])[0]
        src_metas = source.get("metadatas", [[]])[0]
        src_docs = source.get("documents", [[]])[0]

        existing_urls = {
            (m.get("url") or "").lower()
            for m in target.get("metadatas", [[]])[0]
            if m.get("url")
        }

        for i, fid in enumerate(src_ids):
            if len(existing_ids) >= limit:
                break
            if fid in existing_ids:
                continue
            url = (src_metas[i].get("url") or "").lower()
            if url and url in existing_urls:
                continue
            target["ids"][0].append(fid)
            target["metadatas"][0].append(src_metas[i])
            target["documents"][0].append(src_docs[i])
            existing_ids.append(fid)
            if url:
                existing_urls.add(url)

    def retrieve_similar_problems(self, problem_text: str, n_results: int = 3):
        """
        Find problems similar to the given text.
        """
        query_vector = self.embedder.embed(problem_text).tolist()[0]
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=n_results
        )
        return results

_retriever = None

def get_retriever() -> ChromaRetriever:
    global _retriever
    if _retriever is None:
        _retriever = ChromaRetriever()
    return _retriever
