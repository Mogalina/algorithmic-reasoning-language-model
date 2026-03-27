import pickle
from pathlib import Path

import faiss
import numpy as np

from utils.logger import get_logger
from utils.config import load_config, PROJECT_ROOT


logger = get_logger(__name__)


class Searcher:
    """
    Handles similarity search in the Facebook AI Similarity Search (FAISS)
    index.
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize the Searcher.

        Args:
            config_path: Path to the configuration file
        """
        self.config = load_config(config_path)
        database_config = self.config["database"]["faiss"]

        # Index paths
        self.index_directory = PROJECT_ROOT / database_config["path"]
        self.index_path = self.index_directory / "index.faiss"
        self.metadata_path = self.index_directory / "metadata.pkl"
        
        self.index = None
        self.metadata = None

    def _load(self) -> None:
        """
        Lazy load index and metadata.

        Raises:
            FileNotFoundError: If the index or metadata files are not found
        """
        # Load index
        if self.index is None:
            if not self.index_path.exists():
                raise FileNotFoundError("Index not found")
            self.index = faiss.read_index(str(self.index_path))
        
        # Load metadata
        if self.metadata is None:
            if not self.metadata_path.exists():
                raise FileNotFoundError("Metadata store not found")
            with open(self.metadata_path, "rb") as f:
                self.metadata = pickle.load(f)

    def search(
        self, 
        query_vector: np.ndarray, 
        top_k: int = 5
    ) -> list[dict]:
        """
        Search for the most similar chunks.

        Args:
            query_vector: The query vector
            top_k: The number of results to return

        Returns:
            A list of the most similar chunks
        """
        # Load index and metadata
        self._load()

        # Perform search        
        distances, indices = self.index.search(query_vector.astype("float32"), top_k)
        
        # Format results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            # Skip invalid indices
            if idx < 0 or idx >= len(self.metadata):
                continue
            
            entry = self.metadata[idx].copy()
            entry["distance"] = float(dist)
            results.append(entry)
        
        return results
