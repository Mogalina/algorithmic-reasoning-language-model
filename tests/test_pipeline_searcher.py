import pytest
import numpy as np
import pickle
from unittest.mock import MagicMock, patch
from pathlib import Path
from utils.config import load_config
from pipeline.searcher import Searcher


@pytest.fixture
def dummy_config(tmp_path):
    # Create a dummy configuration file
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True)
    config_file = config_dir / "config.yaml"
    config_content = """
    database:
      faiss:
        path: "faiss_data"
    """
    config_file.write_text(config_content)
    
    # Create the search directory within the temporary path
    database_dir = tmp_path / "faiss_data"
    database_dir.mkdir(parents=True)
    
    return config_file, database_dir


def test_searcher_init(dummy_config, tmp_path):
    config_file, database_dir = dummy_config
    load_config.cache_clear()

    with patch("utils.config.PROJECT_ROOT", tmp_path), \
         patch("pipeline.searcher.PROJECT_ROOT", tmp_path):
        searcher = Searcher(config_path=str(config_file))
        assert searcher.index_directory == tmp_path / "faiss_data"
        assert searcher.index_path == tmp_path / "faiss_data" / "index.faiss"
        assert searcher.metadata_path == tmp_path / "faiss_data" / "metadata.pkl"


def test_searcher_load_index(dummy_config, tmp_path):
    config_file, database_dir = dummy_config
    
    # Create dummy index and metadata files
    index_file = database_dir / "index.faiss"
    index_file.write_text("DUMMY INDEX DATA")
    
    metadata_file = database_dir / "metadata.pkl"
    with open(metadata_file, "wb") as f:
        pickle.dump([{"id": 1, "text": "hello world"}], f)
    
    load_config.cache_clear()

    with patch("utils.config.PROJECT_ROOT", tmp_path), \
         patch("pipeline.searcher.PROJECT_ROOT", tmp_path), \
         patch("faiss.read_index") as mock_read_index:
        mock_index = MagicMock()
        mock_read_index.return_value = mock_index
        
        searcher = Searcher(config_path=str(config_file))
        searcher._load()
        
        assert searcher.index == mock_index
        assert len(searcher.metadata) == 1
        assert searcher.metadata[0]["id"] == 1


def test_searcher_search(dummy_config, tmp_path):
    config_file, database_dir = dummy_config
    load_config.cache_clear()
    
    with patch("utils.config.PROJECT_ROOT", tmp_path), \
         patch("pipeline.searcher.PROJECT_ROOT", tmp_path), \
         patch("faiss.read_index") as mock_read_index:
        mock_index = MagicMock()
        
        # Mocking 2 results found for 1 query vector (distance, indices)
        mock_index.search.return_value = (np.array([[0.1, 0.5]]), np.array([[0, 1]]))
        mock_read_index.return_value = mock_index
        
        searcher = Searcher(config_path=str(config_file))
        searcher.metadata = [
            {"id": 1, "text": "first context"},
            {"id": 2, "text": "second context"}
        ]
        searcher.index = mock_index
        
        # Test search with dummy query vector
        query_vector = np.zeros((1, 10))
        results = searcher.search(query_vector, top_k=2)
        
        assert len(results) == 2
        assert results[0]["id"] == 1
        assert results[0]["distance"] == 0.1
        assert results[1]["id"] == 2
        assert results[1]["distance"] == 0.5
