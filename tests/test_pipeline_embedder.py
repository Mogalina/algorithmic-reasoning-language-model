import pytest
import numpy as np
import torch

from unittest.mock import MagicMock, patch
from pipeline.embedder import Embedder


class MockEncoded(dict):
    def to(self, device):
        return self


@pytest.fixture
def mock_transformers():
    with patch("pipeline.embedder.AutoTokenizer.from_pretrained") as mock_tokenizer_load, \
         patch("pipeline.embedder.AutoModel.from_pretrained") as mock_model_load:
        
        # Mock tokenizer behaviors
        mock_tokenizer = MagicMock()
        
        def tokenizer_side_effect(batch, **kwargs):
            batch_size = len(batch) if isinstance(batch, list) else 1
            encoded = MockEncoded({
                "input_ids": torch.zeros((batch_size, 10), dtype=torch.long),
                "attention_mask": torch.ones((batch_size, 10), dtype=torch.long)
            })
            return encoded
            
        mock_tokenizer.side_effect = tokenizer_side_effect
        mock_tokenizer_load.return_value = mock_tokenizer

        # Mock model behaviors
        mock_model = MagicMock()

        def model_side_effect(**kwargs):
            input_ids = kwargs.get("input_ids")
            batch_size = input_ids.shape[0] if input_ids is not None else 1
            mock_outputs = MagicMock()
            mock_outputs.last_hidden_state = torch.zeros((batch_size, 10, 768))
            return mock_outputs

        mock_model.side_effect = model_side_effect
        mock_model_load.return_value = mock_model
        
        yield {
            "load_tokenizer": mock_tokenizer_load,
            "load_model": mock_model_load,
            "tokenizer": mock_tokenizer,
            "model": mock_model,
        }


@pytest.fixture
def dummy_config(tmp_path):
    # Create a dummy config file
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True)
    config_file = config_dir / "config.yaml"
    config_content = """
    embedding:
      model_id: "dummy/model"
      local_model_dir: "./dummy_models"
      batch_size: 2
    """
    
    config_file.write_text(config_content)
    return config_file


def test_embedder_init(dummy_config, mock_transformers, tmp_path):
    with patch("utils.config.PROJECT_ROOT", tmp_path):
        embedder = Embedder(config_path=str(dummy_config))
        assert embedder.batch_size == 2
        assert mock_transformers["load_tokenizer"].called
        assert mock_transformers["load_model"].called


def test_embedder_mean_pool():
    # Create simple tensors
    # Batch size 1, sequence length 2, dim 4
    token_embeddings = torch.tensor([[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]])
    
    # Attention mask
    attention_mask = torch.tensor([[1, 0]])
    
    # Manually call mean pooling through the class
    result = Embedder._mean_pool(None, token_embeddings, attention_mask)
    
    expected = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    assert torch.allclose(result, expected)


def test_embed(dummy_config, mock_transformers, tmp_path):
    with patch("utils.config.PROJECT_ROOT", tmp_path):
        embedder = Embedder(config_path=str(dummy_config))
        
        # Test with single text
        embeddings = embedder.embed("hello world")
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape[0] == 1
        
        # Test with list of texts
        embeddings = embedder.embed(["hello", "world"])
        assert embeddings.shape[0] == 2
