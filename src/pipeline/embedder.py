from pathlib import Path
from typing import Union

import os
import numpy as np
import torch

from transformers import AutoModel, AutoTokenizer, logging

from utils.logger import get_logger
from utils.config import load_config


# Suppress Hugging Face output
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
logging.set_verbosity_error()
logging.disable_progress_bar()

logger = get_logger(__name__)


class Embedder:
    """
    Local text embedder using embedding transformer.

    Uses mean pooling with L2 normalization so embeddings work with FAISS
    IndexFlatIP treated as cosine similarity.
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        config = load_config(config_path)
        embedder_config = config["embedding"]

        # Model parameters
        local_dir = Path(embedder_config["local_model_dir"])
        model_id = embedder_config["model_id"]
        self.batch_size = embedder_config.get("batch_size", 32)

        # Model loading
        model_source = str(local_dir) \
            if local_dir.exists() and any(local_dir.iterdir()) \
            else model_id

        # Model initialization
        logger.info(f"Loading model from: {model_source}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_source)
        self.model = AutoModel.from_pretrained(model_source)
        self.model.eval()

        # Device selection
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
            
        self.model.to(self.device)
        logger.info(f"Device: {self.device}")

    def _mean_pool(
        self, 
        token_embeddings: torch.Tensor, 
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Mean pool token embeddings with attention mask.

        Args:
            token_embeddings: Token embeddings from the model
            attention_mask: Attention mask from the tokenizer

        Returns:
            torch.Tensor of shape (batch_size, dim), dtype float32, L2-normalized
        """
        # Apply attention mask to token embeddings for mean pooling
        mask = attention_mask \
            .unsqueeze(-1) \
            .expand(token_embeddings.size()) \
            .float()
        
        # Perform mean pooling with attention mask to handle padding
        sum_embeddings = torch.sum(token_embeddings * mask, dim=1)
        sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
        
        # Return L2-normalized embeddings
        return sum_embeddings / sum_mask

    def embed(self, texts: Union[str, list[str]]) -> np.ndarray:
        """
        Embed one or more texts into L2-normalized float32 vectors.

        Args:
            texts: A string or list of strings

        Returns:
            np.ndarray of shape (N, dim), dtype float32, L2-normalized
        """
        # Ensure input is a list even if a single string is provided
        if isinstance(texts, str):
            texts = [texts]

        all_embeddings = []
        
        # Process the input texts in batches to prevent out-of-memory errors
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            
            # Tokenize the current batch, padding shorter texts and truncating longer ones
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=2048,
                return_tensors="pt",
            ).to(self.device)

            # Perform the forward pass without tracking gradients
            with torch.no_grad():
                outputs = self.model(**encoded)

            # Apply mean pooling using the attention mask to combine token 
            # embeddings into a single sentence vector
            pooled = self._mean_pool(outputs.last_hidden_state, encoded["attention_mask"])
            pooled = pooled.cpu().float().numpy()

            # Perform L2 normalization for cosine similarity compatibility within FAISS
            norms = np.linalg.norm(pooled, axis=1, keepdims=True)
            pooled = pooled / np.maximum(norms, 1e-9)
            
            all_embeddings.append(pooled)

        # Concatenate all batch results into a single array
        return np.vstack(all_embeddings)
