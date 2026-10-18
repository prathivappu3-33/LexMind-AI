from sentence_transformers import SentenceTransformer
import numpy as np


MODEL_NAME = "BAAI/bge-base-en-v1.5"


class EmbeddingEncoder:

    def __init__(self):

        print(f"\nLoading embedding model: {MODEL_NAME}\n")

        self.model = SentenceTransformer(
            MODEL_NAME,
            trust_remote_code=True
        )

        print("Embedding model loaded successfully.\n")

    def encode(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.
        """

        return self.model.encode(
            text,
            normalize_embeddings=True
        )

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        """
        Generate embeddings for multiple texts.
        """

        return self.model.encode(
            texts,
            batch_size=64,
            normalize_embeddings=True,
            show_progress_bar=False
        )