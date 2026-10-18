"""
Reranker Loader

Loads the BGE Reranker model once and provides
a shared instance across the application.
"""

from sentence_transformers import CrossEncoder

MODEL_NAME = "BAAI/bge-reranker-base"

_model = None


def get_reranker():
    """
    Load the reranker model once.
    """

    global _model

    if _model is None:

        print(f"\nLoading BGE Reranker: {MODEL_NAME}\n")

        _model = CrossEncoder(
            MODEL_NAME,
            trust_remote_code=True,
        )

        print("BGE Reranker loaded successfully.\n")

    return _model