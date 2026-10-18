"""
Response metadata models for LexMindAI.
"""

from pydantic import BaseModel


class ResponseMetadata(BaseModel):
    """
    Metadata about the generated response.
    """

    retrieval_time_ms: float
    reranking_time_ms: float
    generation_time_ms: float
    total_time_ms: float

    contexts_used: int
    vector_contexts: int
    graph_contexts: int

    model: str
    retriever: str
    reranker: str