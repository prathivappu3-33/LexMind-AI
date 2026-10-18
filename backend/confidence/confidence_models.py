"""
Confidence models for LexMindAI.
"""

from pydantic import BaseModel


class ConfidenceBreakdown(BaseModel):
    """Individual confidence components."""

    vector_similarity: float
    reranker: float
    source_agreement: float
    graph_support: float


class ConfidenceResult(BaseModel):
    """Overall confidence result."""

    overall: float
    level: str
    breakdown: ConfidenceBreakdown