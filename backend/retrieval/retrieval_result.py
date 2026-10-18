"""
Retrieval Result

Defines the common data model returned by every retriever in
LexMindAI.

Every retrieval component (Vector, Document Graph, Memory Graph,
future BM25, Web Search, SQL, etc.) should return this object.

This provides a unified interface for Context Fusion,
Fusion Engine, Cross Encoder, and Prompt Builder.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass(slots=True)
class RetrievalResult:
    """
    Standard retrieval result used throughout the retrieval pipeline.
    """

    # ---------------------------------------------------------
    # Identity
    # ---------------------------------------------------------

    source: str
    """
    Origin of the retrieval result.

    Examples:
        vector
        document_graph
        memory_graph
    """

    result_type: str
    """
    Type of retrieved information.

    Examples:
        document
        graph
        memory
    """

    # ---------------------------------------------------------
    # Content
    # ---------------------------------------------------------

    content: str
    """
    Text passed downstream to the reranker and LLM.
    """

    # ---------------------------------------------------------
    # Scores
    # ---------------------------------------------------------

    score: float = 0.0
    """
    Original retrieval score.
    """

    normalized_score: float = 0.0
    """
    Score after normalization.
    """

    fusion_score: float = 0.0
    """
    Final score assigned by the Fusion Engine.
    """

    # ---------------------------------------------------------
    # Metadata
    # ---------------------------------------------------------

    metadata: Dict[str, Any] = field(default_factory=dict)
    """
    Retriever-specific metadata.
    """

    # ---------------------------------------------------------
    # Utility
    # ---------------------------------------------------------

    def unique_key(self) -> str:
        """
        Returns a unique identifier used for duplicate removal.
        """

        chunk_id = self.metadata.get("chunk_id")

        if chunk_id:
            return str(chunk_id)

        graph = self.metadata.get("graph")

        if graph:

            return (
                f"{graph.get('source','')}|"
                f"{graph.get('relation','')}|"
                f"{graph.get('target','')}"
            )

        return self.content