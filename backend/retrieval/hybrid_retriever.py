"""
Hybrid Retriever

Orchestrates vector retrieval and graph retrieval
to provide a unified retrieval interface.
"""

from typing import Dict, List

from backend.retrieval.vector_retriever import VectorRetriever
from backend.retrieval.graph_retriever import GraphRetriever
from backend.retrieval.context_fusion import ContextFusion


class HybridRetriever:
    """
    Main retrieval interface for LexMindAI.
    """

    def __init__(self):

        print("\nInitializing Hybrid Retriever...\n")

        self.vector_retriever = VectorRetriever()

        self.graph_retriever = GraphRetriever()

        self.context_fusion = ContextFusion()

        print("\nHybrid Retriever Ready!\n")

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        min_score: float = 0.70,
        max_hops: int = 2,
    ) -> List[Dict]:
        """
        Perform Hybrid Retrieval.

        Workflow

        Query
            │
            ▼
        Vector Retrieval
            │
            ▼
        Graph Retrieval
            │
            ▼
        Context Fusion
        """

        # -----------------------------
        # Vector Retrieval
        # -----------------------------

        vector_results = self.vector_retriever.retrieve(
            query=query,
            top_k=top_k,
            min_score=min_score
        )

        # -----------------------------
        # Graph Retrieval
        # -----------------------------

        graph_results = self.graph_retriever.retrieve(
            query=query,
            max_hops=max_hops
        )

        # -----------------------------
        # Context Fusion
        # -----------------------------

        fused_context = self.context_fusion.fuse(
            vector_results=vector_results,
            graph_results=graph_results
        )

        return fused_context