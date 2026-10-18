from typing import Dict, List

from backend.retrieval.hybrid_retriever import HybridRetriever
from backend.reranker.reranker import Reranker


class RetrievalPipeline:
    """
    End-to-end retrieval pipeline for LexMindAI.

    Workflow

    Query
        │
        ▼
    Hybrid Retrieval
        │
        ▼
    Context Fusion
        │
        ▼
    BGE Reranker
        │
        ▼
    Ranked Contexts
    """

    def __init__(self):

        print("\nInitializing Retrieval Pipeline...\n")

        self.hybrid_retriever = HybridRetriever()

        self.reranker = Reranker()

        print("\nRetrieval Pipeline Ready!\n")

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        min_score: float = 0.70,
        max_hops: int = 2,
    ) -> List[Dict]:

        contexts = self.hybrid_retriever.retrieve(
            query=query,
            top_k=top_k,
            min_score=min_score,
            max_hops=max_hops,
        )

        ranked_contexts = self.reranker.rerank(
            query=query,
            contexts=contexts,
            top_k=top_k,
        )

        print("\n========== RANKED CONTEXTS ==========\n")

        

        return ranked_contexts