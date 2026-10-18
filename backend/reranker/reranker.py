"""
Reranker

Ranks retrieved contexts using the BGE Reranker model.
"""

from typing import Dict, List

from backend.reranker.loader import get_reranker


class Reranker:
    """
    Reranks hybrid retrieval results.
    """

    def __init__(self):

        self.model = get_reranker()

    def rerank(
        self,
        query: str,
        contexts: List[Dict],
        top_k: int = 10,
    ) -> List[Dict]:
        """
        Rerank retrieved contexts.

        Parameters
        ----------
        query : str
            User query.

        contexts : List[Dict]
            Fused contexts from HybridRetriever.

        top_k : int
            Number of contexts to return.
        """

        if not contexts:
            return []

        passages = [
            context["content"]
            for context in contexts
        ]

        pairs = [
            (query, passage)
            for passage in passages
        ]

        scores = self.model.predict(pairs)

        ranked_contexts = []

        for context, score in zip(contexts, scores):

            context["score"] = float(score)

            ranked_contexts.append(context)

        ranked_contexts.sort(
            key=lambda x: x["score"],
            reverse=True,
        )

        return ranked_contexts[:top_k]