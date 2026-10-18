"""
Confidence Engine for LexMindAI.
"""

from backend.confidence.confidence_models import (
    ConfidenceBreakdown,
    ConfidenceResult,
)


class ConfidenceEngine:
    """
    Computes confidence for generated answers.
    """

    def calculate(self, contexts: list[dict]) -> ConfidenceResult:

        if not contexts:
            return ConfidenceResult(
                overall=0.0,
                level="Very Low",
                breakdown=ConfidenceBreakdown(
                    vector_similarity=0.0,
                    reranker=0.0,
                    source_agreement=0.0,
                    graph_support=0.0,
                ),
            )

        # -------------------------------
        # Vector Similarity
        # -------------------------------

        vector_scores = [
            c.get("metadata", {}).get("score", 0.0)
            for c in contexts
            if c.get("type") == "vector"
        ]

        vector_similarity = (
            sum(vector_scores) / len(vector_scores)
            if vector_scores
            else 0.0
        )

        # -------------------------------
        # Reranker Score
        # -------------------------------

        reranker_scores = [
            c.get("score", 0.0)
            for c in contexts
        ]

        reranker = (
            sum(reranker_scores) / len(reranker_scores)
            if reranker_scores
            else 0.0
        )

        # -------------------------------
        # Source Agreement
        # -------------------------------

        source_agreement = min(
            len(vector_scores) / 5.0,
            1.0,
        )

        # -------------------------------
        # Graph Support
        # -------------------------------

        graph_support = (
            1.0
            if any(c.get("type") == "graph" for c in contexts)
            else 0.0
        )

        # -------------------------------
        # Weighted Confidence
        # -------------------------------

        overall = (
            vector_similarity * 0.40
            + reranker * 0.30
            + source_agreement * 0.20
            + graph_support * 0.10
        )

        overall *= 100

        if overall >= 90:
            level = "Very High"
        elif overall >= 75:
            level = "High"
        elif overall >= 60:
            level = "Medium"
        elif overall >= 40:
            level = "Low"
        else:
            level = "Very Low"

        return ConfidenceResult(
            overall=round(overall, 2),
            level=level,
            breakdown=ConfidenceBreakdown(
                vector_similarity=round(vector_similarity * 100, 2),
                reranker=round(reranker * 100, 2),
                source_agreement=round(source_agreement * 100, 2),
                graph_support=round(graph_support * 100, 2),
            ),
        )