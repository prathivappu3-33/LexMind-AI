"""
Evaluation Metrics

Implements standard Information Retrieval metrics for
evaluating LexMindAI.

Metrics:
- Precision@K
- Recall@K
- MRR
- NDCG
"""

from __future__ import annotations

import math
from typing import Iterable, List, Set


class RetrievalMetrics:
    """
    Computes retrieval evaluation metrics.
    """

    @staticmethod
    def precision_at_k(
        retrieved: List[str],
        relevant: Set[str],
        k: int,
    ) -> float:
        """
        Precision@K
        """

        if k <= 0:
            return 0.0

        retrieved = retrieved[:k]

        hits = sum(
            1
            for item in retrieved
            if item in relevant
        )

        return hits / len(retrieved) if retrieved else 0.0

    @staticmethod
    def recall_at_k(
        retrieved: List[str],
        relevant: Set[str],
        k: int,
    ) -> float:
        """
        Recall@K
        """

        if not relevant:
            return 0.0

        retrieved = retrieved[:k]

        hits = sum(
            1
            for item in retrieved
            if item in relevant
        )

        return hits / len(relevant)

    @staticmethod
    def mean_reciprocal_rank(
        retrieved: Iterable[str],
        relevant: Set[str],
    ) -> float:
        """
        Mean Reciprocal Rank (MRR)
        """

        for rank, item in enumerate(retrieved, start=1):

            if item in relevant:
                return 1.0 / rank

        return 0.0

    @staticmethod
    def ndcg_at_k(
        retrieved: List[str],
        relevant: Set[str],
        k: int,
    ) -> float:
        """
        Normalized Discounted Cumulative Gain
        """

        retrieved = retrieved[:k]

        dcg = 0.0

        for i, item in enumerate(retrieved):

            rel = 1 if item in relevant else 0

            dcg += rel / math.log2(i + 2)

        ideal_hits = min(len(relevant), k)

        if ideal_hits == 0:
            return 0.0

        idcg = sum(
            1 / math.log2(i + 2)
            for i in range(ideal_hits)
        )

        return dcg / idcg