"""
LexMindAI Retrieval Benchmark Evaluator

Evaluates ranked retrieval results against the frozen,
human-verified LexMindAI ground truth.

Metrics:
    - Precision@K
    - Recall@K
    - F1@K
    - Hit Rate@K
    - Reciprocal Rank / MRR
    - DCG@K
    - nDCG@K

Important:
Ground-truth relevance judgments were created from a manually
reviewed candidate pool. Therefore, results should be interpreted
as pooled/judged benchmark metrics unless the relevance judgments
are later expanded to the full corpus.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set


# =============================================================================
# Data Model
# =============================================================================


@dataclass
class QuestionEvaluation:
    """
    Metrics for one benchmark question.
    """

    question_id: Any
    question: str

    retrieved_count: int
    relevant_count: int

    retrieved_chunk_ids: List[str]
    relevant_chunk_ids: List[str]

    metrics: Dict[str, float]


# =============================================================================
# Evaluator
# =============================================================================


class RetrievalEvaluator:
    """
    Evaluate ranked retrieval results against human ground truth.

    Expected ground-truth entry:

        {
            "question_id": 1,
            "question": "What is Docker?",
            "relevant_chunk_ids": [...]
        }

    Expected retrieval result entry:

        {
            "question_id": 1,
            "question": "What is Docker?",
            "retrieved_chunks": [
                {
                    "chunk_id": "...",
                    "score": 0.82
                },
                ...
            ]
        }

    The evaluator also accepts ranked chunk IDs directly.
    """

    def __init__(
        self,
        ground_truth: Sequence[Dict[str, Any]],
        k_values: Sequence[int] = (1, 3, 5, 10),
    ) -> None:

        if not ground_truth:
            raise ValueError(
                "Ground truth cannot be empty."
            )

        cleaned_k = sorted(
            {
                int(k)
                for k in k_values
                if int(k) > 0
            }
        )

        if not cleaned_k:
            raise ValueError(
                "At least one positive K value is required."
            )

        self.k_values = cleaned_k

        self.ground_truth: Dict[str, Dict[str, Any]] = {}

        for entry in ground_truth:

            question_id = str(
                entry.get("question_id")
            )

            if question_id in self.ground_truth:
                raise ValueError(
                    f"Duplicate ground-truth question_id: "
                    f"{question_id}"
                )

            relevant_chunk_ids = entry.get(
                "relevant_chunk_ids",
                []
            )

            if not isinstance(
                relevant_chunk_ids,
                list
            ):
                raise ValueError(
                    f"relevant_chunk_ids must be a list "
                    f"for question {question_id}"
                )

            self.ground_truth[question_id] = {
                "question_id": entry.get(
                    "question_id"
                ),
                "question": entry.get(
                    "question",
                    ""
                ),
                "relevant_chunk_ids": {
                    str(chunk_id)
                    for chunk_id in relevant_chunk_ids
                },
            }

    # =========================================================================
    # Basic Metric Helpers
    # =========================================================================

    @staticmethod
    def precision_at_k(
        retrieved: Sequence[str],
        relevant: Set[str],
        k: int,
    ) -> float:
        """
        Precision@K.

        Number of relevant chunks in the first K positions
        divided by K.

        If fewer than K results were returned, the denominator
        remains K. This penalizes systems that fail to return
        enough results.
        """

        if k <= 0:
            return 0.0

        top_k = list(retrieved[:k])

        relevant_retrieved = sum(
            1
            for chunk_id in top_k
            if chunk_id in relevant
        )

        return relevant_retrieved / k

    @staticmethod
    def recall_at_k(
        retrieved: Sequence[str],
        relevant: Set[str],
        k: int,
    ) -> float:
        """
        Recall@K.

        Number of relevant chunks retrieved in the first K
        positions divided by the total number of known relevant
        chunks for the question.
        """

        if not relevant:
            return 0.0

        top_k = retrieved[:k]

        relevant_retrieved = sum(
            1
            for chunk_id in top_k
            if chunk_id in relevant
        )

        return (
            relevant_retrieved
            /
            len(relevant)
        )

    @staticmethod
    def f1_score(
        precision: float,
        recall: float,
    ) -> float:
        """
        Harmonic mean of precision and recall.
        """

        if precision + recall == 0:
            return 0.0

        return (
            2
            * precision
            * recall
            /
            (precision + recall)
        )

    @staticmethod
    def hit_rate_at_k(
        retrieved: Sequence[str],
        relevant: Set[str],
        k: int,
    ) -> float:
        """
        Hit Rate@K.

        Returns 1 if at least one relevant result occurs
        in the first K positions, otherwise 0.
        """

        top_k = retrieved[:k]

        return float(
            any(
                chunk_id in relevant
                for chunk_id in top_k
            )
        )

    @staticmethod
    def reciprocal_rank(
        retrieved: Sequence[str],
        relevant: Set[str],
    ) -> float:
        """
        Reciprocal rank of the first relevant result.
        """

        for rank, chunk_id in enumerate(
            retrieved,
            start=1,
        ):

            if chunk_id in relevant:
                return 1.0 / rank

        return 0.0

    # =========================================================================
    # DCG / nDCG
    # =========================================================================

    @staticmethod
    def dcg_at_k(
        retrieved: Sequence[str],
        relevant: Set[str],
        k: int,
    ) -> float:
        """
        Binary-relevance DCG@K.

        relevance = 1 if chunk is relevant, otherwise 0.
        """

        score = 0.0

        for rank, chunk_id in enumerate(
            retrieved[:k],
            start=1,
        ):

            relevance = (
                1.0
                if chunk_id in relevant
                else 0.0
            )

            if relevance == 0:
                continue

            score += (
                relevance
                /
                math.log2(rank + 1)
            )

        return score

    @staticmethod
    def ideal_dcg_at_k(
        relevant_count: int,
        k: int,
    ) -> float:
        """
        Ideal binary-relevance DCG@K.
        """

        ideal_hits = min(
            relevant_count,
            k,
        )

        score = 0.0

        for rank in range(
            1,
            ideal_hits + 1,
        ):

            score += (
                1.0
                /
                math.log2(rank + 1)
            )

        return score

    @classmethod
    def ndcg_at_k(
        cls,
        retrieved: Sequence[str],
        relevant: Set[str],
        k: int,
    ) -> float:
        """
        Normalized Discounted Cumulative Gain.
        """

        if not relevant:
            return 0.0

        dcg = cls.dcg_at_k(
            retrieved,
            relevant,
            k,
        )

        idcg = cls.ideal_dcg_at_k(
            len(relevant),
            k,
        )

        if idcg == 0:
            return 0.0

        return dcg / idcg

    # =========================================================================
    # Retrieval Normalization
    # =========================================================================

    @staticmethod
    def extract_chunk_ids(
        retrieved_chunks: Iterable[Any],
    ) -> List[str]:
        """
        Convert retrieval output into ranked chunk IDs.

        Supported forms:

            ["id1", "id2", ...]

        or

            [
                {"chunk_id": "id1", "score": ...},
                {"chunk_id": "id2", "score": ...}
            ]

        Duplicate chunk IDs are removed while preserving
        the first occurrence/ranking.
        """

        chunk_ids: List[str] = []

        seen: Set[str] = set()

        for item in retrieved_chunks:

            chunk_id: Optional[Any]

            if isinstance(item, str):
                chunk_id = item

            elif isinstance(item, dict):
                chunk_id = item.get(
                    "chunk_id"
                )

            else:
                chunk_id = getattr(
                    item,
                    "chunk_id",
                    None,
                )

            if chunk_id is None:
                continue

            chunk_id = str(
                chunk_id
            )

            if chunk_id in seen:
                continue

            seen.add(
                chunk_id
            )

            chunk_ids.append(
                chunk_id
            )

        return chunk_ids

    # =========================================================================
    # Single Question Evaluation
    # =========================================================================

    def evaluate_question(
        self,
        question_id: Any,
        retrieved_chunks: Iterable[Any],
    ) -> QuestionEvaluation:
        """
        Evaluate one ranked retrieval result.
        """

        normalized_question_id = str(
            question_id
        )

        if (
            normalized_question_id
            not in self.ground_truth
        ):
            raise KeyError(
                f"Question ID "
                f"{normalized_question_id} "
                f"not found in ground truth."
            )

        truth = self.ground_truth[
            normalized_question_id
        ]

        relevant: Set[str] = truth[
            "relevant_chunk_ids"
        ]

        retrieved = self.extract_chunk_ids(
            retrieved_chunks
        )

        metrics: Dict[str, float] = {}

        for k in self.k_values:

            precision = self.precision_at_k(
                retrieved,
                relevant,
                k,
            )

            recall = self.recall_at_k(
                retrieved,
                relevant,
                k,
            )

            f1 = self.f1_score(
                precision,
                recall,
            )

            hit_rate = self.hit_rate_at_k(
                retrieved,
                relevant,
                k,
            )

            ndcg = self.ndcg_at_k(
                retrieved,
                relevant,
                k,
            )

            metrics[
                f"precision@{k}"
            ] = precision

            metrics[
                f"recall@{k}"
            ] = recall

            metrics[
                f"f1@{k}"
            ] = f1

            metrics[
                f"hit_rate@{k}"
            ] = hit_rate

            metrics[
                f"ndcg@{k}"
            ] = ndcg

        metrics[
            "reciprocal_rank"
        ] = self.reciprocal_rank(
            retrieved,
            relevant,
        )

        return QuestionEvaluation(
            question_id=truth[
                "question_id"
            ],
            question=truth[
                "question"
            ],
            retrieved_count=len(
                retrieved
            ),
            relevant_count=len(
                relevant
            ),
            retrieved_chunk_ids=retrieved,
            relevant_chunk_ids=sorted(
                relevant
            ),
            metrics=metrics,
        )

    # =========================================================================
    # Full Benchmark Evaluation
    # =========================================================================

    def evaluate(
        self,
        retrieval_results: Sequence[
            Dict[str, Any]
        ],
    ) -> Dict[str, Any]:
        """
        Evaluate all benchmark questions.

        Expected input:

            [
                {
                    "question_id": 1,
                    "retrieved_chunks": [...]
                },
                ...
            ]
        """

        result_index: Dict[
            str,
            Dict[str, Any]
        ] = {}

        for result in retrieval_results:

            question_id = str(
                result.get(
                    "question_id"
                )
            )

            if question_id in result_index:
                raise ValueError(
                    f"Duplicate retrieval result "
                    f"for question_id "
                    f"{question_id}"
                )

            result_index[
                question_id
            ] = result

        evaluations: List[
            QuestionEvaluation
        ] = []

        missing_questions: List[str] = []

        for (
            question_id,
            truth
        ) in self.ground_truth.items():

            retrieval_result = (
                result_index.get(
                    question_id
                )
            )

            if retrieval_result is None:

                missing_questions.append(
                    question_id
                )

                retrieved_chunks = []

            else:

                retrieved_chunks = (
                    retrieval_result.get(
                        "retrieved_chunks"
                    )
                )

                if retrieved_chunks is None:
                    retrieved_chunks = (
                        retrieval_result.get(
                            "chunks",
                            []
                        )
                    )

            evaluation = (
                self.evaluate_question(
                    question_id,
                    retrieved_chunks,
                )
            )

            evaluations.append(
                evaluation
            )

        aggregate_metrics = (
            self.aggregate(
                evaluations
            )
        )

        return {
            "benchmark": {
                "questions": len(
                    self.ground_truth
                ),
                "evaluated_questions": len(
                    evaluations
                ),
                "missing_retrieval_results": (
                    len(
                        missing_questions
                    )
                ),
                "missing_question_ids": (
                    missing_questions
                ),
                "k_values": self.k_values,
            },
            "aggregate_metrics": (
                aggregate_metrics
            ),
            "per_question": [
                {
                    "question_id":
                        evaluation.question_id,

                    "question":
                        evaluation.question,

                    "retrieved_count":
                        evaluation.retrieved_count,

                    "relevant_count":
                        evaluation.relevant_count,

                    "metrics":
                        {
                            key: round(
                                value,
                                6,
                            )
                            for key, value
                            in evaluation.metrics.items()
                        },

                    "retrieved_chunk_ids":
                        evaluation.retrieved_chunk_ids,

                    "relevant_chunk_ids":
                        evaluation.relevant_chunk_ids,
                }
                for evaluation in evaluations
            ],
        }

    # =========================================================================
    # Aggregate Metrics
    # =========================================================================

    def aggregate(
        self,
        evaluations: Sequence[
            QuestionEvaluation
        ],
    ) -> Dict[str, float]:
        """
        Macro-average metrics across benchmark questions.
        """

        if not evaluations:
            return {}

        aggregate_metrics: Dict[
            str,
            float
        ] = {}

        metric_names = list(
            evaluations[0].metrics.keys()
        )

        for metric_name in metric_names:

            values = [
                evaluation.metrics[
                    metric_name
                ]
                for evaluation in evaluations
            ]

            aggregate_metrics[
                metric_name
            ] = round(
                mean(values),
                6,
            )

        # Reciprocal rank averaged across questions
        # is Mean Reciprocal Rank.

        aggregate_metrics[
            "mrr"
        ] = aggregate_metrics.pop(
            "reciprocal_rank"
        )

        return aggregate_metrics