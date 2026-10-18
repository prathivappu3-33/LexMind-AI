"""
Benchmark Runner

Runs retrieval benchmarks on LexMindAI and computes
standard Information Retrieval metrics.
"""

from __future__ import annotations

import time
from statistics import mean
from typing import Dict, List

from backend.evaluation.metrics import RetrievalMetrics


class BenchmarkRunner:
    """
    Executes benchmark experiments.
    """

    def __init__(self, retriever):
        """
        Parameters
        ----------
        retriever
            Any retriever implementing:

                retrieve(query)
        """
        self.retriever = retriever

    def evaluate(
        self,
        benchmark_dataset: List[Dict],
        k: int = 5,
    ) -> Dict:
        """
        Evaluate the retriever on a benchmark dataset.

        Dataset format:

        [
            {
                "question": "...",
                "relevant": {"chunk1","chunk2"}
            }
        ]
        """

        precision_scores = []
        recall_scores = []
        mrr_scores = []
        ndcg_scores = []
        latency_scores = []

        for sample in benchmark_dataset:

            question = sample["question"]
            relevant = sample["relevant"]

            start = time.perf_counter()

            results = self.retriever.retrieve(question)

            latency = time.perf_counter() - start

            retrieved = []

            for result in results:

                if "text" in result:
                    retrieved.append(result["text"])

                elif "content" in result:
                    retrieved.append(result["content"])

            precision_scores.append(
                RetrievalMetrics.precision_at_k(
                    retrieved,
                    relevant,
                    k,
                )
            )

            recall_scores.append(
                RetrievalMetrics.recall_at_k(
                    retrieved,
                    relevant,
                    k,
                )
            )

            mrr_scores.append(
                RetrievalMetrics.mean_reciprocal_rank(
                    retrieved,
                    relevant,
                )
            )

            ndcg_scores.append(
                RetrievalMetrics.ndcg_at_k(
                    retrieved,
                    relevant,
                    k,
                )
            )

            latency_scores.append(latency)

        return {
            "Precision@K": round(mean(precision_scores), 4),
            "Recall@K": round(mean(recall_scores), 4),
            "MRR": round(mean(mrr_scores), 4),
            "NDCG": round(mean(ndcg_scores), 4),
            "Average Latency (s)": round(mean(latency_scores), 4),
        }