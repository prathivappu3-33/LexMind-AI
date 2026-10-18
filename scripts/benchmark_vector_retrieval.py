"""
LexMindAI Vector Retrieval Benchmark

Runs the frozen benchmark questions through LexMindAI's
VectorRetriever and evaluates the ranked results against
the human-verified ground truth.

Metrics:
    Precision@K
    Recall@K
    F1@K
    Hit Rate@K
    MRR
    nDCG@K

Output:
    datasets/evaluation/vector_retrieval_results.json
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List

from backend.evaluation.evaluator import RetrievalEvaluator
from backend.retrieval.vector_retriever import VectorRetriever


# =============================================================================
# Configuration
# =============================================================================

GROUND_TRUTH_PATH = Path(
    "datasets/ground_truth/ground_truth.json"
)

OUTPUT_PATH = Path(
    "datasets/evaluation/vector_retrieval_results.json"
)

K_VALUES = (
    1,
    3,
    5,
    10,
)

# Retrieve enough results for every metric.
TOP_K = max(K_VALUES)

# IMPORTANT:
# For benchmarking we do NOT want VectorRetriever's normal 0.70
# production threshold to remove results before evaluation.
#
# Ranking metrics should evaluate the ranked top-K list itself.
MIN_SCORE = -1.0


# =============================================================================
# Utilities
# =============================================================================

def load_json(
    path: Path,
) -> Any:

    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(file)


def save_json(
    path: Path,
    data: Any,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            data,
            file,
            indent=4,
            ensure_ascii=False,
        )


# =============================================================================
# Retrieval
# =============================================================================

def run_retrieval(
    retriever: VectorRetriever,
    ground_truth: List[Dict[str, Any]],
) -> tuple[List[Dict[str, Any]], Dict[str, float]]:

    retrieval_results: List[
        Dict[str, Any]
    ] = []

    latencies: List[float] = []

    total_questions = len(
        ground_truth
    )

    print(
        "\n" +
        "=" * 100
    )

    print(
        "RUNNING VECTOR RETRIEVAL"
    )

    print(
        "=" * 100
    )

    for index, item in enumerate(
        ground_truth,
        start=1,
    ):

        question_id = item[
            "question_id"
        ]

        question = item[
            "question"
        ]

        print(
            f"\n[{index}/{total_questions}] "
            f"Q{question_id}: {question}"
        )

        start_time = (
            time.perf_counter()
        )

        try:

            results = retriever.retrieve(
                query=question,
                top_k=TOP_K,
                min_score=MIN_SCORE,
            )

            error = None

        except Exception as exc:

            results = []

            error = (
                f"{type(exc).__name__}: "
                f"{exc}"
            )

        elapsed = (
            time.perf_counter()
            -
            start_time
        )

        latencies.append(
            elapsed
        )

        normalized_results = []

        for rank, result in enumerate(
            results,
            start=1,
        ):

            chunk_id = result.get(
                "chunk_id"
            )

            if chunk_id is not None:
                chunk_id = str(
                    chunk_id
                )

            normalized_results.append(
                {
                    "rank": rank,
                    "chunk_id": chunk_id,
                    "score": float(
                        result.get(
                            "score",
                            0.0,
                        )
                    ),
                    "document_id": str(
                        result.get(
                            "document_id",
                            "",
                        )
                    ),
                    "source": result.get(
                        "source"
                    ),
                    "title": result.get(
                        "title"
                    ),
                    "section_heading": (
                        result.get(
                            "section_heading"
                        )
                    ),
                }
            )

        retrieval_results.append(
            {
                "question_id":
                    question_id,

                "question":
                    question,

                "retrieved_chunks":
                    normalized_results,

                "retrieved_count":
                    len(
                        normalized_results
                    ),

                "latency_seconds":
                    round(
                        elapsed,
                        6,
                    ),

                "error":
                    error,
            }
        )

        print(
            f"    Retrieved : "
            f"{len(normalized_results)}"
        )

        print(
            f"    Latency   : "
            f"{elapsed:.3f}s"
        )

        if error:

            print(
                f"    ERROR     : "
                f"{error}"
            )

        else:

            for result in (
                normalized_results[:3]
            ):

                print(
                    f"    #{result['rank']:<2} "
                    f"{result['score']:.4f} | "
                    f"{result['chunk_id']}"
                )

    if latencies:

        latency_summary = {
            "mean_seconds": (
                sum(latencies)
                /
                len(latencies)
            ),
            "min_seconds": min(
                latencies
            ),
            "max_seconds": max(
                latencies
            ),
            "total_seconds": sum(
                latencies
            ),
        }

    else:

        latency_summary = {
            "mean_seconds": 0.0,
            "min_seconds": 0.0,
            "max_seconds": 0.0,
            "total_seconds": 0.0,
        }

    latency_summary = {
        key: round(
            value,
            6,
        )
        for key, value
        in latency_summary.items()
    }

    return (
        retrieval_results,
        latency_summary,
    )


# =============================================================================
# Reporting
# =============================================================================

def print_metrics(
    benchmark_result: Dict[str, Any],
    latency_summary: Dict[str, float],
) -> None:

    metrics = benchmark_result[
        "aggregate_metrics"
    ]

    print(
        "\n" +
        "=" * 100
    )

    print(
        "VECTOR RETRIEVAL BASELINE"
    )

    print(
        "=" * 100
    )

    print(
        f"{'Metric':<20}"
        f"{'Value':>12}"
    )

    print(
        "-" * 32
    )

    metric_order = []

    for k in K_VALUES:

        metric_order.extend(
            [
                f"precision@{k}",
                f"recall@{k}",
                f"f1@{k}",
                f"hit_rate@{k}",
                f"ndcg@{k}",
            ]
        )

    metric_order.append(
        "mrr"
    )

    for metric_name in metric_order:

        if metric_name not in metrics:
            continue

        print(
            f"{metric_name:<20}"
            f"{metrics[metric_name]:>12.4f}"
        )

    print(
        "\n" +
        "-" * 32
    )

    print(
        f"{'Mean latency':<20}"
        f"{latency_summary['mean_seconds']:>11.3f}s"
    )

    print(
        f"{'Total latency':<20}"
        f"{latency_summary['total_seconds']:>11.3f}s"
    )

    print(
        "=" * 100
    )


# =============================================================================
# Main
# =============================================================================

def main() -> None:

    print(
        "\n" +
        "=" * 100
    )

    print(
        "LexMindAI Vector Retrieval Benchmark"
    )

    print(
        "=" * 100
    )

    print(
        f"Ground truth : "
        f"{GROUND_TRUTH_PATH}"
    )

    print(
        f"Output       : "
        f"{OUTPUT_PATH}"
    )

    print(
        f"Top K        : "
        f"{TOP_K}"
    )

    print(
        f"K values     : "
        f"{K_VALUES}"
    )

    print(
        f"Min score    : "
        f"{MIN_SCORE} "
        f"(threshold disabled for benchmark)"
    )

    # -------------------------------------------------------------------------
    # Ground truth
    # -------------------------------------------------------------------------

    ground_truth = load_json(
        GROUND_TRUTH_PATH
    )

    print(
        f"Questions    : "
        f"{len(ground_truth)}"
    )

    # -------------------------------------------------------------------------
    # Initialize retriever
    # -------------------------------------------------------------------------

    print(
        "\nInitializing VectorRetriever..."
    )

    retriever = (
        VectorRetriever()
    )

    print(
        "VectorRetriever ready."
    )

    # -------------------------------------------------------------------------
    # Run retrieval
    # -------------------------------------------------------------------------

    (
        retrieval_results,
        latency_summary,
    ) = run_retrieval(
        retriever,
        ground_truth,
    )

    # -------------------------------------------------------------------------
    # Evaluate
    # -------------------------------------------------------------------------

    print(
        "\nEvaluating retrieval results..."
    )

    evaluator = RetrievalEvaluator(
        ground_truth=ground_truth,
        k_values=K_VALUES,
    )

    benchmark_result = (
        evaluator.evaluate(
            retrieval_results
        )
    )

    # -------------------------------------------------------------------------
    # Add experiment metadata
    # -------------------------------------------------------------------------

    output = {
        "experiment": {
            "name":
                "vector_retrieval_baseline",

            "retriever":
                "VectorRetriever",

            "top_k":
                TOP_K,

            "min_score":
                MIN_SCORE,

            "k_values":
                list(K_VALUES),

            "evaluation_scope":
                "human_verified_pooled_ground_truth",

            "note": (
                "Ground truth contains relevance "
                "judgments from the manually reviewed "
                "candidate pool. Metrics should be "
                "interpreted as pooled benchmark metrics."
            ),
        },

        "latency":
            latency_summary,

        "benchmark":
            benchmark_result[
                "benchmark"
            ],

        "aggregate_metrics":
            benchmark_result[
                "aggregate_metrics"
            ],

        "per_question":
            benchmark_result[
                "per_question"
            ],

        "raw_retrieval_results":
            retrieval_results,
    }

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------

    save_json(
        OUTPUT_PATH,
        output,
    )

    # -------------------------------------------------------------------------
    # Display
    # -------------------------------------------------------------------------

    print_metrics(
        benchmark_result,
        latency_summary,
    )

    print(
        f"\nResults saved to:\n"
        f"{OUTPUT_PATH}"
    )

    print(
        "\nVector baseline benchmark complete.\n"
    )


if __name__ == "__main__":
    main()