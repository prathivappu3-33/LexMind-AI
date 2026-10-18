"""
Context Fusion

Combines vector retrieval and graph retrieval results
into a unified ranked context.

The fusion strategy combines:

1. Reciprocal Rank Fusion (RRF)
2. Graph relevance score
3. Vector + graph provenance overlap
4. Chunk-level evidence merging

The goal is to make graph evidence reinforce relevant
vector evidence instead of appearing as an unrelated
second result.
"""

from __future__ import annotations

from typing import Dict, List, Set, Tuple


class ContextFusion:
    """
    Fuse vector and graph retrieval outputs.

    Vector and graph scores come from different scoring
    systems, so raw scores are not directly compared.

    RRF provides the base ranking signal.

    Additional signals are then applied:

    - graph relevance
    - vector/graph chunk overlap
    - provenance strength

    When a graph relationship points to the same chunk
    retrieved by the vector search, the two results are
    merged into one context item.
    """

    def __init__(
        self,
        vector_weight: float = 0.70,
        graph_weight: float = 0.30,
        rrf_k: int = 60,
    ):
        self.vector_weight = vector_weight
        self.graph_weight = graph_weight
        self.rrf_k = rrf_k

    # ------------------------------------------------------------------
    # RRF
    # ------------------------------------------------------------------

    def _rrf_score(
        self,
        rank: int,
        weight: float,
    ) -> float:
        """
        Calculate weighted Reciprocal Rank Fusion score.

        rank is 1-based.
        """
        return weight / (self.rrf_k + rank)

    # ------------------------------------------------------------------
    # Graph score normalization
    # ------------------------------------------------------------------

    def _normalize_graph_score(
        self,
        score: float,
    ) -> float:
        """
        Convert GraphRetriever's 0-100 relevance score
        into a small normalized fusion bonus.

        The graph score is intentionally NOT used as a raw
        replacement for RRF because vector and graph scores
        live on different scales.

        75 -> approximately 0.0125
        70 -> approximately 0.0117
        50 -> approximately 0.0083
        """
        score = max(0.0, min(float(score), 100.0))

        return (
            score / 100.0
        ) * 0.016

    # ------------------------------------------------------------------
    # Provenance bonus
    # ------------------------------------------------------------------

    def _provenance_bonus(
        self,
        vector_rank: int,
        graph_rank: int,
    ) -> float:
        """
        Bonus when vector and graph retrieval independently
        identify the same underlying chunk.

        Earlier ranks receive a slightly stronger bonus.
        """
        vector_strength = 1.0 / max(vector_rank, 1)
        graph_strength = 1.0 / max(graph_rank, 1)

        return (
            0.025
            + (vector_strength * 0.005)
            + (graph_strength * 0.005)
        )

    # ------------------------------------------------------------------
    # Main fusion
    # ------------------------------------------------------------------

    def fuse(
        self,
        vector_results: List[Dict],
        graph_results: List[Dict],
    ) -> List[Dict]:

        # ==============================================================
        # First collect vector results
        # ==============================================================

        vector_by_chunk: Dict[str, Dict] = {}

        seen_vector_chunks: Set[str] = set()

        for rank, result in enumerate(
            vector_results,
            start=1,
        ):

            chunk_id = result.get("chunk_id")

            if chunk_id is None:
                continue

            chunk_id = str(chunk_id)

            if chunk_id in seen_vector_chunks:
                continue

            seen_vector_chunks.add(chunk_id)

            content = (
                result.get("text")
                or result.get("content")
                or result.get("preview")
                or ""
            )

            vector_item = {
                "chunk_id": chunk_id,
                "type": "vector",
                "source": "vector",
                "content": content,

                "original_score": float(
                    result.get("score", 0.0)
                ),

                "fusion_score": self._rrf_score(
                    rank,
                    self.vector_weight,
                ),

                "vector_rank": rank,
                "metadata": result,

                # Will be populated if graph evidence
                # references this same chunk.
                "graph_matches": [],
            }

            vector_by_chunk[chunk_id] = vector_item

        # ==============================================================
        # Graph results
        # ==============================================================

        graph_items: List[Dict] = []

        seen_graph_relations: Set[
            Tuple[str, str, str]
        ] = set()

        for rank, result in enumerate(
            graph_results,
            start=1,
        ):

            source = str(
                result.get("source", "")
            )

            relation = str(
                result.get("relation", "")
            )

            target = str(
                result.get("target", "")
            )

            graph_key = (
                source.strip().lower(),
                relation.strip().lower(),
                target.strip().lower(),
            )

            if graph_key in seen_graph_relations:
                continue

            seen_graph_relations.add(graph_key)

            relation_text = (
                f"{source} {relation} {target}"
            ).strip()

            graph_score = float(
                result.get(
                    "graph_score",
                    result.get("score", 0.0),
                )
            )

            graph_item = {
                "chunk_id": None,
                "type": "graph",
                "source": "document_graph",
                "content": relation_text,

                "original_score": graph_score,

                "fusion_score": self._rrf_score(
                    rank,
                    self.graph_weight,
                ),

                "graph_rank": rank,
                "graph_data": result,

                "graph_score": graph_score,
            }

            # ----------------------------------------------------------
            # Preserve graph provenance
            # ----------------------------------------------------------

            graph_chunk_id = (
                result.get("chunk_id")
                or result.get("source_chunk_id")
            )

            if graph_chunk_id is None:

                chunk_ids = result.get(
                    "chunk_ids",
                    [],
                )

                if chunk_ids:
                    graph_chunk_id = chunk_ids[0]

            if graph_chunk_id is not None:

                graph_item["chunk_id"] = str(
                    graph_chunk_id
                )

            graph_items.append(
                graph_item
            )

        # ==============================================================
        # Merge graph evidence into matching vector chunks
        # ==============================================================

        merged_chunk_ids: Set[str] = set()

        for graph_item in graph_items:

            chunk_id = graph_item.get(
                "chunk_id"
            )

            if chunk_id is None:
                continue

            chunk_id = str(chunk_id)

            vector_item = vector_by_chunk.get(
                chunk_id
            )

            if vector_item is None:
                continue

            # ----------------------------------------------------------
            # Same chunk found by both retrieval systems.
            # ----------------------------------------------------------

            vector_rank = vector_item.get(
                "vector_rank",
                999,
            )

            graph_rank = graph_item.get(
                "graph_rank",
                999,
            )

            graph_score = graph_item.get(
                "graph_score",
                0.0,
            )

            provenance_bonus = (
                self._provenance_bonus(
                    vector_rank,
                    graph_rank,
                )
            )

            graph_bonus = (
                self._normalize_graph_score(
                    graph_score
                )
            )

            # ----------------------------------------------------------
            # Combine the signals.
            # ----------------------------------------------------------

            vector_item["fusion_score"] += (
                graph_bonus
                + provenance_bonus
            )

            # ----------------------------------------------------------
            # Preserve graph evidence.
            # ----------------------------------------------------------

            vector_item[
                "graph_matches"
            ].append(
                {
                    "relation": graph_item[
                        "content"
                    ],
                    "graph_score": graph_score,
                    "graph_rank": graph_rank,
                    "graph_data": graph_item.get(
                        "graph_data",
                        {},
                    ),
                }
            )

            # ----------------------------------------------------------
            # Mark this graph item as merged.
            # ----------------------------------------------------------

            merged_chunk_ids.add(
                chunk_id
            )

        # ==============================================================
        # Build final result list
        # ==============================================================

        fused_context: List[Dict] = []

        # --------------------------------------------------------------
        # Add vector results
        # --------------------------------------------------------------

        for chunk_id, vector_item in (
            vector_by_chunk.items()
        ):

            # If graph evidence also found this chunk,
            # upgrade the result to hybrid.
            if vector_item["graph_matches"]:

                vector_item["type"] = "hybrid"

                vector_item["source"] = (
                    "vector+graph"
                )

            fused_context.append(
                vector_item
            )

        # --------------------------------------------------------------
        # Add graph-only results
        # --------------------------------------------------------------

        for graph_item in graph_items:

            chunk_id = graph_item.get(
                "chunk_id"
            )

            # If this graph result was merged into a vector
            # result, do not add a duplicate result.
            if (
                chunk_id is not None
                and str(chunk_id)
                in merged_chunk_ids
            ):
                continue

            # ----------------------------------------------------------
            # Graph-only result.
            # ----------------------------------------------------------

            graph_score = graph_item.get(
                "graph_score",
                0.0,
            )

            graph_item["fusion_score"] += (
                self._normalize_graph_score(
                    graph_score
                )
            )

            fused_context.append(
                graph_item
            )

        # ==============================================================
        # Sort by final fusion score
        # ==============================================================

        fused_context.sort(
            key=lambda item: item.get(
                "fusion_score",
                0.0,
            ),
            reverse=True,
        )

        # ==============================================================
        # Debug information
        # ==============================================================

        print(
            "\nContext Fusion:"
        )

        for rank, item in enumerate(
            fused_context[:10],
            start=1,
        ):

            print(
                f"{rank}.",
                item.get("type"),
                "|",
                round(
                    item.get(
                        "fusion_score",
                        0.0,
                    ),
                    6,
                ),
                "| chunk:",
                item.get("chunk_id"),
            )

            if item.get("graph_matches"):

                print(
                    "   Graph evidence:",
                    len(
                        item[
                            "graph_matches"
                        ]
                    ),
                )

        return fused_context