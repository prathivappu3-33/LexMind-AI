"""
Vector Retriever

Handles semantic retrieval using the vector database.
"""

from typing import Dict, List

from backend.retrieval.query_processor import QueryProcessor
from backend.vector_db.search import search_embeddings


class VectorRetriever:
    """
    Retrieves relevant document chunks using semantic vector search.
    """

    def __init__(self):
        self.query_processor = QueryProcessor()

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        min_score: float = 0.70
    ) -> List[Dict]:
        """
        Retrieve relevant chunks for a user query.
        """

        # -----------------------------
        # Process query
        # -----------------------------
        query_data = self.query_processor.process(query)

        embedding = query_data["embedding"]

        # -----------------------------
        # Vector search
        # -----------------------------
        results = search_embeddings(
            embedding=embedding,
            top_k=top_k
        )

        # -----------------------------
        # Post-processing
        # -----------------------------
        results = self._filter_results(
            results,
            min_score
        )

        results = self._deduplicate_results(results)

        return results

    def _filter_results(
        self,
        results: List[Dict],
        min_score: float
    ) -> List[Dict]:
        """
        Remove low-confidence results.
        """

        return [
            result
            for result in results
            if result["score"] >= min_score
        ]

    def _deduplicate_results(
        self,
        results: List[Dict]
    ) -> List[Dict]:
        """
        Remove duplicate chunks.
        """

        unique = []
        seen = set()

        for result in results:

            chunk_id = result["chunk_id"]

            if chunk_id in seen:
                continue

            seen.add(chunk_id)
            unique.append(result)

        return unique   