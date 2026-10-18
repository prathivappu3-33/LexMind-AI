"""
Query Processor

Processes user queries before retrieval.

Pipeline

User Query
    │
    ▼
Entity Extraction
    ▼
Entity Cleaning
    ▼
Phrase Merging
    ▼
BGE Embedding
    ▼
Structured Query Object
"""

from typing import Dict

import numpy as np

from backend.embeddings.encoder import EmbeddingEncoder
from backend.knowledge_graph.entity_extractor import EntityExtractor
from backend.knowledge_graph.entity_postprocessor import EntityPostProcessor
from backend.knowledge_graph.phrase_merger import PhraseMerger


class QueryProcessor:
    """
    Processes a user query into a structured representation
    for Hybrid Retrieval.
    """

    def __init__(self):

        print("\nInitializing Query Processor...\n")

        self.entity_extractor = EntityExtractor()
        self.entity_postprocessor = EntityPostProcessor()
        self.phrase_merger = PhraseMerger()

        self.embedding_encoder = EmbeddingEncoder()

        print("\nQuery Processor Ready!\n")

    def process(self, query: str) -> Dict:

        query = query.strip()

        # -----------------------------
        # Entity Extraction
        # -----------------------------

        raw_entities = self.entity_extractor.extract_entities(
            query
        )

        clean_entities = self.entity_postprocessor.process(
            raw_entities
        )

        merged_entities = self.phrase_merger.merge(
            query,
            clean_entities
        )

        # -----------------------------
        # Embedding
        # -----------------------------

        embedding = self.embedding_encoder.encode(
            query
        )

        return {

            "original_query": query,

            "clean_query": query,

            "entities": merged_entities,

            "embedding": embedding

        }